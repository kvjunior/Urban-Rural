"""
calibration.py  --  Stage 2 of 4.

Fits the channel-resolved, stratum-interacted satisfaction-response model that
turns the multi-agent environment from a toy into an empirically grounded
simulation.

WHAT IS ESTIMATED
-----------------
For each respondent i, satisfaction S_i is modelled as

    S_i = b0
        + sum_c [ beta_c        * channel_{c,i}                 ]   # main effects
        + sum_c [ gamma_c       * channel_{c,i} * rural_i        ]   # x stratum
        + delta_service  * service_i
        + delta_trust    * trust_i
        + delta_integ    * integrity_i
        + delta_ctrl     * ctrl_i
        + delta_rural    * rural_i
        + eps_i

The quantities the paper turns on are the per-channel, per-stratum marginal
effects:  m_{c,urban} = beta_c  and  m_{c,rural} = beta_c + gamma_c.
These become the citizen response coefficients inside environment.py.

ESTIMATION
----------
* Weighted ridge regression (intercept unregularised) for a stable,
  interpretable headline model -- closed form, no external optimiser. The
  survey household weight enters the loss, so the estimate is representative.
* Goodness of fit is reported both unweighted (`r2`) and weighted
  (`r2_weighted`); the weighted figure is the one consistent with the weighted
  estimator, the unweighted figure is comparable with the GBM cross-check.
* A gradient-boosting cross-check reports how much non-linear structure is
  left on the table (a robustness statistic, not used by the simulation).
* Non-parametric bootstrap gives confidence intervals on every marginal
  effect. NOTE: this is a respondent-level i.i.d. bootstrap; it does not
  resample primary sampling units, so for Afrobarometer's clustered design the
  resulting CIs may be somewhat anti-conservative. A design-correct cluster
  bootstrap would require carrying the PSU/EA identifier through
  data_pipeline.py; this limitation should be disclosed in the manuscript.
* Stratum attribute profiles (means/std of every construct by urban/rural) are
  WEIGHTED, matching the manuscript's statement that descriptive estimates use
  the household weight.
* The model is fit POOLED (all retained countries) for the headline
  calibration and once PER COUNTRY for the cross-country robustness sweep.

Output: data/processed/calibration.json  (consumed by environment.py).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Construct names. Kept in sync with data_pipeline.py / environment.py; the
# column order in the design matrix is fixed and named, so coefficients can be
# read back unambiguously regardless of this list's ordering.
CHANNELS: List[str] = ["t_school", "t_localgov", "t_contract"]
MEDIATORS: List[str] = ["service", "trust", "integrity", "ctrl"]


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------
def _design_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Build the regression design matrix. Column order is fixed and named so the
    fitted coefficients can be read back unambiguously:

        intercept | channels | channel x rural | mediators | rural
    """
    rural = (df["stratum"].to_numpy() == 2).astype(float)   # 1=rural, 0=urban
    cols: List[np.ndarray] = [np.ones(len(df))]
    names: List[str] = ["intercept"]

    for ch in CHANNELS:                                   # channel main effects
        cols.append(df[ch].to_numpy()); names.append(ch)
    for ch in CHANNELS:                                   # channel x rural
        cols.append(df[ch].to_numpy() * rural); names.append(f"{ch}:rural")

    for med in MEDIATORS:
        cols.append(df[med].to_numpy()); names.append(med)
    cols.append(rural); names.append("rural")

    return np.column_stack(cols), names


def _ridge_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray,
               lam: float) -> np.ndarray:
    """
    Weighted ridge regression in closed form; the intercept is not penalised.

    Minimises  sum_i w_i (y_i - x_i.beta)^2 + lam * ||beta_{1:}||^2.
    With lam > 0 the normal-equation matrix is positive definite for every
    non-degenerate design, so np.linalg.solve is safe even on small per-country
    or bootstrap subsamples.
    """
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    p = X.shape[1]
    P = np.eye(p) * lam
    P[0, 0] = 0.0                                         # leave intercept free
    return np.linalg.solve(Xw.T @ Xw + P, Xw.T @ yw)


def _fit_stats(X: np.ndarray, y: np.ndarray, w: np.ndarray,
               beta: np.ndarray) -> Tuple[float, float]:
    """
    Return (R^2, weighted R^2) for a fitted model.

    `r2` measures fit to the realised sample; `r2_weighted` uses the survey
    weights and is the figure consistent with the weighted estimator.
    """
    resid = y - X @ beta
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    sw = float(np.sum(w))
    if sw > 0:
        yw_mean = float(np.sum(w * y) / sw)
        wss_res = float(np.sum(w * resid ** 2))
        wss_tot = float(np.sum(w * (y - yw_mean) ** 2))
        r2_w = 1.0 - wss_res / wss_tot if wss_tot > 0 else 0.0
    else:
        r2_w = float("nan")
    return float(r2), float(r2_w)


# ---------------------------------------------------------------------------
# Marginal effects
# ---------------------------------------------------------------------------
def _marginal_effects(beta: np.ndarray, names: List[str]) -> Dict[str, Dict]:
    """
    Convert raw coefficients into the per-channel, per-stratum marginal effects
    the environment needs:  urban = main effect,  rural = main + interaction.
    """
    idx = {n: i for i, n in enumerate(names)}
    eff: Dict[str, Dict] = {}
    for ch in CHANNELS:
        main = float(beta[idx[ch]])
        inter = float(beta[idx[f"{ch}:rural"]])
        eff[ch] = {"urban": main, "rural": main + inter, "interaction": inter}
    return eff


def _bootstrap_ci(df: pd.DataFrame, lam: float, iters: int,
                  rng: np.random.Generator) -> Dict[str, Dict]:
    """
    Non-parametric bootstrap CIs (2.5/97.5%) for every marginal effect.

    Respondents are resampled uniformly with replacement; each resample is
    re-fit with the same weighted ridge. This is a respondent-level bootstrap:
    it does NOT resample primary sampling units, so the CIs may be somewhat
    anti-conservative for Afrobarometer's clustered, multi-stage design. A
    cluster bootstrap would need the PSU/EA identifier (not currently carried
    through the analysis frame); disclose this in the manuscript.
    """
    boot = {ch: {"urban": [], "rural": []} for ch in CHANNELS}
    n = len(df)
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        s = df.iloc[idx]
        X, names = _design_matrix(s)
        beta = _ridge_fit(X, s["satisfaction"].to_numpy(),
                          s["weight"].to_numpy(), lam)
        eff = _marginal_effects(beta, names)
        for ch in CHANNELS:
            boot[ch]["urban"].append(eff[ch]["urban"])
            boot[ch]["rural"].append(eff[ch]["rural"])
    out: Dict[str, Dict] = {}
    for ch in CHANNELS:
        out[ch] = {}
        for strat in ("urban", "rural"):
            arr = np.array(boot[ch][strat])
            out[ch][strat] = [float(np.percentile(arr, 2.5)),
                              float(np.percentile(arr, 97.5))]
    return out


# ---------------------------------------------------------------------------
# Optional non-linear cross-check
# ---------------------------------------------------------------------------
def _gbm_crosscheck(df: pd.DataFrame, cc: dict, seed: int) -> float:
    """
    Report cross-validated GBM R^2 as an upper reference for non-linear
    structure. A robustness statistic only -- never consumed by the simulation.
    Returns NaN if scikit-learn is unavailable.
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import KFold, cross_val_score
    except Exception:                                        # noqa: BLE001
        return float("nan")
    X, _ = _design_matrix(df)
    y = df["satisfaction"].to_numpy()
    gbm = GradientBoostingRegressor(
        random_state=0,
        n_estimators=int(cc.get("gbm_n_estimators", 200)),
        max_depth=int(cc.get("gbm_max_depth", 3)),
    )
    # SHUFFLED CV: the analysis frame is ordered by country, so an unshuffled
    # KFold would place whole countries in single folds and bias the estimate.
    kf = KFold(n_splits=int(cc.get("gbm_cv_folds", 4)),
               shuffle=True, random_state=seed)
    scores = cross_val_score(gbm, X[:, 1:], y, cv=kf,        # drop intercept col
                             scoring="r2")
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Stratum attribute profiles (priors for the environment's citizen agents)
# ---------------------------------------------------------------------------
def _weighted_mean_std(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Survey-weighted mean and (population-style) standard deviation."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    sw = float(np.sum(w))
    if sw <= 0.0:
        return float("nan"), float("nan")
    mu = float(np.sum(w * x) / sw)
    var = float(np.sum(w * (x - mu) ** 2) / sw)
    return mu, float(np.sqrt(max(var, 0.0)))


def _attribute_profiles(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Weighted mean/std of every construct by stratum -> citizen-agent priors.

    Weighted (not raw) to match the manuscript: descriptive estimates use the
    household weight. A degenerate (zero / non-finite) std falls back to 0.1 so
    the environment's prior sampling stays well defined.
    """
    prof: Dict[str, Dict] = {}
    for label, code in (("urban", 1), ("rural", 2)):
        sub = df[df["stratum"] == code]
        if len(sub) == 0:
            raise ValueError(f"[calib] no '{label}' respondents in the pooled "
                             f"frame -- cannot build attribute profiles.")
        w = sub["weight"].to_numpy()
        prof[label] = {}
        for m in MEDIATORS + CHANNELS:
            mu, sd = _weighted_mean_std(sub[m].to_numpy(), w)
            if not np.isfinite(sd) or sd == 0.0:
                sd = 0.1
            prof[label][m] = {"mean": float(mu), "std": float(sd)}
        sat_mu, _ = _weighted_mean_std(sub["satisfaction"].to_numpy(), w)
        prof[label]["satisfaction_mean"] = float(sat_mu)
        prof[label]["n"] = int(len(sub))
    return prof


def _fit_one(df: pd.DataFrame, lam: float) -> Dict:
    """Fit the weighted ridge model on one frame; return effects + fit stats."""
    X, names = _design_matrix(df)
    y = df["satisfaction"].to_numpy()
    w = df["weight"].to_numpy()
    beta = _ridge_fit(X, y, w, lam)
    r2, r2_w = _fit_stats(X, y, w, beta)
    return {
        "coefficients": dict(zip(names, beta.tolist())),
        "marginal_effects": _marginal_effects(beta, names),
        "r2": r2,
        "r2_weighted": r2_w,
        "n": int(len(df)),
    }


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _check_frame(df: pd.DataFrame) -> None:
    """Fail loudly if the analysis frame does not satisfy the model's contract."""
    needed = ["satisfaction", "stratum", "country", "weight"] + CHANNELS + MEDIATORS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"[calib] analysis frame missing columns: {missing}")
    if len(df) == 0:
        raise ValueError("[calib] analysis frame is empty.")
    n_nan = int(df[needed].isna().to_numpy().sum())
    if n_nan:
        raise ValueError(f"[calib] {n_nan} NaN value(s) in modelling columns -- "
                         f"run data_pipeline.py (listwise deletion) first.")
    w = df["weight"].to_numpy(dtype=float)
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("[calib] survey weights must be finite and non-negative.")
    strata = {int(s) for s in df["stratum"].unique()}
    if strata != {1, 2}:
        raise ValueError(f"[calib] the pooled frame must contain both strata "
                         f"(urban=1, rural=2); got {sorted(strata)}.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_calibration(cfg: dict, df: pd.DataFrame) -> Dict:
    """
    Fit the pooled headline model + per-country models, attach bootstrap CIs and
    stratum profiles, and persist everything to calibration.json.
    """
    cc = cfg["calibration"]
    lam = cc["ridge_lambda"]
    seed = int(cfg["project"]["seed"])
    rng = np.random.default_rng(seed)

    _check_frame(df)

    print("[calib] fitting pooled headline model (all retained countries)")
    pooled = _fit_one(df, lam)
    pooled["bootstrap_ci"] = _bootstrap_ci(df, lam, cc["bootstrap_iters"], rng)
    print(f"[calib] ridge R^2={pooled['r2']:.3f}  "
          f"(weighted {pooled['r2_weighted']:.3f})")

    if cc.get("gbm_crosscheck", False):
        pooled["gbm_r2"] = _gbm_crosscheck(df, cc, seed)
        print(f"[calib] GBM cross-check R^2={pooled['gbm_r2']:.3f}  "
              f"(non-linear-structure reference)")

    # Per-country models for the cross-country robustness sweep (E5).
    # The threshold is data.min_country_n -- the SAME knob data_pipeline.py uses
    # to drop small countries -- not a separate hard-coded number. For real
    # data every retained country already clears it (data_pipeline filtered on
    # it); the re-check simply keeps the two stages consistent.
    per_country: Dict[str, Dict] = {}
    if cc.get("per_country", False):
        min_n = cfg["data"]["min_country_n"]
        for c, sub in df.groupby("country"):
            if len(sub) >= min_n:
                per_country[str(int(c))] = _fit_one(sub, lam)
        print(f"[calib] fitted {len(per_country)} per-country models "
              f"(>= {min_n} valid rows each)")
        if not per_country:
            print("[calib] WARNING: no country met the per-country threshold; "
                  "the cross-country sweep (E5) will be empty. (Expected on the "
                  "small synthetic dataset; use the real data for E5.)")

    calibration = {
        "pooled": pooled,
        "per_country": per_country,
        "attribute_profiles": _attribute_profiles(df),
        "channels": CHANNELS,
        "meta": {"n_total": int(len(df)),
                 "n_countries": int(df["country"].nunique()),
                 "n_per_country_models": len(per_country),
                 "ridge_lambda": float(lam),
                 "seed": seed},
    }

    os.makedirs(cfg["paths"]["processed_dir"], exist_ok=True)
    out = cfg["paths"]["calibration"]
    with open(out, "w") as fh:
        json.dump(calibration, fh, indent=2)
    print(f"[calib] calibration written -> {out}")

    # human-readable summary of the headline finding
    me = pooled["marginal_effects"]
    print("[calib] per-channel marginal effects (satisfaction points):")
    for ch in CHANNELS:
        print(f"         {ch:12s} urban={me[ch]['urban']:+6.2f}  "
              f"rural={me[ch]['rural']:+6.2f}")
    return calibration


if __name__ == "__main__":
    import yaml
    from data_pipeline import build_analysis_frame
    with open("config/config.yaml") as fh:
        _cfg = yaml.safe_load(fh)
    _df = build_analysis_frame(_cfg)
    run_calibration(_cfg, _df)

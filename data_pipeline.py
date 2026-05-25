"""
data_pipeline.py  --  Stage 1 of 4.

Ingests the Afrobarometer Round 9 merged dataset (39 countries, 53,444
respondents) and produces the clean, channel-resolved analysis frame consumed
by `calibration.py`.

CENTRAL DESIGN CHOICE
---------------------
Transparency is *not* treated as a single national dial. Afrobarometer Round 9
measures access-to-information through THREE concrete, separately answered
acts (Q35A, Q35B, Q35C). This module keeps them as three distinct CHANNELS,
because the paper's contribution rests on the claim that urban and rural
citizens convert different channels into satisfaction at different rates.

MISSING DATA
------------
The manuscript commits to *listwise deletion*: a respondent is retained only
if every modelling variable is observed. This module enforces that on the FULL
modelling set, then drops countries with fewer than `min_country_n` retained
rows -- so the per-country counts that feed calibration match the manuscript's
Data section exactly.

REPRODUCIBILITY
---------------
If the real .sav file is absent, a synthetic dataset with an identical schema
and a known channel-resolved data-generating process is produced, so the whole
pipeline runs end-to-end before the user obtains the data. If the real file is
*present* but fails to load, the pipeline raises rather than silently falling
back -- synthetic results must never be mislabelled as real.

Data source (free, no registration): https://www.afrobarometer.org/data/
Citation: Afrobarometer Project (2023). Afrobarometer Round 9 merged dataset
[dataset]. DOI: 10.25828/bs9k-zy70.
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Analysis-frame schema -- single source of truth
# ---------------------------------------------------------------------------
CHANNELS: List[str] = ["t_school", "t_localgov", "t_contract"]
MEDIATORS: List[str] = ["service", "trust", "integrity", "ctrl"]

# every column the downstream model conditions on; listwise deletion is applied
# across exactly this set, and the frame is validated to be complete on it.
MODELING_COLUMNS: List[str] = (
    ["satisfaction", "stratum", "country", "weight"] + CHANNELS + MEDIATORS
)
# canonical column order; real and synthetic frames are emitted identically.
OUTPUT_COLUMNS: List[str] = (
    ["country", "stratum", "weight"] + CHANNELS + MEDIATORS + ["satisfaction"]
)
# Afrobarometer transparency-channel keys (config) -> analysis-frame short names
_CHANNEL_SHORT: Dict[str, str] = {
    "budget_school":   "t_school",
    "budget_localgov": "t_localgov",
    "contracts":       "t_contract",
}


# ---------------------------------------------------------------------------
# Recoding helpers
# ---------------------------------------------------------------------------
def _clean_numeric(series: pd.Series, missing_codes: List[int]) -> pd.Series:
    """Coerce to numeric and blank out Afrobarometer missing/DK/refused codes."""
    s = pd.to_numeric(series, errors="coerce")
    # NaN is never `in` missing_codes, so genuine NaNs are preserved by ~isin.
    return s.where(~s.isin(missing_codes))


def _unit_scale(series: pd.Series) -> pd.Series:
    """
    Min-max scale a cleaned series to [0, 1].

    A degenerate (empty or constant) column yields a neutral 0.5 for every
    OBSERVED value, but missing entries are kept as NaN so that listwise
    deletion still removes respondents who did not answer the item.

    Note: scaling uses the observed extrema of the pooled sample. This is the
    intended behaviour, but it means the scale is sample-dependent; it is not a
    bug, simply a modelling choice worth being aware of.
    """
    lo, hi = series.min(), series.max()
    if pd.isna(lo) or hi == lo:
        return pd.Series(np.where(series.isna(), np.nan, 0.5), index=series.index)
    return (series - lo) / (hi - lo)


def _mean_of_items(df: pd.DataFrame, cols: List[str],
                   missing_codes: List[int]) -> pd.Series:
    """
    Average several Likert items into one [0,1] construct score.

    Each item is cleaned and unit-scaled independently, then averaged with NaN
    skipped, so a respondent who answered some-but-not-all items still receives
    a score. A respondent who answered NONE receives NaN (and is dropped by the
    listwise deletion downstream).
    """
    present = [c for c in cols if c in df.columns]
    if not present:
        raise KeyError(f"none of {cols} found in dataset")
    if len(present) < len(cols):
        # Building a construct from a subset silently changes its definition;
        # surface it rather than hide it.
        print(f"[data] WARNING: construct items {sorted(set(cols) - set(present))}"
              f" absent; building construct from {present} only.")
    scaled = [_unit_scale(_clean_numeric(df[c], missing_codes)) for c in present]
    return pd.concat(scaled, axis=1).mean(axis=1)


def _require_columns(available, vm: dict) -> None:
    """
    Fail early, with a single clear message, if the variable map does not match
    the dataset -- far more useful than a deep KeyError mid-recode.
    """
    available = set(available)
    strict = [vm["stratum"], vm["country"], vm["trust"], vm["corruption"]]
    strict += list(vm["transparency_channels"].values())
    missing_strict = [c for c in strict if c not in available]

    problems = []
    if missing_strict:
        problems.append(f"required columns absent from the .sav: {missing_strict}")
    for name in ("satisfaction", "service", "controls"):
        items = vm[name]
        if not any(c in available for c in items):
            problems.append(f"construct '{name}' has no available items "
                             f"(looked for {items})")
    if problems:
        raise KeyError("[data] data.var_map does not match the dataset -- "
                       + "; ".join(problems)
                       + ". Fix data.var_map in config.yaml.")


# ---------------------------------------------------------------------------
# Synthetic fallback -- identical schema, known channel-resolved DGP
# ---------------------------------------------------------------------------
def generate_synthetic(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """
    Build a synthetic analysis frame whose data-generating process embodies the
    paper's hypothesis: the three transparency channels have DIFFERENT marginal
    effects, and those effects DIFFER by urban/rural stratum. Rural citizens are
    more sensitive to the local-government-budget channel (information is
    otherwise scarce there); urban citizens are more sensitive to the contract
    channel. A single-dial model cannot represent this; the channel-resolved
    model recovers it.

    The DGP coefficients below are deliberately hard-coded: they are the known
    ground truth that `test_suite.py` checks calibration can recover, not
    tunable experiment parameters.
    """
    n = cfg["data"]["synthetic_n"]
    urban = rng.binomial(1, 0.45, n)                       # 1 urban, 0 rural

    # three transparency channels, each in [0,1]; urban areas slightly higher
    t_school   = np.clip(0.35 + 0.12 * urban + rng.normal(0, 0.16, n), 0, 1)
    t_localgov = np.clip(0.32 + 0.14 * urban + rng.normal(0, 0.16, n), 0, 1)
    t_contract = np.clip(0.28 + 0.16 * urban + rng.normal(0, 0.16, n), 0, 1)

    service   = np.clip(0.42 + 0.13 * urban + rng.normal(0, 0.17, n), 0, 1)
    trust     = np.clip(0.55 - 0.05 * urban + rng.normal(0, 0.18, n), 0, 1)
    integrity = np.clip(0.50 - 0.04 * urban + rng.normal(0, 0.18, n), 0, 1)
    ctrl      = np.clip(rng.normal(0.5, 0.2, n), 0, 1)

    # TRUE channel-resolved, stratum-interacted satisfaction model.
    # Rural: local-gov-budget channel dominates. Urban: contract channel matters
    # more. This asymmetry is the empirical object the paper is about.
    sat = (
        18.0
        + (10 + 4 * urban)        * t_school          # mild urban tilt
        + (24 - 14 * urban)       * t_localgov         # strong RURAL channel
        + (8 + 14 * urban)        * t_contract         # strong URBAN channel
        + 15 * service + 10 * trust + 12 * integrity
        + 6 * ctrl + 3 * urban
        + rng.normal(0, 8, n)
    )
    sat = np.clip(sat, 0, 100)

    df = pd.DataFrame({
        "country": rng.integers(2, 41, n),             # 39 country codes, 2..40
        "stratum": np.where(urban == 1, 1, 2),         # 1=urban, 2=rural
        "weight": np.ones(n),
        "t_school": t_school, "t_localgov": t_localgov, "t_contract": t_contract,
        "service": service, "trust": trust, "integrity": integrity,
        "ctrl": ctrl, "satisfaction": sat,
    })
    return df[OUTPUT_COLUMNS]


# ---------------------------------------------------------------------------
# Real Afrobarometer .sav loader
# ---------------------------------------------------------------------------
def load_real(cfg: dict) -> pd.DataFrame:
    """
    Read the Afrobarometer .sav file and recode it into the analysis frame.
    Requires pyreadstat (declared in requirements.txt).

    ITEM ORIENTATION -- VERIFY AGAINST THE CODEBOOK.
    Every predictor produced here is assumed to be coded so that HIGHER = MORE
    FAVOURABLE (more transparency, more trust, better service, higher
    satisfaction). Perceived corruption is the one item known to run the other
    way, so `integrity` is taken as its complement. If the Round 9 codebook
    shows any of Q35A/B/C, Q31, Q47C, Q37D or Q46G/I run high=unfavourable,
    reverse it here -- otherwise the calibrated marginal effects are sign-flipped.
    """
    import pyreadstat

    path = cfg["paths"]["raw_data"]
    print(f"[data] reading {path}")
    raw, _ = pyreadstat.read_sav(path)
    vm = cfg["data"]["var_map"]
    miss = cfg["data"]["missing_codes"]

    _require_columns(raw.columns, vm)

    df = pd.DataFrame(index=raw.index)

    # -- stratum: 1=urban, 2=rural; peri-urban (3) folded into urban ---------
    # Clean first so missing/DK codes cannot masquerade as a valid stratum.
    # Anything that is neither rural (2) nor urban/peri-urban (1 or 3) -- a
    # missing value included -- becomes NaN and is removed by the listwise
    # deletion below. (A previous version coerced every non-2 value, NaN
    # included, to urban, silently mis-stratifying respondents whose URBRUR
    # code was missing.)
    strat = _clean_numeric(raw[vm["stratum"]], miss)
    df["stratum"] = np.where(
        strat == 2, 2.0,
        np.where(strat.isin([1, 3]), 1.0, np.nan))

    df["country"] = pd.to_numeric(raw[vm["country"]], errors="coerce")

    # cross-national household weight (unit weights if the column is absent)
    if vm["weight"] in raw.columns:
        df["weight"] = pd.to_numeric(raw[vm["weight"]], errors="coerce")
    else:
        print(f"[data] WARNING: weight column '{vm['weight']}' absent; "
              f"using unit weights.")
        df["weight"] = 1.0

    # -- outcome: citizen satisfaction, rescaled to 0-100 -------------------
    df["satisfaction"] = _mean_of_items(raw, vm["satisfaction"], miss) * 100.0

    # -- the three transparency channels (kept separate) --------------------
    for key, col in vm["transparency_channels"].items():
        df[_CHANNEL_SHORT[key]] = _unit_scale(_clean_numeric(raw[col], miss))

    # -- mediating governance constructs ------------------------------------
    df["service"] = _mean_of_items(raw, vm["service"], miss)
    df["trust"] = _unit_scale(_clean_numeric(raw[vm["trust"]], miss))
    # integrity = complement of perceived corruption (so all predictors are "good")
    df["integrity"] = 1.0 - _unit_scale(_clean_numeric(raw[vm["corruption"]], miss))

    # -- individual controls collapsed to one standardised index -----------
    df["ctrl"] = _mean_of_items(raw, vm["controls"], miss)

    # -- listwise deletion across EVERY modelling column --------------------
    # The manuscript's Data section commits to listwise deletion and filters
    # countries on the count of valid rows AFTER it. Dropping on the full
    # modelling set (not only outcome/stratum/country) keeps the analysis frame
    # complete and makes the per-country counts match the manuscript.
    n_raw = len(df)
    df = df.dropna(subset=MODELING_COLUMNS).reset_index(drop=True)
    print(f"[data] listwise deletion: {n_raw} -> {len(df)} respondents "
          f"({n_raw - len(df)} dropped for missing values)")

    # -- drop countries with too few valid rows -----------------------------
    min_n = cfg["data"]["min_country_n"]
    counts = df["country"].value_counts()
    keep = counts[counts >= min_n].index
    df = df[df["country"].isin(keep)].reset_index(drop=True)
    print(f"[data] {len(df)} respondents across {df['country'].nunique()} "
          f"countries (>= {min_n} valid rows each)")
    return df


# ---------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------
def _validate_frame(df: pd.DataFrame, source: str) -> None:
    """Assert the contract downstream stages rely on; raise loudly on violation."""
    missing = [c for c in MODELING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[data] analysis frame missing columns: {missing}")
    if len(df) == 0:
        raise ValueError("[data] analysis frame is empty -- check min_country_n "
                          "and the dataset.")
    n_nan = int(df[MODELING_COLUMNS].isna().to_numpy().sum())
    if n_nan:
        raise ValueError(f"[data] {n_nan} NaN value(s) remain in modelling "
                          f"columns after construction -- this is a bug.")
    strata = {int(s) for s in df["stratum"].unique()}
    if not strata <= {1, 2}:
        raise ValueError(f"[data] stratum must be in {{1, 2}}; got {sorted(strata)}.")
    tol = 1e-6
    if not df["satisfaction"].between(-tol, 100 + tol).all():
        raise ValueError("[data] satisfaction outside [0, 100].")
    for c in CHANNELS + MEDIATORS:
        if not df[c].between(-tol, 1 + tol).all():
            raise ValueError(f"[data] '{c}' outside [0, 1].")
    if (df["weight"] <= 0).any():
        print("[data] WARNING: non-positive survey weights present.")
    print(f"[data] validation OK [{source}]: {len(df)} rows, "
          f"{df['country'].nunique()} countries, strata={sorted(strata)}.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_analysis_frame(cfg: dict) -> pd.DataFrame:
    """
    Produce the analysis frame: real data if the .sav is present, otherwise the
    synthetic fallback. Validates the frame and persists it for inspection.
    """
    rng = np.random.default_rng(cfg["project"]["seed"])
    raw_path = cfg["paths"]["raw_data"]

    if os.path.exists(raw_path):
        print(f"[data] real dataset found -> loading {raw_path}")
        try:
            df = load_real(cfg)
        except Exception as exc:                              # noqa: BLE001
            # Fail loud. We do NOT fall back to synthetic data when the real
            # file is present: doing so would silently relabel synthetic
            # results as real. The synthetic fallback is reserved for the case
            # where the real file is genuinely absent (the branch below).
            raise RuntimeError(
                f"[data] failed to load the real Afrobarometer dataset at "
                f"{raw_path}. Refusing to fall back to synthetic data while "
                f"the real file is present, to avoid mislabelling synthetic "
                f"results as real. Fix the file or data.var_map in "
                f"config.yaml and retry. Original error: {exc!r}"
            ) from exc
        source = "real"
    else:
        print(f"[data] {raw_path} not found -> generating synthetic dataset.")
        print("[data] (place the real .sav at that path to run on Afrobarometer"
              " data; reported results must use the real dataset.)")
        df = generate_synthetic(cfg, rng)
        source = "synthetic"

    # harmonise dtypes so real and synthetic frames are interchangeable
    df["country"] = df["country"].astype("int64")
    df["stratum"] = df["stratum"].astype("int64")
    df["weight"] = df["weight"].astype("float64")
    df = df[OUTPUT_COLUMNS].reset_index(drop=True)

    _validate_frame(df, source)

    os.makedirs(cfg["paths"]["processed_dir"], exist_ok=True)
    out = cfg["paths"]["analysis_frame"]
    try:
        df.to_parquet(out, index=False)
    except Exception as exc:                                  # noqa: BLE001
        # parquet needs pyarrow; fall back to CSV so the stage still completes.
        out = os.path.splitext(out)[0] + ".csv"
        print(f"[data] parquet unavailable ({exc!r}); writing CSV instead.")
        df.to_csv(out, index=False)
    print(f"[data] analysis frame ({source}, {len(df)} rows) written -> {out}")
    return df


if __name__ == "__main__":
    import yaml
    with open("config/config.yaml") as fh:
        _cfg = yaml.safe_load(fh)
    build_analysis_frame(_cfg)

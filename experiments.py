"""
experiments.py  --  Stage 4a of 4 (run the protocol).

Executes the full experimental protocol and writes raw results to
results/logs/. No figures or LaTeX here -- analysis.py turns these logs into
manuscript artifacts. Separating "run" from "report" keeps expensive runs from
being repeated whenever a figure is restyled.

PROTOCOL
--------
E1  Main comparison      : every baseline + tabular IQL + coordinated MAPPO,
                           each over n_seeds, evaluated on mean satisfaction
                           AND the urban-rural equity gap.
E2  Coordination ablation: difference reward vs naive reward (MAPPO).
E3  Channel ablation     : 3-channel portfolio vs collapsed single dial.
                           This is the experiment that tests the paper's
                           central claim directly.
E4  Equity sensitivity   : sweep lambda_equity -- traces the efficiency/equity
                           frontier and shows whether they truly trade off.
E5  Cross-country        : retrain the policy on each country's OWN calibrated
                           response model (fit per-country in calibration.py)
                           to show the finding is not a pooled artifact.

Every learner is evaluated by the SAME `evaluate()` routine for comparability.

PYTORCH AVAILABILITY
--------------------
The MAPPO contribution needs PyTorch. Availability is probed ONCE, and the
protocol degrades consistently and visibly:
  * E1   -- MAPPO is omitted (tabular is already a separate row; substituting
            would duplicate it).
  * E2   -- skipped entirely (the difference-vs-naive ablation is MAPPO-only).
  * E3-5 -- fall back to the tabular learner so the protocol still runs on a
            GPU-less machine; every affected result records learner='tabular'.
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Dict, List

import numpy as np

from algorithms import build_policy
from calibration import run_calibration
from environment import UrbanRuralGovernanceEnv


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------
def _torch_available() -> bool:
    """True iff PyTorch can be imported (the MAPPO contribution needs it)."""
    try:
        import torch                                         # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Shared evaluation routine
# ---------------------------------------------------------------------------
def evaluate(policy, env, episodes: int) -> Dict[str, float]:
    """
    Roll a (trained or heuristic) policy for several episodes and average the
    end-of-episode welfare diagnostics. Learners act greedily; heuristic rules
    are deterministic and ignore the greedy flag.
    """
    if episodes < 1:
        raise ValueError("evaluate: episodes must be >= 1.")
    sat, gap, urban, rural, glob = [], [], [], [], []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        last = None
        while not done:
            # every policy class shares the act(observations, greedy) signature
            acts = policy.act(obs, greedy=True)
            obs, _, done, info = env.step(acts)
            last = info
        if last is None:                                     # horizon must be >= 1
            raise RuntimeError("evaluate: episode produced no steps "
                               "(environment horizon must be >= 1).")
        sat.append(last["mean_satisfaction"])
        gap.append(last["equity_gap"])
        urban.append(last["urban_satisfaction"])
        rural.append(last["rural_satisfaction"])
        glob.append(last["global_reward"])
    return {
        "mean_satisfaction": float(np.mean(sat)),
        "satisfaction_std": float(np.std(sat)),
        "equity_gap": float(np.mean(gap)),
        "equity_gap_std": float(np.std(gap)),
        "urban_satisfaction": float(np.mean(urban)),
        "rural_satisfaction": float(np.mean(rural)),
        "global_reward": float(np.mean(glob)),
    }


def _make_env(cfg, calibration, seed, channel_resolved=True,
              coordination="difference"):
    return UrbanRuralGovernanceEnv(
        cfg, calibration, seed=seed,
        channel_resolved=channel_resolved, coordination=coordination)


def _train_and_eval(name, cfg, calibration, seed,
                    channel_resolved=True, coordination="difference") -> Dict:
    """Train one policy under one configuration and return its eval metrics."""
    env = _make_env(cfg, calibration, seed, channel_resolved, coordination)
    policy = build_policy(name, env, cfg, seed)
    curve = policy.train() if hasattr(policy, "train") else []
    metrics = evaluate(policy, env, cfg["experiments"]["eval_episodes"])
    metrics["learning_curve"] = curve
    return metrics


# ---------------------------------------------------------------------------
# E1  Main comparison
# ---------------------------------------------------------------------------
def exp_main_comparison(cfg, calibration, mappo_available: bool) -> Dict:
    print("\n=== E1  main comparison ===")
    methods = list(cfg["algorithms"]["baselines"]) + ["tabular"]
    if mappo_available:
        methods.append("mappo")
    else:
        print("[E1] PyTorch unavailable -- MAPPO omitted; "
              "reporting baselines + tabular only.")
    seeds = range(cfg["project"]["n_seeds"])
    results: Dict[str, Dict] = {}
    for m in methods:
        runs = [_train_and_eval(m, cfg, calibration, s) for s in seeds]
        results[m] = _aggregate(runs)
        r = results[m]
        print(f"[E1] {m:13s} sat={r['mean_satisfaction']:.3f}"
              f"+/-{r['mean_satisfaction_sd']:.3f}  "
              f"gap={r['equity_gap']:.3f}")
    return results


# ---------------------------------------------------------------------------
# E2  Coordination ablation  (MAPPO-only)
# ---------------------------------------------------------------------------
def exp_coordination_ablation(cfg, calibration, mappo_available: bool) -> Dict:
    print("\n=== E2  coordination ablation (difference vs naive) ===")
    if not mappo_available:
        print("[E2] skipped -- the coordination ablation requires MAPPO "
              "(PyTorch unavailable).")
        return {}
    out: Dict[str, Dict] = {}
    for coord in cfg["experiments"]["ablations"]["coordination"]:
        runs = [_train_and_eval("mappo", cfg, calibration, s,
                                coordination=coord)
                for s in range(cfg["project"]["n_seeds"])]
        out[coord] = _aggregate(runs)
        print(f"[E2] coordination={coord:11s} "
              f"sat={out[coord]['mean_satisfaction']:.3f}  "
              f"gap={out[coord]['equity_gap']:.3f}")
    return out


# ---------------------------------------------------------------------------
# E3  Channel-resolution ablation  (tests the central claim)
# ---------------------------------------------------------------------------
def exp_channel_ablation(cfg, calibration, mappo_available: bool) -> Dict:
    print("\n=== E3  channel ablation (3-channel portfolio vs single dial) ===")
    learner = "mappo" if mappo_available else "tabular"
    if not mappo_available:
        print("[E3] PyTorch unavailable -- running the ablation with the "
              "tabular learner (recorded as learner='tabular').")
    out: Dict[str, Dict] = {}
    for resolved in cfg["experiments"]["ablations"]["channel_resolved"]:
        label = "channel_resolved" if resolved else "single_dial"
        runs = [_train_and_eval(learner, cfg, calibration, s,
                                channel_resolved=resolved)
                for s in range(cfg["project"]["n_seeds"])]
        out[label] = _aggregate(runs)
        out[label]["learner"] = learner
        print(f"[E3] {label:18s} sat={out[label]['mean_satisfaction']:.3f}  "
              f"gap={out[label]['equity_gap']:.3f}  [{learner}]")
    return out


# ---------------------------------------------------------------------------
# E4  Equity-weight sensitivity
# ---------------------------------------------------------------------------
def exp_equity_sweep(cfg, calibration, mappo_available: bool) -> Dict:
    print("\n=== E4  equity-weight sensitivity ===")
    learner = "mappo" if mappo_available else "tabular"
    if not mappo_available:
        print("[E4] PyTorch unavailable -- sweeping with the tabular learner "
              "(recorded as learner='tabular').")
    out: Dict[str, Dict] = {}
    for lam in cfg["experiments"]["lambda_equity_sweep"]:
        # deep-copy so each lambda gets an isolated config; the shared `cfg` is
        # never mutated.
        cfg_l = copy.deepcopy(cfg)
        cfg_l["environment"]["lambda_equity"] = lam
        runs = [_train_and_eval(learner, cfg_l, calibration, s)
                for s in range(cfg["project"]["n_seeds"])]
        key = f"lambda_{lam}"
        out[key] = _aggregate(runs)
        out[key]["learner"] = learner
        r = out[key]
        print(f"[E4] lambda={lam:<4} sat={r['mean_satisfaction']:.3f}  "
              f"gap={r['equity_gap']:.3f}  [{learner}]")
    return out


# ---------------------------------------------------------------------------
# E5  Cross-country robustness
# ---------------------------------------------------------------------------
def exp_cross_country(cfg, calibration, mappo_available: bool) -> Dict:
    print("\n=== E5  cross-country robustness ===")
    out: Dict[str, Dict] = {}
    per_country = calibration.get("per_country", {})
    if not per_country:
        print("[E5] skipped -- no per-country calibration available. Run "
              "calibration with calibration.per_country=true on the real "
              "dataset (the synthetic dataset has too few rows per country).")
        return out

    learner = "mappo" if mappo_available else "tabular"
    if not mappo_available:
        print("[E5] PyTorch unavailable -- retraining with the tabular learner "
              "(recorded as learner='tabular').")

    n_countries = cfg["experiments"]["robustness_countries"]
    countries = sorted(per_country.keys(),
                       key=lambda c: per_country[c]["n"],
                       reverse=True)[:n_countries]
    # cross-country is expensive; cap the seed count (config-overridable).
    n_seeds = min(int(cfg["experiments"].get("robustness_seeds", 3)),
                  cfg["project"]["n_seeds"])

    for c in countries:
        # A single-country calibration object the environment can consume.
        # NOTE: the per-country MARGINAL EFFECTS are country-specific, but the
        # citizen ATTRIBUTE PROFILES are the pooled ones -- calibration.py does
        # not fit per-country profiles. E5 therefore varies the response model
        # across countries but not the citizen priors. Disclose this in the
        # manuscript, or extend calibration.py to emit per-country profiles.
        cc = {"pooled": per_country[c],
              "per_country": {},
              "attribute_profiles": calibration["attribute_profiles"],
              "channels": calibration["channels"],
              "meta": calibration["meta"]}
        runs = [_train_and_eval(learner, cfg, cc, s) for s in range(n_seeds)]
        out[c] = _aggregate(runs)
        out[c]["learner"] = learner
        print(f"[E5] country {c:>3}  sat={out[c]['mean_satisfaction']:.3f}  "
              f"gap={out[c]['equity_gap']:.3f}  [{learner}]")
    return out


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------
def _aggregate(runs: List[Dict]) -> Dict:
    """Mean + standard deviation across seed repetitions for each metric."""
    if not runs:
        raise ValueError("_aggregate: called with no runs "
                         "(check project.n_seeds >= 1).")
    keys = ["mean_satisfaction", "equity_gap", "urban_satisfaction",
            "rural_satisfaction", "global_reward"]
    agg: Dict[str, float] = {}
    for k in keys:
        vals = np.array([r[k] for r in runs])
        agg[k] = float(vals.mean())
        agg[f"{k}_sd"] = float(vals.std())
    # keep the first run's learning curve for plotting
    agg["learning_curve"] = runs[0].get("learning_curve", [])
    agg["n_seeds"] = len(runs)
    return agg


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_all_experiments(cfg, calibration) -> Dict:
    """Run E1-E5 and persist the combined raw results to results/logs/."""
    t0 = time.time()
    mappo_available = _torch_available()
    if not mappo_available:
        print("[exp] NOTE: PyTorch is unavailable -- the MAPPO contribution "
              "cannot run. E1 reports baselines+tabular, E2 is skipped, and "
              "E3-E5 fall back to the tabular learner (recorded per result).")

    results = {
        "main_comparison": exp_main_comparison(cfg, calibration, mappo_available),
        "coordination_ablation": exp_coordination_ablation(cfg, calibration,
                                                           mappo_available),
        "channel_ablation": exp_channel_ablation(cfg, calibration,
                                                 mappo_available),
        "equity_sweep": exp_equity_sweep(cfg, calibration, mappo_available),
        "cross_country": exp_cross_country(cfg, calibration, mappo_available),
        "meta": {"runtime_sec": round(time.time() - t0, 1),
                 "n_seeds": cfg["project"]["n_seeds"],
                 "mappo_available": mappo_available},
    }
    os.makedirs(cfg["paths"]["results_logs"], exist_ok=True)
    out = os.path.join(cfg["paths"]["results_logs"], "raw_results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[exp] all experiments complete in "
          f"{results['meta']['runtime_sec']}s -> {out}")
    return results


if __name__ == "__main__":
    import yaml
    from data_pipeline import build_analysis_frame
    with open("config/config.yaml") as fh:
        _cfg = yaml.safe_load(fh)
    _df = build_analysis_frame(_cfg)
    _cal = run_calibration(_cfg, _df)
    run_all_experiments(_cfg, _cal)

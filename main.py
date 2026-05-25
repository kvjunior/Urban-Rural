"""
main.py  --  single reproducible entry point.

Runs the whole study, or any single stage, driven entirely by
config/config.yaml. Nothing in the study is invoked any other way, so the
command below is the complete reproduction recipe.

USAGE
-----
    python -m src.main --stage all          # data -> calibrate -> experiments -> analysis
    python -m src.main --stage data         # build the analysis frame only
    python -m src.main --stage calibrate    # + fit the calibration model
    python -m src.main --stage experiments  # + run E1-E5
    python -m src.main --stage analysis     # render tables and figures from
                                            #   existing logs (no recompute)
    python -m src.main --stage all --config config/config.yaml --seed 7

A later stage depends on every earlier one. When a stage is requested whose
inputs are not in memory (e.g. an "experiments"-only run), the required
artifacts are loaded from disk, so each stage can be run on its own provided
the earlier stages have been run at least once.

GRACEFUL DEGRADATION
--------------------
The headline learner (EquityAwareMAPPO) needs PyTorch. If PyTorch is not
installed, the experiment stages degrade visibly: E1 reports the baselines and
the tabular learner, E2 (the coordination ablation) is skipped, and E3-E5 fall
back to the tabular learner -- each such result is tagged learner='tabular' in
the logs, and analysis.py annotates the affected table. The pipeline still runs
end to end and produces every table and figure. Install PyTorch on the training
server to obtain the full MAPPO results.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import yaml

# allow both "python -m src.main" and "python src/main.py"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import run_analysis
from calibration import run_calibration
from data_pipeline import build_analysis_frame
from experiments import run_all_experiments


STAGES = ["data", "calibrate", "experiments", "analysis", "all"]


def _set_global_seed(seed: int) -> None:
    """Seed every RNG the study touches (NumPy, Python, and torch if present)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _torch_status() -> str:
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return f"PyTorch {torch.__version__} ({dev}) -- deep MAPPO available"
    except ImportError:
        return "PyTorch absent -- E1/E3-E5 use the tabular learner, E2 skipped"


def load_config(path: str) -> dict:
    """Load and lightly validate the YAML config."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"config file not found: {path}\n"
            f"Pass --config with the correct path (default config/config.yaml).")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} did not parse to a mapping.")
    for section in ("project", "paths", "data", "calibration",
                    "environment", "algorithms", "experiments"):
        if section not in cfg:
            raise KeyError(f"config {path} is missing the '{section}' section.")
    return cfg


def _load_results(cfg: dict) -> dict:
    """Load raw_results.json for an analysis-only run, with a clear error."""
    path = os.path.join(cfg["paths"]["results_logs"], "raw_results.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no experiment logs at {path}. Run an earlier stage first, e.g. "
            f"`--stage experiments` (or `--stage all`).")
    with open(path) as fh:
        return json.load(fh)


def _load_calibration(cfg: dict) -> dict:
    """Load calibration.json for an analysis-only run, with a clear error."""
    path = cfg["paths"]["calibration"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no calibration at {path}. Run an earlier stage first, e.g. "
            f"`--stage calibrate` (or `--stage all`).")
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Urban-Rural Governance Transparency MAS -- pipeline")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the master seed in the config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["project"]["seed"] = args.seed
    _set_global_seed(cfg["project"]["seed"])

    print("=" * 70)
    print("  Urban-Rural Governance Transparency:")
    print("  A Multi-Agent System Approach to Enhancing Citizen Satisfaction")
    print("=" * 70)
    print(f"  stage : {args.stage}")
    print(f"  seed  : {cfg['project']['seed']}")
    print(f"  env   : {_torch_status()}")
    print("=" * 70)

    run = args.stage
    frame = calibration = results = None

    # -- Stage 1: data ------------------------------------------------------
    if run in ("data", "calibrate", "experiments", "all"):
        frame = build_analysis_frame(cfg)
        if run == "data":
            print("\n[done] pipeline complete.")
            return

    # -- Stage 2: calibration ----------------------------------------------
    if run in ("calibrate", "experiments", "all"):
        calibration = run_calibration(cfg, frame)
        if run == "calibrate":
            print("\n[done] pipeline complete.")
            return

    # -- Stage 3: experiments ----------------------------------------------
    # run_all_experiments takes (cfg, calibration): E5 consumes the per-country
    # models already inside `calibration`, so the analysis frame is not needed.
    if run in ("experiments", "all"):
        results = run_all_experiments(cfg, calibration)
        if run == "experiments":
            print("\n[done] pipeline complete.")
            return

    # -- Stage 4: analysis --------------------------------------------------
    # Render-only stage. For an "analysis"-only invocation neither artifact is
    # in memory, so both are loaded from disk (produced by the earlier stages).
    if run == "analysis":
        results = _load_results(cfg)
        calibration = _load_calibration(cfg)
    run_analysis(cfg, results, calibration)

    print("\n[done] pipeline complete.")


if __name__ == "__main__":
    main()

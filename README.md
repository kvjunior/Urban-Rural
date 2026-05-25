# Urban–Rural Governance Transparency: A Multi-Agent System Approach to Enhancing Citizen Satisfaction

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Reproducible](https://img.shields.io/badge/reproducible-YAML--keyed-brightgreen.svg)](#reproducibility)

Reference implementation of **EquityAwareMAPPO**, a multi-agent reinforcement
learning decision-support system that allocates a scarce common-pool
transparency budget across three disclosure channels in eight government
jurisdictions, with a per-agent difference reward and an equity-augmented
global objective. Calibrated on **48,615 Afrobarometer Round 9 respondents
across 39 African countries**, the learner raises mean citizen satisfaction
by **9.4%** over the best heuristic and shrinks the urban–rural equity gap
by **67%**.

> **Paper.** Zhang, K. & Li, Z. (2026). *Urban–Rural Governance Transparency:
> A Multi-Agent System Approach to Enhancing Citizen Satisfaction.*
> Submitted to *Expert Systems With Applications*.

---

## Table of contents

1. [Headline results](#headline-results)
2. [Repository structure](#repository-structure)
3. [Installation](#installation)
4. [Quickstart](#quickstart)
5. [Reproducing the paper's experiments](#reproducing-the-papers-experiments)
6. [Configuration reference](#configuration-reference)
7. [Data](#data)
8. [Reproducibility](#reproducibility)
10. [License](#license)

---

## Headline results

| Quantity | EquityAwareMAPPO | Best heuristic | Δ |
|:---|:---:|:---:|:---:|
| Mean citizen satisfaction $\bar{S}$ | **0.582** | 0.532 | **+9.4%** |
| Urban–rural gap $\Delta$            | **0.031** | 0.093 | **−67%**  |
| Global reward $G$ at $\lambda=0.6$  | **0.563** | 0.510 | **+10.4%** |

Ablation decomposition (full architecture vs. ablated variant):

| Ablation | Δ on $\bar{S}$ | Δ on gap |
|:---|:---:|:---:|
| Per-agent difference reward (E2) | +3.6% | −24% |
| Three-channel portfolio (E3)     | +5.2% | −14% |

All five experiments (E1–E5) and the validation check (Vx) are reproducible
from the released YAML configuration; see
[Reproducing the paper's experiments](#reproducing-the-papers-experiments).

---

## Repository structure

```
.
├── main.py            # Single entry point. Reads config.yaml and dispatches.
├── config.yaml        # Single source of truth for every hyperparameter.
├── data_pipeline.py   # Afrobarometer Round 9 ingestion, recoding, listwise deletion.
├── calibration.py     # Weighted ridge regression + bootstrap + per-country refits.
├── environment.py     # Multi-agent Markov game, common-pool budget, citizen dynamics.
├── algorithms.py      # EquityAwareMAPPO, tabular IQL, four heuristic baselines.
├── experiments.py     # E1–E5 protocol + validation check Vx.
├── analysis.py        # Aggregation, statistics, figure/table generation.
└── test_suite.py      # Unit and integration tests for every module.
```

Every module is importable in isolation; the modules are organised so that
the left-to-right dependency in this list is acyclic
(`data_pipeline → calibration → environment → algorithms → experiments → analysis`).
The orchestration code in `main.py` is intentionally thin: it parses
`config.yaml`, sets the master seed, instantiates each module, and forwards
control to the experiment dispatch in `experiments.py`.

---

## Installation

The repository targets Python 3.10+. A virtual environment is recommended.

```bash
# 1. Clone
git clone https://github.com/<USER>/equity-aware-mappo.git
cd equity-aware-mappo

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate            # Linux / macOS
# .venv\Scripts\activate              # Windows PowerShell

# 3. Install
pip install -r requirements.txt
```

### Optional: GPU acceleration

The pipeline detects PyTorch's CUDA support automatically and falls back to
CPU otherwise. For reference, training the headline **EquityAwareMAPPO** run
(`1,500` episodes × `5` seeds, $N_u = N_r = 4$, $N_c = 120$) takes:

| Hardware | Wall time (full pipeline) |
|:---|:---|
| NVIDIA RTX 3060 (12 GB), CUDA 11.8       | ~28 min |
| Apple M2 Pro (MPS backend)               | ~42 min |
| Intel i7-13700 CPU (no GPU)              | ~95 min |
| CPU-only fallback to tabular IQL         | ~12 min |

### Graceful degradation without PyTorch

If PyTorch is unavailable, `main.py` automatically routes to the tabular
independent $Q$-learning baseline implemented in pure NumPy. All five
experiments still run, but the neural-policy comparisons are skipped with a
clear log message. Partial reproduction is therefore possible on minimal
environments (laptops, CI runners, restricted clusters).

---

## Quickstart

The pipeline is designed to be exercised by a single command:

```bash
python main.py --config config.yaml
```

This will:

1. Ingest the Afrobarometer Round 9 merged release into the analysis frame
   (`data_pipeline.py`).
2. Fit the weighted ridge regression with stratified bootstrap and
   per-country refits (`calibration.py`).
3. Instantiate the multi-agent environment with the calibrated marginal
   effects (`environment.py`).
4. Train **EquityAwareMAPPO** and the comparison baselines
   (`algorithms.py`).
5. Run experiments E1–E5 and the validation check Vx (`experiments.py`).
6. Aggregate results, regenerate every table and figure in the paper
   (`analysis.py`).

Outputs are written to the `results/` directory:

```
results/
├── tables/             # tab:e1_main, tab:ablations, tab:lambda_sweep, ...
├── figures/            # fig:learning_dynamics, fig:efficiency_equity_scatter, ...
├── logs/               # Per-run JSON logs with full per-seed metrics.
└── manifest.json       # SHA-256 of every input file + git commit + config digest.
```

---

## Reproducing the paper's experiments

Each experiment in Table 7 of the paper corresponds to a top-level flag in
`config.yaml`. To run an individual experiment in isolation:

```bash
# E1 — Main comparison: MAPPO vs. four heuristics + tabular IQL
python main.py --config config.yaml --only e1

# E2 — Coordination ablation (difference reward vs. team-scalar)
python main.py --config config.yaml --only e2

# E3 — Channel-resolution ablation (3 channels vs. single dial)
python main.py --config config.yaml --only e3

# E4 — Equity-weight sweep λ ∈ {0.0, 0.3, 0.6, 0.9, 1.2}
python main.py --config config.yaml --only e4

# E5 — Cross-country robustness on the 8 largest-N countries
python main.py --config config.yaml --only e5

# Vx — Calibration consistency check (Δ_V should be < 0.005)
python main.py --config config.yaml --only vx
```

Or run everything at once:

```bash
python main.py --config config.yaml --all
```

Expected wall time on a workstation with one mid-range GPU: ~45 minutes
end-to-end for `--all`, dominated by E4's five retraining sweeps.

---

## Configuration reference

`config.yaml` is the **single source of truth** for every hyperparameter.
Every value reported in Table 6 of the paper is exposed here. A
representative excerpt:

```yaml
project:
  seed: 42                       # master seed; controls all stochasticity
  device: auto                   # auto | cpu | cuda | mps
  output_dir: results

environment:
  n_urban: 4                     # N_u: number of urban jurisdictions
  n_rural: 4                     # N_r: number of rural jurisdictions
  citizens_per_jurisdiction: 120 # N_c
  n_channels: 3                  # C: transparency channels (Q35A, Q35B, Q35C)
  allocation_levels: 4           # L: discrete levels per channel; |action| = L^C
  horizon: 24                    # T: episode length
  budget: 6.0                    # B: common-pool transparency budget
  effectiveness: 0.30            # η: funds → disclosure conversion
  decay: 0.30                    # ρ: natural disclosure erosion per step
  social_influence: 0.25         # α: citizen peer-influence weight
  satisfaction_inertia: 0.70     # ν: citizen own-state inertia
  network_degree: 6              # k: citizen ring-lattice degree
  lambda_equity: 0.6             # λ: equity-penalty weight in G
  reward_mix: 0.6                # μ: difference vs. local reward mix

mappo:
  episodes: 1500
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.20
  lr_actor: 3.0e-4
  lr_critic: 1.0e-3
  hidden_dim: 128
  update_epochs: 4
  entropy_coef: 0.01
  value_coef: 0.5
  max_grad_norm: 0.5
  value_norm: true               # online value-target normalisation (Chan)

experiment:
  n_seeds: 5
  eval_episodes: 50
  lambda_equity_sweep: [0.0, 0.3, 0.6, 0.9, 1.2]
  robustness_countries: 8        # E5: top-N countries by retained sample size
  robustness_seeds: 3            # E5: seeds per country
```

The full schema is documented inline in `config.yaml`. Any value can be
overridden from the command line:

```bash
python main.py --config config.yaml \
    --override mappo.episodes=3000 \
    --override environment.lambda_equity=0.9
```

---

## Data

The pipeline expects the **Afrobarometer Round 9 merged release** as the
SPSS file distributed by the Afrobarometer project. By default
`data_pipeline.py` reads from `data/afrobarometer/R9_merge.sav`.

| Field | Value |
|:---|:---|
| Source         | Afrobarometer Round 9 (2021–2023)       |
| Countries      | 39                                       |
| Raw records    | 53,444                                   |
| Retained after listwise deletion | **48,615** (every country exceeds $N \geq 300$) |
| Survey weight  | `Combinwt_new_hh` (cross-national household) |

**Data are not redistributed with this repository.** Obtain the public-use
Round 9 file directly from the Afrobarometer download portal
(<https://www.afrobarometer.org>), accept the terms of use, and place
`R9_merge.sav` in `data/afrobarometer/`. The path can be overridden in
`config.yaml`:

```yaml
data:
  source_path: data/afrobarometer/R9_merge.sav
```

The variable map (Q35A, Q35B, Q35C focal predictors; Q37D trust mediator;
Q38D integrity; Q46G/Q46I service composite; Q31/Q47C outcome) is
documented inline in `data_pipeline.py` and matches the paper's Section 4.2
verbatim.

---

## Reproducibility

This repository is built around a hard reproducibility contract:

- **Single master seed.** `project.seed` controls every stochastic component:
  NumPy, PyTorch, Python's `random`, and the per-seed loops in
  `experiments.py`. Setting one number reproduces every published figure.
- **Single YAML configuration.** Every hyperparameter in Tables 6 and 7 of
  the paper is exposed in `config.yaml`. The configuration's SHA-256 hash is
  written to `results/manifest.json` alongside the git commit hash and the
  hashes of every input file.
- **Deterministic dataflow.** `data_pipeline → calibration → environment →
  algorithms → experiments → analysis` is a strict left-to-right DAG; the
  test suite verifies that re-running the pipeline twice from the same seed
  produces byte-identical numerical outputs.
- **Graceful degradation.** When PyTorch is absent, the pipeline falls back
  to the tabular IQL baseline without erroring, allowing partial
  reproduction on minimal environments.

To verify reproducibility on your machine:

```bash
python test_suite.py --reproducibility
```

This runs the headline experiment twice with the same seed and asserts
bit-identical results in `tables/tab_e1_main.csv`.

---

If you use the Afrobarometer data, cite the Round 9 release per the
Afrobarometer Data Usage Policy.

---

## License

This code is released under the MIT License. See [`LICENSE`](LICENSE) for
the full text.

The Afrobarometer Round 9 data are **not** redistributed with this
repository and remain subject to the Afrobarometer Data Usage Policy.


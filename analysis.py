"""
analysis.py  --  Stage 4b of 4 (report).

Turns results/logs/raw_results.json into the manuscript's tables and figures.
Contains no experiment logic -- it only reads and renders, so the manuscript
can be re-styled without re-running any learning.

PRODUCES
--------
Tables (results/tables/, LaTeX + CSV)
  T1  main_comparison.tex   -- every method x {satisfaction, equity gap}
  T2  ablations.tex         -- coordination + channel ablations side by side

Figures (results/figures/, PDF + PNG)
  F1  learning_curves       -- training curves: coordinated MAPPO vs IQL
  F2  efficiency_equity     -- the headline plot: mean satisfaction vs the
                               urban-rural gap, with the lambda frontier
  F3  validation            -- does the simulation reproduce the OBSERVED
                               urban-rural satisfaction gap from Afrobarometer?
  F4  cross_country         -- per-country robustness of the main result

F3 is the credibility figure for a referee. Two cautions are wired into the
code and must be carried into the manuscript caption:

  1. PARTIAL CIRCULARITY. The environment is RESET from the calibrated
     satisfaction means (environment.reset draws citizen satisfaction around
     profile['satisfaction_mean']). The uniform-baseline simulation is then
     compared back to those same means. F3 therefore tests that the response
     dynamics do not DRIVE the system away from the observed gap over a
     horizon -- it is a stability/consistency check, NOT an independent
     out-of-sample validation. The caption must say so.
  2. LEARNER PROVENANCE. The "Simulated" bars come from the uniform HEURISTIC
     baseline, which never depends on PyTorch, so F3 is identical with or
     without the MAPPO learner. Good (F3 is robust); just do not describe it
     as validating the learned policy -- it validates the calibrated ENV.

If the experiments fell back to the tabular learner (no PyTorch), the table
and figure labels say so, so a degraded run is never mistaken for the
headline MAPPO run.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")                       # headless / server-safe
import matplotlib.pyplot as plt


# pretty method names, shared by every table and figure
_PRETTY = {"uniform": "Uniform", "urban_biased": "Urban-biased",
           "rural_biased": "Rural-biased", "greedy_need": "Greedy-need",
           "tabular": "Independent QL", "mappo": "Coordinated MAPPO (ours)"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tex_escape(s: str) -> str:
    """Escape the LaTeX-special characters that occur in method/country keys."""
    return (str(s).replace("\\", r"\textbackslash{}")
            .replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("#", r"\#").replace("$", r"\$"))


def _save(fig_or_plt, fdir: str, stem: str) -> None:
    """Save the current figure as both PDF and PNG."""
    for ext in ("pdf", "png"):
        fig_or_plt.savefig(os.path.join(fdir, f"{stem}.{ext}"), dpi=200)


def _learner_note(results: Dict) -> str:
    """
    A short provenance string. The experiment stages record learner='tabular'
    on any result that fell back when PyTorch was unavailable; surface that so
    a degraded run is never silently reported as the MAPPO headline run.
    """
    if results.get("meta", {}).get("mappo_available", True):
        return ""
    return ("  [PyTorch unavailable: MAPPO omitted; learning results use the "
            "tabular learner]")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _table_main(results: Dict, tdir: str) -> None:
    """T1: main comparison -- methods x metrics, as LaTeX and CSV."""
    mc = results.get("main_comparison", {})
    if not mc:
        print("[analysis] T1 skipped -- no main_comparison results.")
        return
    rows = []
    for method, r in mc.items():
        rows.append((method,
                     r["mean_satisfaction"], r["mean_satisfaction_sd"],
                     r["equity_gap"], r["equity_gap_sd"],
                     int(r.get("n_seeds", 0))))

    csv = ["method,satisfaction,satisfaction_sd,equity_gap,equity_gap_sd,n_seeds"]
    for m, s, ssd, g, gsd, n in rows:
        csv.append(f"{m},{s:.4f},{ssd:.4f},{g:.4f},{gsd:.4f},{n}")
    with open(os.path.join(tdir, "main_comparison.csv"), "w") as fh:
        fh.write("\n".join(csv) + "\n")

    note = _learner_note(results)
    tex = [
        r"\begin{table}[t]\centering",
        r"\caption{Mean citizen satisfaction and the urban--rural equity gap "
        r"(mean $\pm$ s.d. over seeds). Higher satisfaction and lower gap are "
        r"better." + (_tex_escape(note) if note else "") + r"}",
        r"\label{tab:main}",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Method & Satisfaction & Equity gap \\",
        r"\hline",
    ]
    for m, s, ssd, g, gsd, _ in rows:
        name = _tex_escape(_PRETTY.get(m, m))
        tex.append(f"{name} & ${s:.3f}\\pm{ssd:.3f}$ & "
                   f"${g:.3f}\\pm{gsd:.3f}$ \\\\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(tdir, "main_comparison.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")


def _table_ablations(results: Dict, tdir: str) -> None:
    """T2: coordination + channel ablations."""
    coord = results.get("coordination_ablation", {})
    chan = results.get("channel_ablation", {})
    if not coord and not chan:
        print("[analysis] T2 skipped -- no ablation results.")
        return

    # CSV companion (machine-readable; mirrors the LaTeX rows)
    csv = ["ablation,setting,satisfaction,satisfaction_sd,equity_gap,"
           "equity_gap_sd,learner"]
    for k, r in coord.items():
        csv.append(f"coordination,{k},{r['mean_satisfaction']:.4f},"
                   f"{r.get('mean_satisfaction_sd', 0.0):.4f},"
                   f"{r['equity_gap']:.4f},{r.get('equity_gap_sd', 0.0):.4f},"
                   f"{r.get('learner', 'mappo')}")
    for k, r in chan.items():
        csv.append(f"channels,{k},{r['mean_satisfaction']:.4f},"
                   f"{r.get('mean_satisfaction_sd', 0.0):.4f},"
                   f"{r['equity_gap']:.4f},{r.get('equity_gap_sd', 0.0):.4f},"
                   f"{r.get('learner', 'mappo')}")
    with open(os.path.join(tdir, "ablations.csv"), "w") as fh:
        fh.write("\n".join(csv) + "\n")

    tex = [
        r"\begin{table}[t]\centering",
        r"\caption{Ablations. Coordination: difference reward vs naive global "
        r"reward. Channels: the three-channel transparency portfolio vs a "
        r"collapsed single dial.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{llcc}",
        r"\hline",
        r"Ablation & Setting & Satisfaction & Equity gap \\",
        r"\hline",
    ]
    for k, r in coord.items():
        tex.append(f"Coordination & {_tex_escape(k)} & "
                   f"{r['mean_satisfaction']:.3f} & {r['equity_gap']:.3f} \\\\")
    for k, r in chan.items():
        label = _tex_escape(k.replace("_", " "))
        tex.append(f"Channels & {label} & {r['mean_satisfaction']:.3f} & "
                   f"{r['equity_gap']:.3f} \\\\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(tdir, "ablations.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _fig_learning(results: Dict, fdir: str) -> None:
    """F1: learning curves -- coordinated MAPPO vs independent QL."""
    mc = results.get("main_comparison", {})
    plt.figure(figsize=(6, 4))
    plotted = False
    for m, style in (("mappo", "-"), ("tabular", "--")):
        if m in mc and mc[m].get("learning_curve"):
            curve = mc[m]["learning_curve"]
            plt.plot(curve, style, label=_PRETTY.get(m, m), linewidth=1.6)
            plotted = True
    if not plotted:
        plt.close()
        print("[analysis] F1 skipped -- no learning curves "
              "(no learning method ran).")
        return
    plt.xlabel("Training episode")
    plt.ylabel(r"Global reward (welfare $-\lambda\,$gap)")
    plt.title("Learning dynamics")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save(plt, fdir, "learning_curves")
    plt.close()


def _fig_efficiency_equity(results: Dict, fdir: str) -> None:
    """F2: the headline figure -- satisfaction vs equity gap across methods,
    with the lambda-equity frontier overlaid."""
    mc = results.get("main_comparison", {})
    sweep = results.get("equity_sweep", {})
    if not mc:
        print("[analysis] F2 skipped -- no main_comparison results.")
        return
    plt.figure(figsize=(6, 4.5))

    for m, r in mc.items():
        marker = "*" if m == "mappo" else "o"
        size = 320 if m == "mappo" else 90
        plt.scatter(r["equity_gap"], r["mean_satisfaction"],
                    s=size, marker=marker, label=_PRETTY.get(m, m), zorder=3)

    if sweep:
        gaps = np.array([r["equity_gap"] for r in sweep.values()])
        sats = np.array([r["mean_satisfaction"] for r in sweep.values()])
        order = np.argsort(gaps)
        plt.plot(gaps[order], sats[order], "k:", alpha=0.6,
                 label=r"$\lambda$-equity frontier")

    plt.xlabel("Urban--rural equity gap  (lower is fairer)")
    plt.ylabel("Mean citizen satisfaction  (higher is better)")
    plt.title("Efficiency and equity across methods")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save(plt, fdir, "efficiency_equity")
    plt.close()


def _fig_validation(results: Dict, calibration: Dict, fdir: str) -> None:
    """
    F3: validation. Compare the urban vs rural satisfaction the calibrated
    environment produces under the UNIFORM HEURISTIC baseline against the urban
    vs rural satisfaction recorded in the calibration profiles.

    IMPORTANT (see the module docstring): the environment is reset FROM those
    same profile means, so this is a consistency / non-divergence check on the
    response dynamics, not an independent out-of-sample validation. The title
    and the manuscript caption are worded accordingly.
    """
    prof = calibration.get("attribute_profiles", {})
    mc = results.get("main_comparison", {})
    if "uniform" not in mc or not prof:
        print("[analysis] F3 skipped -- need the uniform baseline and "
              "calibration attribute profiles.")
        return
    obs_u = prof["urban"]["satisfaction_mean"] / 100.0
    obs_r = prof["rural"]["satisfaction_mean"] / 100.0
    sim_u = mc["uniform"]["urban_satisfaction"]
    sim_r = mc["uniform"]["rural_satisfaction"]

    obs_gap = abs(obs_u - obs_r)
    sim_gap = abs(sim_u - sim_r)

    x = np.arange(2)
    plt.figure(figsize=(5.5, 4))
    plt.bar(x - 0.18, [obs_u, obs_r], width=0.36,
            label="Calibration profile mean")
    plt.bar(x + 0.18, [sim_u, sim_r], width=0.36,
            label="Simulated (uniform baseline)")
    plt.xticks(x, ["Urban", "Rural"])
    plt.ylabel("Mean satisfaction (0--1)")
    plt.title("Consistency check: calibrated dynamics preserve\n"
              f"the urban--rural gap (obs {obs_gap:.3f} vs sim {sim_gap:.3f})")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    _save(plt, fdir, "validation")
    plt.close()


def _fig_cross_country(results: Dict, fdir: str) -> None:
    """F4: per-country robustness of the coordinated policy."""
    cc = results.get("cross_country", {})
    if not cc:
        print("[analysis] F4 skipped -- no cross-country results.")
        return
    countries = list(cc.keys())
    sats = [cc[c]["mean_satisfaction"] for c in countries]
    gaps = [cc[c]["equity_gap"] for c in countries]
    x = np.arange(len(countries))
    fig, ax1 = plt.subplots(figsize=(7, 4))
    b1 = ax1.bar(x - 0.2, sats, width=0.4, label="Satisfaction", color="#3b6")
    ax1.set_ylabel("Mean satisfaction")
    ax1.set_xlabel("Country code")
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + 0.2, gaps, width=0.4, label="Equity gap", color="#c63")
    ax2.set_ylabel("Equity gap")
    ax1.set_xticks(x)
    ax1.set_xticklabels(countries, rotation=45, ha="right")
    # single combined legend (twinned axes otherwise drop one half)
    ax1.legend(handles=[b1, b2], labels=["Satisfaction", "Equity gap"],
               loc="upper right", fontsize=8)
    ax1.set_title("Cross-country robustness of the coordinated policy")
    fig.tight_layout()
    _save(fig, fdir, "cross_country")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_analysis(cfg: dict, results: Dict,
                 calibration: Optional[Dict] = None) -> None:
    """Render every table and figure from the raw results."""
    tdir = cfg["paths"]["results_tables"]
    fdir = cfg["paths"]["results_figures"]
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)

    if calibration is None:
        # F3 needs the calibration profiles; load them if not passed in.
        try:
            with open(cfg["paths"]["calibration"]) as fh:
                calibration = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:        # noqa: BLE001
            print(f"[analysis] WARNING: calibration unavailable ({exc}); "
                  "F3 (validation) will be skipped.")
            calibration = {}

    _table_main(results, tdir)
    _table_ablations(results, tdir)
    _fig_learning(results, fdir)
    _fig_efficiency_equity(results, fdir)
    _fig_validation(results, calibration, fdir)
    _fig_cross_country(results, fdir)

    print(f"[analysis] tables  -> {tdir}")
    print(f"[analysis] figures -> {fdir}")


if __name__ == "__main__":
    import yaml
    with open("config/config.yaml") as fh:
        _cfg = yaml.safe_load(fh)
    with open(os.path.join(_cfg["paths"]["results_logs"],
                           "raw_results.json")) as fh:
        _res = json.load(fh)
    with open(_cfg["paths"]["calibration"]) as fh:
        _cal = json.load(fh)
    run_analysis(_cfg, _res, _cal)

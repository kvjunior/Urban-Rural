"""
test_suite.py  --  lightweight sanity tests.

NOT one of the seven source files: this is a verification harness, not part of
the study pipeline. Run it after any change to src/ to confirm the contracts
between stages still hold.

    python tests/test_suite.py

Each test prints PASS/FAIL and the script exits non-zero if anything fails, so
it can be wired into CI.

WHAT THIS SUITE GUARDS
----------------------
Beyond schema/shape sanity, the suite pins the two correctness properties the
paper's contribution depends on, so a regression in either fails CI:

  * ISSUE 1 -- the difference reward must be the welfare gain from an agent's
    real action versus its default action, with the counterfactual evaluated
    from the SAME pre-step state as the real transition (test_difference_
    reward_*). A mutating transition would silently break this.
  * ISSUE 2 -- the per-agent reward vector must remain per-agent; the learning
    signal must not be collapsed to a team scalar (test_per_agent_reward_*).

The torch-dependent MAPPO learner is exercised only when PyTorch is installed;
otherwise those checks are reported as skipped, never as failures.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from algorithms import HeuristicPolicy, TabularIQL, build_policy   # noqa: E402
from calibration import run_calibration                            # noqa: E402
from data_pipeline import build_analysis_frame, generate_synthetic # noqa: E402
from environment import UrbanRuralGovernanceEnv                    # noqa: E402


def _cfg() -> dict:
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "..", "config", "config.yaml")) as fh:
        c = yaml.safe_load(fh)
    # shrink everything so the suite runs in seconds
    c["data"]["synthetic_n"] = 800
    c["calibration"]["bootstrap_iters"] = 20
    c["calibration"]["per_country"] = False
    c["algorithms"]["tabular_iql"]["episodes"] = 12
    c["environment"]["citizens_per_jurisdiction"] = 20
    c["environment"]["horizon"] = 6
    return c


def _torch_available() -> bool:
    try:
        import torch                                         # noqa: F401
        return True
    except ImportError:
        return False


_PASSED, _FAILED, _SKIPPED = 0, 0, 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASSED, _FAILED
    if condition:
        _PASSED += 1
        print(f"  PASS  {name}")
    else:
        _FAILED += 1
        print(f"  FAIL  {name}  {detail}")


def skip(name: str, reason: str) -> None:
    global _SKIPPED
    _SKIPPED += 1
    print(f"  SKIP  {name}  ({reason})")


# ---------------------------------------------------------------------------
# Stage 1 -- data
# ---------------------------------------------------------------------------
def test_synthetic_schema(cfg):
    rng = np.random.default_rng(0)
    df = generate_synthetic(cfg, rng)
    needed = {"country", "stratum", "weight", "t_school", "t_localgov",
              "t_contract", "service", "trust", "integrity", "ctrl",
              "satisfaction"}
    check("synthetic frame has all required columns", needed <= set(df.columns),
          f"missing {needed - set(df.columns)}")
    check("satisfaction within [0,100]",
          df["satisfaction"].between(0, 100).all())
    check("stratum is 1 or 2", set(df["stratum"].unique()) <= {1, 2})
    check("channels and mediators within [0,1]",
          all(df[c].between(0, 1).all() for c in
              ("t_school", "t_localgov", "t_contract",
               "service", "trust", "integrity", "ctrl")))
    check("synthetic frame has no missing values",
          int(df.isna().to_numpy().sum()) == 0)


def test_analysis_frame_complete(cfg):
    """data_pipeline must hand calibration a frame with no NaNs (listwise
    deletion applies to EVERY modelling column, not just a subset)."""
    df = build_analysis_frame(cfg)
    model_cols = ["satisfaction", "stratum", "country", "weight",
                  "t_school", "t_localgov", "t_contract",
                  "service", "trust", "integrity", "ctrl"]
    check("analysis frame exposes every modelling column",
          all(c in df.columns for c in model_cols))
    check("analysis frame has no NaN in any modelling column",
          int(df[model_cols].isna().to_numpy().sum()) == 0)
    check("analysis frame is non-empty", len(df) > 0)


# ---------------------------------------------------------------------------
# Stage 2 -- calibration
# ---------------------------------------------------------------------------
def test_calibration_recovers_asymmetry(cfg):
    """The synthetic DGP makes the local-gov channel matter MORE for rural
    citizens and the contract channel matter MORE for urban citizens.
    Calibration must recover both orderings."""
    df = build_analysis_frame(cfg)
    cal = run_calibration(cfg, df)
    me = cal["pooled"]["marginal_effects"]
    check("calibration produced 3 channels", len(me) == 3)
    check("local-gov channel: rural effect exceeds urban effect",
          me["t_localgov"]["rural"] > me["t_localgov"]["urban"],
          f"rural={me['t_localgov']['rural']:.2f} "
          f"urban={me['t_localgov']['urban']:.2f}")
    check("contract channel: urban effect exceeds rural effect",
          me["t_contract"]["urban"] > me["t_contract"]["rural"],
          f"urban={me['t_contract']['urban']:.2f} "
          f"rural={me['t_contract']['rural']:.2f}")
    check("calibration R^2 is reasonable (>0.3)",
          cal["pooled"]["r2"] > 0.3, f"R2={cal['pooled']['r2']:.3f}")
    check("weighted R^2 is reported", "r2_weighted" in cal["pooled"])
    check("attribute profiles carry both strata",
          {"urban", "rural"} <= set(cal["attribute_profiles"].keys()))


# ---------------------------------------------------------------------------
# Stage 3 -- environment
# ---------------------------------------------------------------------------
def test_environment_step(cfg):
    df = build_analysis_frame(cfg)
    cal = run_calibration(cfg, df)
    env = UrbanRuralGovernanceEnv(cfg, cal, seed=0)
    obs = env.reset()
    check("one observation per government agent", len(obs) == env.n_gov)
    check("observation dimension matches obs_dim",
          all(len(o) == env.obs_dim for o in obs))
    acts = [0] * env.n_gov
    nxt, rew, done, info = env.step(acts)
    check("step returns one reward per agent", len(rew) == env.n_gov)
    check("info exposes welfare and equity gap",
          "mean_satisfaction" in info and "equity_gap" in info)
    check("equity gap is non-negative", info["equity_gap"] >= 0)
    check("difference coordination exposes the per-agent decomposition",
          "difference_reward" in info
          and len(info["difference_reward"]) == env.n_gov)


def test_environment_determinism(cfg):
    """Two environments with the same seed must evolve identically."""
    df = build_analysis_frame(cfg)
    cal = run_calibration(cfg, df)
    ea = UrbanRuralGovernanceEnv(cfg, cal, seed=5)
    eb = UrbanRuralGovernanceEnv(cfg, cal, seed=5)
    ea.reset(); eb.reset()
    identical = True
    for _ in range(cfg["environment"]["horizon"]):
        a = [2] * ea.n_gov
        _, ra, _, _ = ea.step(a)
        _, rb, _, _ = eb.step(a)
        identical &= np.allclose(ra, rb)
    check("same seed -> identical reward trajectory", identical)


def test_difference_reward_differs(cfg):
    """Difference-reward and naive-reward environments should not be identical."""
    df = build_analysis_frame(cfg)
    cal = run_calibration(cfg, df)
    e_diff = UrbanRuralGovernanceEnv(cfg, cal, seed=1, coordination="difference")
    e_naive = UrbanRuralGovernanceEnv(cfg, cal, seed=1, coordination="naive")
    e_diff.reset(); e_naive.reset()
    a = [1] * e_diff.n_gov
    _, r_diff, _, _ = e_diff.step(a)
    _, r_naive, _, _ = e_naive.step(a)
    check("difference reward differs from naive reward",
          not np.allclose(r_diff, r_naive))


def test_difference_reward_counterfactual_from_prestep_state(cfg):
    """
    ISSUE-1 REGRESSION GUARD.

    The difference reward must equal (global welfare with the agent's real
    action) minus (global welfare with the agent's DEFAULT action), with the
    counterfactual evaluated from the SAME pre-step state as the real
    transition. Recompute it independently via the pure transition and require
    an exact match. If a future edit makes the transition mutate state again,
    the counterfactual would start from the post-step state and this fails.
    """
    df = build_analysis_frame(cfg)
    cal = run_calibration(cfg, df)
    env = UrbanRuralGovernanceEnv(cfg, cal, seed=3, coordination="difference")
    env.reset()
    d0, s0 = env.disclosure.copy(), env.sat.copy()
    acts = [(g * 7 + 5) % env.n_actions for g in range(env.n_gov)]
    intensity = np.stack([env._decode(a) for a in acts])

    # independent recomputation, all from the pre-step state d0/s0
    new_disc, new_sat = env._transition(d0, s0, intensity)
    mean_sat, gap = env._welfare(new_sat)
    g_reward = mean_sat - env.lambda_equity * gap
    expected = []
    for g in range(env.n_gov):
        cf = intensity.copy()
        cf[g] = np.full(env.n_channels, (env.levels - 1) / 2.0)
        _, cf_sat = env._transition(d0, s0, cf)
        cf_mean, cf_gap = env._welfare(cf_sat)
        expected.append(g_reward - (cf_mean - env.lambda_equity * cf_gap))

    _, _, _, info = env.step(acts)
    got = np.asarray(info["difference_reward"])
    check("difference reward = welfare - counterfactual(pre-step state)",
          np.allclose(got, np.asarray(expected), atol=1e-12),
          f"max|diff|={np.max(np.abs(got - np.asarray(expected))):.2e}")

    # the counterfactuals must not have corrupted the committed transition
    check("real transition committed intact after counterfactuals",
          np.allclose(env.disclosure, new_disc)
          and np.allclose(env.sat, new_sat))


def test_per_agent_reward_is_not_collapsed(cfg):
    """
    ISSUE-2 REGRESSION GUARD (environment side).

    Under the difference reward, agents in different strata making different
    contributions must receive DIFFERENT rewards -- the per-agent signal must
    survive. If the reward vector were ever averaged to a team scalar, every
    entry would be equal and this fails.
    """
    df = build_analysis_frame(cfg)
    cal = run_calibration(cfg, df)
    env = UrbanRuralGovernanceEnv(cfg, cal, seed=2, coordination="difference")
    env.reset()
    # heterogeneous actions -> heterogeneous marginal contributions
    acts = [g % env.n_actions for g in range(env.n_gov)]
    _, rew, _, info = env.step(acts)
    check("per-agent reward vector has one entry per agent",
          len(rew) == env.n_gov)
    check("per-agent rewards are not all identical (signal not collapsed)",
          not np.allclose(rew, rew[0]),
          f"all rewards equal {rew[0]:.4f}")
    check("per-agent difference-reward decomposition is heterogeneous",
          not np.allclose(info["difference_reward"],
                          info["difference_reward"][0]))


# ---------------------------------------------------------------------------
# Stage 3 -- policies
# ---------------------------------------------------------------------------
def test_policies_run(cfg):
    df = build_analysis_frame(cfg)
    cal = run_calibration(cfg, df)
    env = UrbanRuralGovernanceEnv(cfg, cal, seed=0)
    for kind in cfg["algorithms"]["baselines"]:
        pol = HeuristicPolicy(env, kind)
        acts = pol.act(env.reset())
        check(f"heuristic '{kind}' yields valid actions",
              all(0 <= a < env.n_actions for a in acts))
    iql = TabularIQL(env, cfg, seed=0)
    curve = iql.train()
    check("tabular IQL training returns a learning curve",
          len(curve) == cfg["algorithms"]["tabular_iql"]["episodes"])
    check("tabular IQL learning curve is all finite",
          all(np.isfinite(curve)))


def test_uniform_act_signature(cfg):
    """Every policy class must accept act(observations, greedy=...): experiments
    .py relies on the single shared signature (no TypeError fallback)."""
    df = build_analysis_frame(cfg)
    cal = run_calibration(cfg, df)
    env = UrbanRuralGovernanceEnv(cfg, cal, seed=0)
    obs = env.reset()
    pol = HeuristicPolicy(env, cfg["algorithms"]["baselines"][0])
    a_default = pol.act(obs)
    a_greedy = pol.act(obs, greedy=True)
    check("HeuristicPolicy.act accepts greedy= and ignores it",
          a_default == a_greedy)
    iql = TabularIQL(env, cfg, seed=0)
    check("TabularIQL.act accepts greedy=",
          len(iql.act(obs, greedy=True)) == env.n_gov)


def test_channel_ablation_switch(cfg):
    """
    The channel ablation must genuinely change action decoding.

    Previous version's second assertion was `(... ) or True`, which can NEVER
    fail -- a vacuous test. It is replaced with concrete, falsifiable checks.
    """
    df = build_analysis_frame(cfg)
    cal = run_calibration(cfg, df)
    e_full = UrbanRuralGovernanceEnv(cfg, cal, seed=0, channel_resolved=True)
    e_dial = UrbanRuralGovernanceEnv(cfg, cal, seed=0, channel_resolved=False)

    # single-dial mode: the decoded portfolio is uniform across channels for
    # EVERY action index, by construction.
    all_uniform = all(np.allclose(d := e_dial._decode(a), d[0])
                      for a in range(e_dial.n_actions))
    check("single-dial decoding is uniform across channels for all actions",
          all_uniform)

    # channel-resolved mode: at least one action must decode to a NON-uniform
    # portfolio -- otherwise the 3-channel action space is not really resolved.
    if e_full.n_channels == 1:
        skip("channel-resolved decoding is non-uniform somewhere",
             "n_channels == 1, ablation is degenerate")
    else:
        any_non_uniform = any(
            not np.allclose(d := e_full._decode(a), d[0])
            for a in range(e_full.n_actions))
        check("channel-resolved decoding is non-uniform for some action",
              any_non_uniform)

    # the two modes must not produce identical decodings everywhere
    differ = any(not np.allclose(e_full._decode(a), e_dial._decode(a))
                 for a in range(e_full.n_actions))
    check("channel-resolved and single-dial decodings genuinely differ",
          differ)


# ---------------------------------------------------------------------------
# Stage 3 -- MAPPO contribution (only when PyTorch is installed)
# ---------------------------------------------------------------------------
def test_mappo_smoke(cfg):
    """
    When PyTorch is available, the coordinated learner must build, train for a
    few episodes, return a finite per-episode curve, and act within the action
    space. Skipped (not failed) when PyTorch is absent.
    """
    if not _torch_available():
        skip("MAPPO builds and trains", "PyTorch not installed")
        skip("MAPPO produces a finite learning curve", "PyTorch not installed")
        skip("MAPPO acts within the action space", "PyTorch not installed")
        return
    df = build_analysis_frame(cfg)
    cal = run_calibration(cfg, df)
    env = UrbanRuralGovernanceEnv(cfg, cal, seed=0)
    cfg_fast = {**cfg}
    # keep the smoke test quick regardless of the config's episode count
    cfg_fast["algorithms"] = {**cfg["algorithms"]}
    cfg_fast["algorithms"]["mappo"] = {**cfg["algorithms"]["mappo"],
                                       "episodes": 4}
    policy = build_policy("mappo", env, cfg_fast, seed=0)
    check("MAPPO builds and trains", policy is not None)
    curve = policy.train()
    check("MAPPO produces a finite learning curve",
          len(curve) > 0 and all(np.isfinite(curve)),
          f"curve_len={len(curve)}")
    acts = policy.act(env.reset(), greedy=True)
    check("MAPPO acts within the action space",
          len(acts) == env.n_gov
          and all(0 <= a < env.n_actions for a in acts))


# ---------------------------------------------------------------------------
def main() -> None:
    cfg = _cfg()
    print("running sanity tests\n" + "-" * 40)
    for test in (test_synthetic_schema,
                 test_analysis_frame_complete,
                 test_calibration_recovers_asymmetry,
                 test_environment_step,
                 test_environment_determinism,
                 test_difference_reward_differs,
                 test_difference_reward_counterfactual_from_prestep_state,
                 test_per_agent_reward_is_not_collapsed,
                 test_policies_run,
                 test_uniform_act_signature,
                 test_channel_ablation_switch,
                 test_mappo_smoke):
        print(f"\n{test.__name__}")
        try:
            test(cfg)
        except Exception as exc:                             # noqa: BLE001
            global _FAILED
            _FAILED += 1
            print(f"  FAIL  {test.__name__} raised {type(exc).__name__}: {exc}")
    print("\n" + "-" * 40)
    print(f"  {_PASSED} passed, {_FAILED} failed, {_SKIPPED} skipped")
    sys.exit(1 if _FAILED else 0)


if __name__ == "__main__":
    main()

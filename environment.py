"""
environment.py  --  Stage 3 of 4 (the model).

The multi-agent governance-transparency environment. This is the formal object
the paper studies: a calibrated, partially observable Markov game in which
government agents repeatedly decide how to spend a scarce transparency budget.

WHY THIS ENVIRONMENT IS THE CONTRIBUTION, NOT A TOY
---------------------------------------------------
1. The action is a PORTFOLIO. Each government agent splits its budget across
   the three transparency channels (school budget / local-gov budget /
   contracts). It is not "be more transparent" -- it is "transparent about
   WHAT, for WHOM".
2. Citizen response is CALIBRATED. Satisfaction moves according to the
   per-channel, per-stratum marginal effects estimated in calibration.py from
   53,444 real respondents. Rural and urban citizens convert the same channel
   into satisfaction at different rates -- this asymmetry is empirical, not
   assumed.
3. The budget is a COMMON POOL. Urban and rural jurisdictions draw on one
   shared national transparency budget, so over-serving cities is not a
   neutral act -- it has an opportunity cost measured in rural satisfaction.
4. The environment exposes a DIFFERENCE-REWARD signal: each agent is credited
   with its marginal contribution to global welfare (welfare minus the
   counterfactual welfare had it taken a default action). The counterfactual
   is evaluated by the PURE transition `_transition`, from the SAME pre-step
   state as the real transition -- so the difference reward isolates the
   agent's action, nothing else. This is what the coordinated learner in
   algorithms.py consumes.

CALIBRATION SEMANTICS -- READ THIS
----------------------------------
The marginal effects from calibration.py are clamped at zero and then divided
by their global maximum (see __init__). The environment therefore inherits the
RELATIVE per-channel, per-stratum response structure -- which channel matters
more, and for whom -- but NOT the absolute satisfaction scale, and it cannot
represent a channel whose calibrated effect is negative. The transition's other
constants (decay, effectiveness, the inertia/peer blend) are environment design
parameters, not calibrated quantities. "Calibrated environment" should be read,
and described in the manuscript, in exactly this sense.

The environment is deliberately framework-agnostic (pure NumPy): the same
object is driven by the tabular learner and by the PyTorch MAPPO policy.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

CHANNELS = ["t_school", "t_localgov", "t_contract"]


def _validate_calibration(calibration: dict) -> None:
    """Fail early and clearly if the calibration dict is missing what the env needs."""
    try:
        me = calibration["pooled"]["marginal_effects"]
        prof = calibration["attribute_profiles"]
    except (KeyError, TypeError) as exc:                     # noqa: BLE001
        raise KeyError("environment: calibration must contain "
                       "'pooled.marginal_effects' and 'attribute_profiles' "
                       f"(missing {exc}).") from exc
    for ch in CHANNELS:
        if ch not in me or not {"urban", "rural"} <= set(me[ch]):
            raise KeyError("environment: calibration marginal_effects missing "
                           f"urban/rural entry for channel '{ch}'.")
    for strat in ("urban", "rural"):
        if strat not in prof:
            raise KeyError(f"environment: attribute_profiles missing '{strat}'.")
        p = prof[strat]
        if "satisfaction_mean" not in p:
            raise KeyError(f"environment: attribute_profiles['{strat}'] missing "
                           "'satisfaction_mean'.")
        for ch in CHANNELS:
            if ch not in p or "mean" not in p[ch] or "std" not in p[ch]:
                raise KeyError(f"environment: attribute_profiles['{strat}'] "
                               f"missing mean/std for channel '{ch}'.")


class UrbanRuralGovernanceEnv:
    """
    A Markov game with G government agents (urban + rural jurisdictions) and
    G * citizens_per_jurisdiction citizen sub-agents whose satisfaction evolves
    under the calibrated channel-resolved response model.
    """

    # -- construction --------------------------------------------------------
    def __init__(self, cfg: dict, calibration: dict,
                 seed: int = 0,
                 channel_resolved: bool = True,
                 coordination: str = "difference"):
        if coordination not in ("difference", "naive"):
            raise ValueError("coordination must be 'difference' or 'naive'; "
                             f"got {coordination!r}.")
        _validate_calibration(calibration)

        ec = cfg["environment"]
        self.rng = np.random.default_rng(seed)

        self.n_urban = ec["n_urban"]
        self.n_rural = ec["n_rural"]
        self.n_gov = self.n_urban + self.n_rural
        self.n_cit = ec["citizens_per_jurisdiction"]
        self.horizon = ec["horizon"]
        self.budget = ec["budget"]
        self.n_channels = ec["n_channels"]
        self.levels = ec["allocation_levels"]
        self.effectiveness = ec["effectiveness"]
        self.decay = ec["decay"]
        self.social = ec["social_influence"]
        self.degree = ec["network_degree"]
        self.lambda_equity = ec["lambda_equity"]
        self.reward_mix = ec["reward_mix"]
        # transition constants -- config-overridable, defaults preserve behaviour
        self.sat_inertia = ec.get("satisfaction_inertia", 0.7)   # own-state weight
        self.sat_init_std = ec.get("citizen_sat_init_std", 0.12)  # reset noise

        # n_channels MUST match the CHANNELS list, or the coefficient loops and
        # the disclosure matrix disagree.
        if self.n_channels != len(CHANNELS):
            raise ValueError(f"environment.n_channels ({self.n_channels}) must "
                             f"equal len(CHANNELS) ({len(CHANNELS)}).")

        # ABLATION SWITCHES -------------------------------------------------
        # channel_resolved=False collapses the 3-channel portfolio into one
        # dial -> reproduces the "transparency is a single dial" baseline.
        # coordination="naive" replaces the difference reward with raw global
        # welfare -> isolates the value of the coordination mechanism.
        self.channel_resolved = channel_resolved
        self.coordination = coordination

        # stratum label per government agent (0 urban, 1 rural)
        self.is_rural = np.array([0] * self.n_urban + [1] * self.n_rural)

        # per-channel, per-stratum marginal effects from calibration ---------
        # Negative effects are clamped to zero, then the whole matrix is scaled
        # by its global maximum. See the CALIBRATION SEMANTICS note in the
        # module docstring: this keeps the RELATIVE channel/stratum structure,
        # not the absolute satisfaction scale.
        me = calibration["pooled"]["marginal_effects"]
        self.coef = np.zeros((self.n_gov, self.n_channels))
        for g in range(self.n_gov):
            strat = "rural" if self.is_rural[g] else "urban"
            for c, ch in enumerate(CHANNELS):
                self.coef[g, c] = max(me[ch][strat], 0.0)
        self.coef /= max(self.coef.max(), 1e-6)              # normalise

        self.profiles = calibration["attribute_profiles"]
        self._build_citizen_networks()
        self.action_space_n = self.levels ** self.n_channels
        self.reset()

    # -- citizen social networks --------------------------------------------
    def _build_citizen_networks(self) -> None:
        """One ring-lattice social network per jurisdiction (peer influence)."""
        self.neighbours: List[np.ndarray] = []
        for _ in range(self.n_gov):
            adj = np.zeros((self.n_cit, self.n_cit))
            half = max(1, self.degree // 2)
            for i in range(self.n_cit):
                for d in range(1, half + 1):
                    adj[i, (i + d) % self.n_cit] = 1
                    adj[i, (i - d) % self.n_cit] = 1
            # row-normalise so (neighbours @ sat) is the mean neighbour value;
            # the maximum(.,1) guards the degenerate small-n_cit case.
            self.neighbours.append(adj / np.maximum(adj.sum(1, keepdims=True), 1))

    # -- episode reset -------------------------------------------------------
    def reset(self) -> List[np.ndarray]:
        """Initialise disclosure levels and citizen satisfaction from priors."""
        self.t = 0
        # current disclosure on each channel per jurisdiction
        self.disclosure = np.zeros((self.n_gov, self.n_channels))
        for g in range(self.n_gov):
            prof = self.profiles["rural" if self.is_rural[g] else "urban"]
            for c, ch in enumerate(CHANNELS):
                self.disclosure[g, c] = np.clip(
                    self.rng.normal(prof[ch]["mean"], prof[ch]["std"]), 0, 1)
        # citizen satisfaction matrix (G x n_cit), scaled to [0,1]
        self.sat = np.zeros((self.n_gov, self.n_cit))
        for g in range(self.n_gov):
            prof = self.profiles["rural" if self.is_rural[g] else "urban"]
            base = prof["satisfaction_mean"] / 100.0
            self.sat[g] = np.clip(
                self.rng.normal(base, self.sat_init_std, self.n_cit), 0, 1)
        return self._observations()

    # -- action decoding -----------------------------------------------------
    def _decode(self, a: int) -> np.ndarray:
        """
        Map a discrete action index to a 3-channel intensity request.

        Channel-resolved mode: `a` is read as a base-`levels` number, one digit
        per channel. Single-dial mode: only `a % levels` is used and spread
        evenly over the channels -- so the 3-channel portfolio collapses to one
        dial. Note that the action space size (`levels ** n_channels`) is held
        CONSTANT across the ablation, so both arms get networks/Q-tables of
        identical capacity; in single-dial mode this means actions alias onto
        `levels` effective settings (a known, deliberate property of the
        ablation, not a bug).
        """
        if not self.channel_resolved:
            lvl = a % self.levels
            share = np.full(self.n_channels, float(lvl))
        else:
            share = np.zeros(self.n_channels)
            x = a
            for c in range(self.n_channels):
                share[c] = x % self.levels
                x //= self.levels
        # Return RAW per-channel intensity on a [0, levels-1] scale.
        # It is deliberately NOT normalised: the total intensity a jurisdiction
        # requests is what drives its claim on the shared common-pool budget,
        # so an "invest hard" action genuinely pulls more budget than a
        # "minimal" action. Within-jurisdiction channel proportions are
        # preserved downstream in _transition().
        return share

    # -- observations --------------------------------------------------------
    def _observations(self) -> List[np.ndarray]:
        """Per-agent observation: own disclosure, own mean satisfaction,
        stratum flag, national mean satisfaction, normalised time."""
        gmean = float(self.sat.mean())
        obs = []
        for g in range(self.n_gov):
            obs.append(np.concatenate([
                self.disclosure[g],
                [self.sat[g].mean(), float(self.is_rural[g]),
                 gmean, self.t / self.horizon],
            ]).astype(np.float32))
        return obs

    # -- core transition (PURE) ---------------------------------------------
    def _transition(self, disclosure: np.ndarray, sat: np.ndarray,
                    intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        One round of transparency dynamics, as a PURE function.

        Given the current `disclosure` and `sat` and an `intensity` demand
        matrix (G x n_channels, raw from `_decode`), return the resulting
        (new_disclosure, new_satisfaction). This method does NOT mutate any
        environment state -- which is what makes the difference-reward
        counterfactual in `step` correct: the real transition and every
        counterfactual transition are evaluated from the identical pre-step
        state, so the difference reward isolates the agent's action alone.

        A jurisdiction's claim on the shared common-pool budget is proportional
        to its TOTAL requested intensity, so urban-/rural-biased policies
        genuinely move budget between strata. Within a jurisdiction, the budget
        is split across channels in proportion to the per-channel intensity.
        """
        total = intensity.sum(1)                            # demand per agent
        claim = total / max(total.sum(), 1e-6)              # common-pool share
        gov_budget = claim * self.budget                    # sums to self.budget

        new_disc = disclosure.copy()
        new_sat = sat.copy()
        for g in range(self.n_gov):
            # split this jurisdiction's budget across channels by intensity
            if total[g] > 0:
                channel_split = intensity[g] / total[g]
            else:
                channel_split = np.full(self.n_channels,
                                        1.0 / self.n_channels)
            spend = channel_split * gov_budget[g]
            # disclosure: investment raises it, decay erodes it
            new_disc[g] = np.clip(
                disclosure[g] * (1 - self.decay)
                + self.effectiveness * spend, 0, 1)
            # citizen satisfaction responds via the calibrated channel effects
            drive = float(new_disc[g] @ self.coef[g])
            target = np.clip(drive, 0, 1)
            peer = self.neighbours[g] @ sat[g]
            new_sat[g] = np.clip(
                (1 - self.social) * (self.sat_inertia * sat[g]
                                     + (1 - self.sat_inertia) * target)
                + self.social * peer, 0, 1)
        return new_disc, new_sat

    # -- welfare + equity ----------------------------------------------------
    def _welfare(self, sat: np.ndarray) -> Tuple[float, float]:
        """Return (mean satisfaction, urban-rural gap) for a satisfaction matrix."""
        urban_mean = sat[self.is_rural == 0].mean()
        rural_mean = sat[self.is_rural == 1].mean()
        gap = abs(urban_mean - rural_mean)
        return float(sat.mean()), float(gap)

    # -- environment step ----------------------------------------------------
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray],
                                                np.ndarray, bool, Dict]:
        """
        Advance the game one round.

        Returns
        -------
        obs     : next per-agent observations
        rewards : per-agent reward vector (length n_gov)
        done    : episode-termination flag -- a HORIZON TRUNCATION, never an
                  absorbing terminal (the governance process does not "finish",
                  it is cut off). Learners must bootstrap the value at this
                  boundary rather than treat it as a true terminal.
        info    : diagnostics -- global welfare, the equity gap, per-stratum
                  satisfaction, and (under difference coordination) the
                  per-agent difference-reward decomposition.
        """
        intensity = np.stack([self._decode(a) for a in actions])

        # real transition -- evaluated, but NOT yet committed to self.*
        new_disc, new_sat = self._transition(self.disclosure, self.sat, intensity)
        mean_sat, gap = self._welfare(new_sat)
        # global objective: raise mean satisfaction, penalise the urban-rural gap
        global_reward = mean_sat - self.lambda_equity * gap

        rewards = np.zeros(self.n_gov)
        diff_rewards = np.zeros(self.n_gov)
        for g in range(self.n_gov):
            local = float(new_sat[g].mean())
            if self.coordination == "difference":
                # Counterfactual: agent g instead takes the DEFAULT action -- a
                # uniform mid-level request across channels -- while every
                # other agent keeps its real action. This default is the
                # average-action (mean-field) clamp of the Wonderful-Life
                # Utility (Wolpert & Tumer 2001): it is a counterfactual
                # device, not an action any agent takes.
                #
                # The counterfactual is evaluated from self.disclosure /
                # self.sat -- the PRE-step state, still uncommitted -- via the
                # pure _transition. So `shaped` is welfare-with-g's-action
                # minus welfare-with-g's-default, holding everything else
                # fixed: agent g's true marginal contribution.
                cf_intensity = intensity.copy()
                cf_intensity[g] = np.full(self.n_channels,
                                          (self.levels - 1) / 2.0)
                _, cf_sat = self._transition(self.disclosure, self.sat,
                                             cf_intensity)
                cf_mean, cf_gap = self._welfare(cf_sat)
                cf_global = cf_mean - self.lambda_equity * cf_gap
                shaped = global_reward - cf_global        # difference reward
                diff_rewards[g] = shaped
                rewards[g] = (self.reward_mix * shaped
                              + (1 - self.reward_mix) * local)
            else:                                          # naive coordination
                rewards[g] = (self.reward_mix * global_reward
                              + (1 - self.reward_mix) * local)

        # commit the real transition only AFTER every counterfactual has been
        # evaluated against the (still pre-step) self.disclosure / self.sat.
        self.disclosure = new_disc
        self.sat = new_sat
        self.t += 1
        done = self.t >= self.horizon

        info = {"mean_satisfaction": mean_sat, "equity_gap": gap,
                "global_reward": global_reward,
                "urban_satisfaction": float(new_sat[self.is_rural == 0].mean()),
                "rural_satisfaction": float(new_sat[self.is_rural == 1].mean())}
        if self.coordination == "difference":
            info["difference_reward"] = diff_rewards.tolist()
        return self._observations(), rewards, done, info

    # -- convenience ---------------------------------------------------------
    @property
    def obs_dim(self) -> int:
        return self.n_channels + 4

    @property
    def n_actions(self) -> int:
        return self.action_space_n

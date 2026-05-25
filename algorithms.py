"""
algorithms.py  --  the policies (baselines, reproducible baseline learner,
and the paper's contribution).

THREE TIERS
-----------
1. Heuristic baselines (`HeuristicPolicy`): non-learning allocation rules --
   uniform, urban-biased, rural-biased, greedy-need. They show what a
   reasonable but unsophisticated government would do.

2. `TabularIQL`: independent tabular Q-learning. A genuine learner, but with
   NO coordination mechanism -- each agent optimises its own reward. Included
   as the reference baseline that isolates the value of coordination, and as a
   dependency-light path so reviewers can reproduce the study without a GPU.

3. `EquityAwareMAPPO` (THE CONTRIBUTION): multi-agent PPO with a centralised,
   AGENT-SPECIFIC critic and a DIFFERENCE-REWARD training signal. The critic
   sees agent i's own observation together with a compact global summary
   (system-wide disclosure and the urban-rural satisfaction gap) and outputs a
   PER-AGENT value V_i. Pairing per-agent values with the environment's
   per-agent difference reward yields a genuine per-agent advantage, so each
   government agent learns a transparency PORTFOLIO that accounts for its
   marginal effect on national welfare *and* on the urban-rural gap. The
   novelty is not "PPO" -- it is the pairing of a portfolio action space with a
   coordination signal derived from the equity-augmented global objective.

PyTorch (tier 3 only) is imported lazily: if it is unavailable the MAPPO class
raises a clear ImportError and the pipeline (main.py / experiments.py) falls
back to TabularIQL, so every other stage remains runnable.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


# ===========================================================================
# Value-target normaliser  (MAPPO "value normalisation"; Yu et al. 2022)
# ===========================================================================
class _RunningNorm:
    """
    Online mean/variance of a scalar quantity via Chan's parallel-variance
    update. Used to normalise the critic's regression targets: the critic
    predicts a standardised value, which keeps its loss well scaled when the
    reward magnitude shifts (e.g. across the equity-weight sweep E4).

    Pure NumPy/float -- `normalize` / `denormalize` also accept torch tensors,
    since they are plain affine maps with Python-float parameters.
    """

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4                                   # tiny non-zero prior

    def update(self, x) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return
        b_mean, b_var, b_count = float(x.mean()), float(x.var()), int(x.size)
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean += delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        m2 = m_a + m_b + delta * delta * self.count * b_count / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self) -> float:
        return math.sqrt(self.var + 1e-8)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std + self.mean


# ===========================================================================
# 1. Heuristic baselines
# ===========================================================================
class HeuristicPolicy:
    """Non-learning allocation rules over the 3-channel portfolio."""

    def __init__(self, env, kind: str):
        self.env = env
        self.kind = kind

    def _portfolio_to_action(self, share: np.ndarray) -> int:
        """Encode a per-channel level vector into a discrete action index."""
        lv = self.env.levels
        idx = 0
        for c in range(self.env.n_channels):
            idx += int(np.clip(share[c], 0, lv - 1)) * (lv ** c)
        return idx

    def act(self, observations: List[np.ndarray],
            greedy: bool = False) -> List[int]:
        # `greedy` is accepted only so all three policy classes share one
        # act(observations, greedy) signature; heuristic rules are already
        # deterministic, so it is ignored.
        acts = []
        top = self.env.levels - 1
        for g, _ in enumerate(observations):
            rural = self.env.is_rural[g] == 1
            if self.kind == "uniform":
                share = np.full(self.env.n_channels, top // 2 + 1)
            elif self.kind == "urban_biased":
                # invest hard only in urban jurisdictions
                share = (np.full(self.env.n_channels, top) if not rural
                         else np.full(self.env.n_channels, 1))
            elif self.kind == "rural_biased":
                share = (np.full(self.env.n_channels, top) if rural
                         else np.full(self.env.n_channels, 1))
            elif self.kind == "greedy_need":
                # spend where this jurisdiction's disclosure is currently lowest
                disc = self.env.disclosure[g]
                share = np.ones(self.env.n_channels)
                share[int(np.argmin(disc))] = top
            else:
                raise ValueError(f"unknown heuristic '{self.kind}'")
            acts.append(self._portfolio_to_action(share))
        return acts


# ===========================================================================
# 2. Tabular Independent Q-Learning  (reference baseline learner)
# ===========================================================================
class TabularIQL:
    """
    Independent tabular Q-learning. Continuous observations are discretised
    into a small key. No agent sees any other's state -> no coordination. This
    is the control condition against which the coordinated learner is judged.
    """

    def __init__(self, env, cfg: dict, seed: int = 0):
        ac = cfg["algorithms"]["tabular_iql"]
        self.env = env
        self.alpha = ac["alpha"]
        self.gamma = ac["gamma"]
        self.eps = ac["eps_start"]
        self.eps_min = ac["eps_min"]
        self.eps_decay = ac["eps_decay"]
        self.episodes = ac["episodes"]
        self.rng = np.random.default_rng(seed)
        # one Q-table (dict) per government agent
        self.Q: List[Dict] = [dict() for _ in range(env.n_gov)]

    def _key(self, obs: np.ndarray) -> tuple:
        """Coarse discretisation of a continuous observation vector."""
        return tuple(np.round(obs * 3).astype(int))

    def _row(self, g: int, key: tuple) -> np.ndarray:
        if key not in self.Q[g]:
            self.Q[g][key] = np.zeros(self.env.n_actions)
        return self.Q[g][key]

    def act(self, observations: List[np.ndarray],
            greedy: bool = False) -> List[int]:
        acts = []
        for g, obs in enumerate(observations):
            row = self._row(g, self._key(obs))
            if (not greedy) and self.rng.random() < self.eps:
                acts.append(int(self.rng.integers(self.env.n_actions)))
            else:
                acts.append(int(np.argmax(row)))
        return acts

    def train(self) -> List[float]:
        """Run tabular Q-learning; return the per-episode global-reward curve."""
        curve = []
        for _ in range(self.episodes):
            obs = self.env.reset()
            ep_r = 0.0
            done = False
            while not done:
                keys = [self._key(o) for o in obs]
                acts = self.act(obs)
                nxt, rew, done, info = self.env.step(acts)
                for g in range(self.env.n_gov):
                    row = self._row(g, keys[g])
                    nrow = self._row(g, self._key(nxt[g]))
                    # The episode ends by HORIZON TRUNCATION, never by reaching
                    # an absorbing terminal (see environment.py): the governance
                    # process continues past the horizon. The bootstrap term is
                    # therefore ALWAYS kept -- the old `* (not done)` factor
                    # zeroed it at the horizon and taught a spurious
                    # end-of-episode penalty.
                    target = rew[g] + self.gamma * np.max(nrow)
                    row[acts[g]] += self.alpha * (target - row[acts[g]])
                obs = nxt
                ep_r += info["global_reward"]
            self.eps = max(self.eps_min, self.eps * self.eps_decay)
            curve.append(ep_r / self.env.horizon)
        return curve


# ===========================================================================
# 3. Equity-Aware Coordinated MAPPO  (THE CONTRIBUTION)
# ===========================================================================
class EquityAwareMAPPO:
    """
    Multi-Agent PPO with a centralised agent-specific critic and a
    difference-reward training signal derived from the equity-augmented global
    objective.

    Design
    ------
    * Actors are decentralised: each government agent maps its LOCAL observation
      to a distribution over the 3-channel transparency portfolio. Parameters
      are shared across agents; the stratum flag in the observation lets the
      one network specialise to urban vs rural. At deployment a jurisdiction
      needs only local information -- realistic for governance.
    * The critic is CENTRALISED but AGENT-SPECIFIC (the "AS" value input of
      Yu et al. 2022): for agent i it takes i's own observation together with a
      compact, count-invariant global summary -- mean disclosure per channel,
      urban and rural mean satisfaction, and the urban-rural gap -- and outputs
      a PER-AGENT value V_i. A per-agent value is what makes a genuine
      per-agent advantage possible.
    * Training uses the environment's PER-AGENT difference reward. Generalised
      advantage estimation is run per agent (agent i's own reward paired with
      its own V_i), so each agent is credited with its MARGINAL contribution to
      the equity-augmented welfare -- nothing is averaged into a team scalar.
    * The episode ends by horizon truncation, so the advantage at the final
      step bootstraps with V_i(s_T), not zero.
    * Value targets are normalised online (optional, on by default) to keep the
      critic loss well scaled.

    The combination -- portfolio action space + centralised critic + per-agent
    difference reward on an equity-augmented objective -- is the paper's
    methodological contribution.

    Config keys (cfg["algorithms"]["mappo"]): the standard PPO/MAPPO knobs,
    plus two optional ones, `episodes_per_update` (default 1) and `value_norm`
    (default true), so neither requires editing an existing config.
    """

    def __init__(self, env, cfg: dict, seed: int = 0):
        try:
            import torch                                     # noqa: F401
        except Exception as exc:                             # noqa: BLE001
            raise ImportError(
                "EquityAwareMAPPO requires PyTorch. Install it on the training "
                "server, or run with --algo tabular for a torch-free baseline."
            ) from exc
        import torch
        import torch.nn as nn

        mc = cfg["algorithms"]["mappo"]
        self.env = env
        self.episodes = mc["episodes"]
        self.gamma = mc["gamma"]
        self.lam = mc["gae_lambda"]
        self.clip = mc["clip_eps"]
        self.update_epochs = mc["update_epochs"]
        self.ent_coef = mc["entropy_coef"]
        self.val_coef = mc["value_coef"]
        self.max_grad = mc["max_grad_norm"]
        # Several episodes per PPO update (Yu et al. 2022, larger-batch finding).
        # Default 1 reproduces the original batch size; raise it in config for a
        # lower-variance update. The total episode budget is held at `episodes`.
        self.episodes_per_update = max(1, int(mc.get("episodes_per_update", 1)))
        # Online value-target normalisation; on by default ("never hurts").
        self.value_norm = bool(mc.get("value_norm", True))

        dev = mc["device"]
        if dev == "cuda" and not torch.cuda.is_available():
            print("[mappo] WARNING: device='cuda' requested but CUDA is "
                  "unavailable; falling back to CPU.")
            dev = "cpu"
        use_cuda = dev in ("auto", "cuda") and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        torch.manual_seed(seed)

        obs_dim = env.obs_dim
        n_act = env.n_actions
        hid = mc["hidden_dim"]
        n_ch = env.n_channels

        # decentralised actor (shared parameters)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
            nn.Linear(hid, n_act),
        ).to(self.device)

        # centralised, agent-specific critic -> per-agent value V_i.
        # input = own observation (obs_dim)  ++  global summary (n_ch + 3:
        # mean disclosure per channel, urban mean sat, rural mean sat, gap).
        self.global_dim = n_ch + 3
        self.critic_in_dim = obs_dim + self.global_dim
        self.critic = nn.Sequential(
            nn.Linear(self.critic_in_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
            nn.Linear(hid, 1),
        ).to(self.device)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(),
                                          lr=mc["lr_actor"])
        self.opt_critic = torch.optim.Adam(self.critic.parameters(),
                                           lr=mc["lr_critic"])

        # fixed urban/rural agent indices for the global-summary computation
        is_rural = np.asarray(env.is_rural)
        urban = np.where(is_rural == 0)[0]
        rural = np.where(is_rural == 1)[0]
        if urban.size == 0:                                  # degenerate guard
            urban = np.arange(env.n_gov)
        if rural.size == 0:
            rural = np.arange(env.n_gov)
        self._urban_idx = torch.tensor(urban, dtype=torch.long,
                                       device=self.device)
        self._rural_idx = torch.tensor(rural, dtype=torch.long,
                                       device=self.device)

        self.vnorm = _RunningNorm()
        self._torch = torch
        self._nn = nn

    # -- agent-specific critic input ----------------------------------------
    def _global_summary(self, o):
        """
        Compact global state from a (..., G, obs_dim) observation batch ->
        (..., global_dim). Observation layout (see environment.py):
        [disclosure(n_channels), own_sat_mean, is_rural, gmean, time].
        """
        torch = self._torch
        n_ch = self.env.n_channels
        disc_mean = o[..., :n_ch].mean(dim=-2)               # (..., n_ch)
        sat = o[..., n_ch]                                   # (..., G)
        urban_sat = sat[..., self._urban_idx].mean(dim=-1, keepdim=True)
        rural_sat = sat[..., self._rural_idx].mean(dim=-1, keepdim=True)
        gap = (urban_sat - rural_sat).abs()                  # (..., 1)
        return torch.cat([disc_mean, urban_sat, rural_sat, gap], dim=-1)

    def _critic_input(self, o):
        """(..., G, obs_dim) -> (..., G, critic_in_dim): own obs ++ global summary."""
        torch = self._torch
        gsum = self._global_summary(o)                       # (..., global_dim)
        gsum = gsum.unsqueeze(-2).expand(*o.shape[:-1], gsum.shape[-1])
        return torch.cat([o, gsum], dim=-1)

    def _critic_value(self, o, denormalize: bool):
        """
        Per-agent value for a (..., G, obs_dim) batch -> (..., G).

        denormalize=True  -> value mapped back to reward units (for GAE).
        denormalize=False -> the raw (normalised-space) prediction, kept with
                             gradient for the critic regression in _update.
        """
        raw = self.critic(self._critic_input(o)).squeeze(-1)  # (..., G)
        if denormalize and self.value_norm:
            return self.vnorm.denormalize(raw)
        return raw

    # -- action selection ----------------------------------------------------
    def act(self, observations: List[np.ndarray],
            greedy: bool = False) -> List[int]:
        torch = self._torch
        obs = torch.tensor(np.stack(observations),
                           dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.actor(obs)
            if greedy:
                return [int(a) for a in logits.argmax(-1).cpu().numpy()]
            dist = torch.distributions.Categorical(logits=logits)
            return [int(a) for a in dist.sample().cpu().numpy()]

    # -- rollout -------------------------------------------------------------
    def _run_episode(self) -> Dict:
        """Roll out one episode; store the tensors a PPO update needs plus the
        bootstrap value of the post-horizon state."""
        torch = self._torch
        env = self.env
        obs = env.reset()
        ep: Dict[str, list] = {k: [] for k in ("obs", "act", "logp", "rew", "val")}
        done = False
        ep_global = 0.0
        while not done:
            o = torch.tensor(np.stack(obs), dtype=torch.float32,
                             device=self.device)
            with torch.no_grad():
                logits = self.actor(o)
                dist = torch.distributions.Categorical(logits=logits)
                acts = dist.sample()
                logp = dist.log_prob(acts)
                val = self._critic_value(o, denormalize=True)   # (G,) real units
            nxt, rew, done, info = env.step([int(a) for a in acts.cpu().numpy()])
            ep["obs"].append(o)
            ep["act"].append(acts)
            ep["logp"].append(logp)
            ep["rew"].append(torch.tensor(rew, dtype=torch.float32,
                                          device=self.device))
            ep["val"].append(val)
            obs = nxt
            ep_global += info["global_reward"]
        # Bootstrap: the horizon is a TRUNCATION, not an absorbing terminal,
        # so the value beyond it is V(s_T), never 0.
        o_T = torch.tensor(np.stack(obs), dtype=torch.float32,
                           device=self.device)
        with torch.no_grad():
            ep["boot_val"] = self._critic_value(o_T, denormalize=True)  # (G,)
        ep["ep_global"] = ep_global / env.horizon
        return ep

    def _collect(self) -> List[Dict]:
        """Roll out `episodes_per_update` episodes for one PPO update."""
        return [self._run_episode() for _ in range(self.episodes_per_update)]

    # -- generalised advantage estimation ------------------------------------
    def _episode_gae(self, ep: Dict):
        """
        PER-AGENT generalised advantage estimation for one episode.
        Returns (adv, ret), each of shape (T, G).

        Issue-2 fix: agent i's own difference reward is paired with its own
        value V_i; the advantage stays per-agent. The previous implementation
        averaged the reward vector to a team scalar (`rew[t].mean()`) and
        broadcast one shared advantage to all agents, discarding exactly the
        per-agent credit the difference reward exists to provide.
        """
        torch = self._torch
        val = torch.stack(ep["val"])           # (T, G)  denormalised values
        rew = torch.stack(ep["rew"])           # (T, G)  per-agent diff. reward
        boot = ep["boot_val"]                  # (G,)
        T = val.shape[0]
        adv = torch.zeros_like(val)            # (T, G)
        last = torch.zeros_like(boot)          # (G,)
        for t in reversed(range(T)):
            v_next = val[t + 1] if t + 1 < T else boot
            delta = rew[t] + self.gamma * v_next - val[t]    # (G,)
            last = delta + self.gamma * self.lam * last      # (G,)
            adv[t] = last
        ret = adv + val                        # (T, G)
        return adv, ret

    # -- PPO update ----------------------------------------------------------
    def _update(self, episodes: List[Dict]) -> None:
        torch, nn = self._torch, self._nn

        advs, rets, obss, acts_, logps = [], [], [], [], []
        for ep in episodes:
            adv, ret = self._episode_gae(ep)                 # (T,G), (T,G)
            advs.append(adv)
            rets.append(ret)
            obss.append(torch.stack(ep["obs"]))              # (T,G,obs_dim)
            acts_.append(torch.stack(ep["act"]))             # (T,G)
            logps.append(torch.stack(ep["logp"]))            # (T,G)
        adv = torch.cat(advs, dim=0)                         # (B*T, G)
        ret = torch.cat(rets, dim=0)                         # (B*T, G)
        obs = torch.cat(obss, dim=0)                         # (B*T, G, obs_dim)
        acts = torch.cat(acts_, dim=0)                       # (B*T, G)
        old_logp = torch.cat(logps, dim=0).detach()          # (B*T, G)

        # Refresh the value-target normaliser on this batch's returns, then have
        # the critic regress toward the NORMALISED return.
        if self.value_norm:
            self.vnorm.update(ret.detach().cpu().numpy())
            ret_target = self.vnorm.normalize(ret).detach()
        else:
            ret_target = ret.detach()

        # advantage normalisation, pooled over all (timestep, agent) pairs
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).detach()

        for _ in range(self.update_epochs):
            logits = self.actor(obs)                         # (B*T, G, n_act)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(acts)                       # (B*T, G)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - old_logp)               # (B*T, G)
            # PER-AGENT advantage weighting -- no shared-scalar broadcast.
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - self.clip,
                                  1 + self.clip) * adv
            actor_loss = -(torch.min(unclipped, clipped).mean()
                           + self.ent_coef * entropy)

            self.opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad)
            self.opt_actor.step()

            value = self._critic_value(obs, denormalize=False)  # (B*T, G)
            critic_loss = self.val_coef * ((value - ret_target) ** 2).mean()
            self.opt_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad)
            self.opt_critic.step()

    # -- training loop -------------------------------------------------------
    def train(self) -> List[float]:
        """Train the coordinated policy; return the per-episode reward curve."""
        curve: List[float] = []
        n_updates = max(1, self.episodes // self.episodes_per_update)
        milestone = max(1, n_updates // 10)
        for u in range(n_updates):
            episodes = self._collect()
            self._update(episodes)
            for ep in episodes:
                curve.append(ep["ep_global"])
            if (u + 1) % milestone == 0:
                recent = float(np.mean(curve[-50:]))
                print(f"[mappo] update {u + 1}/{n_updates}  "
                      f"({len(curve)} episodes)  global_reward={recent:.4f}")
        return curve


# ===========================================================================
# Factory
# ===========================================================================
def build_policy(name: str, env, cfg: dict, seed: int = 0):
    """Construct a policy by name (used by experiments.py and main.py)."""
    if name in cfg["algorithms"]["baselines"]:
        return HeuristicPolicy(env, name)
    if name == "tabular":
        return TabularIQL(env, cfg, seed)
    if name == "mappo":
        return EquityAwareMAPPO(env, cfg, seed)
    raise ValueError(f"unknown policy '{name}'")

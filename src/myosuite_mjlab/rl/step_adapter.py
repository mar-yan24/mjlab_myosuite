"""VecEnv wrapper that reconciles step() arity between mjlab and rsl_rl.

mjlab's ``RslRlVecEnvWrapper`` and current rsl-rl-lib both use the 4-tuple
``(obs, rewards, dones, extras)``. A future rsl_rl may return 5. This adapter
forwards whatever the underlying env returns so either side can evolve without
breaking the other.
"""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from typing import Any, cast


class RslRlStepAdapter(VecEnv):
    """Wraps a VecEnv so step() always returns (obs, rewards, dones, infos, extras)."""

    def __init__(self, env: VecEnv) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.num_actions = env.num_actions

    @property
    def unwrapped(self) -> VecEnv:
        """Return the innermost env (ManagerBasedRlEnv) for runner state access."""
        out = self.env
        while hasattr(out, "unwrapped"):
            out = out.unwrapped  # type: ignore[union-attr]
        return out

    @property
    def cfg(self):  # type: ignore[no-any-return]
        """Pass the underlying env's cfg through so rsl_rl's OnPolicyRunner can
        read it (rsl_rl >=4 reads ``env.cfg`` during __init__)."""
        return getattr(self.env, "cfg", None)

    @property
    def observation_space(self):  # type: ignore[no-any-return]
        return self.env.observation_space

    @property
    def action_space(self):  # type: ignore[no-any-return]
        return self.env.action_space

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf  # type: ignore[union-attr]

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:  # type: ignore[override]
        self.env.episode_length_buf = value  # type: ignore[union-attr]

    def get_observations(self) -> TensorDict:
        return self.env.get_observations()  # type: ignore[union-attr]

    def reset(self) -> tuple[TensorDict, dict]:
        return self.env.reset()  # type: ignore[union-attr]

    def step(  # type: ignore[override]
        self, actions: torch.Tensor
    ) -> tuple[Any, ...]:
        """Pass the underlying env's step return through unchanged."""
        return cast("tuple[Any, ...]", self.env.step(actions))  # type: ignore[union-attr]

    def seed(self, seed: int = -1) -> int:
        return self.env.seed(seed)  # type: ignore[union-attr]

    def close(self) -> None:
        self.env.close()  # type: ignore[union-attr]

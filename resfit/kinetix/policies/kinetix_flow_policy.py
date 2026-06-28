# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from resfit.kinetix.utils.deps import configure_jax_gpu, jax_gpu_device, place_nnx_on_device

configure_jax_gpu()

import pickle
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import Tensor

import flax.nnx as nnx

from resfit.kinetix.policies.configuration_kinetix_flow import KinetixFlowConfig
from resfit.kinetix.policies.flow_model import FlowPolicy
from resfit.kinetix.policies.model_config import ModelConfig
from resfit.kinetix.utils.checkpoints import download_kinetix_checkpoint
from resfit.lerobot.policies.pretrained import PreTrainedPolicy


class KinetixFlowPolicy(PreTrainedPolicy):
    """Adapter exposing rtc-kinetix flow policies through the LeRobot policy API."""

    config_class = KinetixFlowConfig
    name = "kinetix_flow"

    def __init__(self, config: KinetixFlowConfig, flow_policy: FlowPolicy):
        super().__init__(config)
        self._flow_policy = flow_policy
        self._rng = jax.random.key(0)
        self._jax_device = jax_gpu_device()
        self._action_queues: list[deque[np.ndarray]] = []
        self._device = torch.device("cpu")

        num_steps = config.num_flow_steps

        @nnx.jit
        def _jit_action(key, obs):
            return flow_policy.action(key, obs, num_steps)

        self._jit_action = _jit_action

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_path: str | Path,
        obs_dim: int,
        action_dim: int,
        model_config: ModelConfig | None = None,
        n_action_steps: int = 1,
        num_flow_steps: int = 5,
    ) -> KinetixFlowPolicy:
        checkpoint_file = download_kinetix_checkpoint(checkpoint_path)
        with checkpoint_file.open("rb") as f:
            state_dict = pickle.load(f)

        model_config = model_config or ModelConfig()
        rng = jax.random.key(0)
        flow_policy = FlowPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=model_config,
            rngs=nnx.Rngs(rng),
        )
        graphdef, state = nnx.split(flow_policy)
        state.replace_by_pure_dict(state_dict)
        flow_policy = place_nnx_on_device(nnx.merge(graphdef, state))

        config = KinetixFlowConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_action_steps=n_action_steps,
            chunk_size=model_config.action_chunk_size,
            num_flow_steps=num_flow_steps,
            model=model_config,
        )
        return cls(config, flow_policy)

    def reset(self, env_ids: list[int] | Tensor | None = None) -> None:
        if env_ids is None:
            self._action_queues = []
            return

        if isinstance(env_ids, Tensor):
            env_ids_list = env_ids.detach().cpu().tolist()
        else:
            env_ids_list = env_ids
        if isinstance(env_ids_list, int):
            env_ids_list = [env_ids_list]
        for env_id in env_ids_list:
            if 0 <= env_id < len(self._action_queues):
                self._action_queues[env_id] = deque()

    def get_optim_params(self) -> dict:
        raise NotImplementedError("Kinetix flow base policy is inference-only in residual RL.")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("Kinetix flow base policy is inference-only in residual RL.")

    def to(self, device=None, dtype=None, non_blocking=False):
        del dtype, non_blocking
        if device is not None:
            self._device = torch.device(device)
        return self

    def eval(self):
        return self

    @staticmethod
    def _batch_size(batch: dict[str, Tensor]) -> int:
        state = batch["observation.state"]
        return state.shape[0]

    def _ensure_queues(self, batch_size: int) -> None:
        if len(self._action_queues) < batch_size:
            self._action_queues.extend(deque() for _ in range(batch_size - len(self._action_queues)))

    def _predict_chunk(self, obs_np: np.ndarray) -> np.ndarray:
        self._rng, key = jax.random.split(self._rng)
        key = jax.device_put(key, self._jax_device)
        obs = jax.device_put(jnp.asarray(obs_np[None], dtype=jnp.float32), self._jax_device)
        chunk = self._jit_action(key, obs)
        return np.asarray(jax.device_get(chunk[0]), dtype=np.float32)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], chunk_length: int) -> Tensor:
        """Predict the first ``chunk_length`` actions from a single-environment batch."""
        state = batch["observation.state"]
        if isinstance(state, Tensor):
            if state.ndim == 2:
                state_np = state[0].detach().cpu().numpy()
                device = state.device
            else:
                state_np = state.detach().cpu().numpy()
                device = state.device
        else:
            state_np = np.asarray(state, dtype=np.float32)
            device = self._device

        chunk_np = self._predict_chunk(state_np)
        return torch.as_tensor(chunk_np[:chunk_length].copy(), dtype=torch.float32, device=device)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        del noise
        batch_size = self._batch_size(batch)
        self._ensure_queues(batch_size)

        actions: list[Tensor] = []
        for index in range(batch_size):
            if len(self._action_queues[index]) == 0:
                state = batch["observation.state"][index]
                if isinstance(state, Tensor):
                    state_np = state.detach().cpu().numpy()
                else:
                    state_np = np.asarray(state, dtype=np.float32)
                chunk = self._predict_chunk(state_np)
                for action in chunk[: self.config.n_action_steps]:
                    self._action_queues[index].append(action.astype(np.float32, copy=False))
            action_np = self._action_queues[index].popleft()
            actions.append(torch.as_tensor(action_np.copy(), device=self._device))

        return torch.stack(actions, dim=0)

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from torch import Tensor

from resfit.lerobot.policies.pi0.configuration_pi0 import PI0Config
from resfit.lerobot.policies.pretrained import PreTrainedPolicy


def lerobot_pi0_config_for_openpi(openpi_train_config) -> PI0Config:
    action_horizon = openpi_train_config.model.action_horizon
    return PI0Config(
        input_features={
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        },
        n_action_steps=action_horizon,
        chunk_size=action_horizon,
    )


class OpenPIPi0AlohaSimPolicy(PreTrainedPolicy):
    """Adapter that exposes an OpenPI pi0 policy through the LeRobot API."""

    config_class = PI0Config
    name = "pi0"

    def __init__(self, openpi_policy, config: PI0Config):
        super().__init__(config)
        self._openpi_policy = openpi_policy
        self._action_queues: list[deque[np.ndarray]] = []
        self._device = torch.device("cpu")

    @classmethod
    def from_openpi(cls, openpi_policy, *, openpi_config_name: str) -> OpenPIPi0AlohaSimPolicy:
        from openpi.training import config as openpi_config

        train_config = openpi_config.get_config(openpi_config_name)
        return cls(openpi_policy, lerobot_pi0_config_for_openpi(train_config))

    @property
    def _action_queue(self) -> deque[np.ndarray]:
        if not self._action_queues:
            return deque()
        return self._action_queues[0]

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
        raise NotImplementedError("OpenPI pi0 base policy is inference-only in residual RL.")

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("OpenPI pi0 base policy is inference-only in residual RL.")

    def to(self, device=None, dtype=None, non_blocking=False):
        del dtype, non_blocking
        if device is not None:
            self._device = torch.device(device)
        return self

    def eval(self):
        return self

    @staticmethod
    def _batch_size(batch: dict[str, Tensor]) -> int:
        for value in batch.values():
            if isinstance(value, Tensor):
                return value.shape[0]
        raise ValueError(f"Could not infer batch size from batch keys: {list(batch.keys())}")

    def needs_inference(self, batch: dict[str, Tensor]) -> bool:
        batch_size = self._batch_size(batch)
        self._ensure_queues(batch_size)
        return any(len(self._action_queues[index]) == 0 for index in range(batch_size))

    @staticmethod
    def _batch_to_openpi_obs(batch: dict, index: int) -> dict:
        state = batch["observation.state"][index]
        image = batch["observation.images.top"][index]
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        obs = {
            "state": state,
            "images": {"cam_high": image},
        }
        if "task" in batch:
            tasks = batch["task"]
            if isinstance(tasks, (list, tuple)):
                obs["prompt"] = tasks[index]
            else:
                obs["prompt"] = tasks
        return obs

    def _ensure_queues(self, batch_size: int) -> None:
        if len(self._action_queues) < batch_size:
            self._action_queues.extend(deque() for _ in range(batch_size - len(self._action_queues)))

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        del noise
        batch_size = self._batch_size(batch)
        self._ensure_queues(batch_size)

        actions: list[Tensor] = []
        for index in range(batch_size):
            if len(self._action_queues[index]) == 0:
                result = self._openpi_policy.infer(self._batch_to_openpi_obs(batch, index))
                action_chunk = result["actions"][: self.config.n_action_steps]
                for action in action_chunk:
                    self._action_queues[index].append(np.asarray(action, dtype=np.float32))
            action_np = self._action_queues[index].popleft()
            actions.append(torch.as_tensor(action_np, device=self._device))

        return torch.stack(actions, dim=0)

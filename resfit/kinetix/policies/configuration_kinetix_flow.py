# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

from resfit.kinetix.policies.model_config import ModelConfig
from resfit.lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("kinetix_flow")
@dataclass
class KinetixFlowConfig(PreTrainedConfig):
    """Configuration for rtc-kinetix flow-matching base policies."""

    obs_dim: int = 0
    action_dim: int = 0
    n_action_steps: int = 1
    chunk_size: int = 8
    num_flow_steps: int = 5
    model: ModelConfig = field(default_factory=ModelConfig)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    def __post_init__(self):
        super().__post_init__()

        if self.chunk_size is None:
            self.chunk_size = self.model.action_chunk_size
        if self.n_action_steps is None:
            self.n_action_steps = 1

        self.input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(self.obs_dim,)),
        }
        self.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,)),
        }

    def get_optimizer_preset(self) -> AdamWConfig:
        raise NotImplementedError("Kinetix flow base policy is inference-only in residual RL.")

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError("Kinetix flow policy requires observation.state.")
        if self.action_feature is None:
            raise ValueError("Kinetix flow policy requires action outputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

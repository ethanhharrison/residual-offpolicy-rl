# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

from resfit.lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("pi0")
@dataclass
class PI0Config(PreTrainedConfig):
    """Minimal PI0 config for the OpenPI adapter used in residual RL."""

    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=1e-4)

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("PI0 requires at least one image input feature.")

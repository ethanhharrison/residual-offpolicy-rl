# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    channel_dim: int = 256
    channel_hidden_dim: int = 512
    token_hidden_dim: int = 64
    num_layers: int = 4
    action_chunk_size: int = 8
    simulated_delay: int | None = None

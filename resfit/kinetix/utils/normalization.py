# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import numpy as np


def kinetix_dataset_stats(obs_dim: int, action_dim: int) -> dict[str, dict[str, np.ndarray]]:
    """Default normalization stats for online-only Kinetix residual RL."""
    return {
        "observation.state": {
            "mean": np.zeros(obs_dim, dtype=np.float32),
            "std": np.ones(obs_dim, dtype=np.float32),
        },
        "action": {
            "min": np.full(action_dim, -1.0, dtype=np.float32),
            "max": np.full(action_dim, 1.0, dtype=np.float32),
        },
    }

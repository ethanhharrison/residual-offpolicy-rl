# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# Adapted from Physical-Intelligence/real-time-chunking-kinetix (MIT License).

from __future__ import annotations

from resfit.kinetix.utils.deps import configure_jax_gpu

configure_jax_gpu()

import jax
import jax.numpy as jnp
import kinetix.environment.wrappers as wrappers

from resfit.kinetix.constants import ACTION_NOISE_STD


class NoisyActionWrapper(wrappers.UnderspecifiedEnvWrapper):
    """Adds Gaussian noise to actions, matching rtc-kinetix evaluation."""

    def step_env(self, key, state, action, params):
        key1, key2 = jax.random.split(key)
        action = action + jax.random.normal(key1, action.shape) * ACTION_NOISE_STD
        return self._env.step_env(key2, state, action, params)

    def reset_to_level(self, rng, level, params):
        return self._env.reset_to_level(rng, level, params)

    def action_space(self, params):
        return self._env.action_space(params)

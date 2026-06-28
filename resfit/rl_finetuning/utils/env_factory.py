# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from resfit.aloha.environments.aloha_sim import create_vectorized_env as create_aloha_vectorized_env
from resfit.dexmg.environments.dexmg import create_vectorized_env as create_dexmg_vectorized_env
from resfit.kinetix.constants import KINETIX_TASKS

ALOHA_TASKS = {"AlohaTransferCube", "AlohaInsertion"}


def _create_kinetix_vectorized_env(**kwargs):
    from resfit.kinetix.environments.kinetix_env import create_vectorized_env

    return create_vectorized_env(**kwargs)


def create_vectorized_env(
    *,
    env_type: str,
    env_name: str,
    num_envs: int,
    device: str = "cpu",
    camera_size: int | None = None,
    debug: bool = False,
    video_key: str = "observation.images.agentview",
    seed: int | None = None,
):
    if env_type == "kinetix" or env_name in KINETIX_TASKS:
        return _create_kinetix_vectorized_env(
            env_name=env_name,
            num_envs=num_envs,
            device=device,
            debug=debug,
            video_key=video_key,
            seed=seed,
        )

    if env_type == "aloha" or env_name in ALOHA_TASKS:
        return create_aloha_vectorized_env(
            env_name=env_name,
            num_envs=num_envs,
            device=device,
            rl_camera_size=camera_size or 84,
            debug=debug,
            video_key=video_key,
        )

    return create_dexmg_vectorized_env(
        env_name=env_name,
        num_envs=num_envs,
        device=device,
        camera_size=camera_size or 84,
        debug=debug,
        video_key=video_key,
    )

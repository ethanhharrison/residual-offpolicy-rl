# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import collections
import logging
from typing import Callable

import gymnasium as gym
import gym_aloha  # noqa: F401 - registers gym_aloha/* env IDs with Gymnasium
import numpy as np
import torch

from resfit.aloha.constants import PI0_BASE_CAMERA_HEIGHT, PI0_BASE_CAMERA_WIDTH

logger = logging.getLogger(__name__)

# Mapping from user-facing task names to gym-aloha environment IDs.
ALOHA_TASKS = {
    "AlohaTransferCube": "AlohaTransferCube-v0",
    "AlohaInsertion": "AlohaInsertion-v0",
}

ALOHA_HORIZONS = {
    "AlohaTransferCube-v0": 400,
    "AlohaInsertion-v0": 400,
}


def _patch_aloha_task_observation(task, rl_height: int, rl_width: int) -> None:
    """Render one low-res camera per step instead of three full-size views."""

    def get_observation(physics):
        obs = collections.OrderedDict()
        obs["qpos"] = task.get_qpos(physics)
        obs["qvel"] = task.get_qvel(physics)
        obs["env_state"] = task.get_env_state(physics)
        obs["images"] = {
            "top": physics.render(height=rl_height, width=rl_width, camera_id="top"),
        }
        return obs

    task.get_observation = get_observation


class AlohaGymWrapper:
    """Gymnasium wrapper for gym-aloha environments in LeRobot observation format."""

    def __init__(
        self,
        task: str,
        render_size: tuple[int, int] | int | None = None,
        rl_camera_size: int = 84,
        env_id: int = 0,
    ):
        gym_task = ALOHA_TASKS.get(task, task)
        if gym_task not in ALOHA_HORIZONS:
            raise ValueError(f"Unknown Aloha task: {task}. Supported: {list(ALOHA_TASKS)}")

        self.task = gym_task
        self.original_task = task
        self.rl_camera_size = rl_camera_size
        if render_size is None:
            self.render_size = (PI0_BASE_CAMERA_HEIGHT, PI0_BASE_CAMERA_WIDTH)
        elif isinstance(render_size, int):
            self.render_size = (render_size, render_size)
        else:
            self.render_size = render_size
        self.env_id = env_id
        self.horizon = ALOHA_HORIZONS[gym_task]
        self.video_key = "observation.images.top"

        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 50, "horizon": self.horizon}
        self.spec = None
        self.render_mode = "rgb_array"
        self.episode_steps = 0

        self.env = gym.make(
            f"gym_aloha/{gym_task}",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            max_episode_steps=self.horizon,
            disable_env_checker=True,
        )
        _patch_aloha_task_observation(
            self.env.unwrapped._env.task,
            rl_height=self.rl_camera_size,
            rl_width=self.rl_camera_size,
        )

        action_dim = self.env.action_space.shape[0]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        sample_obs, _ = self.env.reset()
        processed_obs = self._process_obs(sample_obs)
        obs_spaces = {}
        for key, value in processed_obs.items():
            obs_spaces[key] = gym.spaces.Box(
                low=-np.inf if "state" in key else 0.0,
                high=np.inf if "state" in key else 1.0,
                shape=value.shape,
                dtype=value.dtype,
            )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _process_image(self, img: np.ndarray) -> np.ndarray:
        """Convert HWC uint8 image to CHW float32 in [0, 1]."""
        img = img.astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))

    def _process_obs(self, obs: dict) -> dict:
        processed: dict[str, np.ndarray] = {}

        agent_pos = obs["agent_pos"]
        if agent_pos.ndim == 0:
            agent_pos = np.array([agent_pos])
        processed["observation.state"] = agent_pos.astype(np.float32)

        pixels = obs["pixels"]
        if isinstance(pixels, dict):
            for cam_name, img in pixels.items():
                processed[f"observation.images.{cam_name}"] = self._process_image(img)
        else:
            processed["observation.images.top"] = self._process_image(pixels)

        return processed

    def seed(self, seed=None):
        return [seed]

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        processed_obs = self._process_obs(obs)
        self.episode_steps = 0
        return processed_obs, info

    def step(self, action):
        if hasattr(action, "cpu"):
            action = action.cpu().numpy()
        if action.ndim > 1:
            action = action[0]

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        processed_obs = self._process_obs(obs)

        is_success = bool(info.get("is_success", False))
        reward_scalar = 1.0 if is_success else 0.0
        terminated_scalar = is_success
        truncated_scalar = bool(terminated or truncated)

        if terminated_scalar or truncated_scalar:
            info = {
                **info,
                "success": is_success,
                "episode_steps": self.episode_steps,
            }
            self.episode_steps = 0

        return processed_obs, reward_scalar, terminated_scalar, truncated_scalar, info

    def render(self):
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("No frame returned from Aloha environment")
        return frame

    def render_base_policy_image(self) -> np.ndarray:
        """Render the top camera at pi0 resolution (only when base policy re-infers)."""
        physics = self.env.unwrapped._env.physics
        return physics.render(
            height=PI0_BASE_CAMERA_HEIGHT,
            width=PI0_BASE_CAMERA_WIDTH,
            camera_id="top",
        )

    def set_video_key(self, video_key: str):
        self.video_key = video_key

    def close(self):
        self.env.close()

    @property
    def unwrapped(self):
        return self

    def get_wrapper_attr(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    def set_wrapper_attr(self, name: str, value):
        setattr(self, name, value)


def make_aloha_env(
    task: str,
    render_size: tuple[int, int] | int | None = None,
    rl_camera_size: int = 84,
    env_id: int = 0,
) -> Callable[[], AlohaGymWrapper]:
    def _make():
        return AlohaGymWrapper(
            task=task,
            render_size=render_size,
            rl_camera_size=rl_camera_size,
            env_id=env_id,
        )

    return _make


class VectorizedEnvWrapper:
    """Wrapper around gymnasium vectorized environments with torch observation conversion."""

    def __init__(
        self,
        vec_env: gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
        video_key: str,
        device: str = "cpu",
    ):
        self.vec_env = vec_env
        self.video_key = video_key
        self._last_obs = None
        self.device = device

    def reset(self, **kwargs):
        obs, info = self.vec_env.reset(**kwargs)
        self._last_obs = obs
        obs = self._convert_obs_to_torch(obs, self.device)
        return obs, info

    def step(self, actions):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        obs, rewards, terminated, truncated, info = self.vec_env.step(actions)
        self._last_obs = obs

        obs = self._convert_obs_to_torch(obs, self.device)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool)
        truncated = torch.tensor(truncated, device=self.device, dtype=torch.bool)
        return obs, rewards, terminated, truncated, info

    def render(self) -> np.ndarray:
        frames: np.ndarray | None = self.vec_env.render()
        if frames is None:
            raise RuntimeError("No frames returned from vectorized environment")
        return frames

    @property
    def fps(self):
        return self.vec_env.metadata["render_fps"]

    @property
    def metadata(self):
        return self.vec_env.metadata

    @property
    def num_envs(self):
        return self.vec_env.num_envs

    def close(self):
        return self.vec_env.close()

    def __getattr__(self, name: str):
        return getattr(self.vec_env, name)

    def _convert_obs_to_torch(self, obs, device):
        non_blocking = device != "cpu"
        if isinstance(obs, dict):
            torch_obs = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    tensor = torch.from_numpy(value)
                    if tensor.dtype == torch.float64:
                        tensor = tensor.float()
                    torch_obs[key] = tensor.to(device, non_blocking=non_blocking)
                else:
                    torch_obs[key] = value
            return torch_obs
        if isinstance(obs, np.ndarray):
            tensor = torch.from_numpy(obs)
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            return tensor.to(device, non_blocking=non_blocking)
        return obs

    def render_base_policy_images(self) -> torch.Tensor:
        """Return pi0-resolution top-camera frames as float32 CHW tensors on *device*."""
        frames = self.vec_env.call("render_base_policy_image")
        batch = []
        non_blocking = self.device != "cpu"
        for frame in frames:
            array = np.asarray(frame)
            if not array.flags["C_CONTIGUOUS"]:
                array = array.copy()
            tensor = torch.from_numpy(array).to(self.device, non_blocking=non_blocking).float().div_(255.0)
            batch.append(tensor.permute(2, 0, 1))
        return torch.stack(batch, dim=0)


def create_vectorized_env(
    env_name: str,
    num_envs: int,
    device: str = "cpu",
    render_size: tuple[int, int] | int | None = None,
    rl_camera_size: int = 84,
    debug: bool = False,
    video_key: str = "observation.images.top",
) -> VectorizedEnvWrapper:
    """Create a vectorized Aloha Sim environment."""
    import gymnasium as gym

    env_fns = [
        make_aloha_env(env_name, render_size, rl_camera_size=rl_camera_size, env_id=i) for i in range(num_envs)
    ]

    use_sync = debug or num_envs == 1
    if use_sync:
        vec_env = gym.vector.SyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    else:
        vec_env = gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=True,
            copy=True,
            context="spawn",
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )

    vec_env.call("set_wrapper_attr", "video_key", video_key)

    wrapped_env = VectorizedEnvWrapper(vec_env, video_key, device)
    wrapped_env.env_name = env_name
    wrapped_env.render_size = render_size

    logger.debug(f"Created {num_envs} vectorized {env_name} Aloha environments")
    return wrapped_env

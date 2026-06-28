# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from resfit.kinetix.utils.deps import configure_jax_gpu, jax_gpu_device

configure_jax_gpu()

import logging
from typing import Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as kinetix_wrappers
import kinetix.util.saving as saving
import numpy as np
import torch
from kinetix.render.renderer_pixels import make_render_pixels

from resfit.kinetix.constants import FRAME_SKIP, LARGE_ENV_PARAMS, RENDER_SCREEN_DIM, task_to_level_path
from resfit.kinetix.environments.wrappers import NoisyActionWrapper

logger = logging.getLogger(__name__)


def _build_kinetix_env():
    static_env_params = kenv_state.StaticEnvParams(**LARGE_ENV_PARAMS, frame_skip=FRAME_SKIP)
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    env = kinetix_wrappers.LogWrapper(kinetix_wrappers.AutoReplayWrapper(NoisyActionWrapper(env)))
    return env, static_env_params


def _load_level(level_path, static_env_params, env_params):
    level, level_static_env_params, level_env_params = saving.load_from_json_file(str(level_path))
    assert level_static_env_params == static_env_params, (
        f"Level static params mismatch for {level_path}: {level_static_env_params} != {static_env_params}"
    )
    assert level_env_params == env_params, (
        f"Level env params mismatch for {level_path}: {level_env_params} != {env_params}"
    )
    return level


class KinetixGymWrapper:
    """Single Kinetix environment exposing LeRobot-style state observations."""

    def __init__(self, task: str, env_id: int = 0, seed: int | None = None):
        self.task = task
        self.env_id = env_id
        self.level_path = task_to_level_path(task)

        self._env, self._static_env_params = _build_kinetix_env()
        self._env_params = kenv_state.EnvParams()
        self._level = _load_level(self.level_path, self._static_env_params, self._env_params)
        self.horizon = int(self._env_params.max_timesteps)

        self._rng = jax.random.key(seed if seed is not None else env_id)
        self._state = None
        self.episode_steps = 0

        self.metadata = {"render_fps": 15, "horizon": self.horizon}
        self.spec = None
        self.render_mode = None
        self.video_key = "observation.state"

        action_dim = int(self._env.action_space(self._env_params).shape[0])
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        self._jax_device = jax_gpu_device()
        self._jit_reset = jax.jit(self._reset_impl, device=self._jax_device)
        self._jit_step = jax.jit(self._step_impl, device=self._jax_device)

        render_static_params = self._static_env_params.replace(screen_dim=RENDER_SCREEN_DIM)
        render_pixels = make_render_pixels(self._env_params, render_static_params)

        def _render_impl(state):
            env_state = state
            while not isinstance(env_state, kenv_state.EnvState):
                env_state = env_state.env_state
            pixels = render_pixels(env_state)
            return pixels.round().astype(jnp.uint8).transpose(1, 0, 2)[::-1]

        self._jit_render = jax.jit(_render_impl, device=self._jax_device)

        self._rng, reset_key = jax.random.split(self._rng)
        reset_key = jax.device_put(reset_key, self._jax_device)
        obs, self._state = self._jit_reset(reset_key)
        obs = np.asarray(obs, dtype=np.float32)
        state_dim = obs.shape[0]
        self.observation_space = gym.spaces.Dict(
            {
                "observation.state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
                ),
            }
        )

    def _reset_impl(self, rng):
        return self._env.reset_to_level(rng, self._level, self._env_params)

    def _step_impl(self, rng, state, action):
        return self._env.step(rng, state, action, self._env_params)

    def seed(self, seed=None):
        if seed is not None:
            self._rng = jax.random.key(seed + self.env_id)
        return [seed]

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            self.seed(seed)
        self._rng, reset_key = jax.random.split(self._rng)
        reset_key = jax.device_put(reset_key, self._jax_device)
        obs, self._state = self._jit_reset(reset_key)
        obs = np.asarray(obs, dtype=np.float32)
        self.episode_steps = 0
        return {"observation.state": obs}, {}

    def step(self, action):
        if hasattr(action, "cpu"):
            action = action.detach().cpu().numpy()
        if action.ndim > 1:
            action = action[0]
        action = np.asarray(action, dtype=np.float32)

        self._rng, step_key = jax.random.split(self._rng)
        step_key = jax.device_put(step_key, self._jax_device)
        action = jax.device_put(jnp.asarray(action, dtype=jnp.float32), self._jax_device)
        obs, self._state, reward, done, info = self._jit_step(step_key, self._state, action)
        obs = np.asarray(obs, dtype=np.float32)
        reward_scalar = float(reward)
        done_bool = bool(done)
        self.episode_steps += 1

        terminated = done_bool and reward_scalar > 0.0
        truncated = done_bool and reward_scalar <= 0.0

        step_info = {
            "success": terminated,
            "episode_steps": self.episode_steps,
            "raw_reward": reward_scalar,
        }
        if isinstance(info, dict):
            for key, value in info.items():
                if value is None:
                    step_info[key] = None
                elif np.ndim(value) == 0:
                    step_info[key] = float(value)
                else:
                    step_info[key] = value

        if done_bool:
            self.episode_steps = 0

        return {"observation.state": obs}, reward_scalar, terminated, truncated, step_info

    def render(self) -> np.ndarray:
        """Return an RGB frame (H, W, 3, uint8) for video recording."""
        if self._state is None:
            raise RuntimeError("Cannot render before reset")
        return np.asarray(jax.device_get(self._jit_render(self._state)))

    def close(self):
        return None

    @property
    def unwrapped(self):
        return self

    def get_wrapper_attr(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    def set_wrapper_attr(self, name: str, value):
        setattr(self, name, value)


def make_kinetix_env(task: str, env_id: int = 0, seed: int | None = None) -> Callable[[], KinetixGymWrapper]:
    def _make():
        return KinetixGymWrapper(task=task, env_id=env_id, seed=seed)

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
        """Return RGB frames from all environments (num_envs, H, W, 3, uint8)."""
        frames = self.vec_env.call("render")
        return np.stack(frames, axis=0)

    @property
    def fps(self):
        return self.metadata.get("render_fps", 15)

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


def create_vectorized_env(
    env_name: str,
    num_envs: int,
    device: str = "cpu",
    debug: bool = False,
    video_key: str = "observation.state",
    seed: int | None = None,
) -> VectorizedEnvWrapper:
    """Create a vectorized Kinetix environment."""
    import gymnasium as gym

    env_fns = [make_kinetix_env(env_name, env_id=i, seed=seed) for i in range(num_envs)]

    # Kinetix and the flow base policy share one JAX GPU runtime in-process.
    vec_env = gym.vector.SyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    vec_env.call("set_wrapper_attr", "video_key", video_key)

    wrapped_env = VectorizedEnvWrapper(vec_env, video_key, device)
    wrapped_env.env_name = env_name
    logger.debug(f"Created {num_envs} vectorized Kinetix {env_name} environments")
    return wrapped_env

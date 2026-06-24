# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Environment wrapper that includes a base policy, enabling residual RL training
to be done in a standard way without explicit base policy handling in the training loop.

This wrapper assumes:
- The environment is already vectorized (batched)
- All inputs/outputs are torch tensors
- The environment comes from create_vectorized_env
"""

from collections import deque
import gymnasium as gym
import numpy as np
import torch

from resfit.lerobot.policies.pretrained import PreTrainedPolicy


class BasePolicyVecEnvWrapper:
    """
    Wraps a vectorized environment with a base policy to enable standard RL training of residual policies.

    This wrapper:
    1. Takes raw observations from the vectorized environment
    2. Passes them through the base policy to get base actions
    3. Augments observations with base actions for the residual policy
    4. Combines base + residual actions before stepping the environment
    5. Returns augmented observations that include base actions

    Assumes the environment is already vectorized and works with torch tensors.
    """

    def __init__(
        self,
        vec_env,
        base_policy: PreTrainedPolicy,
        action_scaler,
        state_standardizer,
        language_instruction: str | None = None,
        inference_delay: int = 0,
    ):
        """
        Args:
            vec_env: Vectorized environment from create_vectorized_env
            base_policy: Base policy (e.g., ACTPolicy or OpenPIPi0AlohaSimPolicy)
            action_scaler: ActionScaler object for scaling/unscaling actions (REQUIRED)
            state_standardizer: StateStandardizer object for standardizing states (REQUIRED)
            language_instruction: Task prompt for language-conditioned base policies (pi0).
            inference_delay: Number of environment steps to simulate base-policy inference delay.
        """
        assert action_scaler is not None, "action_scaler is required for consistent normalization"
        assert state_standardizer is not None, "state_standardizer is required for consistent normalization"
        assert inference_delay >= 0, f"inference_delay must be >= 0, got {inference_delay}"
        if inference_delay > 0:
            max_chunk = getattr(base_policy.config, "chunk_size", base_policy.config.n_action_steps)
            required_chunk = inference_delay + base_policy.config.n_action_steps
            assert required_chunk <= max_chunk, (
                f"inference_delay ({inference_delay}) + n_action_steps "
                f"({base_policy.config.n_action_steps}) must be <= chunk_size ({max_chunk})"
            )
        assert getattr(base_policy.config, "temporal_ensemble_coeff", None) is None or inference_delay == 0, (
            "base_policy inference_delay is not supported with ACT temporal ensembling"
        )

        self.vec_env = vec_env
        self.base_policy = base_policy
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer
        self.language_instruction = language_instruction
        self.inference_delay = inference_delay
        self.requires_language = getattr(base_policy.config, "type", "") == "pi0"

        # Get action dimension from the environment
        self.action_dim = vec_env.action_space.shape[-1]

        # Store image keys from base policy config
        self.image_keys = list(base_policy.config.image_features.keys())

        if inference_delay > 0:
            self._init_delay_state()

        # Create modified observation space that includes base actions
        self._setup_observation_space()

    @property
    def num_envs(self):
        return self.vec_env.num_envs

    def _init_delay_state(self):
        self._obs_history = [[] for _ in range(self.num_envs)]
        self._episode_step = [0 for _ in range(self.num_envs)]
        self._base_action_queues = [deque() for _ in range(self.num_envs)]

    def _setup_observation_space(self):
        """Setup observation space to include base actions in the state."""

        # Get original observation space
        orig_obs_space = self.vec_env.observation_space

        # Copy the action space
        self.action_space = self.vec_env.action_space

        # Create new observation space with augmented state
        obs_spaces = {}
        for key, space in orig_obs_space.spaces.items():
            if key == "observation.state":
                # Augment state dimension with base actions
                orig_shape = list(space.shape)
                new_shape = orig_shape.copy()
                # new_shape[-1] += self.action_dim  # Not anymore
                obs_spaces[key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=tuple(new_shape), dtype=space.dtype)
            else:
                # Keep other observations unchanged
                obs_spaces[key] = space

        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _prepare_base_policy_obs(self, raw_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.requires_language:
            return raw_obs
        obs = raw_obs.copy()
        batch_size = next(iter(obs.values())).shape[0]
        obs["task"] = [self.language_instruction] * batch_size
        return obs

    def _maybe_attach_base_policy_images(self, raw_obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Attach full-resolution camera frames only when pi0 must re-infer."""
        obs = self._prepare_base_policy_obs(raw_obs)
        needs_infer = getattr(self.base_policy, "needs_inference", None)
        if needs_infer is None or not needs_infer(obs):
            return obs

        image_key = self.image_keys[0]
        if hasattr(self.vec_env, "render_base_policy_images"):
            obs[image_key] = self.vec_env.render_base_policy_images()
        return obs

    def _reset_delay_state(self, env_ids=None):
        if self.inference_delay == 0:
            return

        if env_ids is None:
            env_ids_list = list(range(self.num_envs))
        elif isinstance(env_ids, torch.Tensor):
            env_ids_list = env_ids.detach().cpu().tolist()
        else:
            env_ids_list = list(env_ids)

        for env_id in env_ids_list:
            self._obs_history[env_id] = []
            self._episode_step[env_id] = 0
            self._base_action_queues[env_id] = deque()

    def _append_obs_to_history(self, raw_obs):
        pi0_images = None
        if self.requires_language and hasattr(self.vec_env, "render_base_policy_images"):
            pi0_images = self.vec_env.render_base_policy_images()

        for env_idx in range(self.num_envs):
            snapshot = {}
            for key, value in raw_obs.items():
                if isinstance(value, torch.Tensor):
                    snapshot[key] = value[env_idx : env_idx + 1].clone()
                else:
                    snapshot[key] = value

            if pi0_images is not None:
                snapshot[self.image_keys[0]] = pi0_images[env_idx : env_idx + 1].clone()

            self._obs_history[env_idx].append(snapshot)

    def _obs_from_history(self, env_idx: int, time_idx: int):
        return self._obs_history[env_idx][time_idx]

    def _predict_action_chunk(self, infer_obs) -> torch.Tensor:
        """Return action chunk for a single environment."""
        infer_obs = infer_obs.copy()
        if self.requires_language:
            infer_obs["task"] = [self.language_instruction]

        chunk_length = self.base_policy.config.n_action_steps
        if self.inference_delay > 0:
            chunk_length += self.inference_delay

        if self.base_policy.config.type == "pi0":
            result = self.base_policy._openpi_policy.infer(self.base_policy._batch_to_openpi_obs(infer_obs, 0))
            action_chunk = result["actions"][:chunk_length]
            device = next(iter(infer_obs.values())).device if infer_obs else self.base_policy._device
            return torch.stack([torch.as_tensor(action, dtype=torch.float32, device=device) for action in action_chunk], dim=0)

        batch = self.base_policy.normalize_inputs(infer_obs)
        if self.base_policy.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = [batch[key] for key in self.base_policy.config.image_features]

        actions_seq = self.base_policy.model(batch)[0][:, :chunk_length]
        actions_seq = self.base_policy.unnormalize_outputs({"action": actions_seq})["action"]
        return actions_seq[0]

    def _fill_base_action_queue(self, env_idx: int) -> None:
        step = self._episode_step[env_idx]
        obs_index = max(0, step - self.inference_delay)
        infer_obs = self._obs_from_history(env_idx, obs_index)
        action_offset = step - obs_index
        chunk_end = action_offset + self.base_policy.config.n_action_steps

        action_chunk = self._predict_action_chunk(infer_obs)
        assert chunk_end <= action_chunk.shape[0], (
            f"Action chunk range [{action_offset}, {chunk_end}) exceeds chunk length "
            f"{action_chunk.shape[0]} for env {env_idx} at step {self._episode_step[env_idx]}"
        )

        for action in action_chunk[action_offset:chunk_end].unbind(0):
            self._base_action_queues[env_idx].append(action)

    def _select_base_action(self, raw_obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.inference_delay == 0:
            with torch.no_grad():
                return self.base_policy.select_action(self._maybe_attach_base_policy_images(raw_obs))

        actions = []
        with torch.no_grad():
            for env_idx in range(self.num_envs):
                if len(self._base_action_queues[env_idx]) == 0:
                    self._fill_base_action_queue(env_idx)
                actions.append(self._base_action_queues[env_idx].popleft())
        return torch.stack(actions, dim=0)

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        """Reset environment and base policy."""
        # Reset the underlying vectorized environment
        raw_obs, info = self.vec_env.reset(**kwargs)

        # Reset base policy
        self.base_policy.reset()

        if self.inference_delay > 0:
            self._reset_delay_state()
            self._append_obs_to_history(raw_obs)

        # Get base action from the base policy
        base_action = self._select_base_action(raw_obs)

        base_naction = self.action_scaler.scale(base_action)

        # Augment observations with base action and apply state standardization
        augmented_obs = self._augment_obs(raw_obs, base_naction)

        # Store for later use in step
        self._last_base_naction = base_naction

        return augmented_obs, info

    def step(
        self, residual_naction: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step the environment with residual action.

        Args:
            residual_action: The residual action from the residual policy

        Returns:
            augmented_obs: Observations augmented with base actions
            reward: Reward tensor
            terminated: Terminated tensor
            truncated: Truncated tensor
            info: Info dict
        """
        # Combine base and residual actions
        # Residual action is already scaled inside the Actor class
        # To ensure that we can use the same exploration for all dimensions,
        # we use the normalized actions as the action space
        # The normalized base action is stored as [-1, 1] in the replay buffer
        # and the residual action is predicted as action_scale * [-1, 1]
        combined_naction = self._last_base_naction + residual_naction

        # Unscale back to original action space for environment execution
        env_action = self.action_scaler.unscale(combined_naction)

        # Step the underlying vectorized environment
        raw_obs, reward, terminated, truncated, info = self.vec_env.step(env_action)

        # Store the scaled action for replay buffer (already computed above)
        info["scaled_action"] = combined_naction

        # Clear stale action chunks before querying the base policy. With vector-env
        # autoreset, raw_obs is already the *next* episode's first observation when
        # done=True. Aloha marks failures as truncated-only, so we must reset on both.
        done = terminated | truncated
        if done.any():
            self.base_policy.reset(env_ids=torch.where(done)[0])
            if self.inference_delay > 0:
                self._reset_delay_state(torch.where(done)[0])

        if self.inference_delay > 0:
            self._append_obs_to_history(raw_obs)
            for env_idx in range(self.num_envs):
                if not done[env_idx]:
                    self._episode_step[env_idx] += 1

        # Get next base action for the returned observation (fresh chunk after reset).
        base_action = self._select_base_action(raw_obs)

        base_naction = self.action_scaler.scale(base_action)

        # Augment observations with base action and apply state standardization
        augmented_obs = self._augment_obs(raw_obs, base_naction)

        # Handle final_obs in info dict to ensure consistent shapes
        if "final_obs" in info:
            info = self._process_final_obs_in_info(info, combined_naction.device)

        # Store for next step
        self._last_base_naction = base_naction

        return augmented_obs, reward, terminated, truncated, info

    def _augment_obs(self, raw_obs: dict[str, torch.Tensor], base_naction: torch.Tensor) -> dict[str, torch.Tensor]:
        """Augment observations with base actions."""

        # New way to do this is to just add the base action to the state under its own key
        augmented_obs = raw_obs.copy()
        augmented_obs["observation.base_action"] = base_naction
        augmented_obs["observation.state"] = self.state_standardizer.standardize(augmented_obs["observation.state"])

        return augmented_obs

    def _process_final_obs_in_info(self, info: dict, device: torch.device) -> dict:
        """Pad final_obs state with zeros to match augmented observation format."""
        if "final_obs" not in info or info["final_obs"] is None:
            return info

        for final_obs_dict in info["final_obs"]:
            if final_obs_dict is not None and "observation.state" in final_obs_dict:
                # Pad with zeros (no action taken at terminal state)
                final_obs_dict["observation.base_action"] = torch.zeros(
                    self.action_dim, device=device, dtype=torch.float32
                )

        return info

    def render(self):
        """Pass through to underlying environment."""
        return self.vec_env.render()

    def close(self):
        """Close the environment."""
        return self.vec_env.close()

    # Pass through any other attributes/methods to the underlying environment
    def __getattr__(self, name: str):
        """Delegate unknown attributes to the underlying vectorized environment."""
        return getattr(self.vec_env, name)

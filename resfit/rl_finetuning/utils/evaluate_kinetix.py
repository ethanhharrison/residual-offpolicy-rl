# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

import wandb
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent


def run_kinetix_evaluation(
    *,
    env,
    agent: QAgent,
    num_episodes: int = 20,
    device: torch.device | str = "cpu",
    global_step: int | None = None,
    save_video: bool = False,
    save_q_plots: bool = False,
    run_name: str | None = None,
    output_dir: str | Path | None = "outputs",
) -> dict[str, float]:
    """Evaluate a residual policy on Kinetix tasks with DexMG-style logging."""

    def _annotate_frame(
        frame: np.ndarray,
        *,
        env_idx: int,
        episode_num: int,
        total_episodes: int,
        step_idx: int,
        is_success: bool,
        q_value: float,
        font=None,
    ) -> np.ndarray:
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)

        status_text = "SUCCESS" if is_success else "FAIL"
        status_color = (0, 255, 0) if is_success else (255, 0, 0)

        y = 10
        dy = 15
        draw.text((10, y), f"Env {env_idx + 1}", fill=(255, 255, 255), font=font)
        y += dy
        draw.text((10, y), f"Episode {episode_num}/{total_episodes}", fill=(255, 255, 255), font=font)
        y += dy
        draw.text((10, y), f"Step {step_idx}", fill=(255, 255, 255), font=font)
        y += dy
        draw.text((10, y), status_text, fill=status_color, font=font)
        y += dy
        draw.text((10, y), f"Q = {q_value:.2f}", fill=(255, 255, 255), font=font)

        return np.asarray(pil_img)

    def _create_q_trajectory_plots(
        trajectories: list[list[float]],
        episode_lengths: list[int],
        successes: list[bool],
        output_path: Path,
        global_step: int | None = None,
    ) -> None:
        if not trajectories:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        successful_trajs = [traj for i, traj in enumerate(trajectories) if successes[i]]
        failed_trajs = [traj for i, traj in enumerate(trajectories) if not successes[i]]

        for i, traj in enumerate(successful_trajs):
            steps = list(range(len(traj)))
            ax1.plot(steps, traj, "g-", alpha=0.6, linewidth=1, label="Success" if i == 0 else "")

        for i, traj in enumerate(failed_trajs):
            steps = list(range(len(traj)))
            ax1.plot(steps, traj, "r-", alpha=0.6, linewidth=1, label="Failure" if i == 0 else "")

        ax1.set_xlabel("Episode Step")
        ax1.set_ylabel("Q-Value")
        ax1.set_title(f"Q-Value Trajectories Over Time (Step {global_step or 'N/A'})")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        progress_points = [0.25, 0.5, 0.75, 1.0]
        q_values_at_progress = {f"{int(p * 100)}%": [] for p in progress_points}

        for traj in trajectories:
            traj_len = len(traj)
            for p in progress_points:
                step_idx = min(int(p * traj_len), traj_len - 1)
                if step_idx < len(traj):
                    q_values_at_progress[f"{int(p * 100)}%"].append(traj[step_idx])

        box_data = [q_values_at_progress[f"{int(p * 100)}%"] for p in progress_points]
        box_labels = [f"{int(p * 100)}%" for p in progress_points]

        ax2.boxplot(box_data)
        ax2.set_xticklabels(box_labels)
        ax2.set_xlabel("Episode Progress")
        ax2.set_ylabel("Q-Value")
        ax2.set_title("Q-Value Distribution at Different Episode Progress Points")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved Q-trajectory plots to: {output_path}")

    device = torch.device(device)
    agent.eval()

    num_envs: int = env.num_envs if hasattr(env, "num_envs") else 1

    ep_rewards: list[list[float]] = [[] for _ in range(num_envs)]
    ep_q_preds: list[list[float]] = [[] for _ in range(num_envs)]

    successes: list[bool] = []
    returns: list[float] = []

    all_q_trajectories: list[list[float]] = []
    all_episode_lengths: list[int] = []

    frame_buffer: list[list[np.ndarray]] | None = [[] for _ in range(num_envs)] if save_video else None
    all_frames: list[np.ndarray] | None = [] if save_video else None

    done_episodes = 0
    obs, _ = env.reset()

    progress_dots = ["."] * num_episodes
    print(f"Evaluating {num_episodes} episodes: {''.join(progress_dots)}", end="", flush=True)

    while done_episodes < num_episodes:
        with torch.no_grad():
            actions, q_pred = agent.act_and_q_value(
                obs, eval_mode=True, stddev=0.0, cpu=False, compute_q=True
            )
            assert q_pred is not None

        next_obs, reward, terminated, truncated, _ = env.step(actions)
        done_flags = terminated | truncated

        if save_video and frame_buffer is not None:
            frame = env.render()
            for env_idx in range(num_envs):
                frame_buffer[env_idx].append(frame[env_idx])

        for env_idx in range(num_envs):
            ep_rewards[env_idx].append(reward[env_idx].item())
            ep_q_preds[env_idx].append(q_pred[env_idx].item())

            if done_flags[env_idx]:
                ep_return = float(sum(ep_rewards[env_idx]))
                is_success = bool(terminated[env_idx].item())

                progress_dots[done_episodes] = "✓" if is_success else "✗"
                print(f"\rEvaluating {num_episodes} episodes: {''.join(progress_dots)}", end="", flush=True)

                successes.append(is_success)
                returns.append(ep_return)

                if save_q_plots:
                    all_q_trajectories.append(ep_q_preds[env_idx].copy())

                all_episode_lengths.append(len(ep_q_preds[env_idx]))

                if save_video and frame_buffer is not None and all_frames is not None:
                    episode_frames = frame_buffer[env_idx]
                    episode_qs = ep_q_preds[env_idx]
                    episode_global_idx = done_episodes + 1

                    for step_idx, fr in enumerate(episode_frames):
                        annotated_fr = _annotate_frame(
                            fr,
                            env_idx=env_idx,
                            episode_num=episode_global_idx,
                            total_episodes=num_episodes,
                            step_idx=step_idx + 1,
                            is_success=is_success,
                            q_value=episode_qs[step_idx],
                        )
                        all_frames.append(annotated_fr)

                    frame_buffer[env_idx].clear()

                ep_rewards[env_idx].clear()
                ep_q_preds[env_idx].clear()

                done_episodes += 1

                if done_episodes == num_episodes:
                    break

        obs = next_obs

    print("Done")

    if len(all_episode_lengths) != len(successes):
        raise RuntimeError(
            f"Episode length/success misalignment: lengths={len(all_episode_lengths)} successes={len(successes)}"
        )

    success_rate: float = float(np.mean(successes)) if successes else 0.0
    mean_return: float = float(np.mean(returns)) if returns else 0.0

    successful_episode_lengths = [length for length, is_success in zip(all_episode_lengths, successes) if is_success]
    mean_successful_episode_length: float = (
        float(np.mean(successful_episode_lengths)) if successful_episode_lengths else 0.0
    )

    metrics: dict[str, float] = {
        "eval/success_rate": success_rate,
        "eval/mean_return": mean_return,
        "eval/mean_successful_episode_length": mean_successful_episode_length,
    }

    if wandb.run is not None:
        wandb.log(metrics, step=global_step)

    if save_q_plots and all_q_trajectories and run_name is not None:
        parent = Path(str(output_dir or "outputs")) / run_name.split("__")[0]
        parent.mkdir(parents=True, exist_ok=True)

        plot_name = f"eval_q_trajectories_{run_name}_step_{global_step if global_step is not None else 'NA'}.png"
        plot_path = parent / plot_name

        _create_q_trajectory_plots(
            trajectories=all_q_trajectories,
            episode_lengths=all_episode_lengths,
            successes=successes,
            output_path=plot_path,
            global_step=global_step,
        )

        if wandb.run is not None:
            wandb.log({"value/q_trajectories": wandb.Image(str(plot_path))}, step=global_step)

    if save_video and all_frames is not None and run_name is not None:
        parent = Path(str(output_dir or "outputs")) / run_name.split("__")[0]
        parent.mkdir(parents=True, exist_ok=True)

        vid_name = f"eval_{run_name}_step_{global_step if global_step is not None else 'NA'}.mp4"
        video_path = parent / vid_name

        fps_val = getattr(env, "fps", 15)

        writer = imageio.get_writer(video_path, fps=fps_val)
        for fr in all_frames:
            writer.append_data(fr)
        writer.close()

        if wandb.run is not None:
            wandb.log({"eval/video": wandb.Video(str(video_path), format="mp4")}, step=global_step)

    agent.train(True)

    return metrics

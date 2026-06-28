# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from pathlib import Path

WORLDS_DIR = Path(__file__).resolve().parent / "environments" / "worlds" / "l"

# User-facing task names map to level JSON files from real-time-chunking-kinetix.
KINETIX_TASKS: dict[str, str] = {
    "mjc_swimmer": "worlds/l/mjc_swimmer.json",
    "mjc_walker": "worlds/l/mjc_walker.json",
    "car_launch": "worlds/l/car_launch.json",
}

RTC_BC_CHECKPOINT_ROOT = "https://storage.googleapis.com/rtc-assets/bc/24"

LARGE_ENV_PARAMS = {
    "num_polygons": 12,
    "num_circles": 4,
    "num_joints": 6,
    "num_thrusters": 2,
    "num_motor_bindings": 4,
    "num_thruster_bindings": 2,
}
FRAME_SKIP = 2
ACTION_NOISE_STD = 0.1
RENDER_SCREEN_DIM = (512, 512)


def task_to_level_path(task: str) -> Path:
    if task not in KINETIX_TASKS:
        raise ValueError(f"Unknown Kinetix task: {task}. Supported: {list(KINETIX_TASKS)}")
    level_path = WORLDS_DIR / f"{task}.json"
    if not level_path.exists():
        raise FileNotFoundError(f"Kinetix level file not found: {level_path}")
    return level_path


def task_to_level_name(task: str) -> str:
    level_path = KINETIX_TASKS[task]
    return level_path.replace("/", "_").replace(".json", "")


def default_bc_checkpoint(task: str) -> str:
    level_name = task_to_level_name(task)
    return f"{RTC_BC_CHECKPOINT_ROOT}/policies/{level_name}.pkl"

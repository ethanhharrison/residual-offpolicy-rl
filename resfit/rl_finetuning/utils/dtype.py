# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def maybe_resize_rl_images(obs_dict: dict, keys: list[str], size: int | None) -> None:
    """Downsample RL camera observations to size x size in-place."""
    if size is None:
        return

    for key in keys:
        if key not in obs_dict:
            continue
        value = obs_dict[key]
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value).float()
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[-2] == size and tensor.shape[-1] == size:
                continue
            resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
            obs_dict[key] = resized.squeeze(0).numpy()
        elif isinstance(value, torch.Tensor):
            tensor = value.float()
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[-2] == size and tensor.shape[-1] == size:
                continue
            resized = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
            obs_dict[key] = resized.squeeze(0)


def to_uint8(obs_dict: dict, keys: list[str]):
    """In-place conversion of image entries in *obs_dict* to uint8.

    Assumes images are float32 in [0,1].  Non-image entries are left
    untouched.  Supports torch.Tensor and numpy.ndarray inputs.
    """

    for _k in keys:
        if _k not in obs_dict:
            continue
        _v = obs_dict[_k]
        if isinstance(_v, torch.Tensor):
            if _v.dtype == torch.uint8:
                continue  # already uint8
            # Avoid inplace on shared tensors
            obs_dict[_k] = (_v * 255.0).clamp_(0, 255).to(torch.uint8)
        elif isinstance(_v, np.ndarray):
            if _v.dtype == np.uint8:
                continue
            obs_dict[_k] = (_v * 255.0).clip(0, 255).astype(np.uint8)
    return obs_dict

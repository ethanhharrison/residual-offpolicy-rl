# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import os
import site
from pathlib import Path


def _discover_pip_cuda_nvcc_root() -> Path | None:
    for site_dir in site.getsitepackages():
        candidate = Path(site_dir) / "nvidia" / "cuda_nvcc"
        if candidate.is_dir():
            return candidate
    return None


def _configure_cuda_root() -> None:
    if os.environ.get("CUDA_ROOT"):
        return

    try:
        from nvidia import cuda_nvcc
    except ImportError:
        pip_root = _discover_pip_cuda_nvcc_root()
        if pip_root is not None:
            os.environ["CUDA_ROOT"] = str(pip_root)
        return

    if cuda_nvcc.__file__ is not None:
        os.environ.setdefault("CUDA_ROOT", str(Path(cuda_nvcc.__file__).parent))
        return

    namespace_path = getattr(cuda_nvcc, "__path__", None)
    if namespace_path:
        os.environ["CUDA_ROOT"] = str(Path(next(iter(namespace_path))))
        return

    pip_root = _discover_pip_cuda_nvcc_root()
    if pip_root is not None:
        os.environ["CUDA_ROOT"] = str(pip_root)


def configure_jax_gpu() -> None:
    """Configure JAX for GPU use (Kinetix env + flow base policy)."""
    _configure_cuda_root()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if "JAX_PLATFORMS" in os.environ and os.environ["JAX_PLATFORMS"] == "cpu":
        del os.environ["JAX_PLATFORMS"]


def jax_gpu_device():
    """Return the default JAX GPU device."""
    import jax

    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    if not gpu_devices:
        raise RuntimeError(
            "JAX GPU backend is not available. Source resfit/kinetix/jax_cuda_env.sh and run on a GPU node."
        )
    return gpu_devices[0]


def place_on_jax_device(value, device=None):
    """Recursively place a JAX pytree on the target JAX device."""
    import jax

    if device is None:
        device = jax_gpu_device()
    return jax.tree.map(lambda leaf: jax.device_put(leaf, device), value)


def place_nnx_on_device(module, device=None):
    """Place all parameters of an Flax NNX module on the target JAX device."""
    import flax.nnx as nnx

    graphdef, state = nnx.split(module)
    state = place_on_jax_device(state, device)
    return nnx.merge(graphdef, state)


def log_jax_devices(context: str = "JAX") -> None:
    """Log the active JAX backend/devices once at startup."""
    import jax

    devices = jax.devices()
    print(f"[{context}] backend={jax.default_backend()}, devices={[str(d) for d in devices]}")


# Backwards-compatible aliases
configure_jax_policy = configure_jax_gpu
configure_jax_env = configure_jax_gpu
configure_jax_cuda = configure_jax_gpu

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Literal

import torch
import wandb

from resfit.lerobot.policies.act.modeling_act import ACTPolicy
from resfit.lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from resfit.lerobot.policies.pi0.modeling_openpi_pi0_aloha_sim import OpenPIPi0AlohaSimPolicy
from resfit.lerobot.policies.pretrained import PreTrainedPolicy

OPENPI_PI0_ALOHA_SIM_CHECKPOINT = "s3://openpi-assets/checkpoints/pi0_aloha_sim"
OPENPI_PALIGEMMA_TOKENIZER_GS = "gs://big_vision/paligemma_tokenizer.model"
OPENPI_PALIGEMMA_TOKENIZER_HTTPS = (
    "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
)
BasePolicyType = Literal["act", "pi0"]


def _patch_openpi_restore_params() -> None:
    """Patch OpenPI checkpoint restore for orbax>=0.11.14 StepMetadata API."""
    import pathlib

    import jax
    import jax.numpy as jnp
    import numpy as np
    import orbax.checkpoint as ocp
    from flax import traverse_util

    import openpi.models.model as openpi_model

    if getattr(openpi_model.restore_params, "_resfit_orbax_compat", False):
        return

    def restore_params(
        params_path: pathlib.Path | str,
        *,
        restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
        dtype: jnp.dtype | None = None,
        sharding: jax.sharding.Sharding | None = None,
    ):
        params_path = (
            pathlib.Path(params_path).resolve()
            if not str(params_path).startswith("gs://")
            else params_path
        )

        if restore_type is jax.Array and sharding is None:
            mesh = jax.sharding.Mesh(jax.devices(), ("x",))
            sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        with ocp.PyTreeCheckpointer() as ckptr:
            metadata = ckptr.metadata(params_path)
            if hasattr(metadata, "item_metadata") and metadata.item_metadata is not None:
                params_tree = metadata.item_metadata.tree["params"]
            else:
                params_tree = metadata["params"]
            item = {"params": params_tree}

            params = ckptr.restore(
                params_path,
                ocp.args.PyTreeRestore(
                    item=item,
                    restore_args=jax.tree.map(
                        lambda _: ocp.ArrayRestoreArgs(
                            sharding=sharding, restore_type=restore_type, dtype=dtype
                        ),
                        item,
                    ),
                ),
            )["params"]

        flat_params = traverse_util.flatten_dict(params)
        if flat_params and all(kp[-1] == "value" for kp in flat_params):
            flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
        return traverse_util.unflatten_dict(flat_params)

    restore_params._resfit_orbax_compat = True  # type: ignore[attr-defined]
    openpi_model.restore_params = restore_params


def _download_openpi_checkpoint(openpi_checkpoint: str, download) -> Path:
    parsed = urllib.parse.urlparse(openpi_checkpoint)
    if parsed.scheme == "":
        path = Path(openpi_checkpoint).expanduser()
        if not path.is_dir() or not (path / "params").is_dir():
            raise FileNotFoundError(f"OpenPI checkpoint not found at {openpi_checkpoint}")
        return download.maybe_download(str(path))

    download_kwargs: dict = {}
    if parsed.scheme == "s3":
        download_kwargs["anon"] = True
    checkpoint_dir = download.maybe_download(openpi_checkpoint, **download_kwargs)
    checkpoint_path = Path(checkpoint_dir)
    if not (checkpoint_path / "params").is_dir():
        raise FileNotFoundError(f"OpenPI JAX checkpoint not found at {checkpoint_path} (missing params/)")
    return checkpoint_path


def _ensure_openpi_gcs_assets(download) -> None:
    """Pre-cache public GCS assets over HTTPS when gcsfs SSL fails on HPC nodes."""
    cache_dir = download.get_cache_dir()
    tokenizer_path = cache_dir / "big_vision" / "paligemma_tokenizer.model"
    if tokenizer_path.exists() and tokenizer_path.stat().st_size > 0:
        return

    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tokenizer_path.with_suffix(".partial")
    try:
        import certifi
        import ssl

        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(OPENPI_PALIGEMMA_TOKENIZER_HTTPS, context=context) as response:
            tmp_path.write_bytes(response.read())
        tmp_path.replace(tokenizer_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        # Fall back to OpenPI's gcsfs path if HTTPS fails for some reason.
        download.maybe_download(OPENPI_PALIGEMMA_TOKENIZER_GS, gs={"token": "anon"})


def download_policy_from_wandb(
    run_id: str,
    *,
    step: str | None = None,
    artifact_version: str = "latest",
) -> tuple[Path, str]:
    """Download a policy checkpoint logged on W&B and return its folder.

    The policy is expected to have been created with the training utilities in
    `train_hf.py` and therefore to contain a `config.json` in the root of the
    downloaded artifact.
    """
    api = wandb.Api()
    project, id_ = run_id.split("/")

    if step is None or str(step).lower() == "latest":
        artifact_name = f"run_{id_}_latest:{artifact_version}"
        checkpoint_step = "latest"
    elif str(step).lower() == "best":
        artifact_name = f"run_{id_}_best:{artifact_version}"
        checkpoint_step = "best"
    else:
        artifact_name = f"run_{id_}_model_step_{step}:{artifact_version}"
        checkpoint_step = str(step)

    artifact_path = f"{project}/{artifact_name}"
    artifact = api.artifact(artifact_path)

    art_dir = Path(artifact.download())
    policy_dir = art_dir / "policy"  # The artifact root already contains the policy files.

    if not (policy_dir / "config.json").exists():
        raise FileNotFoundError(f"Policy directory not found inside downloaded artifact: {policy_dir}")

    return policy_dir, checkpoint_step


def load_openpi_pi0_policy(
    *,
    openpi_config_name: str = "pi0_aloha_sim",
    openpi_checkpoint: str = OPENPI_PI0_ALOHA_SIM_CHECKPOINT,
) -> OpenPIPi0AlohaSimPolicy:
    """Load a PI0 JAX checkpoint from OpenPI's public S3 bucket."""
    try:
        from openpi.policies import policy_config
        from openpi.shared import download
        from openpi.training import config as openpi_config
    except ImportError as exc:
        raise RuntimeError(
            "OpenPI is required to load pi0 base policies. Install the GitHub package (not PyPI stub):\n"
            "  uv pip install --reinstall --no-deps "
            "'openpi @ git+https://github.com/Physical-Intelligence/openpi.git'\n"
            "OpenPI requires Python >= 3.11."
        ) from exc

    _patch_openpi_restore_params()
    train_config = openpi_config.get_config(openpi_config_name)
    _ensure_openpi_gcs_assets(download)
    checkpoint_dir = _download_openpi_checkpoint(openpi_checkpoint, download)
    openpi_policy = policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
    )
    return OpenPIPi0AlohaSimPolicy.from_openpi(
        openpi_policy,
        openpi_config_name=openpi_config_name,
    )


def load_policy(policy_dir: Path) -> ACTPolicy:
    """Infer policy type (diffusion / act) from `config.json` and load weights."""

    with (policy_dir / "config.json").open() as f:
        cfg_dict = json.load(f)

    policy_name_field = str(cfg_dict.get("type", "")).lower()

    # TODO: improve policy-type inference logic when additional policies are added
    if "diffusion" in policy_name_field:
        raise NotImplementedError("Diffusion policy not implemented")
        return DiffusionPolicy.from_pretrained(policy_dir)
    if "use_vae" in cfg_dict:
        return ACTPolicy.from_pretrained(policy_dir)

    raise ValueError(f"Unknown policy type: {policy_name_field}")


def load_base_policy(
    *,
    policy_type: BasePolicyType,
    device: str | torch.device,
    wandb_id: str | None = None,
    wt_type: str = "best",
    wt_version: str = "latest",
    openpi_config_name: str = "pi0_aloha_sim",
    openpi_checkpoint: str = OPENPI_PI0_ALOHA_SIM_CHECKPOINT,
) -> PreTrainedPolicy:
    """Load the configured base policy for residual RL."""
    if policy_type == "pi0":
        policy = load_openpi_pi0_policy(
            openpi_config_name=openpi_config_name,
            openpi_checkpoint=openpi_checkpoint,
        )
        return policy.to(device)

    if policy_type == "act":
        if not wandb_id or wandb_id == "TODO":
            raise ValueError("base_policy.wandb_id is required when base_policy.type='act'")
        policy_dir, _ = download_policy_from_wandb(
            wandb_id,
            step=wt_type,
            artifact_version=wt_version,
        )
        policy = load_policy(policy_dir)
        return policy.to(device)

    raise ValueError(f"Unsupported base policy type: {policy_type}")


def save_checkpoint(ckpt_dir: Path, step: int, policy, optimizer) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Save model weights + config
    policy.save_pretrained(ckpt_dir / "policy")
    # Save optimizer & misc state
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
        },
        ckpt_dir / "trainer_state.pt",
    )


def load_checkpoint(ckpt_dir: Path, policy, optimizer):
    state_pth = ckpt_dir / "trainer_state.pt"
    if not state_pth.exists():
        raise FileNotFoundError(state_pth)
    state = torch.load(state_pth, map_location="cpu")
    policy_loaded = policy.from_pretrained(ckpt_dir / "policy")
    optimizer.load_state_dict(state["optimizer"])
    return state["step"], policy_loaded, optimizer

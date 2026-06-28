# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def _gs_uri_to_https(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme != "gs":
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    return f"https://storage.googleapis.com/{parsed.netloc}{parsed.path}"


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def download_kinetix_checkpoint(checkpoint_path: str | Path) -> Path:
    """Download a rtc-kinetix flow policy checkpoint if needed."""
    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        return path

    parsed = urlparse(str(checkpoint_path))
    if parsed.scheme in {"", "file"}:
        if not path.is_file():
            raise FileNotFoundError(f"Kinetix checkpoint not found: {checkpoint_path}")
        return path

    cache_root = Path.home() / ".cache" / "resfit" / "kinetix_checkpoints"
    cache_root.mkdir(parents=True, exist_ok=True)
    cached = cache_root / Path(parsed.path.lstrip("/")).name
    if cached.exists() and cached.stat().st_size > 0:
        return cached

    if parsed.scheme == "gs":
        gsutil = shutil.which("gsutil")
        if gsutil is not None:
            subprocess.run([gsutil, "-m", "cp", str(checkpoint_path), str(cached)], check=True)
            return cached

        https_url = _gs_uri_to_https(str(checkpoint_path))
        try:
            _download_file(https_url, cached)
            return cached
        except OSError as exc:
            raise RuntimeError(
                f"Cannot download {checkpoint_path}: gsutil not found and HTTPS download failed "
                f"({https_url}). Download manually with curl/wget and pass a local path via "
                "base_policy.kinetix_checkpoint."
            ) from exc

    if parsed.scheme in {"http", "https"}:
        _download_file(str(checkpoint_path), cached)
        return cached

    raise ValueError(f"Unsupported checkpoint URI scheme: {parsed.scheme}")

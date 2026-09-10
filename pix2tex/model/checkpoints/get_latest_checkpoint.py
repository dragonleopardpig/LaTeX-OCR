"""Download the published pix2tex checkpoints into a verified user cache."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import requests
from platformdirs import user_cache_path
from tqdm import tqdm

RELEASE_TAG = "v0.0.1"
RELEASE_ROOT = (
    "https://github.com/lukas-blecher/LaTeX-OCR/releases/download/" + RELEASE_TAG
)
CHECKPOINTS = {
    "weights.pth": {
        "url": f"{RELEASE_ROOT}/weights.pth",
        "sha256": "a63d9141c53d266cb682fb5a8bd83bd5cbe283145e0e78ebdc0f895195a1dfaa",
        "size": 102_113_875,
    },
    "image_resizer.pth": {
        "url": f"{RELEASE_ROOT}/image_resizer.pth",
        "sha256": "1c3820659985ad142b526490bb25c23d977176ac2073591b3bddada692718458",
        "size": 19_441_973,
    },
}
DOWNLOAD_TIMEOUT = (10, 120)


def checkpoint_directory() -> Path:
    """Return the writable directory used for downloaded model assets."""
    configured = os.environ.get("PIX2TEX_CACHE_DIR")
    root = Path(configured).expanduser() if configured else user_cache_path("pix2tex")
    return root / "checkpoints"


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_is_valid(path: Path, metadata: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == metadata["size"]
        and sha256sum(path) == metadata["sha256"]
    )


def download_checkpoint(name: str, destination: Path | None = None) -> Path:
    """Download one checkpoint, verify it, and atomically install it."""
    if name not in CHECKPOINTS:
        raise ValueError(f"Unknown checkpoint: {name}")

    metadata = CHECKPOINTS[name]
    destination = Path(destination) if destination else checkpoint_directory() / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_is_valid(destination, metadata):
        return destination

    temporary_path: Path | None = None
    try:
        with requests.get(
            metadata["url"],
            stream=True,
            allow_redirects=True,
            timeout=DOWNLOAD_TIMEOUT,
        ) as response:
            response.raise_for_status()
            advertised_size = response.headers.get("content-length")
            if advertised_size and int(advertised_size) != metadata["size"]:
                raise RuntimeError(
                    f"Unexpected Content-Length for {name}: {advertised_size}"
                )

            digest = hashlib.sha256()
            downloaded = 0
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{name}.",
                suffix=".part",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                with tqdm(
                    desc=f"Downloading {name}",
                    total=metadata["size"],
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        downloaded += len(chunk)
                        if downloaded > metadata["size"]:
                            raise RuntimeError(f"Download exceeded expected size for {name}")
                        digest.update(chunk)
                        temporary.write(chunk)
                        progress.update(len(chunk))
                temporary.flush()
                os.fsync(temporary.fileno())

        if downloaded != metadata["size"]:
            raise RuntimeError(
                f"Incomplete checkpoint {name}: got {downloaded} bytes, "
                f"expected {metadata['size']}"
            )
        if digest.hexdigest() != metadata["sha256"]:
            raise RuntimeError(f"SHA-256 verification failed for {name}")

        temporary_path.chmod(0o600)
        os.replace(temporary_path, destination)
        return destination
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def download_checkpoints(destination: Path | None = None) -> dict[str, Path]:
    """Ensure that both model files are present and return their paths."""
    directory = Path(destination) if destination else checkpoint_directory()
    return {
        name: download_checkpoint(name, directory / name)
        for name in CHECKPOINTS
    }


if __name__ == "__main__":
    for checkpoint_name, checkpoint_path in download_checkpoints().items():
        print(f"{checkpoint_name}: {checkpoint_path}")

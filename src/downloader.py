"""Video downloading for the Dockerized task-runner interface.

Supports both:

1. Local files
   C:/Users/.../video.mp4

2. Remote URLs
   https://....

The returned value is always a local Path, so the rest of the pipeline
does not care where the video originally came from.
"""

from __future__ import annotations

import mimetypes
import shutil
from pathlib import Path
from urllib.parse import urlparse

import requests

_DEFAULT_EXTENSION = ".mp4"
_CHUNK_SIZE = 1 << 20  # 1 MiB
_DOWNLOAD_TIMEOUT_SECONDS = 60


def _infer_extension(url: str, content_type: str | None) -> str:
    """Infer a video extension from URL or Content-Type."""

    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix:
        return suffix

    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            return guessed

    return _DEFAULT_EXTENSION


def download_video(url: str, output_path: str | Path) -> Path:
    """Obtain a video as a local file.

    Supports both:
    - Local filesystem paths
    - HTTP/HTTPS URLs

    Returns:
        Local path to the downloaded/copied video.
    """

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # LOCAL FILE SUPPORT
    # ----------------------------------------------------------

    local_path = Path(url)

    if local_path.exists():
        if not dest.suffix:
            dest = dest.with_suffix(local_path.suffix or _DEFAULT_EXTENSION)

        shutil.copy2(local_path, dest)
        return dest

    # ----------------------------------------------------------
    # REMOTE URL SUPPORT
    # ----------------------------------------------------------

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Unsupported video source: {url}\n"
            "Expected either:\n"
            " - a local file path\n"
            " - an http/https URL"
        )

    with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
        response.raise_for_status()

        if not dest.suffix:
            dest = dest.with_suffix(
                _infer_extension(url, response.headers.get("Content-Type"))
            )

        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    return dest

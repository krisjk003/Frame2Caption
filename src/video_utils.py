"""Local video file validation and lightweight metadata extraction.

Uses OpenCV (a pip-installable dependency, no system binaries required) to
read the video's frame rate and frame count in order to compute duration.
"""
from __future__ import annotations

import mimetypes
from pathlib import Path

import cv2

from config import Config
from src.models import VideoMetadata
from src.utils import VideoValidationError

_MIME_OVERRIDES = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".m4v": "video/x-m4v",
}


def _guess_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    return _MIME_OVERRIDES.get(path.suffix.lower(), "application/octet-stream")


def _read_duration_seconds(path: Path) -> float:
    """Return the video's duration in seconds using OpenCV.

    Raises:
        VideoValidationError: If the file cannot be opened or its
            frame-rate / frame-count metadata is missing or invalid.
    """

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoValidationError(
            f"Could not open '{path}' as a video file. It may be corrupted "
            "or in an unsupported container/codec."
        )
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        capture.release()

    if fps <= 0 or frame_count <= 0:
        raise VideoValidationError(
            f"Could not determine the duration of '{path}'. The file's "
            "metadata may be missing or malformed."
        )
    return frame_count / fps


def validate_video(video_path: Path, config: Config) -> VideoMetadata:
    """Validate a local video file against pipeline requirements.

    Checks existence, extension, non-zero size, readability, and duration
    bounds (with a small tolerance to account for encoder rounding).

    Args:
        video_path: Path to the candidate video file.
        config: Active pipeline configuration (duration bounds, allowed
            extensions).

    Returns:
        A populated :class:`VideoMetadata`.

    Raises:
        VideoValidationError: If any validation check fails.
    """

    if not video_path.exists():
        raise VideoValidationError(f"Video file not found: {video_path}")
    if not video_path.is_file():
        raise VideoValidationError(f"Path is not a file: {video_path}")

    extension = video_path.suffix.lower()
    if extension not in config.allowed_video_extensions:
        allowed = ", ".join(config.allowed_video_extensions)
        raise VideoValidationError(
            f"Unsupported video extension '{extension}'. Allowed: {allowed}"
        )

    size_bytes = video_path.stat().st_size
    if size_bytes <= 0:
        raise VideoValidationError(f"Video file is empty: {video_path}")

    duration_seconds = _read_duration_seconds(video_path)

    lower_bound = config.min_video_duration_seconds - config.duration_tolerance_seconds
    upper_bound = config.max_video_duration_seconds + config.duration_tolerance_seconds
    if not (lower_bound <= duration_seconds <= upper_bound):
        raise VideoValidationError(
            "Video duration is out of the allowed range: "
            f"{duration_seconds:.1f}s (allowed "
            f"{config.min_video_duration_seconds:.0f}s-"
            f"{config.max_video_duration_seconds:.0f}s, "
            f"tolerance +/-{config.duration_tolerance_seconds:.1f}s)."
        )

    mime_type = _guess_mime_type(video_path)

    return VideoMetadata(
        path=video_path,
        duration_seconds=duration_seconds,
        size_bytes=size_bytes,
        mime_type=mime_type,
        extension=extension,
    )

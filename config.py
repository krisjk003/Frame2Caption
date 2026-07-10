"""Configuration management for the video captioning pipeline.

All runtime configuration is loaded from environment variables (optionally
via a local .env file). Nothing is ever hardcoded — in particular, the
Gemini API key is never embedded in source code.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when the application configuration is invalid or incomplete."""


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration, populated from environment variables."""

    gemini_api_key: str
    model_name: str = "gemini-2.5-flash"

    # Retry behavior for transient Gemini API failures.
    max_retries: int = 3
    retry_backoff_base: float = 2.0

    # Video upload / processing polling and timeouts.
    upload_poll_interval_seconds: float = 3.0
    upload_timeout_seconds: float = 300.0
    generation_timeout_seconds: float = 180.0

    # Accepted input video duration bounds (30 seconds to 2 minutes).
    min_video_duration_seconds: float = 30.0
    max_video_duration_seconds: float = 120.0
    duration_tolerance_seconds: float = 1.5

    # Generation sampling parameters.
    summary_temperature: float = 0.2
    rewrite_temperature: float = 0.8
    max_output_tokens: int = 4096

    # Filesystem locations.
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    sample_videos_dir: Path = field(default_factory=lambda: Path("sample_videos"))
    log_dir: Path = field(default_factory=lambda: Path("outputs") / "logs")

    log_level: str = "INFO"
    keep_remote_file: bool = False

    allowed_video_extensions: tuple[str, ...] = (
        ".mp4",
        ".mov",
        ".avi",
        ".webm",
        ".mkv",
        ".m4v",
    )

    def __post_init__(self) -> None:
        if not self.gemini_api_key or not self.gemini_api_key.strip():
            raise ConfigError("GEMINI_API_KEY must not be empty.")
        if self.min_video_duration_seconds <= 0 or self.max_video_duration_seconds <= 0:
            raise ConfigError("Video duration bounds must be positive.")
        if self.min_video_duration_seconds > self.max_video_duration_seconds:
            raise ConfigError(
                "min_video_duration_seconds cannot exceed max_video_duration_seconds."
            )
        if self.max_retries < 1:
            raise ConfigError("max_retries must be at least 1.")
        if self.upload_timeout_seconds <= 0 or self.generation_timeout_seconds <= 0:
            raise ConfigError("Timeout values must be positive.")

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "Config":
        """Build a :class:`Config` from environment variables.

        Loads ``env_file`` (defaulting to ``.env`` in the current working
        directory) if it exists, then falls back to any already-exported
        environment variables.

        Raises:
            ConfigError: If ``GEMINI_API_KEY`` is missing or empty.
        """

        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ConfigError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and set "
                "your Gemini API key, or export GEMINI_API_KEY in your shell. "
                "Get a key at https://aistudio.google.com/apikey"
            )

        return cls(
            gemini_api_key=api_key,
            model_name=os.getenv("GEMINI_MODEL_NAME", cls.model_name).strip()
            or cls.model_name,
            max_retries=_get_int("MAX_RETRIES", cls.max_retries),
            retry_backoff_base=_get_float("RETRY_BACKOFF_BASE", cls.retry_backoff_base),
            upload_poll_interval_seconds=_get_float(
                "UPLOAD_POLL_INTERVAL_SECONDS", cls.upload_poll_interval_seconds
            ),
            upload_timeout_seconds=_get_float(
                "UPLOAD_TIMEOUT_SECONDS", cls.upload_timeout_seconds
            ),
            generation_timeout_seconds=_get_float(
                "GENERATION_TIMEOUT_SECONDS", cls.generation_timeout_seconds
            ),
            min_video_duration_seconds=_get_float(
                "MIN_VIDEO_DURATION_SECONDS", cls.min_video_duration_seconds
            ),
            max_video_duration_seconds=_get_float(
                "MAX_VIDEO_DURATION_SECONDS", cls.max_video_duration_seconds
            ),
            duration_tolerance_seconds=_get_float(
                "DURATION_TOLERANCE_SECONDS", cls.duration_tolerance_seconds
            ),
            summary_temperature=_get_float("SUMMARY_TEMPERATURE", cls.summary_temperature),
            rewrite_temperature=_get_float("REWRITE_TEMPERATURE", cls.rewrite_temperature),
            max_output_tokens=_get_int("MAX_OUTPUT_TOKENS", cls.max_output_tokens),
            output_dir=Path(os.getenv("OUTPUT_DIR", "outputs")),
            sample_videos_dir=Path(os.getenv("SAMPLE_VIDEOS_DIR", "sample_videos")),
            log_dir=Path(os.getenv("OUTPUT_DIR", "outputs")) / "logs",
            log_level=(os.getenv("LOG_LEVEL", cls.log_level).strip().upper() or cls.log_level),
            keep_remote_file=_get_bool("KEEP_REMOTE_FILE", cls.keep_remote_file),
        )

    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a copy of this configuration with the given fields
        overridden (e.g. from CLI arguments). Re-validates on construction.
        """

        return dataclasses.replace(self, **overrides)

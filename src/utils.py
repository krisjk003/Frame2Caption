"""General-purpose helpers: custom exceptions, retry logic, JSON parsing,
output directory management, and a lightweight terminal progress indicator.
"""
from __future__ import annotations

import functools
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class VideoCaptioningError(Exception):
    """Base exception for all application-specific errors."""


class VideoValidationError(VideoCaptioningError):
    """Raised when the input video fails validation (missing, wrong
    format, wrong duration, unreadable, etc.).
    """


class GeminiAPIError(VideoCaptioningError):
    """Raised when the Gemini API returns an unrecoverable error."""


class UploadTimeoutError(VideoCaptioningError):
    """Raised when a video upload does not finish processing in time."""


class CaptionValidationError(VideoCaptioningError):
    """Raised when the model's caption output is missing, malformed, or
    does not match the expected JSON schema.
    """


REQUIRED_CAPTION_KEYS: tuple[str, ...] = (
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
)


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------
def retry(
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    logger: logging.Logger | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Return a decorator that retries a function with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts before giving up.
        backoff_base: Base for the exponential backoff delay (``base **
            (attempt - 1)`` seconds).
        exceptions: Tuple of exception types that should trigger a retry.
            Any other exception propagates immediately.
        logger: Optional logger to report retry attempts on. Falls back to
            a module-level logger named after the wrapped function.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            log = logger or logging.getLogger(func.__module__)
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001 - intentional broad catch
                    last_exc = exc
                    if attempt >= max_attempts:
                        log.error(
                            "%s failed after %d attempt(s): %s",
                            func.__name__,
                            attempt,
                            exc,
                        )
                        break
                    delay = backoff_base ** (attempt - 1)
                    log.warning(
                        "%s failed on attempt %d/%d (%s). Retrying in %.1fs...",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
            assert last_exc is not None  # for type-checkers; loop always sets it
            raise last_exc

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def ensure_directory(path: Path) -> Path:
    """Create ``path`` (and any missing parents) if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_slug() -> str:
    """Return a filesystem-friendly UTC timestamp, e.g. '20260710T153000Z'."""

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def human_readable_duration(seconds: float) -> str:
    """Format a duration in seconds as e.g. '1m 12s' or '48s'."""

    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# JSON extraction & validation
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from a model response, tolerating markdown
    code fences or minor surrounding text.

    Raises:
        CaptionValidationError: If no valid JSON object can be recovered.
    """

    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise CaptionValidationError(
                "Model response did not contain a JSON object."
            )
        candidate = cleaned[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise CaptionValidationError(
                f"Failed to parse JSON from model response: {exc}"
            ) from exc

    if not isinstance(parsed, dict):
        raise CaptionValidationError("Expected a JSON object for captions.")
    return parsed


def validate_caption_schema(data: dict[str, Any]) -> None:
    """Validate that ``data`` matches the expected four-caption schema.

    Raises:
        CaptionValidationError: If keys are missing or values are not
            non-empty strings.
    """

    if not isinstance(data, dict):
        raise CaptionValidationError("Expected a JSON object for captions.")

    missing = [key for key in REQUIRED_CAPTION_KEYS if key not in data]
    if missing:
        raise CaptionValidationError(
            f"Missing required caption key(s): {', '.join(missing)}"
        )

    for key in REQUIRED_CAPTION_KEYS:
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            raise CaptionValidationError(
                f"Caption '{key}' must be a non-empty string."
            )


# ---------------------------------------------------------------------------
# Terminal progress indicator
# ---------------------------------------------------------------------------
class ProgressIndicator:
    """A minimal, dependency-free terminal spinner for long-running steps.

    Falls back to a single static line when stdout is not a TTY (e.g. when
    output is redirected to a file or captured by another process).

    Usage:
        with ProgressIndicator("Uploading video"):
            do_slow_thing()
    """

    _FRAMES = "|/-\\"

    def __init__(
        self, message: str, interval: float = 0.15, stream: Any = sys.stdout
    ) -> None:
        self._message = message
        self._interval = interval
        self._stream = stream
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _spin(self) -> None:
        i = 0
        while not self._stop_event.is_set():
            frame = self._FRAMES[i % len(self._FRAMES)]
            self._stream.write(f"\r{self._message}... {frame}")
            self._stream.flush()
            i += 1
            time.sleep(self._interval)
        clear_width = len(self._message) + 6
        self._stream.write("\r" + " " * clear_width + "\r")
        self._stream.flush()

    def __enter__(self) -> "ProgressIndicator":
        is_tty = bool(getattr(self._stream, "isatty", lambda: False)())
        if is_tty:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        else:
            self._stream.write(f"{self._message}...\n")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=1.0)
        return False

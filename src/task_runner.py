"""Orchestrates the Dockerized batch-task interface.

For every task in tasks.json this module:
    1. Downloads the video referenced by ``video_url`` (src/downloader.py).
    2. Runs it through the existing, unmodified caption-generation logic
       (src/caption_pipeline.py -> src/gemini_client.py).
    3. Selects the requested caption styles (defaulting to all four).

All per-task results are then written to results.json (src/io.py).

This is the only module that knows about the tasks.json/results.json
contract itself; CaptionPipeline remains a general-purpose "one video in,
captions out" component, unaware of Docker, batches, or the evaluator.
"""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from config import Config
from src.caption_pipeline import CaptionPipeline
from src.downloader import download_video
from src.io import load_tasks, save_results
from src.utils import REQUIRED_CAPTION_KEYS, VideoCaptioningError
from pathlib import Path

if Path("/input").exists() and Path("/output").exists():
    # Running inside the hackathon Docker container
    DEFAULT_TASKS_PATH = Path("/input/tasks.json")
    DEFAULT_RESULTS_PATH = Path("/output/results.json")
else:
    # Running locally
    DEFAULT_TASKS_PATH = Path("input/tasks.json")
    DEFAULT_RESULTS_PATH = Path("output/results.json")
    _SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_stem(task_id: str) -> str:
    """Sanitize a task_id for safe use as part of a local filename."""

    cleaned = _SAFE_CHARS_RE.sub("_", str(task_id)).strip("_")
    return cleaned or "task"


def _select_styles(
    captions: dict[str, str], styles: list[str] | None
) -> dict[str, str]:
    """Return only the requested caption styles, in canonical order.

    Unknown style names are ignored. If ``styles`` is omitted/empty, or
    none of the requested names are recognized, all four captions are
    returned (matches the spec: "If styles is omitted, generate all four
    captions.").
    """

    if not styles:
        return dict(captions)

    selected = {key: captions[key] for key in REQUIRED_CAPTION_KEYS if key in styles}
    return selected or dict(captions)


def run_tasks(
    tasks_path: str | Path = DEFAULT_TASKS_PATH,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    config: Config | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    """Process every task in ``tasks_path`` and write results to
    ``results_path``.

    Returns:
        The path results were written to.

    Raises:
        ConfigError: If the Gemini API key / configuration is invalid.
        FileNotFoundError: If ``tasks_path`` does not exist.
        ValueError: If ``tasks_path`` is not a well-formed task list.
    """

    log = logger or logging.getLogger("video_captioning.task_runner")
    cfg = config or Config.from_env()
    pipeline = CaptionPipeline(cfg, logger=log)

    tasks = load_tasks(tasks_path)
    log.info("Loaded %d task(s) from %s", len(tasks), tasks_path)

    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="frame2caption_") as tmp_dir:
        for task in tasks:
            task_id = task["task_id"]
            video_url = task["video_url"]
            styles = task.get("styles")

            log.info("Processing task '%s': %s", task_id, video_url)

            local_video_path: Path | None = None
            try:
                # No suffix here on purpose: download_video infers the
                # correct extension from the URL/Content-Type.
                download_target = Path(tmp_dir) / f"{_safe_stem(task_id)}_video"
                local_video_path = download_video(video_url, download_target)

                result = pipeline.run(local_video_path, styles=styles)
                captions = _select_styles(result.captions.to_dict(), styles)

                results.append({"task_id": task_id, "captions": captions})
                log.info("Task '%s' completed successfully.", task_id)
            except VideoCaptioningError as exc:
                # Known pipeline failure (bad video, Gemini error, etc.) —
                # record it and keep processing the remaining tasks.
                log.error("Task '%s' failed: %s", task_id, exc)
                results.append({"task_id": task_id, "error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - one bad task must not kill the batch
                log.exception("Task '%s' failed unexpectedly.", task_id)
                results.append({"task_id": task_id, "error": str(exc)})
            finally:
                if local_video_path is not None:
                    local_video_path.unlink(missing_ok=True)

    output_path = save_results(results_path, results)
    log.info("Wrote %d result(s) to %s", len(results), output_path)
    return output_path

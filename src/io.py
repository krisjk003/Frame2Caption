"""JSON I/O for the Dockerized hackathon-evaluator interface.

Reads /input/tasks.json and writes /output/results.json (or any paths the
caller supplies). This is unrelated to — and does not replace — the
existing per-run outputs/*.json + outputs/*.txt reporting produced by
src/caption_pipeline.py, which is left untouched for backward
compatibility.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    """Load and minimally validate the task list from a JSON file.

    Expects a JSON array of objects, each with at least ``task_id`` and
    ``video_url``. An optional ``styles`` list may also be present.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is not a JSON array of well-formed task
            objects.
    """

    tasks_path = Path(path)
    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_path}")

    with tasks_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse {tasks_path} as JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array of tasks in {tasks_path}, got "
            f"{type(data).__name__}."
        )

    for index, task in enumerate(data):
        if not isinstance(task, dict):
            raise ValueError(f"Task at index {index} is not a JSON object.")
        if not task.get("task_id"):
            raise ValueError(f"Task at index {index} is missing 'task_id'.")
        if not task.get("video_url"):
            raise ValueError(
                f"Task '{task.get('task_id', index)}' is missing 'video_url'."
            )

    return data


def save_results(path: str | Path, results: list[dict[str, Any]]) -> Path:
    """Write ``results`` to ``path`` as pretty-printed JSON.

    Creates parent directories if they do not already exist.

    Returns:
        The path that was written.
    """

    results_path = Path(path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results_path

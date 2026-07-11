#!/usr/bin/env python3
"""Docker entry point for the Frame2Caption hackathon-evaluator interface.

On startup this reads every task from /input/tasks.json, runs each one
through the existing (unmodified) captioning pipeline, and writes
/output/results.json. Kept intentionally tiny — all orchestration lives in
src/task_runner.py; all caption-generation logic is untouched in
src/caption_pipeline.py and src/gemini_client.py.

Usage (inside the container):
    python app.py

No CLI arguments are read or required.
"""
from __future__ import annotations

import sys

from config import ConfigError
from src.logger import setup_logger
from src.task_runner import run_tasks

EXIT_OK = 0
EXIT_UNEXPECTED_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_TASKS_ERROR = 3


def main() -> int:
    logger = setup_logger()

    try:
        output_path = run_tasks()
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except (FileNotFoundError, ValueError) as exc:
        # Malformed or missing /input/tasks.json.
        logger.error("Tasks file error: %s", exc)
        print(f"[TASKS ERROR] {exc}", file=sys.stderr)
        return EXIT_TASKS_ERROR
    except Exception as exc:  # noqa: BLE001 - top-level safety net
        logger.exception("Unexpected error while running tasks.")
        print(f"[UNEXPECTED ERROR] {exc}", file=sys.stderr)
        return EXIT_UNEXPECTED_ERROR

    logger.info("Done. Results written to: %s", output_path)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
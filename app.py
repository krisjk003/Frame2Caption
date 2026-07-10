#!/usr/bin/env python3
"""Command-line entry point for the AI video captioning pipeline.

Generates four styled captions (formal, sarcastic, humorous-tech,
humorous-non-tech) for a short video using Gemini 2.5 Flash, driven
entirely by prompt engineering (no fine-tuning or training).

Usage:
    python app.py --video sample.mp4
    python app.py --video sample.mp4 --output-dir results --log-level DEBUG
    python app.py --video sample.mp4 --keep-remote-file
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from config import Config, ConfigError
from src.caption_pipeline import CaptionPipeline
from src.logger import setup_logger
from src.utils import (
    CaptionValidationError,
    GeminiAPIError,
    UploadTimeoutError,
    VideoValidationError,
)

EXIT_OK = 0
EXIT_UNEXPECTED_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_VIDEO_ERROR = 3
EXIT_TIMEOUT_ERROR = 4
EXIT_CAPTION_ERROR = 5
EXIT_GEMINI_ERROR = 6


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description=(
            "Generate formal, sarcastic, humorous-tech, and "
            "humorous-non-tech captions for a short video using Gemini 2.5 "
            "Flash and prompt engineering (no fine-tuning)."
        ),
    )
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        help="Path to the input video file (30 seconds to 2 minutes long).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory (default: outputs/).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override the log level (default: from .env or INFO).",
    )
    parser.add_argument(
        "--keep-remote-file",
        action="store_true",
        help="Do not delete the uploaded video from Gemini's File API "
        "after the run completes.",
    )
    return parser


def _print_result(captions_json: str) -> None:
    banner = "=" * 70
    print(f"\n{banner}\nCAPTIONS GENERATED\n{banner}")
    print(captions_json)
    print(f"{banner}\n")


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = Config.from_env()
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    overrides: dict[str, Any] = {}
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
        overrides["log_dir"] = args.output_dir / "logs"
    if args.log_level is not None:
        overrides["log_level"] = args.log_level
    if args.keep_remote_file:
        overrides["keep_remote_file"] = True
    if overrides:
        try:
            config = config.with_overrides(**overrides)
        except ConfigError as exc:
            print(f"[CONFIG ERROR] {exc}", file=sys.stderr)
            return EXIT_CONFIG_ERROR

    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(level=config.log_level, log_dir=config.log_dir)

    pipeline = CaptionPipeline(config, logger=logger)

    try:
        result = pipeline.run(args.video)
    except VideoValidationError as exc:
        logger.error("Video validation failed: %s", exc)
        print(f"[VIDEO ERROR] {exc}", file=sys.stderr)
        return EXIT_VIDEO_ERROR
    except UploadTimeoutError as exc:
        logger.error("Upload/processing timed out: %s", exc)
        print(f"[TIMEOUT ERROR] {exc}", file=sys.stderr)
        return EXIT_TIMEOUT_ERROR
    except CaptionValidationError as exc:
        logger.error("Caption generation failed validation: %s", exc)
        print(f"[CAPTION ERROR] {exc}", file=sys.stderr)
        return EXIT_CAPTION_ERROR
    except GeminiAPIError as exc:
        logger.error("Gemini API error: %s", exc)
        print(f"[GEMINI ERROR] {exc}", file=sys.stderr)
        return EXIT_GEMINI_ERROR
    except Exception as exc:  # noqa: BLE001 - top-level safety net
        logger.exception("Unexpected error while running the pipeline.")
        print(f"[UNEXPECTED ERROR] {exc}", file=sys.stderr)
        return EXIT_UNEXPECTED_ERROR

    _print_result(result.captions.to_json())
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

"""Orchestrates the end-to-end captioning pipeline:

    validate video -> upload -> wait for processing -> summarize
    -> rewrite into 4 styles -> persist JSON + TXT
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from config import Config
from src.gemini_client import GeminiClient
from src.models import PipelineResult
from src.utils import ensure_directory, timestamp_slug
from src.video_utils import validate_video


class CaptionPipeline:
    """High-level pipeline that turns a local video file into four styled
    captions plus a saved JSON/TXT report.
    """

    def __init__(
        self,
        config: Config,
        gemini_client: GeminiClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._logger = logger or logging.getLogger(__name__)
        self._client = gemini_client or GeminiClient(config, logger=self._logger)

    def run(self, video_path: Path) -> PipelineResult:
        """Run the full pipeline for a single video and persist the result.

        Args:
            video_path: Path to a local video file (30s-2min by default).

        Returns:
            The completed :class:`PipelineResult`.

        Raises:
            VideoValidationError: If the input video fails validation.
            UploadTimeoutError: If Gemini does not finish processing the
                video in time.
            GeminiAPIError: If the Gemini API fails unrecoverably.
            CaptionValidationError: If the final caption JSON is invalid
                after all retries.
        """

        self._logger.info("Starting captioning pipeline for: %s", video_path)

        video_metadata = validate_video(video_path, self._config)
        self._logger.info(
            "Video validated: %.1fs, %s, %d bytes",
            video_metadata.duration_seconds,
            video_metadata.mime_type,
            video_metadata.size_bytes,
        )

        uploaded_file = self._client.upload_video(video_metadata)
        active_file = None
        try:
            active_file = self._client.wait_until_active(uploaded_file)
            summary = self._client.generate_summary(active_file, video_metadata)
            captions = self._client.generate_captions(summary)
        finally:
            if not self._config.keep_remote_file:
                cleanup_target = active_file or uploaded_file
                self._client.delete_file(cleanup_target)

        result = PipelineResult(
            video_metadata=video_metadata,
            summary=summary,
            captions=captions,
            model_name=self._config.model_name,
        )

        json_path, txt_path = self._save_outputs(result)
        result = dataclasses.replace(result, json_path=json_path, txt_path=txt_path)
        self._logger.info("Pipeline completed successfully.")
        return result

    def _save_outputs(self, result: PipelineResult) -> tuple[Path, Path]:
        """Persist the pipeline result as both JSON and plain-text files.

        Returns:
            A ``(json_path, txt_path)`` tuple of the written files.
        """

        output_dir = ensure_directory(self._config.output_dir)
        stem = result.video_metadata.path.stem
        slug = timestamp_slug()

        json_path = output_dir / f"{stem}_{slug}.json"
        txt_path = output_dir / f"{stem}_{slug}.txt"

        json_path.write_text(result.to_json(), encoding="utf-8")
        txt_path.write_text(result.to_text(), encoding="utf-8")

        self._logger.info("Saved JSON output to: %s", json_path)
        self._logger.info("Saved text output to: %s", txt_path)
        return json_path, txt_path
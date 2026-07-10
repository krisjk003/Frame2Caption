"""Thin, robust wrapper around the Google GenAI SDK for uploading videos and
generating factual summaries / styled captions with Gemini 2.5 Flash.

Everything here is prompt engineering plus API orchestration: no model
weights are trained, fine-tuned, or modified.
"""
from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any

from google import genai
from google.genai import errors, types

from config import Config
from prompts.rewrite_prompt import build_rewrite_prompt
from prompts.summary_prompt import build_summary_prompt
from prompts.system_prompt import SYSTEM_PROMPT
from src.models import CaptionSet, UploadedFile, VideoMetadata
from src.utils import (
    CaptionValidationError,
    GeminiAPIError,
    ProgressIndicator,
    UploadTimeoutError,
    extract_json,
    retry,
)

_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    errors.APIError,
    ConnectionError,
    TimeoutError,
    OSError,
)

_ACTIVE_STATES = {"ACTIVE"}
_FAILED_STATES = {"FAILED"}


def _state_name(file_obj: Any) -> str:
    """Normalize a Gemini File's ``state`` (enum or string) to a plain
    upper-case string such as 'ACTIVE', 'PROCESSING', or 'FAILED'.
    """

    state = getattr(file_obj, "state", None)
    if state is None:
        return "UNKNOWN"
    name = getattr(state, "name", state)
    return str(name).upper()


class GeminiClient:
    """Encapsulates all interaction with the Gemini API for this pipeline."""

    def __init__(self, config: Config, logger: logging.Logger | None = None) -> None:
        self._config = config
        self._logger = logger or logging.getLogger(__name__)
        self._client = genai.Client(api_key=config.gemini_api_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _call_with_timeout(
        self, func: Any, timeout: float, *args: Any, **kwargs: Any
    ) -> Any:
        """Run a blocking SDK call on a worker thread and enforce a hard
        wall-clock timeout, raising :class:`TimeoutError` if exceeded.

        Note: the underlying network call cannot be forcibly killed if it
        times out (a fundamental Python limitation with blocking I/O), but
        the caller is unblocked immediately and the retry/error-handling
        logic proceeds as if the call failed.
        """

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as exc:
                func_name = getattr(func, "__name__", str(func))
                raise TimeoutError(
                    f"Gemini call '{func_name}' did not complete within "
                    f"{timeout:.0f}s."
                ) from exc

    # ------------------------------------------------------------------
    # Upload & processing
    # ------------------------------------------------------------------
    def upload_video(self, video_metadata: VideoMetadata) -> UploadedFile:
        """Upload a local video file to the Gemini File API.

        Raises:
            GeminiAPIError / errors.APIError: On unrecoverable API failure
                after exhausting retries.
        """

        @retry(
            max_attempts=self._config.max_retries,
            backoff_base=self._config.retry_backoff_base,
            exceptions=_RETRYABLE_EXCEPTIONS,
            logger=self._logger,
        )
        def _do_upload() -> Any:
            try:
                upload_config = types.UploadFileConfig(
                    mime_type=video_metadata.mime_type
                )
                return self._client.files.upload(
                    file=str(video_metadata.path), config=upload_config
                )
            except (TypeError, AttributeError):
                # Defensive fallback for SDK versions with a different
                # UploadFileConfig signature; mime type is auto-detected
                # from the file extension in this case.
                return self._client.files.upload(file=str(video_metadata.path))

        self._logger.info("Uploading video to Gemini: %s", video_metadata.path)
        with ProgressIndicator("Uploading video"):
            raw_file = _do_upload()
        self._logger.info("Upload complete. Remote file name: %s", raw_file.name)

        return UploadedFile(
            name=raw_file.name,
            uri=getattr(raw_file, "uri", "") or "",
            mime_type=getattr(raw_file, "mime_type", video_metadata.mime_type),
            state=_state_name(raw_file),
            raw=raw_file,
        )

    def wait_until_active(self, uploaded_file: UploadedFile) -> UploadedFile:
        """Poll the Gemini File API until the uploaded video finishes
        processing (state becomes ACTIVE).

        Raises:
            UploadTimeoutError: If processing does not complete within the
                configured timeout.
            GeminiAPIError: If the remote file enters a FAILED state.
        """

        deadline = time.monotonic() + self._config.upload_timeout_seconds
        current = uploaded_file.raw
        self._logger.info("Waiting for Gemini to finish processing the video...")

        with ProgressIndicator("Processing video on Gemini"):
            while True:
                state = _state_name(current)
                if state in _ACTIVE_STATES:
                    break
                if state in _FAILED_STATES:
                    raise GeminiAPIError(
                        "Gemini failed to process the uploaded video "
                        f"'{uploaded_file.name}' (state=FAILED)."
                    )
                if time.monotonic() >= deadline:
                    raise UploadTimeoutError(
                        f"Timed out after {self._config.upload_timeout_seconds:.0f}s "
                        f"waiting for '{uploaded_file.name}' to finish processing "
                        f"(last state={state})."
                    )
                time.sleep(self._config.upload_poll_interval_seconds)
                try:
                    current = self._client.files.get(name=uploaded_file.name)
                except _RETRYABLE_EXCEPTIONS as exc:
                    self._logger.warning(
                        "Transient error polling file status: %s", exc
                    )
                    continue

        self._logger.info("Video is ACTIVE and ready for analysis.")
        return UploadedFile(
            name=uploaded_file.name,
            uri=getattr(current, "uri", uploaded_file.uri) or uploaded_file.uri,
            mime_type=getattr(current, "mime_type", uploaded_file.mime_type),
            state=_state_name(current),
            raw=current,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate_summary(
        self, uploaded_file: UploadedFile, video_metadata: VideoMetadata
    ) -> str:
        """Generate a single, factual, hallucination-free video summary.

        Raises:
            GeminiAPIError: If Gemini returns an empty response after
                exhausting retries.
        """

        prompt = build_summary_prompt(video_metadata.duration_seconds)

        @retry(
            max_attempts=self._config.max_retries,
            backoff_base=self._config.retry_backoff_base,
            exceptions=_RETRYABLE_EXCEPTIONS,
            logger=self._logger,
        )
        def _do_generate() -> str:
            response = self._call_with_timeout(
                self._client.models.generate_content,
                self._config.generation_timeout_seconds,
                model=self._config.model_name,
                contents=[uploaded_file.raw, prompt],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=self._config.summary_temperature,
                    max_output_tokens=self._config.max_output_tokens,
                ),
            )
            text = (response.text or "").strip()
            if not text:
                raise GeminiAPIError("Gemini returned an empty summary.")
            return text

        self._logger.info(
            "Requesting factual summary from %s...", self._config.model_name
        )
        with ProgressIndicator("Analyzing video content"):
            summary = _do_generate()
        self._logger.info("Summary generated (%d characters).", len(summary))
        return summary

    def generate_captions(self, summary: str) -> CaptionSet:
        """Rewrite a factual summary into four styled captions.

        Raises:
            CaptionValidationError: If Gemini's JSON output is malformed or
                incomplete after exhausting retries.
        """

        prompt = build_rewrite_prompt(summary)
        retryable = _RETRYABLE_EXCEPTIONS + (CaptionValidationError,)

        @retry(
            max_attempts=self._config.max_retries,
            backoff_base=self._config.retry_backoff_base,
            exceptions=retryable,
            logger=self._logger,
        )
        def _do_generate() -> CaptionSet:
            response = self._call_with_timeout(
                self._client.models.generate_content,
                self._config.generation_timeout_seconds,
                model=self._config.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self._config.rewrite_temperature,
                    max_output_tokens=self._config.max_output_tokens,
                    response_mime_type="application/json",
                ),
            )
            text = (response.text or "").strip()
            if not text:
                raise CaptionValidationError(
                    "Gemini returned an empty caption response."
                )
            data = extract_json(text)
            return CaptionSet.from_dict(data)

        self._logger.info("Rewriting summary into four caption styles...")
        with ProgressIndicator("Generating styled captions"):
            captions = _do_generate()
        self._logger.info("Captions generated successfully.")
        return captions

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def delete_file(self, uploaded_file: UploadedFile) -> None:
        """Best-effort deletion of the remote uploaded file. Never raises."""

        try:
            self._client.files.delete(name=uploaded_file.name)
            self._logger.info("Deleted remote file: %s", uploaded_file.name)
        except Exception as exc:  # noqa: BLE001 - cleanup must never crash the run
            self._logger.warning(
                "Could not delete remote file '%s': %s", uploaded_file.name, exc
            )

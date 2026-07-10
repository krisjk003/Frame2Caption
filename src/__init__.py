"""Core application package for the AI video captioning pipeline.

Modules:
    models: Typed dataclasses shared across the pipeline.
    utils: Exceptions, retry logic, JSON helpers, and CLI progress display.
    logger: Centralized logging configuration.
    video_utils: Local video file validation and metadata extraction.
    gemini_client: Thin wrapper around the Google GenAI SDK.
    caption_pipeline: End-to-end orchestration of the captioning workflow.
"""

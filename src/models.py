"""Typed data structures shared across the captioning pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils import validate_caption_schema


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata describing a validated local video file."""

    path: Path
    duration_seconds: float
    size_bytes: int
    mime_type: str
    extension: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "duration_seconds": round(self.duration_seconds, 2),
            "size_bytes": self.size_bytes,
            "mime_type": self.mime_type,
            "extension": self.extension,
        }


@dataclass(frozen=True)
class UploadedFile:
    """A handle to a file that has been uploaded to the Gemini File API."""

    name: str
    uri: str
    mime_type: str
    state: str
    raw: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class CaptionSet:
    """The four generated caption styles."""

    formal: str
    sarcastic: str
    humorous_tech: str
    humorous_non_tech: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaptionSet":
        """Build a :class:`CaptionSet` from a parsed JSON object.

        Raises:
            CaptionValidationError: If the schema is invalid (propagated
                from :func:`src.utils.validate_caption_schema`).
        """

        validate_caption_schema(data)
        return cls(
            formal=data["formal"].strip(),
            sarcastic=data["sarcastic"].strip(),
            humorous_tech=data["humorous_tech"].strip(),
            humorous_non_tech=data["humorous_non_tech"].strip(),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "formal": self.formal,
            "sarcastic": self.sarcastic,
            "humorous_tech": self.humorous_tech,
            "humorous_non_tech": self.humorous_non_tech,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass(frozen=True)
class PipelineResult:
    """The complete output of a single end-to-end pipeline run."""

    video_metadata: VideoMetadata
    summary: str
    captions: CaptionSet
    model_name: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video_metadata.to_dict(),
            "model": self.model_name,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "captions": self.captions.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_text(self) -> str:
        """Render a human-readable plain-text report of this result."""

        c = self.captions
        lines = [
            f"Video: {self.video_metadata.path}",
            f"Duration: {self.video_metadata.duration_seconds:.1f}s",
            f"Model: {self.model_name}",
            f"Generated at: {self.generated_at}",
            "",
            "Factual Summary:",
            self.summary,
            "",
            "Formal:",
            c.formal,
            "",
            "Sarcastic:",
            c.sarcastic,
            "",
            "Humorous (Tech):",
            c.humorous_tech,
            "",
            "Humorous (Non-Tech):",
            c.humorous_non_tech,
            "",
        ]
        return "\n".join(lines)

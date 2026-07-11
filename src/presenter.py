"""Human-friendly terminal presentation of a completed pipeline run.

The pipeline already writes a clean JSON + TXT report to disk (see
``CaptionPipeline._save_outputs``). This module is only responsible for
what gets echoed to the *console* at the end of a run: instead of a raw
JSON blob, it renders the summary and all four caption styles as a
readable, color-coded report, then points at the saved files.
"""
from __future__ import annotations

import os
import shutil
import sys
import textwrap
from typing import TextIO

from src.models import PipelineResult

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_HEADER_COLOR = "\033[96m"  # bright cyan
_RULE_COLOR = "\033[90m"  # grey

# (attribute on CaptionSet, display label, ANSI color)
_STYLES: tuple[tuple[str, str, str], ...] = (
    ("formal", "FORMAL", "\033[34m"),  # blue
    ("sarcastic", "SARCASTIC", "\033[35m"),  # magenta
    ("humorous_tech", "HUMOROUS - TECH", "\033[36m"),  # cyan
    ("humorous_non_tech", "HUMOROUS - NON-TECH", "\033[33m"),  # yellow
)


def _enable_windows_ansi() -> None:
    """Best-effort: turn on ANSI escape processing in legacy Windows
    consoles (cmd.exe). No-op on other platforms; never raises.
    """

    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:  # noqa: BLE001 - purely cosmetic, must never crash
        pass


def supports_color(stream: TextIO = sys.stdout) -> bool:
    """Whether ``stream`` should receive ANSI color codes.

    Respects the ``NO_COLOR`` / ``FORCE_COLOR`` conventions and otherwise
    falls back to a TTY check, matching the approach already used in
    :mod:`src.logger`.
    """

    if os.getenv("NO_COLOR") is not None:
        return False
    if os.getenv("FORCE_COLOR") is not None:
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def _terminal_width(default: int = 78) -> int:
    columns = shutil.get_terminal_size(fallback=(default + 2, 24)).columns
    return max(40, min(columns - 2, 88))


def _wrap(text: str, width: int, indent: str = "  ") -> str:
    lines = textwrap.wrap(
        " ".join(text.split()),
        width=width,
        initial_indent=indent,
        subsequent_indent=indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(lines) if lines else indent.rstrip()


def render_captions(
    result: PipelineResult,
    use_color: bool | None = None,
    width: int | None = None,
    show_summary: bool = True,
) -> str:
    """Render a :class:`PipelineResult` as a readable console report."""

    if use_color is None:
        use_color = supports_color()
    if width is None:
        width = _terminal_width()

    def c(code: str, text: str) -> str:
        return f"{code}{text}{_RESET}" if use_color else text

    rule = c(_RULE_COLOR, "\u2500" * width)
    video_name = result.video_metadata.path.name
    duration = result.video_metadata.duration_seconds

    out: list[str] = []
    out.append(rule)
    out.append(
        c(_BOLD + _HEADER_COLOR, "VIDEO CAPTIONS")
        + c(_DIM, f"  ·  {video_name}  ·  {duration:.1f}s  ·  {result.model_name}")
    )
    out.append(rule)
    out.append("")

    if show_summary:
        out.append(c(_DIM, "SUMMARY"))
        out.append(c(_DIM, _wrap(result.summary, width - 2)))
        out.append("")

    captions = result.captions.to_dict()
    for attr, label, color in _STYLES:
        text = captions[attr]
        out.append(c(_BOLD + color, label))
        out.append(c(color, "\u2500" * len(label)))
        out.append(_wrap(text, width - 2))
        out.append("")

    out.append(rule)
    if result.json_path or result.txt_path:
        saved = "  ·  ".join(
            str(p) for p in (result.json_path, result.txt_path) if p is not None
        )
        out.append(c(_DIM, f"Saved -> {saved}"))
        out.append(rule)

    return "\n".join(out)


def print_captions(
    result: PipelineResult,
    use_color: bool | None = None,
    width: int | None = None,
    show_summary: bool = True,
    stream: TextIO = sys.stdout,
) -> None:
    """Print the formatted report for ``result`` to ``stream``."""

    _enable_windows_ansi()
    report = render_captions(
        result, use_color=use_color, width=width, show_summary=show_summary
    )
    print(f"\n{report}\n", file=stream)
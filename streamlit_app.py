"""Frame2Caption — control room.

A single-file, drop-in replacement for the original demo UI. It talks
directly to the existing, unmodified pipeline (config.py, src/*) — nothing
in the backend or the Docker/evaluator contract (app.py, src/task_runner.py,
src/io.py) is touched or required to change.

Two modes:
  - Single Take  — fastest path, calls the pipeline components directly
                   in-memory with a live, real staged progress console.
  - Batch Run    — writes input/tasks.json and calls src.task_runner.run_tasks(),
                   the exact same code path the Docker evaluator uses.

Run it exactly like before:
    streamlit run streamlit_app.py

For the dark theme to apply correctly, keep the accompanying
.streamlit/config.toml next to this file (same folder as before).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import queue
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from config import Config, ConfigError
from src.caption_pipeline import CaptionPipeline
from src.downloader import download_video
from src.task_runner import run_tasks
from src.utils import (
    CaptionValidationError,
    GeminiAPIError,
    UploadTimeoutError,
    VideoCaptioningError,
    VideoValidationError,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STYLE_META: dict[str, dict[str, str]] = {
    "formal": {"label": "Formal", "var": "--f2c-formal"},
    "sarcastic": {"label": "Sarcastic", "var": "--f2c-sarcastic"},
    "humorous_tech": {"label": "Humorous · Tech", "var": "--f2c-tech"},
    "humorous_non_tech": {"label": "Humorous · General", "var": "--f2c-general"},
}
STYLE_ORDER = tuple(STYLE_META.keys())

_FALLBACK_EXTENSIONS = (".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v")

SINGLE_STAGE_MAP: list[tuple[str, int]] = [
    ("preparing local video file", 5),
    ("video validated", 15),
    ("uploading video to gemini", 25),
    ("upload complete", 35),
    ("waiting for gemini", 45),
    ("is active and ready", 60),
    ("requesting factual summary", 68),
    ("summary generated", 82),
    ("rewriting summary into four caption styles", 88),
    ("captions generated successfully", 97),
]

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --f2c-bg: #0A0C10;
  --f2c-surface: #12151B;
  --f2c-surface-2: #171B22;
  --f2c-border: #262B33;
  --f2c-text: #E8EAED;
  --f2c-muted: #8B93A1;
  --f2c-formal: #6C8EF5;
  --f2c-sarcastic: #F5A623;
  --f2c-tech: #2DD4BF;
  --f2c-general: #F45D8C;
  --f2c-rec: #E5484D;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; }

#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }

div[data-testid="stButton"] button {
  border-radius: 8px !important;
  font-weight: 600 !important;
  letter-spacing: .01em;
  transition: transform .12s ease, box-shadow .12s ease;
}
div[data-testid="stButton"] button:hover { transform: translateY(-1px); }

div[data-testid="stCodeBlock"] pre,
div[data-testid="stCodeBlock"] code {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.95rem !important;
  white-space: pre-wrap !important;
}

section[data-testid="stSidebar"] { border-right: 1px solid var(--f2c-border); }

/* --- hero slate --- */
.f2c-slate {
  position: relative; border: 1px solid var(--f2c-border); border-radius: 14px;
  overflow: hidden; background: linear-gradient(180deg, var(--f2c-surface), var(--f2c-surface-2));
  margin-bottom: 26px;
}
.f2c-slate-stripes {
  height: 10px;
  background: repeating-linear-gradient(-45deg, #1A1D23 0 18px, #2A2E37 18px 36px);
}
.f2c-slate-body {
  display: flex; justify-content: space-between; align-items: flex-end;
  padding: 22px 28px 26px; gap: 24px; flex-wrap: wrap;
}
.f2c-eyebrow {
  font-family: 'JetBrains Mono', monospace; font-size: .72rem; letter-spacing: .14em;
  color: var(--f2c-muted); margin-bottom: 6px; text-transform: uppercase;
}
.f2c-title { font-size: 2.5rem; font-weight: 700; margin: 0; line-height: 1.05; color: var(--f2c-text); }
.f2c-title-accent { color: var(--f2c-rec); }
.f2c-tagline { color: var(--f2c-muted); margin: 8px 0 0; font-size: 1rem; }
.f2c-slate-right { display: flex; flex-direction: column; gap: 6px; font-family: 'JetBrains Mono', monospace; font-size: .74rem; }
.f2c-slate-field { display: flex; gap: 8px; }
.f2c-slate-field span { color: var(--f2c-muted); letter-spacing: .08em; width: 54px; display: inline-block; }
.f2c-slate-field b { color: var(--f2c-text); }

/* --- chips / metadata --- */
.f2c-chiprow { display: flex; gap: 10px; flex-wrap: wrap; margin: 4px 0 18px; }
.f2c-chip {
  background: var(--f2c-surface-2); border: 1px solid var(--f2c-border); border-radius: 8px;
  padding: 6px 12px; font-family: 'JetBrains Mono', monospace; font-size: .72rem;
}
.f2c-chip-k { color: var(--f2c-muted); margin-right: 6px; letter-spacing: .05em; }
.f2c-chip-v { color: var(--f2c-text); font-weight: 600; }

/* --- caption card label --- */
.f2c-card-label {
  font-family: 'JetBrains Mono', monospace; font-size: .74rem; letter-spacing: .08em;
  color: var(--f2c-text); margin-bottom: 8px; display: flex; align-items: center;
}
.f2c-dot { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 8px; }

/* --- spec sheet --- */
.f2c-spec { font-family: 'JetBrains Mono', monospace; font-size: .72rem; }
.f2c-spec-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px dashed var(--f2c-border); gap: 8px; }
.f2c-spec-row span { color: var(--f2c-muted); }
.f2c-spec-row b { color: var(--f2c-text); text-align: right; }

/* --- live console --- */
.f2c-console {
  font-family: 'JetBrains Mono', monospace; background: #05070A; border: 1px solid var(--f2c-border);
  border-radius: 10px; padding: 14px 16px; font-size: .78rem; line-height: 1.55;
  color: #9FE3D6; max-height: 220px; overflow-y: auto;
}
.f2c-console-head { color: var(--f2c-text); font-weight: 600; margin-bottom: 8px; display: flex; align-items: center; gap: 6px; }
.f2c-console-time { margin-left: auto; color: var(--f2c-muted); }
.f2c-console-row { white-space: pre-wrap; opacity: .9; }
.f2c-console-dim { color: var(--f2c-muted); font-style: italic; }
.f2c-rec-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--f2c-rec); animation: f2c-pulse 1.1s infinite; }
@keyframes f2c-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .2; } }

/* --- progress bar --- */
.f2c-progress { height: 8px; border-radius: 4px; background: var(--f2c-surface-2); overflow: hidden; margin-top: 10px; }
.f2c-progress-fill { height: 100%; background: linear-gradient(90deg, var(--f2c-rec), var(--f2c-sarcastic)); transition: width .25s ease; }
.f2c-progress-pct { font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: var(--f2c-muted); margin-top: 4px; text-align: right; }

.f2c-empty {
  border: 1px dashed var(--f2c-border); border-radius: 12px; padding: 28px; text-align: center;
  color: var(--f2c-muted); font-size: .95rem; margin-top: 8px;
}
</style>
"""


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    mm, ss = divmod(total, 60)
    return f"{mm:02d}:{ss:02d}"


def format_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def default_config_values() -> dict[str, Any]:
    """Dataclass field defaults for Config, used when no API key is set yet
    so the sidebar can still show something meaningful.
    """
    values: dict[str, Any] = {}
    for f in dataclasses.fields(Config):
        if f.default is not dataclasses.MISSING:
            values[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            values[f.name] = f.default_factory()
    return values


def select_styles(captions: dict[str, str], styles: list[str] | None) -> dict[str, str]:
    """Mirrors src/task_runner.py's _select_styles: filter to the requested
    styles, defaulting to all four if none were requested.
    """
    if not styles:
        return dict(captions)
    selected = {k: captions[k] for k in STYLE_ORDER if k in styles and k in captions}
    return selected or dict(captions)


def describe_error(err: BaseException) -> dict[str, str]:
    if isinstance(err, ConfigError):
        return {"title": "Configuration error", "detail": str(err)}
    if isinstance(err, VideoValidationError):
        return {"title": "Video didn't pass validation", "detail": str(err)}
    if isinstance(err, UploadTimeoutError):
        return {"title": "Upload to Gemini timed out", "detail": str(err)}
    if isinstance(err, GeminiAPIError):
        return {"title": "Gemini API error", "detail": str(err)}
    if isinstance(err, CaptionValidationError):
        return {"title": "Model output didn't validate", "detail": str(err)}
    if isinstance(err, VideoCaptioningError):
        return {"title": "Pipeline error", "detail": str(err)}
    if isinstance(err, FileNotFoundError):
        return {"title": "File not found", "detail": str(err)}
    if isinstance(err, ValueError):
        return {"title": "Bad input", "detail": str(err)}
    return {"title": "Unexpected error", "detail": str(err)}


# ---------------------------------------------------------------------------
# Logging bridge (background thread -> live UI)
# ---------------------------------------------------------------------------

def make_logger(run_id: str) -> tuple[logging.Logger, "queue.Queue[str]"]:
    logger = logging.getLogger(f"f2c.ui.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []
    q: "queue.Queue[str]" = queue.Queue()

    class _QueueHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                q.put(record.getMessage())
            except Exception:
                pass

    handler = _QueueHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger, q


def render_console_html(lines: list[str], elapsed: float) -> str:
    mm, ss = divmod(int(elapsed), 60)
    if lines:
        rows = "".join(f'<div class="f2c-console-row">&rsaquo; {l}</div>' for l in lines)
    else:
        rows = '<div class="f2c-console-row f2c-console-dim">waiting for first signal…</div>'
    return (
        '<div class="f2c-console">'
        f'<div class="f2c-console-head"><span class="f2c-rec-dot"></span>REC '
        f'<span class="f2c-console-time">{mm:02d}:{ss:02d}</span></div>'
        f"{rows}"
        "</div>"
    )


def render_progress_html(pct: int) -> str:
    pct = max(0, min(100, pct))
    return (
        f'<div class="f2c-progress"><div class="f2c-progress-fill" style="width:{pct}%"></div></div>'
        f'<div class="f2c-progress-pct">{pct}%</div>'
    )


def poll_and_render(
    thread: threading.Thread,
    q: "queue.Queue[str]",
    console_ph: Any,
    progress_ph: Any,
    progress_fn: Callable[[list[str]], int],
) -> list[str]:
    logs: list[str] = []
    start = time.monotonic()
    pct = 0
    while thread.is_alive() or not q.empty():
        got_new = False
        while True:
            try:
                line = q.get_nowait()
            except queue.Empty:
                break
            logs.append(line)
            got_new = True
        if got_new:
            pct = max(pct, progress_fn(logs))
        elapsed = time.monotonic() - start
        console_ph.markdown(render_console_html(logs[-10:], elapsed), unsafe_allow_html=True)
        progress_ph.markdown(render_progress_html(pct), unsafe_allow_html=True)
        time.sleep(0.2)
    thread.join()
    while True:
        try:
            logs.append(q.get_nowait())
        except queue.Empty:
            break
    return logs


def single_progress_fn(lines: list[str]) -> int:
    blob = "\n".join(lines).lower()
    pct = 0
    for needle, value in SINGLE_STAGE_MAP:
        if needle in blob:
            pct = max(pct, value)
    return pct


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def run_single_take(
    source: str, styles: list[str], cfg: Config, logger: logging.Logger
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="f2c_ui_") as tmp:
        logger.info("Preparing local video file...")
        target = Path(tmp) / "input_video"
        local_path = download_video(source, target)
        pipeline = CaptionPipeline(cfg, logger=logger)
        result = pipeline.run(local_path, styles=styles)
        data = result.to_dict()
        data["captions"] = select_styles(data["captions"], styles)
        return data


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_hero(next_take_no: int, model_label: str) -> None:
    st.markdown(
        f"""
        <div class="f2c-slate">
          <div class="f2c-slate-stripes"></div>
          <div class="f2c-slate-body">
            <div>
              <div class="f2c-eyebrow">AI video captioning</div>
              <h1 class="f2c-title">Frame<span class="f2c-title-accent">2</span>Caption</h1>
              <p class="f2c-tagline">One factual pass. Four honest voices.</p>
            </div>
            <div class="f2c-slate-right">
              <div class="f2c-slate-field"><span>PROD</span><b>FRAME2CAPTION</b></div>
              <div class="f2c-slate-field"><span>MODEL</span><b>{model_label}</b></div>
              <div class="f2c-slate-field"><span>TAKE</span><b>{next_take_no:03d}</b></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_meta_chips(video_meta: dict[str, Any], model_name: str, generated_at: str) -> None:
    mime = video_meta.get("mime_type", "")
    chips = [
        ("DURATION", format_duration(video_meta.get("duration_seconds", 0))),
        ("SIZE", format_bytes(video_meta.get("size_bytes", 0))),
        ("FORMAT", mime.split("/")[-1].upper() if mime else "—"),
        ("MODEL", model_name),
        ("GENERATED", generated_at.replace("T", " ").split("+")[0]),
    ]
    html = '<div class="f2c-chiprow">' + "".join(
        f'<div class="f2c-chip"><span class="f2c-chip-k">{k}</span><span class="f2c-chip-v">{v}</span></div>'
        for k, v in chips
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_spec(values: dict[str, Any]) -> None:
    rows = [
        ("Model", values.get("model_name", "—")),
        (
            "Duration window",
            f"{int(values.get('min_video_duration_seconds', 0))}s – "
            f"{int(values.get('max_video_duration_seconds', 0))}s",
        ),
        ("Max retries", values.get("max_retries", "—")),
        ("Upload timeout", f"{int(values.get('upload_timeout_seconds', 0))}s"),
        ("Gen. timeout", f"{int(values.get('generation_timeout_seconds', 0))}s"),
        ("Extensions", ", ".join(values.get("allowed_video_extensions", _FALLBACK_EXTENSIONS))),
    ]
    html = "".join(f'<div class="f2c-spec-row"><span>{k}</span><b>{v}</b></div>' for k, v in rows)
    st.markdown(f'<div class="f2c-spec">{html}</div>', unsafe_allow_html=True)


def render_caption_card(style_key: str, text: str) -> None:
    meta = STYLE_META.get(style_key, {"label": style_key, "var": "--f2c-muted"})
    words = len(text.split())
    chars = len(text)
    with st.container(border=True):
        st.markdown(
            f'<div class="f2c-card-label">'
            f'<span class="f2c-dot" style="background:var({meta["var"]})"></span>'
            f'{meta["label"].upper()}'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.code(text, language=None, wrap_lines=True)
        st.caption(f"{words} words · {chars} chars")


def render_caption_grid(captions: dict[str, str]) -> None:
    present = [k for k in STYLE_ORDER if k in captions]
    for i in range(0, len(present), 2):
        cols = st.columns(2)
        for j, key in enumerate(present[i : i + 2]):
            with cols[j]:
                render_caption_card(key, captions[key])


def build_markdown_report(take: dict[str, Any]) -> str:
    result = take["result"]
    v = result["video"]
    lines = [
        f"# Frame2Caption — Take {take['n']:03d}",
        "",
        f"- **Source:** {take.get('source', '')}",
        f"- **Duration:** {format_duration(v.get('duration_seconds', 0))}",
        f"- **Size:** {format_bytes(v.get('size_bytes', 0))}",
        f"- **Format:** {v.get('mime_type', '')}",
        f"- **Model:** {result.get('model', '')}",
        f"- **Generated at:** {result.get('generated_at', '')}",
        "",
        "## Factual summary",
        result.get("summary", ""),
        "",
    ]
    for key in STYLE_ORDER:
        if key in result.get("captions", {}):
            lines += [f"## {STYLE_META[key]['label']}", result["captions"][key], ""]
    return "\n".join(lines)


def render_batch_results(take: dict[str, Any]) -> None:
    results = take["results"]
    total = len(results)
    ok = sum(1 for r in results if "error" not in r)
    failed = total - ok
    c1, c2, c3 = st.columns(3)
    c1.metric("Tasks", total)
    c2.metric("Succeeded", ok)
    c3.metric("Failed", failed)
    st.download_button(
        "⬇ Download results.json",
        json.dumps(results, indent=2, ensure_ascii=False),
        file_name="results.json",
        mime="application/json",
        key=f"dl_batch_{take['n']}",
    )
    for r in results:
        ok_task = "error" not in r
        with st.expander(f"{'🟢' if ok_task else '🔴'}  {r.get('task_id', '?')}"):
            if ok_task:
                render_caption_grid(r.get("captions", {}))
            else:
                st.error(r.get("error", "Unknown error"))


def render_take(take: dict[str, Any]) -> None:
    ok = "error" not in take
    st.markdown(f"### Take {take['n']:03d} — {'🟢 Success' if ok else '🔴 Failed'}")
    st.caption(f"{take.get('ts', '')} · {take.get('mode', 'single').title()} mode")

    if not ok:
        err = take["error"]
        st.error(f"**{err['title']}**\n\n{err['detail']}")
        return

    if take["mode"] == "single":
        result = take["result"]
        left, right = st.columns([3, 2])
        with left:
            render_meta_chips(result["video"], result["model"], result["generated_at"])
            with st.expander("🎞️ Factual summary"):
                st.write(result["summary"])
        with right:
            try:
                st.video(take.get("source", ""))
            except Exception:
                st.caption("Preview unavailable for this source.")
        render_caption_grid(result["captions"])
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "⬇ Download JSON",
                json.dumps(result, indent=2, ensure_ascii=False),
                file_name=f"take_{take['n']:03d}.json",
                mime="application/json",
                use_container_width=True,
                key=f"dl_json_{take['n']}",
            )
        with d2:
            st.download_button(
                "⬇ Download Markdown report",
                build_markdown_report(take),
                file_name=f"take_{take['n']:03d}.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"dl_md_{take['n']}",
            )
    else:
        render_batch_results(take)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Frame2Caption",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

for key, default in (("takes", []), ("active_take", None), ("take_counter", 0)):
    if key not in st.session_state:
        st.session_state[key] = default

try:
    base_cfg: Config | None = Config.from_env()
    config_error: str | None = None
except ConfigError as exc:
    base_cfg = None
    config_error = str(exc)

spec_values = dataclasses.asdict(base_cfg) if base_cfg else default_config_values()  # type: ignore[arg-type]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("#### 🎬 Control room")
    if base_cfg is not None:
        st.success(f"Gemini online · {base_cfg.model_name}", icon="🟢")
    else:
        st.error("GEMINI_API_KEY not set", icon="🔴")
        st.caption(config_error or "Add it to your .env file.")

    with st.expander("Pipeline spec"):
        render_spec(spec_values)

    summary_temp = spec_values.get("summary_temperature", 0.2)
    rewrite_temp = spec_values.get("rewrite_temperature", 0.8)
    max_tokens = spec_values.get("max_output_tokens", 4096)
    keep_remote = spec_values.get("keep_remote_file", False)

    with st.expander("Advanced generation settings"):
        if base_cfg is None:
            st.caption("Add GEMINI_API_KEY to unlock these.")
        summary_temp = st.slider(
            "Summary temperature", 0.0, 1.0, float(summary_temp), 0.05,
            help="Lower = more literal factual summaries.",
            disabled=base_cfg is None,
        )
        rewrite_temp = st.slider(
            "Rewrite temperature", 0.0, 1.2, float(rewrite_temp), 0.05,
            help="Higher = more creative/varied caption tone.",
            disabled=base_cfg is None,
        )
        max_tokens = st.number_input(
            "Max output tokens", min_value=256, max_value=8192, step=256,
            value=int(max_tokens), disabled=base_cfg is None,
        )
        keep_remote = st.checkbox(
            "Keep uploaded file on Gemini's servers", value=bool(keep_remote),
            disabled=base_cfg is None, help="Useful for debugging; off by default.",
        )

    run_cfg = (
        base_cfg.with_overrides(
            summary_temperature=summary_temp,
            rewrite_temperature=rewrite_temp,
            max_output_tokens=int(max_tokens),
            keep_remote_file=keep_remote,
        )
        if base_cfg is not None
        else None
    )

    st.markdown("---")
    st.markdown("#### 🎞️ Takes")
    if not st.session_state.takes:
        st.caption("No takes yet.")
    else:
        for idx, take in reversed(list(enumerate(st.session_state.takes))):
            ok = "error" not in take
            label = f"{'🟢' if ok else '🔴'} Take {take['n']:03d} · {take.get('ts', '')}"
            if st.button(label, key=f"take_btn_{idx}", use_container_width=True):
                st.session_state.active_take = idx
                st.rerun()

    st.markdown("---")
    st.caption("Talks directly to src/caption_pipeline.py and src/task_runner.py. Nothing in the backend was touched.")

# ---------------------------------------------------------------------------
# Main — hero
# ---------------------------------------------------------------------------

render_hero(st.session_state.take_counter + 1, base_cfg.model_name if base_cfg else spec_values.get("model_name", "gemini-2.5-flash"))

mode = st.segmented_control(
    "Mode",
    options=["single", "batch"],
    format_func=lambda k: "🎬 Single Take" if k == "single" else "📦 Batch Run",
    default="single",
    label_visibility="collapsed",
    key="mode_switch",
)

allowed_exts = tuple(spec_values.get("allowed_video_extensions", _FALLBACK_EXTENSIONS))
sample_dir = Path("sample_videos")
sample_files = (
    sorted(p.name for p in sample_dir.glob("*") if p.suffix.lower() in allowed_exts)
    if sample_dir.exists()
    else []
)

# ---------------------------------------------------------------------------
# Single Take mode
# ---------------------------------------------------------------------------

if mode == "single":
    left_col, right_col = st.columns([3, 2])

    with left_col:
        source_options = ["url", "upload"] + (["sample"] if sample_files else [])
        source_labels = {"url": "🔗 URL", "upload": "📁 Upload", "sample": "🎞️ Sample"}
        source_mode = st.segmented_control(
            "Video source",
            options=source_options,
            format_func=lambda k: source_labels[k],
            default="url",
            label_visibility="collapsed",
            key="source_switch",
        )

        video_url = ""
        uploaded = None
        sample_choice = None

        if source_mode == "url":
            video_url = st.text_input("Video URL", placeholder="https://....mp4 or a local path")
        elif source_mode == "upload":
            uploaded = st.file_uploader(
                "Upload a video", type=[e.lstrip(".") for e in allowed_exts]
            )
        else:
            sample_choice = st.selectbox("Sample clip", options=sample_files)

        selected_styles = st.pills(
            "Caption styles",
            options=list(STYLE_META.keys()),
            format_func=lambda k: STYLE_META[k]["label"],
            selection_mode="multi",
            default=list(STYLE_META.keys()),
            key="single_styles",
        )
        st.caption("Deselect all to generate every style — that's the pipeline's own default.")

        source: str | None = None
        source_label: str | None = None

        if source_mode == "url" and video_url.strip():
            source = video_url.strip()
            source_label = source
        elif source_mode == "upload" and uploaded is not None:
            cache_key = f"upload_path::{getattr(uploaded, 'file_id', uploaded.name)}"
            cached = st.session_state.get(cache_key)
            if cached and Path(cached).exists():
                source = cached
            else:
                suffix = Path(uploaded.name).suffix or ".mp4"
                upload_dir = Path(tempfile.gettempdir()) / "f2c_uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                upload_path = upload_dir / f"{int(time.time() * 1000)}{suffix}"
                upload_path.write_bytes(uploaded.getvalue())
                source = str(upload_path)
                st.session_state[cache_key] = source
            source_label = uploaded.name
        elif source_mode == "sample" and sample_choice:
            source = str(sample_dir / sample_choice)
            source_label = sample_choice

        ready = source is not None and run_cfg is not None

        generate_clicked = st.button(
            "🎬  Generate captions", type="primary", use_container_width=True, disabled=not ready
        )
        if run_cfg is None:
            st.caption("⚠️ Set GEMINI_API_KEY in your .env to enable generation.")
        elif source is None:
            st.caption("Add a video source above to enable generation.")

    with right_col:
        with st.container(border=True):
            st.markdown('<div class="f2c-eyebrow">Preview</div>', unsafe_allow_html=True)
            try:
                if source_mode == "url" and video_url.strip():
                    st.video(video_url.strip())
                elif source_mode == "upload" and uploaded is not None:
                    st.video(uploaded)
                elif source_mode == "sample" and sample_choice:
                    st.video(str(sample_dir / sample_choice))
                else:
                    st.caption("Drop in a source on the left to preview it here.")
            except Exception:
                st.caption("Couldn't preview this source directly — it may still process fine.")

    if generate_clicked and ready and source is not None and run_cfg is not None:
        run_logger, log_q = make_logger(f"single-{time.time()}")
        box: dict[str, Any] = {}

        def _target(src=source, styles=list(selected_styles), cfg=run_cfg, logger=run_logger) -> None:
            try:
                box["result"] = run_single_take(src, styles, cfg, logger)
            except Exception as exc:  # noqa: BLE001
                box["error"] = exc

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()

        with st.status("Rolling camera…", expanded=True) as status:
            console_ph = st.empty()
            progress_ph = st.empty()
            poll_and_render(thread, log_q, console_ph, progress_ph, single_progress_fn)
            if "error" in box:
                status.update(label="Cut — something went wrong", state="error")
            else:
                progress_ph.markdown(render_progress_html(100), unsafe_allow_html=True)
                status.update(label="That's a wrap", state="complete", expanded=False)

        take_no = st.session_state.take_counter + 1
        ts = datetime.now().strftime("%H:%M:%S")
        if "error" in box:
            st.session_state.takes.append(
                {"n": take_no, "ts": ts, "mode": "single", "source": source_label, "error": describe_error(box["error"])}
            )
        else:
            st.session_state.takes.append(
                {"n": take_no, "ts": ts, "mode": "single", "source": source_label, "result": box["result"]}
            )
        st.session_state.take_counter += 1
        st.session_state.active_take = len(st.session_state.takes) - 1
        st.rerun()

# ---------------------------------------------------------------------------
# Batch Run mode
# ---------------------------------------------------------------------------

else:
    st.caption(
        "Runs the exact input/tasks.json → output/results.json contract your Docker evaluator uses — "
        "good for a dry run before submission."
    )

    if "batch_rows" not in st.session_state:
        st.session_state.batch_rows = [{"task_id": "task_1", "video_url": ""}]

    edited = st.data_editor(
        st.session_state.batch_rows,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "task_id": st.column_config.TextColumn("Task ID", required=True),
            "video_url": st.column_config.TextColumn("Video URL / path", required=True, width="large"),
        },
        key="batch_editor",
    )
    if isinstance(edited, list):
        rows_list = edited
    else:
        try:
            rows_list = edited.to_dict("records")
        except AttributeError:
            rows_list = list(edited)

    batch_styles = st.pills(
        "Caption styles (applied to every task)",
        options=list(STYLE_META.keys()),
        format_func=lambda k: STYLE_META[k]["label"],
        selection_mode="multi",
        default=list(STYLE_META.keys()),
        key="batch_styles",
    )

    run_clicked = st.button(
        "📦  Run batch", type="primary", use_container_width=True, disabled=run_cfg is None
    )
    if run_cfg is None:
        st.caption("⚠️ Set GEMINI_API_KEY in your .env to enable generation.")

    if run_clicked and run_cfg is not None:
        rows = [r for r in rows_list if r.get("task_id") and r.get("video_url")]
        if not rows:
            st.warning("Add at least one task with a Task ID and Video URL/path.")
        else:
            tasks = [
                {"task_id": r["task_id"], "video_url": r["video_url"], "styles": list(batch_styles)}
                for r in rows
            ]
            Path("input").mkdir(exist_ok=True)
            Path("output").mkdir(exist_ok=True)
            Path("input/tasks.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")

            run_logger, log_q = make_logger(f"batch-{time.time()}")
            box = {}

            def _target(cfg=run_cfg, logger=run_logger) -> None:
                try:
                    run_tasks(config=cfg, logger=logger)
                    box["results"] = json.loads(Path("output/results.json").read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001
                    box["error"] = exc

            thread = threading.Thread(target=_target, daemon=True)
            thread.start()

            def batch_progress_fn(lines: list[str], _total=len(tasks)) -> int:
                done = sum(
                    1 for l in lines if "completed successfully" in l.lower() or " failed" in l.lower()
                )
                return min(100, round(100 * done / max(1, _total)))

            with st.status("Rolling camera on the batch…", expanded=True) as status:
                console_ph = st.empty()
                progress_ph = st.empty()
                poll_and_render(thread, log_q, console_ph, progress_ph, batch_progress_fn)
                if "error" in box:
                    status.update(label="Cut — batch run failed", state="error")
                else:
                    progress_ph.markdown(render_progress_html(100), unsafe_allow_html=True)
                    status.update(label="That's a wrap on the batch", state="complete", expanded=False)

            take_no = st.session_state.take_counter + 1
            ts = datetime.now().strftime("%H:%M:%S")
            if "error" in box:
                st.session_state.takes.append(
                    {"n": take_no, "ts": ts, "mode": "batch", "error": describe_error(box["error"])}
                )
            else:
                st.session_state.takes.append(
                    {"n": take_no, "ts": ts, "mode": "batch", "results": box["results"]}
                )
            st.session_state.take_counter += 1
            st.session_state.active_take = len(st.session_state.takes) - 1
            st.rerun()

# ---------------------------------------------------------------------------
# Results panel
# ---------------------------------------------------------------------------

st.markdown("---")

if st.session_state.active_take is not None and 0 <= st.session_state.active_take < len(st.session_state.takes):
    render_take(st.session_state.takes[st.session_state.active_take])
else:
    st.markdown(
        '<div class="f2c-empty">No takes yet — set your scene above and hit generate.</div>',
        unsafe_allow_html=True,
    )
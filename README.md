# AI Video Captioning (Gemini 2.5 Flash + Prompt Engineering)

Generate four distinct, factually-grounded captions for a short video —
**Formal**, **Sarcastic**, **Humorous-Tech**, and **Humorous-Non-Tech** —
using Google's **Gemini 2.5 Flash** model.

This project uses **prompt engineering only**. There is no fine-tuning, no
training loop, and no modification of model weights anywhere in this
codebase. All behavior is controlled through the prompts in `prompts/`.

---

## How it works

```
Local video file (30s – 2min)
        │
        ▼
 Validate file (exists, format, duration) ── src/video_utils.py
        │
        ▼
 Upload video to Gemini File API ── src/gemini_client.py
        │
        ▼
 Poll until Gemini finishes processing (state: ACTIVE)
        │
        ▼
 Prompt #1 — Factual Summary
   • Watch the entire video, beginning → middle → end
   • Track scene changes, actions, objects, people, emotions, context
   • Preserve chronology, avoid hallucination        ── prompts/system_prompt.py
                                                          prompts/summary_prompt.py
        │
        ▼
 Prompt #2 — Style Rewrite
   • Rewrite the frozen factual summary into 4 styles
   • Every fact must be preserved; nothing new invented ── prompts/rewrite_prompt.py
        │
        ▼
 Parse + validate JSON  ── src/utils.py
        │
        ▼
 Save outputs/<video>_<timestamp>.json
 Save outputs/<video>_<timestamp>.txt
        │
        ▼
 Print captions to the console
```

The pipeline deliberately separates **"what happened"** (one factual,
hallucination-resistant summary) from **"how to say it"** (four stylistic
rewrites that are contractually forbidden from inventing new facts). This
two-stage design is the core piece of prompt engineering in this project.

---

## Project structure

```
video_captioning/
├── app.py                    # CLI entry point
├── config.py                 # Environment-driven configuration (dataclass)
├── requirements.txt
├── README.md
├── .env.example
│
├── prompts/
│   ├── system_prompt.py       # System instruction for factual analysis
│   ├── summary_prompt.py      # Builds the "describe the whole video" prompt
│   └── rewrite_prompt.py      # Builds the "rewrite into 4 styles" prompt
│
├── src/
│   ├── gemini_client.py       # Google GenAI SDK wrapper (upload/poll/generate)
│   ├── caption_pipeline.py    # Orchestrates the end-to-end pipeline
│   ├── video_utils.py         # Local video validation & metadata (OpenCV)
│   ├── logger.py              # Colored console + file logging
│   ├── utils.py                # Exceptions, retry decorator, JSON helpers, spinner
│   └── models.py              # Dataclasses: VideoMetadata, CaptionSet, etc.
│
├── outputs/                    # JSON + TXT results are written here
└── sample_videos/               # Put your own test videos here
```

---

## Requirements

- Python 3.10+
- A Gemini API key — get one for free at
  <https://aistudio.google.com/apikey>
- `ffmpeg` is **not** required. Video duration is measured with OpenCV
  (`opencv-python-headless`), a pure pip dependency.

---

## Setup

```bash
# 1. Clone / unzip the project, then enter it
cd video_captioning

# 2. Create and activate a virtual environment (recommended)
python3.10 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# then edit .env and set:
#   GEMINI_API_KEY=your_real_key_here
```

The `GEMINI_API_KEY` is **never hardcoded**. It is read from the `.env`
file (or your shell environment) at runtime via `config.py`.

---

## Usage

```bash
python app.py --video sample_videos/sample.mp4
```

Optional flags:

```bash
python app.py --video sample.mp4 --output-dir results
python app.py --video sample.mp4 --log-level DEBUG
python app.py --video sample.mp4 --keep-remote-file   # skip deleting the
                                                        # uploaded file from
                                                        # Gemini afterward
```

On success, the console prints the four captions as JSON, e.g.:

```json
{
  "formal": "A presenter demonstrates a new wireless charger on a desk, explaining its features to the camera.",
  "sarcastic": "Oh good, another wireless charger. Truly the invention we've all been waiting for.",
  "humorous_tech": "Behold: another attempt to solve the 'plugging in a cable' problem with more cables.",
  "humorous_non_tech": "Someone's very excited about a little pad that charges your phone. Bless them."
}
```

Two files are also written to `outputs/`:

- `outputs/<video-name>_<timestamp>.json` — full structured result
  (video metadata, factual summary, and all four captions)
- `outputs/<video-name>_<timestamp>.txt` — the same content in a
  human-readable plain-text report

---

## Configuration reference

All settings can be overridden via environment variables (see
`.env.example` for the full list with defaults):

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(required)* | Your Gemini API key |
| `GEMINI_MODEL_NAME` | `gemini-2.5-flash` | Model used for both stages |
| `MAX_RETRIES` | `3` | Retry attempts for transient failures |
| `RETRY_BACKOFF_BASE` | `2.0` | Exponential backoff base (1s, 2s, 4s, ...) |
| `UPLOAD_POLL_INTERVAL_SECONDS` | `3.0` | How often to poll file processing status |
| `UPLOAD_TIMEOUT_SECONDS` | `300.0` | Max time to wait for Gemini to process the video |
| `GENERATION_TIMEOUT_SECONDS` | `180.0` | Max time to wait for a single generate_content call |
| `MIN_VIDEO_DURATION_SECONDS` | `30.0` | Minimum accepted video length |
| `MAX_VIDEO_DURATION_SECONDS` | `120.0` | Maximum accepted video length |
| `DURATION_TOLERANCE_SECONDS` | `1.5` | Tolerance around the bounds above |
| `SUMMARY_TEMPERATURE` | `0.2` | Lower = more literal/factual summary |
| `REWRITE_TEMPERATURE` | `0.8` | Higher = more creative caption rewrites |
| `MAX_OUTPUT_TOKENS` | `4096` | Cap on generated tokens per call |
| `OUTPUT_DIR` | `outputs` | Where JSON/TXT results are saved |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `KEEP_REMOTE_FILE` | `false` | Skip deleting the uploaded file from Gemini |

---

## Prompt engineering details

**Stage 1 — Factual summary** (`prompts/system_prompt.py` +
`prompts/summary_prompt.py`): Gemini is instructed to watch the entire
video, explicitly track beginning/middle/end, note every scene change,
list actions/objects/people/emotions/context in chronological order, and
explicitly avoid inventing anything unclear or unseen. Temperature is kept
low (`0.2` by default) to favor literal accuracy over creativity.

**Stage 2 — Style rewrite** (`prompts/rewrite_prompt.py`): The factual
summary from Stage 1 is quoted back to the model as frozen ground truth.
Gemini is instructed to rewrite it into four styles — formal, sarcastic,
humorous-tech, humorous-non-tech — while being explicitly forbidden from
adding any fact not already present in the summary. The model is asked to
return strict JSON (`response_mime_type="application/json"`), which is
then parsed and schema-validated in `src/utils.py`. If validation fails,
the whole rewrite call is retried automatically.

No dataset is used, no gradient updates occur, and no model checkpoint is
produced — every behavior above comes purely from prompt text and
generation parameters (temperature, `response_mime_type`, etc.).

---

## Error handling & exit codes

`app.py` returns a distinct exit code depending on what failed, which is
useful for scripting/CI:

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Unexpected/unhandled error |
| `2` | Configuration error (e.g. missing `GEMINI_API_KEY`) |
| `3` | Video validation error (missing file, bad format, wrong duration) |
| `4` | Upload/processing timeout |
| `5` | Caption JSON failed validation after all retries |
| `6` | Unrecoverable Gemini API error |

Every stage logs to both the console (colorized) and
`outputs/logs/app.log`.

---

## Engineering features

- **Retries with exponential backoff** on transient network/API failures
  (`src/utils.py::retry`), applied to uploads, summary generation, and
  caption generation.
- **Hard timeouts** on both the upload/processing wait and each individual
  Gemini call (`GeminiClient._call_with_timeout`), so the CLI never hangs
  indefinitely.
- **JSON schema validation** for the four-caption output, with automatic
  retry if the model's JSON is malformed or incomplete.
- **A dependency-free terminal spinner** (`ProgressIndicator`) shows
  progress during uploads and generation, and degrades to a single static
  log line when output isn't a TTY (e.g. in CI).
- **Immutable, validated configuration** via a frozen dataclass
  (`config.Config`), loaded once from environment variables with
  `Config.from_env()` and safely overridable via `Config.with_overrides()`.
- **Full type hints** and dataclasses throughout; clean separation between
  I/O (`gemini_client`, `video_utils`), orchestration (`caption_pipeline`),
  and pure logic (`utils`, `models`).
- **Best-effort cleanup**: the uploaded remote file is deleted from
  Gemini's File API after each run unless `--keep-remote-file` is passed,
  and cleanup failures are logged as warnings rather than crashing the run.

---

## Troubleshooting

**"GEMINI_API_KEY is not set"** — Make sure you copied `.env.example` to
`.env` and filled in a real key, and that you're running `python app.py`
from the project root (so `.env` is found).

**"Could not determine the duration of ..."** — The video's container may
be missing frame-rate/frame-count metadata that OpenCV needs. Try
re-encoding it, e.g. `ffmpeg -i input.mov -c copy fixed.mp4`, if you have
ffmpeg installed locally (optional — not a runtime dependency of this
project).

**Video duration rejected** — By default only videos between 30 and 120
seconds (±1.5s tolerance) are accepted. Adjust `MIN_VIDEO_DURATION_SECONDS`
/ `MAX_VIDEO_DURATION_SECONDS` in `.env` if you need different bounds.

**Timeouts on large videos** — Increase `UPLOAD_TIMEOUT_SECONDS` and/or
`GENERATION_TIMEOUT_SECONDS` in `.env`.

---

## License

This project template is provided as-is for you to use and modify freely.

# 🎬 Frame2Caption

> AI-powered video captioning agent that generates multiple caption styles from short videos using **Google Gemini 2.5 Flash**.

Frame2Caption is a modular multimodal AI pipeline built for automated video caption generation. Given a video, it produces captions in four distinct styles while maintaining factual consistency with the video content.

Designed for the **AMD Developer Hackathon – Track 2**, the project follows a Docker-based batch processing architecture, making it suitable for automated evaluation as well as local development.

---

# ✨ Features

- 🎥 Processes videos from **30 seconds to 2 minutes**
- 🧠 Uses **Google Gemini 2.5 Flash** for multimodal video understanding
- ✍️ Generates four caption styles
  - Formal
  - Sarcastic
  - Humorous-Tech
  - Humorous-Non-Tech
- 📦 Docker-compatible batch processing
- 🌐 Supports both
  - Local video files
  - Remote video URLs
- 🔄 Automatic retries and robust error handling
- 📄 Produces evaluator-compatible JSON output
- 🏗 Modular, reusable architecture

---

# 📂 Project Structure

```text
Frame2Caption/
│
├── app.py
├── config.py
├── Dockerfile
├── requirements.txt
│
├── prompts/
│
├── src/
│   ├── caption_pipeline.py
│   ├── downloader.py
│   ├── gemini_client.py
│   ├── io.py
│   ├── logger.py
│   ├── models.py
│   ├── task_runner.py
│   ├── utils.py
│   └── video_utils.py
│
├── input/
├── output/
├── sample_videos/
└── README.md
```

---

# ⚙️ Architecture

```text
                app.py
                   │
                   ▼
            Task Runner
                   │
      ┌────────────┴────────────┐
      ▼                         ▼
 Video Downloader        Caption Pipeline
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
          Video Validation             Gemini 2.5 Flash
                                                │
                                                ▼
                                   Caption Generation
                                                │
                                                ▼
                                   output/results.json
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/krisjk003/Frame2Caption.git
cd Frame2Caption
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Configuration

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

# ▶️ Local Usage

Create

```text
input/tasks.json
```

Example

```json
[
  {
    "task_id": "test1",
    "video_url": "sample_videos/video.mp4",
    "styles": [
      "formal",
      "sarcastic",
      "humorous_tech",
      "humorous_non_tech"
    ]
  }
]
```

Run

```bash
python app.py
```

Generated output

```text
output/results.json
```

---

# 🐳 Docker Usage

Build

```bash
docker build -t frame2caption .
```

Run

```bash
docker run --rm \
    -e GEMINI_API_KEY=YOUR_API_KEY \
    -v $(pwd)/input:/input \
    -v $(pwd)/output:/output \
    frame2caption
```

---

# 📤 Output Format

```json
[
  {
    "task_id": "test1",
    "captions": {
      "formal": "...",
      "sarcastic": "...",
      "humorous_tech": "...",
      "humorous_non_tech": "..."
    }
  }
]
```

---

# 🛠 Tech Stack

- Python 3.10+
- Google Gemini 2.5 Flash
- Prompt Engineering
- Requests
- python-dotenv
- Docker

---

# 📋 Pipeline

1. Read task list
2. Download or locate the video
3. Validate video metadata
4. Upload video to Gemini
5. Wait until processing completes
6. Generate a factual video summary
7. Rewrite into four caption styles
8. Save evaluator-compatible JSON output

---

# 🎯 Supported Caption Styles

| Style | Description |
|--------|-------------|
| Formal | Professional, objective and factual |
| Sarcastic | Dry, ironic and lightly mocking |
| Humorous-Tech | Technology/programming themed humor |
| Humorous-Non-Tech | Everyday humor without technical references |

---

# 📌 Notes

- No model fine-tuning is performed.
- Caption quality is achieved entirely through prompt engineering.
- Supports both local videos and remote URLs.
- Fully compatible with Docker-based batch evaluation.
- Designed to process multiple tasks sequentially from a single input file.

---

# 📄 License

This project is released under the MIT License.

---

# 👨‍💻 Author

**Jyothish Kumar JS**

GitHub: https://github.com/krisjk003

# 🎬 Frame2Caption

**Frame2Caption** is a modular AI-powered video captioning system that generates four distinct caption styles from short videos using **Google Gemini 2.5 Flash** and **prompt engineering**—without fine-tuning.

Built for multimodal AI applications and hackathons, the project processes videos end-to-end and produces captions tailored to different audiences.

---

## ✨ Features

- 🎥 Accepts short MP4 videos (10 seconds – 2 minutes)
- 🧠 Uses Gemini 2.5 Flash for multimodal understanding
- ✍️ Generates four caption styles:
  - Formal
  - Sarcastic
  - Humorous-Tech
  - Humorous-Non-Tech
- 🔄 Modular pipeline with reusable components
- ⚡ Automatic retries and robust error handling
- 📄 Saves generated captions as structured JSON

---

## 📂 Project Structure

```text
Frame2Caption/
├── prompts/          # Prompt templates
├── sample_videos/    # Sample videos
├── src/              # Core pipeline
│   ├── caption_pipeline.py
│   ├── gemini_client.py
│   ├── logger.py
│   ├── models.py
│   ├── utils.py
│   └── video_utils.py
├── app.py
├── config.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

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

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Configuration

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

## ▶️ Usage

Run the caption generator

```bash
python app.py --video sample_videos/video.mp4
```

Or specify a custom output directory

```bash
python app.py --video sample_videos/video.mp4 --output-dir outputs
```

---

## 📤 Example Output

```json
{
  "formal": "...",
  "sarcastic": "...",
  "humorous_tech": "...",
  "humorous_non_tech": "..."
}
```

---

## ⚙️ Technologies

- Python 3.10+
- Google Gemini 2.5 Flash
- Prompt Engineering
- python-dotenv

---

## 📌 Notes

- No model fine-tuning is used.
- Caption quality is achieved through prompt engineering.
- Supports videos between 10 seconds and 2 minutes.

---

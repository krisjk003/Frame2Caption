# -----------------------------
# Frame2Caption
# AMD Developer Hackathon - Track 2
# -----------------------------

FROM python:3.10-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Directories expected by the evaluator
RUN mkdir -p /input /output

# Start the agent
ENTRYPOINT ["python", "app.py"]
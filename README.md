# Offline Subtitle Generator 🎬

A fully **offline**, **CPU-optimized** subtitle generation system that combines
**speech recognition** (Faster-Whisper) and **sound event detection** (YAMNet)
to produce rich `.srt` subtitle files with contextual captions.

## Features

- 🗣️ **Speech-to-Text** — Accurate transcription using Faster-Whisper (INT8 quantized)
- 🔊 **Sound Event Captions** — Detects 60+ sound events (door slams, crying, engines, etc.)
- 🧠 **Smart Segmentation** — Silero VAD separates speech from non-speech regions
- ⚡ **CPU Optimized** — INT8 quantization, VAD-based skip, voluntary CPU throttling
- 💾 **Result Caching** — Skip re-processing unchanged files
- 🔌 **100% Offline** — No cloud APIs, no internet required after setup

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/macOS

# Install Python packages
pip install -r requirements.txt
```

### 2. Install FFmpeg

- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html), add to PATH
- **Linux:** `sudo apt install ffmpeg`
- **macOS:** `brew install ffmpeg`

### 3. Download Models

See [models/README.md](models/README.md) for download instructions.

### 4. Generate Subtitles

```bash
# Basic usage — generates movie.srt alongside the video
python main.py movie.mp4

# Custom output path
python main.py movie.mp4 -o subtitles.srt

# Use faster (but less accurate) model
python main.py movie.mp4 --model base

# Parallel processing (faster, higher CPU)
python main.py movie.mp4 --parallel

# Force language detection
python main.py movie.mp4 --language en

# Allow higher CPU usage
python main.py movie.mp4 --max-cpu 80

# Verbose logging
python main.py movie.mp4 -v
```

## Architecture

```
Video File → Audio Extract → VAD Segment → ASR + SED → Merge → .srt
               (FFmpeg)      (Silero)     (Whisper)   (YAMNet)
                                          (parallel/sequential)
```

| Stage | Time (10 min video) | CPU | RAM |
|---|---|---|---|
| Audio extraction | 2–5 sec | 10–20% | 50 MB |
| VAD segmentation | 1–3 sec | 15–25% | 30 MB |
| ASR (Whisper small INT8) | 3–4 min | 50–70% | 600 MB |
| Sound event detection | 5–15 sec | 20–40% | 50 MB |
| **Total** | **~4–5 min** | **Peak 70%** | **~880 MB** |

## Configuration

Edit `config.yaml` to adjust:
- Model size (`tiny` / `base` / `small`)
- CPU usage limit
- Confidence thresholds
- Threading mode (sequential/parallel)
- VAD sensitivity

## Project Structure

```
offline-subtitles/
├── main.py                # CLI entry point
├── config.py              # Configuration loader
├── config.yaml            # Settings
├── requirements.txt       # Dependencies
├── pipeline/              # Core modules
│   ├── orchestrator.py    # Pipeline coordinator
│   ├── audio_extractor.py # FFmpeg wrapper
│   ├── vad.py             # Silero VAD
│   ├── asr_worker.py      # Faster-Whisper ASR
│   ├── sed_worker.py      # YAMNet SED
│   ├── merger.py          # Timeline merger
│   ├── srt_writer.py      # SRT formatter
│   └── cpu_throttle.py    # CPU limiter
├── models/                # Model files
├── cache/                 # Processing cache
└── tests/                 # Unit tests
```

## Output Example

```srt
1
00:00:01,200 --> 00:00:04,800
Hello everyone, welcome to the show.

2
00:00:05,100 --> 00:00:06,300
(Audience clapping)

3
00:00:06,500 --> 00:00:10,200
Today we're going to talk about something amazing.

4
00:00:10,500 --> 00:00:11,800
(Door slams)
```

## Requirements

- **Python** 3.10+
- **FFmpeg** (system-installed)
- **Hardware:** Intel i5 class CPU, 8+ GB RAM
- **Disk:** ~200 MB for models
- **No GPU required**

## License

MIT

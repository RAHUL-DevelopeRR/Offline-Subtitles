# Model Files

This directory contains pre-downloaded ML model files used by the pipeline.

## Required Models

### 1. YAMNet TFLite (Sound Event Detection)

Download the quantized TFLite model:

1. Visit: https://tfhub.dev/google/lite-model/yamnet/tflite/1
2. Download `yamnet.tflite` (or `1.tflite` — rename to `yamnet.tflite`)
3. Place it in this `models/` directory

**Alternative (direct download):**
```bash
curl -L "https://tfhub.dev/google/lite-model/yamnet/tflite/1?lite-format=tflite" -o models/yamnet.tflite
```

### 2. YAMNet Class Map

Download the class name mapping CSV:

1. Visit: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
2. Download `yamnet_class_map.csv`
3. Place it in this `models/` directory

**Alternative (direct download):**
```bash
curl -L "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv" -o models/yamnet_class_map.csv
```

### 3. Faster-Whisper ASR Model

The Faster-Whisper model is downloaded **automatically** on first use.
It will be cached in your user directory (~/.cache/huggingface/).

Models available:
| Model | Download Size | RAM Usage |
|---|---|---|
| `tiny` | ~40 MB | ~300 MB |
| `base` | ~75 MB | ~400 MB |
| `small` (default) | ~250 MB | ~600 MB |

To pre-download:
```python
from faster_whisper import WhisperModel
model = WhisperModel("small", device="cpu", compute_type="int8")
```

### 4. Silero VAD Model

The Silero VAD model is downloaded **automatically** on first use via `torch.hub`.
It will be cached in your torch hub directory.

Size: ~2 MB

## Directory Structure After Setup

```
models/
├── yamnet.tflite           # ~4 MB (must download manually)
├── yamnet_class_map.csv    # ~30 KB (must download manually)
└── README.md               # This file
```

## Verification

Run this to verify all models are accessible:
```bash
python -c "
from pathlib import Path
models = ['models/yamnet.tflite', 'models/yamnet_class_map.csv']
for m in models:
    p = Path(m)
    status = '✅' if p.exists() else '❌ MISSING'
    size = f'({p.stat().st_size / 1024:.0f} KB)' if p.exists() else ''
    print(f'  {status} {m} {size}')
"
```

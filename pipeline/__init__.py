"""
Offline Subtitle Generator — Pipeline Package

Modular processing pipeline for offline subtitle generation:
  - audio_extractor: FFmpeg-based audio extraction
  - vad: Voice activity detection and segmentation
  - asr_worker: Speech-to-text via Faster-Whisper
  - sed_worker: Sound event detection via YAMNet
  - merger: Timeline merging with overlap resolution
  - srt_writer: Standard SRT file output
  - cpu_throttle: CPU usage monitoring and throttling
"""

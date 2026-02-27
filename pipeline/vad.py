"""
Voice Activity Detection — Silero VAD-based segmenter.

Splits audio into speech, non-speech, and silence regions using the
Silero VAD model. This enables the pipeline to:
  - Send only speech regions to ASR (saves ~30-50% processing time)
  - Send only non-speech regions to SED (avoids false positives)
  - Skip silence entirely
"""

import logging
import numpy as np
import torch
import soundfile as sf
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class AudioSegment:
    """A labeled chunk of audio with absolute timestamps."""
    start_sec: float
    end_sec: float
    segment_type: str       # "speech" | "non_speech" | "silence"
    audio_data: np.ndarray  # Float32 PCM samples

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    def __repr__(self):
        return (f"AudioSegment({self.segment_type}, "
                f"{self.start_sec:.2f}–{self.end_sec:.2f}s, "
                f"{self.duration:.2f}s)")


class VADSegmenter:
    """
    Segments audio into speech and non-speech regions using Silero VAD.

    The model is lazily loaded on first use to avoid startup cost
    when it may not be needed (e.g., cached results).
    """

    def __init__(self, config):
        self.threshold = getattr(config, "threshold", 0.5)
        self.min_speech_sec = getattr(config, "min_speech_sec", 0.25)
        self.min_silence_sec = getattr(config, "min_silence_sec", 0.3)
        self.padding_sec = getattr(config, "padding_sec", 0.2)
        self.energy_threshold = getattr(config, "energy_threshold", 0.001)

        # Lazy-loaded
        self._model = None
        self._utils = None

    def _load_model(self):
        """Load the Silero VAD model (JIT/ONNX) on first use."""
        if self._model is not None:
            return

        logger.info("Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            onnx=True,
            trust_repo=True
        )
        self._model = model
        (self._get_speech_timestamps,
         _, self._read_audio, *_) = utils
        logger.info("Silero VAD loaded successfully.")

    def segment(self, audio_path: Path) -> List[AudioSegment]:
        """
        Segment an audio file into speech and non-speech regions.

        Args:
            audio_path: Path to 16kHz mono WAV file.

        Returns:
            List of AudioSegment objects sorted by start time.
        """
        self._load_model()

        # Read audio
        audio_np, sr = sf.read(str(audio_path), dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(
                f"Expected {SAMPLE_RATE}Hz audio, got {sr}Hz. "
                f"Re-extract with correct sample rate."
            )

        # Ensure 1-D
        if audio_np.ndim > 1:
            audio_np = audio_np[:, 0]

        wav_tensor = torch.from_numpy(audio_np)
        total_duration = len(audio_np) / SAMPLE_RATE

        logger.info(f"Running VAD on {total_duration:.1f}s audio...")

        # Get speech timestamps
        speech_timestamps = self._get_speech_timestamps(
            wav_tensor,
            self._model,
            threshold=self.threshold,
            min_speech_duration_ms=int(self.min_speech_sec * 1000),
            min_silence_duration_ms=int(self.min_silence_sec * 1000),
            return_seconds=True,
            sampling_rate=SAMPLE_RATE
        )

        # Build segment list
        segments = self._build_segments(audio_np, speech_timestamps, total_duration)

        speech_count = sum(1 for s in segments if s.segment_type == "speech")
        nonspeech_count = sum(1 for s in segments if s.segment_type == "non_speech")
        speech_duration = sum(s.duration for s in segments if s.segment_type == "speech")

        logger.info(
            f"VAD complete: {speech_count} speech segments "
            f"({speech_duration:.1f}s), {nonspeech_count} non-speech segments"
        )

        return segments

    def _build_segments(
        self,
        audio_np: np.ndarray,
        speech_timestamps: list,
        total_duration: float
    ) -> List[AudioSegment]:
        """Build speech and non-speech segments from VAD timestamps."""
        segments = []
        prev_end = 0.0

        for ts in speech_timestamps:
            speech_start = max(0.0, ts["start"] - self.padding_sec)
            speech_end = min(total_duration, ts["end"] + self.padding_sec)

            # Non-speech gap before this speech segment
            if speech_start > prev_end + 0.05:  # 50ms minimum gap
                gap_audio = self._slice_audio(audio_np, prev_end, speech_start)
                if self._has_energy(gap_audio):
                    segments.append(AudioSegment(
                        start_sec=prev_end,
                        end_sec=speech_start,
                        segment_type="non_speech",
                        audio_data=gap_audio
                    ))
                # else: silence — skip entirely

            # Speech segment
            speech_audio = self._slice_audio(audio_np, speech_start, speech_end)
            segments.append(AudioSegment(
                start_sec=speech_start,
                end_sec=speech_end,
                segment_type="speech",
                audio_data=speech_audio
            ))
            prev_end = speech_end

        # Trailing non-speech after last speech segment
        if prev_end < total_duration - 0.05:
            tail_audio = self._slice_audio(audio_np, prev_end, total_duration)
            if self._has_energy(tail_audio):
                segments.append(AudioSegment(
                    start_sec=prev_end,
                    end_sec=total_duration,
                    segment_type="non_speech",
                    audio_data=tail_audio
                ))

        return segments

    def _slice_audio(self, audio_np: np.ndarray, start_sec: float, end_sec: float) -> np.ndarray:
        """Extract a slice of audio samples by time range."""
        start_idx = int(start_sec * SAMPLE_RATE)
        end_idx = int(end_sec * SAMPLE_RATE)
        return audio_np[start_idx:end_idx]

    def _has_energy(self, samples: np.ndarray) -> bool:
        """Check if audio samples have meaningful energy (not silence)."""
        if len(samples) == 0:
            return False
        return float(np.abs(samples).mean()) > self.energy_threshold

"""
ASR Worker — Speech-to-text using Faster-Whisper.

Uses the CTranslate2 backend with INT8 quantization for efficient
CPU-based transcription. Processes speech segments identified by VAD.

Optimized for both speed and accuracy:
  - condition_on_previous_text=False for faster independent chunks
  - Temperature fallback for difficult audio segments
  - Word-level timestamps for precise boundaries
  - Confidence filtering to suppress hallucinated text
"""

import os
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """A single transcribed speech segment with absolute timestamps."""
    start_sec: float
    end_sec: float
    text: str
    confidence: float    # Average log probability
    language: str

    def __repr__(self):
        return (f"TranscriptionResult({self.start_sec:.2f}–{self.end_sec:.2f}s, "
                f"'{self.text[:40]}...', conf={self.confidence:.2f})")


class ASRWorker:
    """
    Automatic Speech Recognition using Faster-Whisper.

    Features:
      - INT8 quantized inference for CPU efficiency
      - Lazy model loading with auto-detected thread count
      - Automatic language detection (first segment)
      - Temperature fallback for difficult audio
      - Word-level timestamps for precise segment boundaries
      - Confidence-based filtering to suppress hallucinations
      - Handles long segments by splitting at quiet points
    """

    def __init__(self, config):
        self.model_size = getattr(config, "model", "small")
        self.compute_type = getattr(config, "compute_type", "int8")
        self.beam_size = getattr(config, "beam_size", 3)
        self.language = getattr(config, "language", None)
        self.max_segment_sec = getattr(config, "max_segment_sec", 30)

        # Speed tuning
        self.best_of = getattr(config, "best_of", 1)
        self.patience = getattr(config, "patience", 1.0)

        # Accuracy tuning
        self.no_speech_threshold = getattr(config, "no_speech_threshold", 0.6)
        self.log_prob_threshold = getattr(config, "log_prob_threshold", -1.0)
        self.min_confidence = getattr(config, "min_confidence", -1.0)
        self.word_timestamps = getattr(config, "word_timestamps", True)
        self.initial_prompt = getattr(config, "initial_prompt", None)

        # Thread count: 0 = auto-detect
        raw_threads = getattr(config, "threads", 0)
        if raw_threads <= 0:
            self.cpu_threads = os.cpu_count() or 4
            logger.info(f"Auto-detected {self.cpu_threads} CPU threads")
        else:
            self.cpu_threads = raw_threads

        # Temperature fallback: parse comma-separated string or use default
        raw_temp = getattr(config, "temperature", "0.0,0.2,0.4,0.6,0.8,1.0")
        if isinstance(raw_temp, str):
            self.temperature = [float(t.strip()) for t in raw_temp.split(",")]
        elif isinstance(raw_temp, (list, tuple)):
            self.temperature = [float(t) for t in raw_temp]
        else:
            self.temperature = [float(raw_temp)]

        # Lazy-loaded
        self._model = None
        self._detected_language = None

    def _load_model(self):
        """Load the Faster-Whisper model on first use."""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        logger.info(
            f"Loading Faster-Whisper model '{self.model_size}' "
            f"(compute_type={self.compute_type}, threads={self.cpu_threads})"
        )

        self._model = WhisperModel(
            self.model_size,
            device="cpu",
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads
        )

        logger.info(f"Faster-Whisper model loaded successfully.")

    def transcribe(self, segment) -> List[TranscriptionResult]:
        """
        Transcribe a single speech AudioSegment.

        Args:
            segment: AudioSegment with segment_type == "speech".

        Returns:
            List of TranscriptionResult with absolute timestamps.
        """
        self._load_model()

        audio_data = segment.audio_data.astype(np.float32)

        # Skip very short segments (<0.1s)
        if len(audio_data) < 1600:  # 0.1s at 16kHz
            return []

        # Split long segments into chunks for better accuracy
        chunks = self._split_if_needed(segment)

        all_results = []
        for chunk_audio, chunk_offset in chunks:
            results = self._transcribe_chunk(chunk_audio, chunk_offset)
            all_results.extend(results)

        return all_results

    def _transcribe_chunk(
        self,
        audio_data: np.ndarray,
        time_offset: float
    ) -> List[TranscriptionResult]:
        """Transcribe a single audio chunk with all optimizations."""
        try:
            segments_iter, info = self._model.transcribe(
                audio_data,
                beam_size=self.beam_size,
                best_of=self.best_of,
                patience=self.patience,
                language=self.language or self._detected_language,
                temperature=self.temperature,
                vad_filter=False,              # We already did VAD externally
                word_timestamps=self.word_timestamps,
                condition_on_previous_text=False,  # Faster: no sequential dependency
                no_speech_threshold=self.no_speech_threshold,
                log_prob_threshold=self.log_prob_threshold,
                initial_prompt=self.initial_prompt,
            )

            # Cache detected language for subsequent segments
            if self._detected_language is None:
                self._detected_language = info.language
                logger.info(
                    f"Detected language: {info.language} "
                    f"(probability: {info.language_probability:.2f})"
                )

            results = []
            for seg in segments_iter:
                text = seg.text.strip()
                if not text:
                    continue

                # Confidence filtering: skip hallucinated/low-quality segments
                if seg.avg_logprob < self.min_confidence:
                    logger.debug(
                        f"Filtered low-confidence segment: '{text[:40]}' "
                        f"(avg_logprob={seg.avg_logprob:.2f} < {self.min_confidence})"
                    )
                    continue

                # Use word-level timestamps for precise boundaries if available
                if self.word_timestamps and seg.words:
                    abs_start = time_offset + seg.words[0].start
                    abs_end = time_offset + seg.words[-1].end
                else:
                    # Fallback to segment-level timestamps
                    abs_start = time_offset + seg.start
                    abs_end = time_offset + seg.end

                results.append(TranscriptionResult(
                    start_sec=abs_start,
                    end_sec=abs_end,
                    text=text,
                    confidence=seg.avg_logprob,
                    language=info.language
                ))

            return results

        except Exception as e:
            logger.error(f"ASR error at offset {time_offset:.1f}s: {e}")
            return []

    def _split_if_needed(self, segment) -> List[tuple]:
        """
        Split a long segment into chunks for Whisper processing.
        Whisper works best with segments under 30 seconds.

        Returns:
            List of (audio_data, time_offset) tuples.
        """
        audio = segment.audio_data.astype(np.float32)
        max_samples = int(self.max_segment_sec * 16000)

        if len(audio) <= max_samples:
            return [(audio, segment.start_sec)]

        chunks = []
        offset = 0
        while offset < len(audio):
            end = min(offset + max_samples, len(audio))

            # Try to split at a quiet point near the boundary
            if end < len(audio):
                search_start = max(offset, end - 8000)  # Search last 0.5s
                quiet_point = self._find_quiet_point(audio[search_start:end])
                if quiet_point is not None:
                    end = search_start + quiet_point

            chunk_audio = audio[offset:end]
            chunk_time = segment.start_sec + (offset / 16000)
            chunks.append((chunk_audio, chunk_time))
            offset = end

        logger.debug(
            f"Split {segment.duration:.1f}s segment into {len(chunks)} chunks"
        )
        return chunks

    def _find_quiet_point(self, audio: np.ndarray, window: int = 160) -> Optional[int]:
        """
        Find the quietest point in an audio buffer for splitting.

        Args:
            audio: Audio samples to search.
            window: RMS window size in samples (10ms at 16kHz).

        Returns:
            Sample index of the quietest point, or None if audio is too short.
        """
        if len(audio) < window * 2:
            return None

        # Compute sliding RMS energy
        min_energy = float("inf")
        min_pos = None

        for i in range(0, len(audio) - window, window // 2):
            energy = np.sqrt(np.mean(audio[i:i + window] ** 2))
            if energy < min_energy:
                min_energy = energy
                min_pos = i + window // 2

        return min_pos


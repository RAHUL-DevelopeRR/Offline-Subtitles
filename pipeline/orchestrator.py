"""
Pipeline Orchestrator — Coordinates the entire subtitle generation pipeline.

Stages:
  1. Audio Extraction (FFmpeg)
  2. VAD Segmentation (Silero)
  3. ASR + SED Analysis (Faster-Whisper + YAMNet)
  4. Timeline Merge + SRT Output
"""

import time
import hashlib
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, List, Tuple

from .audio_extractor import AudioExtractor
from .vad import VADSegmenter, AudioSegment
from .asr_worker import ASRWorker, TranscriptionResult
from .sed_worker import SEDWorker, SoundEventResult
from .merger import TimelineMerger, SubtitleEntry
from .srt_writer import SRTWriter
from .cpu_throttle import CPUThrottle

logger = logging.getLogger(__name__)

# Type alias for progress callbacks: (message: str, percent: int) -> None
ProgressCallback = Optional[Callable[[str, int], None]]


class SubtitlePipeline:
    """
    Main pipeline orchestrator for offline subtitle generation.

    Usage:
        config = load_config()
        pipeline = SubtitlePipeline(config)
        pipeline.process("video.mp4", "video.srt")
    """

    def __init__(self, config):
        self.config = config

        # Initialize pipeline components (models loaded lazily)
        self.extractor = AudioExtractor(
            sample_rate=config.audio.sample_rate,
            channels=config.audio.channels
        )
        self.vad = VADSegmenter(config.vad)
        self.asr = ASRWorker(config.asr)
        self.sed = SEDWorker(config.sed)
        self.merger = TimelineMerger(config.merge)
        self.writer = SRTWriter()
        self.throttle = CPUThrottle(
            max_percent=config.max_cpu_percent,
            check_interval=config.threading.throttle_check_interval
        )

    def process(
        self,
        video_path: Path,
        output_path: Path,
        progress_cb: ProgressCallback = None
    ) -> List[SubtitleEntry]:
        """
        Run the full subtitle generation pipeline.

        Args:
            video_path: Path to the input video file.
            output_path: Path for the output .srt file.
            progress_cb: Optional callback for progress updates.

        Returns:
            List of generated SubtitleEntry objects.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        start_time = time.monotonic()

        logger.info(f"{'='*60}")
        logger.info(f"Offline Subtitle Generator")
        logger.info(f"Input:  {video_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Mode:   {self.config.threading_mode}")
        logger.info(f"ASR:    Faster-Whisper {self.config.asr.model} ({self.config.asr.compute_type})")
        logger.info(f"{'='*60}")

        # Check cache
        cached = self._check_cache(video_path, output_path)
        if cached:
            self._report(progress_cb, "Using cached result.", 100)
            return cached

        # ── Stage 1: Audio Extraction ──
        self._report(progress_cb, "Extracting audio from video...", 5)
        audio_path = self.extractor.extract(video_path)

        try:
            # ── Stage 2: VAD Segmentation ──
            self._report(progress_cb, "Detecting speech regions...", 10)
            segments = self.vad.segment(audio_path)
            speech_segs = [s for s in segments if s.segment_type == "speech"]
            nonspeech_segs = [s for s in segments if s.segment_type == "non_speech"]

            logger.info(
                f"Segments: {len(speech_segs)} speech, "
                f"{len(nonspeech_segs)} non-speech, "
                f"{sum(s.duration for s in speech_segs):.1f}s total speech"
            )

            # ── Stage 3: Analysis ──
            if self.config.threading_mode == "parallel":
                transcriptions, sound_events = self._run_parallel(
                    speech_segs, nonspeech_segs, progress_cb
                )
            else:
                transcriptions, sound_events = self._run_sequential(
                    speech_segs, nonspeech_segs, progress_cb
                )

            # ── Stage 4: Merge and Write SRT ──
            self._report(progress_cb, "Merging timeline and generating subtitles...", 90)
            entries = self.merger.merge(transcriptions, sound_events)
            self.writer.write(entries, output_path)

            # Cache result
            self._save_cache(video_path)

            # Summary
            elapsed = time.monotonic() - start_time
            self._report(progress_cb, f"Done! ({elapsed:.1f}s)", 100)

            logger.info(f"{'='*60}")
            logger.info(f"Pipeline complete in {elapsed:.1f}s")
            logger.info(f"  Subtitles: {len(entries)} entries")
            logger.info(f"  Transcriptions: {len(transcriptions)}")
            logger.info(f"  Sound events: {len(sound_events)}")
            logger.info(f"  CPU throttles: {self.throttle.total_throttles}")
            logger.info(f"  Output: {output_path}")
            logger.info(f"{'='*60}")

            # Print preview
            preview = self.writer.write_preview(entries, max_entries=5)
            if preview:
                logger.info(f"Preview:\n{preview}")

            return entries

        finally:
            # Always clean up temp audio
            self.extractor.cleanup(audio_path)

    def _run_sequential(
        self,
        speech_segs: List[AudioSegment],
        nonspeech_segs: List[AudioSegment],
        progress_cb: ProgressCallback
    ) -> Tuple[List[TranscriptionResult], List[SoundEventResult]]:
        """Run ASR then SED sequentially (default, CPU-friendly)."""

        # ASR phase
        self._report(progress_cb, "Transcribing speech (sequential)...", 15)
        transcriptions = []
        total_speech = len(speech_segs)

        for i, seg in enumerate(speech_segs):
            result = self.asr.transcribe(seg)
            transcriptions.extend(result)
            self.throttle.throttle_if_needed()

            pct = 15 + int(65 * (i + 1) / max(total_speech, 1))
            self._report(
                progress_cb,
                f"Transcribing... {i + 1}/{total_speech} segments",
                pct
            )

        # SED phase
        self._report(progress_cb, "Detecting sound events...", 82)
        sound_events = []
        total_nonspeech = len(nonspeech_segs)

        for i, seg in enumerate(nonspeech_segs):
            result = self.sed.detect(seg)
            sound_events.extend(result)
            self.throttle.throttle_if_needed()

            pct = 82 + int(8 * (i + 1) / max(total_nonspeech, 1))
            self._report(
                progress_cb,
                f"Detecting sounds... {i + 1}/{total_nonspeech} segments",
                pct
            )

        return transcriptions, sound_events

    def _run_parallel(
        self,
        speech_segs: List[AudioSegment],
        nonspeech_segs: List[AudioSegment],
        progress_cb: ProgressCallback
    ) -> Tuple[List[TranscriptionResult], List[SoundEventResult]]:
        """Run ASR and SED in parallel (faster but higher CPU usage)."""

        self._report(progress_cb, "Analyzing audio (parallel mode)...", 15)

        with ThreadPoolExecutor(max_workers=2) as executor:
            asr_future = executor.submit(
                self._batch_asr, speech_segs
            )
            sed_future = executor.submit(
                self._batch_sed, nonspeech_segs
            )

            transcriptions = asr_future.result()
            sound_events = sed_future.result()

        return transcriptions, sound_events

    def _batch_asr(self, segments: List[AudioSegment]) -> List[TranscriptionResult]:
        """Process all speech segments through ASR."""
        results = []
        for seg in segments:
            results.extend(self.asr.transcribe(seg))
            self.throttle.throttle_if_needed()
        return results

    def _batch_sed(self, segments: List[AudioSegment]) -> List[SoundEventResult]:
        """Process all non-speech segments through SED."""
        results = []
        for seg in segments:
            results.extend(self.sed.detect(seg))
        return results

    # ── Caching ──

    def _file_hash(self, filepath: Path) -> str:
        """Compute SHA-256 hash of a file for cache keying."""
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                sha.update(chunk)
        return sha.hexdigest()[:16]

    def _cache_path(self, video_path: Path) -> Path:
        """Get the cache file path for a video."""
        file_hash = self._file_hash(video_path)
        cache_dir = Path(self.config.cache.directory)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{video_path.stem}_{file_hash}.cache.json"

    def _check_cache(self, video_path: Path, output_path: Path) -> Optional[List[SubtitleEntry]]:
        """Check if a cached SRT exists for this video file."""
        if not self.config.cache.enabled:
            return None

        cache_file = self._cache_path(video_path)
        if not cache_file.exists():
            return None

        # Check if output SRT already exists and matches
        if output_path.exists():
            logger.info(f"Cache hit: {cache_file.name}")
            return []  # Empty list signals "skip processing"

        return None

    def _save_cache(self, video_path: Path):
        """Save a cache marker for this video file."""
        if not self.config.cache.enabled:
            return

        cache_file = self._cache_path(video_path)
        cache_data = {
            "video": str(video_path),
            "hash": self._file_hash(video_path),
            "model": self.config.asr.model,
            "timestamp": time.time()
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)

        logger.debug(f"Cache saved: {cache_file}")

    # ── Utilities ──

    @staticmethod
    def _report(cb: ProgressCallback, msg: str, pct: int):
        """Report progress to logger and optional callback."""
        logger.info(f"[{pct:3d}%] {msg}")
        if cb:
            cb(msg, pct)

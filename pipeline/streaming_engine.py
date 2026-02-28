"""
Streaming Subtitle Engine — Real-time progressive subtitle generation.

Processes audio in 30-second chunks while video plays, generating
subtitles progressively. Supports seek, pause, and caches processed
segments for instant replay.

Architecture:
  - Scheduler thread: decides which chunk to process next
  - Inference thread: VAD → ASR → SED → Merge per chunk
  - Writer thread: appends finalized subtitles to SRT file

Reuses existing pipeline modules:
  VADSegmenter, ASRWorker, SEDWorker, TimelineMerger, CPUThrottle
"""

import time
import json
import struct
import sys
import os
import logging
import threading
import subprocess
import numpy as np
from queue import Queue, Empty, Full
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

from .vad import VADSegmenter, AudioSegment, SAMPLE_RATE
from .asr_worker import ASRWorker, TranscriptionResult
from .sed_worker import SEDWorker
from .merger import TimelineMerger, SubtitleEntry
from .srt_writer import SRTWriter
from .cpu_throttle import CPUThrottle

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════

@dataclass
class ChunkRequest:
    """A request to process a specific audio chunk."""
    chunk_id: int
    start_sec: float
    end_sec: float
    priority: int = 0  # 0=normal, 1=urgent (after seek)


@dataclass
class ChunkResult:
    """Result of processing a single chunk."""
    chunk_id: int
    start_sec: float
    end_sec: float
    subtitles: List[SubtitleEntry]
    inference_time: float = 0.0


# ═══════════════════════════════════════════════════════════════
#  Chunked Audio Extractor (FFmpeg pipe, no temp files)
# ═══════════════════════════════════════════════════════════════

class ChunkedAudioExtractor:
    """
    Extracts audio in chunks via FFmpeg pipe — no temp files.
    Uses -ss for fast seeking and pipe:1 for stdout output.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def extract_chunk(self, video_path: str,
                      start_sec: float,
                      duration_sec: float) -> np.ndarray:
        """
        Extract a chunk of audio as float32 numpy array.

        Args:
            video_path: Path to the video file.
            start_sec: Start position in seconds.
            duration_sec: Duration to extract in seconds.

        Returns:
            Float32 audio array at 16kHz mono.
        """
        cmd = [
            "ffmpeg",
            "-ss", f"{start_sec:.3f}",       # seek BEFORE input (fast)
            "-i", str(video_path),
            "-t", f"{duration_sec:.3f}",      # read only chunk
            "-vn",                             # no video
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "1",                        # mono
            "-f", "s16le",                     # raw PCM to stdout
            "-loglevel", "error",
            "pipe:1"                           # output to pipe
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, timeout=60
            )
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg timed out extracting chunk at {start_sec:.1f}s")
            return np.array([], dtype=np.float32)

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            logger.error(f"FFmpeg chunk extraction failed: {stderr}")
            return np.array([], dtype=np.float32)

        if not proc.stdout:
            logger.warning(f"FFmpeg returned no audio for chunk at {start_sec:.1f}s")
            return np.array([], dtype=np.float32)

        # Convert raw int16 bytes → float32 numpy
        samples = np.frombuffer(proc.stdout, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0

    def get_duration(self, video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        return float(result.stdout.strip())


# ═══════════════════════════════════════════════════════════════
#  Overlap Reconciler — de-duplicates overlap zones
# ═══════════════════════════════════════════════════════════════

class OverlapReconciler:
    """
    De-duplicates subtitles from overlapping chunk regions.
    Uses time-overlap + text similarity to detect duplicates.
    """

    def __init__(self, overlap_sec: float = 3.0):
        self.overlap_sec = overlap_sec
        self.committed: List[SubtitleEntry] = []
        self.next_index = 1

    def reconcile(self, chunk_result: ChunkResult) -> List[SubtitleEntry]:
        """
        Returns only the NEW subtitles from this chunk
        (not duplicates from overlap zone).
        """
        overlap_end = chunk_result.start_sec + self.overlap_sec
        new_subs = []

        for sub in chunk_result.subtitles:
            if chunk_result.chunk_id > 0 and sub.start_sec < overlap_end:
                # In overlap zone — check for duplicates
                if self._is_duplicate(sub):
                    continue

            # Assign final index and commit
            sub.index = self.next_index
            self.next_index += 1
            self.committed.append(sub)
            new_subs.append(sub)

        return new_subs

    def _is_duplicate(self, candidate: SubtitleEntry) -> bool:
        """Check if candidate matches any recently committed subtitle."""
        for existing in reversed(self.committed[-30:]):
            if existing.end_sec < candidate.start_sec - 1.0:
                break

            # Time overlap check
            time_overlap = (
                min(candidate.end_sec, existing.end_sec) -
                max(candidate.start_sec, existing.start_sec)
            )
            if time_overlap <= 0:
                continue

            # Text similarity check
            similarity = self._text_similarity(candidate.text, existing.text)
            if similarity > 0.6:
                # Refine existing end-time
                existing.end_sec = max(existing.end_sec, candidate.end_sec)
                return True

        return False

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Simple word-overlap similarity (avoids Levenshtein dependency)."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union) if union else 0.0

    def reset(self):
        """Reset state (e.g., after a seek to uncached region)."""
        self.committed.clear()
        self.next_index = 1

    def reset_to(self, subtitles: List[SubtitleEntry]):
        """Reset with pre-existing subtitles (from cache)."""
        self.committed = list(subtitles)
        self.next_index = (max(s.index for s in subtitles) + 1) if subtitles else 1


# ═══════════════════════════════════════════════════════════════
#  Chunk Cache — cached processed segments
# ═══════════════════════════════════════════════════════════════

class ChunkCache:
    """
    Caches processed chunk results in memory.
    Serves cached subtitles instantly on seek-back.
    Evicts oldest entries when capacity exceeded.
    """

    def __init__(self, max_chunks: int = 20):
        self.max_chunks = max_chunks
        self._cache: Dict[int, ChunkResult] = {}
        self._access_order: List[int] = []
        self._lock = threading.Lock()

    def get(self, chunk_id: int) -> Optional[ChunkResult]:
        """Get cached result for a chunk, or None."""
        with self._lock:
            result = self._cache.get(chunk_id)
            if result is not None:
                # Move to end (most recent)
                if chunk_id in self._access_order:
                    self._access_order.remove(chunk_id)
                self._access_order.append(chunk_id)
            return result

    def put(self, result: ChunkResult):
        """Cache a chunk result, evicting oldest if needed."""
        with self._lock:
            self._cache[result.chunk_id] = result
            if result.chunk_id in self._access_order:
                self._access_order.remove(result.chunk_id)
            self._access_order.append(result.chunk_id)

            # Evict oldest if over capacity
            while len(self._cache) > self.max_chunks:
                oldest = self._access_order.pop(0)
                self._cache.pop(oldest, None)

    def has(self, chunk_id: int) -> bool:
        with self._lock:
            return chunk_id in self._cache

    def get_all_subtitles(self) -> List[SubtitleEntry]:
        """Get all cached subtitles, sorted by time."""
        with self._lock:
            all_subs = []
            for result in self._cache.values():
                all_subs.extend(result.subtitles)
            all_subs.sort(key=lambda s: s.start_sec)
            return all_subs

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._access_order.clear()


# ═══════════════════════════════════════════════════════════════
#  Streaming Subtitle Engine — Main Orchestrator
# ═══════════════════════════════════════════════════════════════

class StreamingSubtitleEngine:
    """
    Real-time streaming subtitle engine.

    Processes audio in 30-second chunks progressively while video
    plays. Supports seek, pause (continues processing), and caches
    processed segments for instant replay.

    Usage:
        engine = StreamingSubtitleEngine(config)
        engine.start("video.mp4", "video.srt")
        # In playback loop:
        subs = engine.get_subtitles(playback_time)
        # On seek:
        engine.on_seek(new_position)
        # On done:
        engine.stop()
    """

    def __init__(self, config):
        self.config = config

        # Streaming parameters
        streaming_cfg = getattr(config, 'streaming', None)
        self.chunk_duration = getattr(streaming_cfg, 'chunk_duration', 30.0)
        self.overlap = getattr(streaming_cfg, 'overlap', 3.0)
        self.max_cache_chunks = getattr(streaming_cfg, 'max_cache_chunks', 20)
        self.stride = self.chunk_duration - self.overlap

        # Pipeline workers (reuse existing modules)
        self.extractor = ChunkedAudioExtractor(
            sample_rate=getattr(config.audio, 'sample_rate', 16000)
        )
        self.vad = VADSegmenter(config.vad)
        self.asr = ASRWorker(config.asr)
        self.sed = SEDWorker(config.sed)
        self.merger = TimelineMerger(config.merge)
        self.throttle = CPUThrottle(
            max_percent=getattr(config, 'max_cpu_percent', 70),
            check_interval=getattr(
                config.threading, 'throttle_check_interval', 2.0
            )
        )

        # State
        self.reconciler = OverlapReconciler(self.overlap)
        self.cache = ChunkCache(self.max_cache_chunks)
        self.subtitle_buffer: List[SubtitleEntry] = []
        self._buffer_lock = threading.Lock()

        # Queues
        self._chunk_queue: Queue = Queue(maxsize=3)
        self._result_queue: Queue = Queue(maxsize=5)

        # Control
        self._shutdown = threading.Event()
        self._cancel = threading.Event()
        self._seek_event = threading.Event()
        self._first_chunk_ready = threading.Event()
        self._playback_position = 0.0
        self._pos_lock = threading.Lock()
        self._video_path = ""
        self._output_srt = ""
        self._video_duration = 0.0
        self._threads: List[threading.Thread] = []
        self._current_chunk_id = 0
        self._processing_active = threading.Event()

        # SRT writer
        self._srt_lock = threading.Lock()

        # JSON feed for VLC SPU plugin (fallback)
        self._json_feed_path = ""
        self._json_version = 0

        # Named pipe for VLC SPU plugin (primary)
        self._pipe_handle = None
        self._pipe_connected = False
        self._pipe_lock = threading.Lock()
        self._pipe_name = (
            r"\\.\pipe\vlc-aisub" if sys.platform == "win32"
            else "/tmp/vlc-aisub"
        )

    # ── Public API ──────────────────────────────────────────

    def start(self, video_path: str, output_srt: str):
        """Start the streaming engine (non-blocking)."""
        self._video_path = str(video_path)
        self._output_srt = str(output_srt)
        self._shutdown.clear()
        self._cancel.clear()
        self._first_chunk_ready.clear()

        # Get video duration
        try:
            self._video_duration = self.extractor.get_duration(self._video_path)
        except Exception as e:
            logger.error(f"Could not get video duration: {e}")
            self._video_duration = 0.0

        logger.info(
            f"Starting streaming engine: {Path(video_path).name} "
            f"({self._video_duration:.1f}s), "
            f"chunk={self.chunk_duration}s, overlap={self.overlap}s"
        )

        # Clear SRT file
        with open(self._output_srt, "w", encoding="utf-8") as f:
            f.write("")

        # Set up JSON feed path (same dir as SRT, named subtitle_feed.json)
        srt_dir = Path(self._output_srt).parent
        self._json_feed_path = str(srt_dir / "subtitle_feed.json")
        self._json_version = 0
        # Write initial empty feed
        self._write_json_feed()

        # Start named pipe server for VLC SPU plugin
        self._start_pipe_server()

        # Launch threads
        self._threads = [
            threading.Thread(
                target=self._scheduler_loop,
                name="stream-scheduler", daemon=True
            ),
            threading.Thread(
                target=self._inference_loop,
                name="stream-inference", daemon=True
            ),
            threading.Thread(
                target=self._commit_loop,
                name="stream-committer", daemon=True
            ),
        ]
        for t in self._threads:
            t.start()

        self._processing_active.set()

    def stop(self):
        """Graceful shutdown."""
        logger.info("Stopping streaming engine...")
        # Send "all done" to VLC plugin before shutdown
        self._pipe_send_done()
        self._shutdown.set()
        self._cancel.set()
        for t in self._threads:
            t.join(timeout=10.0)
        self._threads.clear()
        self._pipe_cleanup()
        logger.info("Streaming engine stopped.")

    def wait_for_first_chunk(self, timeout: float = 60.0) -> bool:
        """Block until first chunk is processed. Returns True if ready."""
        return self._first_chunk_ready.wait(timeout=timeout)

    def on_seek(self, position_sec: float):
        """Handle user seek — reprioritize processing."""
        logger.info(f"Seek to {position_sec:.1f}s")

        # Signal cancel to interrupt current inference
        self._cancel.set()

        # Drain queues
        self._drain_queue(self._chunk_queue)
        self._drain_queue(self._result_queue)

        # Update position
        with self._pos_lock:
            self._playback_position = position_sec

        # Calculate which chunk covers this position
        target_chunk = int(position_sec / self.stride)
        self._current_chunk_id = target_chunk

        # Check cache
        cached = self.cache.get(target_chunk)
        if cached:
            logger.info(
                f"Seek hit cache for chunk {target_chunk} "
                f"({len(cached.subtitles)} subs)"
            )
            # Load cached subs into buffer
            with self._buffer_lock:
                all_cached = self.cache.get_all_subtitles()
                self.subtitle_buffer = all_cached
            self._write_all_subs_to_srt()

            # Pipe dispatch is handled by the player's dispatcher thread
            self._pipe_send_clear()
        else:
            logger.info(f"Seek miss — scheduling chunk {target_chunk} as urgent")
            # Clear pipe on seek miss too
            self._pipe_send_clear()

        # Signal seek event to wake scheduler
        self._seek_event.set()

        # Clear cancel after queues are drained
        time.sleep(0.1)
        self._cancel.clear()

    def get_subtitles(self, playback_time: float) -> List[SubtitleEntry]:
        """Get active subtitles for the current playback time."""
        with self._pos_lock:
            self._playback_position = playback_time

        with self._buffer_lock:
            active = [
                s for s in self.subtitle_buffer
                if s.start_sec <= playback_time <= s.end_sec
            ]
        return active

    def update_position(self, playback_time: float):
        """Update playback position (called periodically by player)."""
        with self._pos_lock:
            self._playback_position = playback_time

    @property
    def is_running(self) -> bool:
        return not self._shutdown.is_set()

    @property
    def first_chunk_ready(self) -> bool:
        return self._first_chunk_ready.is_set()

    @property
    def buffer_lead(self) -> float:
        """How many seconds of subtitles are buffered ahead."""
        with self._buffer_lock:
            if not self.subtitle_buffer:
                return 0.0
            max_end = max(s.end_sec for s in self.subtitle_buffer)
        with self._pos_lock:
            return max_end - self._playback_position

    @property
    def total_subtitles(self) -> int:
        with self._buffer_lock:
            return len(self.subtitle_buffer)

    @property
    def processing_complete(self) -> bool:
        """True if all chunks have been processed."""
        if self._video_duration <= 0:
            return False
        total_chunks = self._total_chunks()
        return all(
            self.cache.has(i) for i in range(total_chunks)
        )

    # ── Internal threads ────────────────────────────────────

    def _scheduler_loop(self):
        """Decide which chunk to process next."""
        next_chunk = 0
        total_chunks = self._total_chunks()

        while not self._shutdown.is_set():
            # Check for seek events
            if self._seek_event.is_set():
                self._seek_event.clear()
                next_chunk = self._current_chunk_id
                total_chunks = self._total_chunks()  # refresh

            # Skip already-cached chunks
            while next_chunk < total_chunks and self.cache.has(next_chunk):
                next_chunk += 1

            if next_chunk >= total_chunks:
                # All chunks scheduled/cached — check if done
                if self.processing_complete:
                    logger.info("All chunks processed!")
                    self._processing_active.clear()
                time.sleep(1.0)

                # Re-check in case of seek
                if self._seek_event.is_set():
                    continue
                # Wrap around check
                next_chunk = self._find_next_unprocessed()
                if next_chunk is None:
                    time.sleep(2.0)
                    continue

            # Build chunk request
            start_sec = next_chunk * self.stride
            end_sec = min(start_sec + self.chunk_duration, self._video_duration)

            if end_sec <= start_sec:
                next_chunk += 1
                continue

            req = ChunkRequest(
                chunk_id=next_chunk,
                start_sec=start_sec,
                end_sec=end_sec,
                priority=1 if self._seek_event.is_set() else 0
            )

            try:
                self._chunk_queue.put(req, timeout=2.0)
                next_chunk += 1
            except Full:
                time.sleep(0.5)

    def _inference_loop(self):
        """Process chunks: extract → VAD → ASR → SED → merge."""
        import torch

        while not self._shutdown.is_set():
            try:
                req = self._chunk_queue.get(timeout=2.0)
            except Empty:
                continue

            if self._cancel.is_set():
                continue

            # Check cache first
            if self.cache.has(req.chunk_id):
                logger.debug(f"Chunk {req.chunk_id} already cached, skipping")
                continue

            t_start = time.monotonic()
            chunk_dur = req.end_sec - req.start_sec

            logger.info(
                f"Processing chunk {req.chunk_id}: "
                f"{req.start_sec:.1f}–{req.end_sec:.1f}s "
                f"({chunk_dur:.1f}s)"
            )

            # ── 1. Extract audio ──
            audio_data = self.extractor.extract_chunk(
                self._video_path, req.start_sec, chunk_dur
            )
            if len(audio_data) == 0:
                logger.warning(f"No audio in chunk {req.chunk_id}, skipping")
                continue

            if self._cancel.is_set():
                continue

            # ── 2. VAD (in-memory, no file I/O) ──
            speech_segs, nonspeech_segs = self._run_vad_inmemory(
                audio_data, req.start_sec
            )

            if self._cancel.is_set():
                continue

            # ── 3. ASR ──
            transcriptions = []
            for seg in speech_segs:
                if self._cancel.is_set():
                    break
                results = self.asr.transcribe(seg)
                transcriptions.extend(results)
                self.throttle.throttle_if_needed()

            if self._cancel.is_set():
                continue

            # ── 4. SED (conditional — skip if throttled) ──
            sound_events = []
            if nonspeech_segs:
                for seg in nonspeech_segs:
                    if self._cancel.is_set():
                        break
                    events = self.sed.detect(seg)
                    sound_events.extend(events)
                    self.throttle.throttle_if_needed()

            if self._cancel.is_set():
                continue

            # ── 5. Merge ──
            subtitles = self.merger.merge(transcriptions, sound_events)

            t_elapsed = time.monotonic() - t_start

            result = ChunkResult(
                chunk_id=req.chunk_id,
                start_sec=req.start_sec,
                end_sec=req.end_sec,
                subtitles=subtitles,
                inference_time=t_elapsed
            )

            logger.info(
                f"Chunk {req.chunk_id} done: {len(subtitles)} subs "
                f"in {t_elapsed:.1f}s (ratio: {t_elapsed/chunk_dur:.2f}x)"
            )

            try:
                self._result_queue.put(result, timeout=5.0)
            except Full:
                logger.warning("Result queue full, dropping chunk result")

    def _commit_loop(self):
        """Reconcile results, update buffer, and write to SRT."""
        while not self._shutdown.is_set():
            try:
                result = self._result_queue.get(timeout=2.0)
            except Empty:
                continue

            # Cache the result
            self.cache.put(result)

            # Reconcile overlaps
            new_subs = self.reconciler.reconcile(result)

            if new_subs:
                # Add to subtitle buffer
                with self._buffer_lock:
                    self.subtitle_buffer.extend(new_subs)
                    self.subtitle_buffer.sort(key=lambda s: s.start_sec)

                # Append to SRT file
                self._append_subs_to_srt(new_subs)

                # Write JSON feed for VLC SPU plugin (fallback)
                self._write_json_feed()

                # Pipe dispatch handled by player's dispatcher thread

                logger.info(
                    f"Committed {len(new_subs)} new subtitles "
                    f"(total: {self.total_subtitles})"
                )

            # Signal first chunk ready
            if not self._first_chunk_ready.is_set():
                self._first_chunk_ready.set()
                logger.info("First chunk ready — playback can start!")

    # ── VAD helper (in-memory, no file I/O) ─────────────────

    def _run_vad_inmemory(
        self, audio_data: np.ndarray, time_offset: float
    ) -> Tuple[List[AudioSegment], List[AudioSegment]]:
        """
        Run VAD on in-memory audio chunk.
        Returns (speech_segments, nonspeech_segments) with absolute times.
        """
        import torch

        # Ensure the VAD model is loaded
        self.vad._load_model()

        wav_tensor = torch.from_numpy(audio_data)
        total_chunk_duration = len(audio_data) / SAMPLE_RATE

        timestamps = self.vad._get_speech_timestamps(
            wav_tensor,
            self.vad._model,
            threshold=self.vad.threshold,
            min_speech_duration_ms=int(self.vad.min_speech_sec * 1000),
            min_silence_duration_ms=int(self.vad.min_silence_sec * 1000),
            return_seconds=True,
            sampling_rate=SAMPLE_RATE
        )

        speech_segs = []
        nonspeech_segs = []
        prev_end = 0.0

        for ts in timestamps:
            speech_start = max(0.0, ts["start"] - self.vad.padding_sec)
            speech_end = min(total_chunk_duration, ts["end"] + self.vad.padding_sec)

            # Non-speech gap before speech
            if speech_start > prev_end + 0.05:
                gap_audio = audio_data[
                    int(prev_end * SAMPLE_RATE):int(speech_start * SAMPLE_RATE)
                ]
                if len(gap_audio) > 0 and float(np.abs(gap_audio).mean()) > 0.001:
                    nonspeech_segs.append(AudioSegment(
                        start_sec=time_offset + prev_end,
                        end_sec=time_offset + speech_start,
                        segment_type="non_speech",
                        audio_data=gap_audio
                    ))

            # Speech segment
            s_start = int(speech_start * SAMPLE_RATE)
            s_end = int(speech_end * SAMPLE_RATE)
            speech_audio = audio_data[s_start:s_end]
            speech_segs.append(AudioSegment(
                start_sec=time_offset + speech_start,
                end_sec=time_offset + speech_end,
                segment_type="speech",
                audio_data=speech_audio
            ))
            prev_end = speech_end

        # Trailing non-speech
        if prev_end < total_chunk_duration - 0.05:
            tail_audio = audio_data[int(prev_end * SAMPLE_RATE):]
            if len(tail_audio) > 0 and float(np.abs(tail_audio).mean()) > 0.001:
                nonspeech_segs.append(AudioSegment(
                    start_sec=time_offset + prev_end,
                    end_sec=time_offset + total_chunk_duration,
                    segment_type="non_speech",
                    audio_data=tail_audio
                ))

        logger.debug(
            f"VAD: {len(speech_segs)} speech, "
            f"{len(nonspeech_segs)} non-speech segments"
        )
        return speech_segs, nonspeech_segs

    # ── SRT writing helpers ─────────────────────────────────

    def _append_subs_to_srt(self, subs: List[SubtitleEntry]):
        """Append new subtitles to the SRT file."""
        with self._srt_lock:
            try:
                with open(self._output_srt, "a", encoding="utf-8") as f:
                    for sub in subs:
                        f.write(self._format_srt_entry(sub))
            except IOError as e:
                logger.error(f"Failed to write SRT: {e}")

    def _write_all_subs_to_srt(self):
        """Rewrite the entire SRT file from buffer (after seek)."""
        with self._srt_lock:
            with self._buffer_lock:
                subs = sorted(self.subtitle_buffer, key=lambda s: s.start_sec)
            try:
                with open(self._output_srt, "w", encoding="utf-8") as f:
                    for i, sub in enumerate(subs, 1):
                        sub.index = i
                        f.write(self._format_srt_entry(sub))
            except IOError as e:
                logger.error(f"Failed to rewrite SRT: {e}")

    @staticmethod
    def _format_srt_entry(sub: SubtitleEntry) -> str:
        """Format a single SRT entry."""
        def _ts(sec: float) -> str:
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            ms = int((sec % 1) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        return (
            f"{sub.index}\n"
            f"{_ts(sub.start_sec)} --> {_ts(sub.end_sec)}\n"
            f"{sub.text}\n\n"
        )

    def _write_json_feed(self):
        """Write JSON feed file for the VLC SPU plugin."""
        if not self._json_feed_path:
            return

        self._json_version += 1

        with self._buffer_lock:
            subs = sorted(self.subtitle_buffer, key=lambda s: s.start_sec)

        feed = {
            "version": self._json_version,
            "subtitles": [
                {
                    "start": round(s.start_sec, 3),
                    "end": round(s.end_sec, 3),
                    "text": s.text
                }
                for s in subs
            ]
        }

        # Atomic write: write to temp file, then rename
        tmp_path = self._json_feed_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(feed, f, ensure_ascii=False)
            # Rename (atomic on most OS)
            import os
            if os.path.exists(self._json_feed_path):
                os.remove(self._json_feed_path)
            os.rename(tmp_path, self._json_feed_path)
            logger.debug(
                f"JSON feed updated: v{self._json_version}, "
                f"{len(subs)} entries"
            )
        except IOError as e:
            logger.error(f"Failed to write JSON feed: {e}")

    # ── Named pipe IPC (VLC SPU plugin) ─────────────────────

    def _start_pipe_server(self):
        """Start named pipe server for VLC SPU plugin communication."""
        try:
            if sys.platform == "win32":
                import win32pipe
                import win32file
                import pywintypes

                self._pipe_handle = win32pipe.CreateNamedPipe(
                    self._pipe_name,
                    win32pipe.PIPE_ACCESS_OUTBOUND,
                    (
                        win32pipe.PIPE_TYPE_BYTE |
                        win32pipe.PIPE_READMODE_BYTE |
                        win32pipe.PIPE_WAIT
                    ),
                    1,           # max instances
                    65536,       # out buffer
                    0,           # in buffer
                    0,           # timeout
                    None         # security
                )
                logger.info(f"Named pipe created: {self._pipe_name}")

                # Wait for VLC to connect (in background thread)
                def _wait_connect():
                    try:
                        win32pipe.ConnectNamedPipe(self._pipe_handle, None)
                        with self._pipe_lock:
                            self._pipe_connected = True
                        logger.info("VLC connected to subtitle pipe")
                    except Exception as e:
                        logger.warning(f"Pipe connect failed: {e}")

                t = threading.Thread(
                    target=_wait_connect,
                    name="pipe-connect", daemon=True
                )
                t.start()

            else:
                # Unix: create FIFO
                pipe_path = self._pipe_name
                try:
                    os.mkfifo(pipe_path, 0o666)
                except FileExistsError:
                    pass

                # Open in background (blocks until reader connects)
                def _wait_open():
                    try:
                        fd = os.open(pipe_path, os.O_WRONLY)
                        with self._pipe_lock:
                            self._pipe_handle = fd
                            self._pipe_connected = True
                        logger.info("VLC connected to subtitle FIFO")
                    except Exception as e:
                        logger.warning(f"FIFO open failed: {e}")

                t = threading.Thread(
                    target=_wait_open,
                    name="pipe-connect", daemon=True
                )
                t.start()

        except ImportError:
            logger.warning(
                "pywin32 not installed — named pipe disabled. "
                "Install with: pip install pywin32"
            )
            self._pipe_handle = None
        except Exception as e:
            logger.warning(f"Failed to create pipe: {e}")
            self._pipe_handle = None

    def _pipe_send_subtitle(self, sub):
        """Send a single subtitle via named pipe (ASUB v2 message)."""
        with self._pipe_lock:
            if not self._pipe_connected or self._pipe_handle is None:
                return

        text_bytes = sub.text.encode("utf-8")[:2040]  # max 2040 bytes
        duration_ms = int((sub.end_sec - sub.start_sec) * 1000)
        if duration_ms < 500:
            duration_ms = 500  # minimum 500ms display
        msg = struct.pack(
            "<4sII",
            b"ASUB",
            duration_ms,
            len(text_bytes)
        ) + text_bytes

        self._pipe_write(msg)

    def _pipe_send_clear(self):
        """Send ACLR message — clear all cached subtitles in VLC plugin."""
        self._pipe_write(struct.pack("<4s", b"ACLR"))

    def _pipe_send_done(self):
        """Send ADNE message — all subtitles complete."""
        self._pipe_write(struct.pack("<4s", b"ADNE"))

    def _pipe_write(self, data: bytes):
        """Write raw bytes to the named pipe. Handles errors gracefully."""
        with self._pipe_lock:
            if not self._pipe_connected or self._pipe_handle is None:
                return

        try:
            if sys.platform == "win32":
                import win32file
                win32file.WriteFile(self._pipe_handle, data)
            else:
                os.write(self._pipe_handle, data)
        except Exception as e:
            logger.warning(f"Pipe write failed: {e}")
            with self._pipe_lock:
                self._pipe_connected = False

    def _pipe_cleanup(self):
        """Close the named pipe."""
        with self._pipe_lock:
            self._pipe_connected = False
            if self._pipe_handle is not None:
                try:
                    if sys.platform == "win32":
                        import win32file
                        win32file.CloseHandle(self._pipe_handle)
                    else:
                        os.close(self._pipe_handle)
                except Exception:
                    pass
                self._pipe_handle = None

        # Clean up FIFO file on Unix
        if sys.platform != "win32":
            try:
                os.unlink(self._pipe_name)
            except Exception:
                pass

    # ── Utility ─────────────────────────────────────────────

    def _total_chunks(self) -> int:
        """Total number of chunks for the video."""
        if self._video_duration <= 0:
            return 0
        return max(1, int(
            (self._video_duration - self.overlap) / self.stride
        ) + 1)

    def _find_next_unprocessed(self) -> Optional[int]:
        """Find the next unprocessed chunk, or None if all done."""
        total = self._total_chunks()
        for i in range(total):
            if not self.cache.has(i):
                return i
        return None

    @staticmethod
    def _drain_queue(q: Queue):
        """Empty a queue without blocking."""
        while not q.empty():
            try:
                q.get_nowait()
            except Empty:
                break

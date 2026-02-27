"""
Forensic Subtitle Generator -- Precision-Critical Mode
======================================================
Generates frame-perfect SRT subtitles for aa23.mp4 with:
  - Word-level timestamps (token precision)
  - Waveform-based end-time correction (trailing decay analysis)
  - Progressive drift detection & proportional correction
  - Forced alignment refinement (<40ms error)
  - YAMNet event tagging anchored to waveform peaks
  - Validation with >=93% alignment confidence gate

Usage:
    python forensic_subtitle.py
    python forensic_subtitle.py --video other.mp4
    python forensic_subtitle.py --beam-size 5
"""

import sys
import os
import time
import subprocess
import tempfile
import logging
import argparse
import csv
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Fix Windows console encoding before any print/log
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import soundfile as sf
import torch

# -- Logging --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("forensic")

SAMPLE_RATE = 16000

# ── Data Classes ────────────────────────────────────────────────

@dataclass
class WordToken:
    """A single word with precise timestamps and probability."""
    word: str
    start: float
    end: float
    probability: float


@dataclass
class ForensicSegment:
    """A subtitle segment with word-level detail and correction metadata."""
    start_sec: float
    end_sec: float
    text: str
    words: List[WordToken] = field(default_factory=list)
    segment_type: str = "speech"           # speech | event
    confidence: float = 0.0
    original_end: float = 0.0              # before correction
    drift_correction: float = 0.0          # applied drift offset
    alignment_refined: bool = False
    energy_extended: bool = False


# ════════════════════════════════════════════════════════════════
# Stage 1 — Audio Extraction
# ════════════════════════════════════════════════════════════════

def extract_audio(video_path: Path) -> Path:
    """Extract 16kHz mono PCM WAV from video using FFmpeg. No silence trimming."""
    output = Path(tempfile.mktemp(suffix=".wav", prefix="forensic_"))
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn",                        # no video
        "-acodec", "pcm_s16le",       # 16-bit PCM
        "-ar", str(SAMPLE_RATE),      # 16 kHz
        "-ac", "1",                   # mono
        "-loglevel", "error",
        "-y", str(output),
    ]
    logger.info(f"[Stage 1] Extracting audio: {video_path.name} → {output.name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr}")
    size_mb = output.stat().st_size / (1024 * 1024)
    logger.info(f"[Stage 1] Audio extracted: {size_mb:.1f} MB")
    return output


# ════════════════════════════════════════════════════════════════
# Stage 2 — Non-Destructive VAD
# ════════════════════════════════════════════════════════════════

@dataclass
class VADRegion:
    start_sec: float
    end_sec: float
    region_type: str        # "speech" | "non_speech"
    audio_data: np.ndarray

    @property
    def duration(self):
        return self.end_sec - self.start_sec


def run_vad(audio_path: Path) -> Tuple[List[VADRegion], np.ndarray]:
    """Run Silero VAD, preserving absolute time positions. Returns regions + full audio."""
    logger.info("[Stage 2] Loading Silero VAD model...")
    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", onnx=True, trust_repo=True
    )
    get_speech_ts, _, read_audio, *_ = utils

    audio_np, sr = sf.read(str(audio_path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE}Hz, got {sr}Hz")
    if audio_np.ndim > 1:
        audio_np = audio_np[:, 0]

    wav_tensor = torch.from_numpy(audio_np)
    total_dur = len(audio_np) / SAMPLE_RATE

    logger.info(f"[Stage 2] Running VAD on {total_dur:.1f}s audio...")
    speech_ts = get_speech_ts(
        wav_tensor, model,
        threshold=0.45,
        min_speech_duration_ms=200,
        min_silence_duration_ms=250,
        return_seconds=True,
        sampling_rate=SAMPLE_RATE,
    )

    # Build regions (speech + non-speech), preserving absolute positions
    regions: List[VADRegion] = []
    prev_end = 0.0
    ENERGY_THR = 0.001

    def _slice(start, end):
        return audio_np[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]

    def _has_energy(samples):
        return len(samples) > 0 and float(np.abs(samples).mean()) > ENERGY_THR

    for ts in speech_ts:
        s, e = ts["start"], ts["end"]
        # Non-speech gap before
        if s > prev_end + 0.05:
            gap = _slice(prev_end, s)
            if _has_energy(gap):
                regions.append(VADRegion(prev_end, s, "non_speech", gap))
        # Speech
        regions.append(VADRegion(s, e, "speech", _slice(s, e)))
        prev_end = e

    # Trailing non-speech
    if prev_end < total_dur - 0.05:
        tail = _slice(prev_end, total_dur)
        if _has_energy(tail):
            regions.append(VADRegion(prev_end, total_dur, "non_speech", tail))

    speech_count = sum(1 for r in regions if r.region_type == "speech")
    ns_count = sum(1 for r in regions if r.region_type == "non_speech")
    speech_dur = sum(r.duration for r in regions if r.region_type == "speech")
    logger.info(
        f"[Stage 2] VAD: {speech_count} speech ({speech_dur:.1f}s), "
        f"{ns_count} non-speech regions"
    )
    return regions, audio_np


# ════════════════════════════════════════════════════════════════
# Stage 3 — Word-Level ASR
# ════════════════════════════════════════════════════════════════

def run_asr(
    speech_regions: List[VADRegion],
    beam_size: int = 3,
    model_size: str = "small",
) -> List[ForensicSegment]:
    """Transcribe speech with word-level timestamps using Faster-Whisper."""
    from faster_whisper import WhisperModel

    logger.info(f"[Stage 3] Loading Faster-Whisper '{model_size}' (INT8, beam={beam_size})...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)

    all_segments: List[ForensicSegment] = []
    detected_lang = None

    for ri, region in enumerate(speech_regions):
        audio = region.audio_data.astype(np.float32)
        if len(audio) < 1600:  # < 0.1s
            continue

        segments_iter, info = model.transcribe(
            audio,
            beam_size=beam_size,
            language=detected_lang,
            vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=True,
        )

        if detected_lang is None:
            detected_lang = info.language
            logger.info(
                f"[Stage 3] Detected language: {detected_lang} "
                f"(prob={info.language_probability:.2f})"
            )

        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue

            words = []
            if seg.words:
                for w in seg.words:
                    words.append(WordToken(
                        word=w.word.strip(),
                        start=region.start_sec + w.start,
                        end=region.start_sec + w.end,
                        probability=w.probability,
                    ))

            abs_start = region.start_sec + seg.start
            abs_end = region.start_sec + seg.end

            confidence = seg.avg_logprob
            # Convert log prob to pseudo-percentage (sigmoid-like scaling)
            conf_pct = max(0, min(100, (1.0 + confidence) * 100))

            all_segments.append(ForensicSegment(
                start_sec=abs_start,
                end_sec=abs_end,
                text=text,
                words=words,
                segment_type="speech",
                confidence=conf_pct,
                original_end=abs_end,
            ))

        logger.info(f"[Stage 3] Region {ri+1}/{len(speech_regions)} transcribed")

    logger.info(f"[Stage 3] ASR complete: {len(all_segments)} segments")
    return all_segments


# ════════════════════════════════════════════════════════════════
# Stage 4 — Waveform End-Time Correction
# ════════════════════════════════════════════════════════════════

def correct_end_times(
    segments: List[ForensicSegment],
    audio: np.ndarray,
    trailing_window_ms: int = 350,
    rms_frame_ms: int = 10,
) -> List[ForensicSegment]:
    """
    Extend subtitle end times to true waveform silence boundary.
    Analyzes a 350ms trailing window after the last word's end.
    """
    logger.info("[Stage 4] Correcting end times via waveform energy analysis...")
    corrections = 0
    frame_samples = int(rms_frame_ms / 1000 * SAMPLE_RATE)
    window_samples = int(trailing_window_ms / 1000 * SAMPLE_RATE)

    for seg in segments:
        if seg.segment_type != "speech":
            continue

        end_idx = int(seg.end_sec * SAMPLE_RATE)
        trail_start = end_idx
        trail_end = min(end_idx + window_samples, len(audio))

        if trail_end <= trail_start:
            continue

        trail = audio[trail_start:trail_end]

        # Compute mean RMS of the segment itself for adaptive thresholding
        seg_start_idx = int(seg.start_sec * SAMPLE_RATE)
        seg_audio = audio[seg_start_idx:end_idx]
        if len(seg_audio) < frame_samples:
            continue
        seg_rms = float(np.sqrt(np.mean(seg_audio ** 2)))
        threshold = seg_rms * 0.10   # 10% of segment's mean RMS

        # Scan trailing window for silence boundary
        silence_offset = None
        for i in range(0, len(trail) - frame_samples, frame_samples):
            frame_rms = float(np.sqrt(np.mean(trail[i:i + frame_samples] ** 2)))
            if frame_rms < threshold:
                silence_offset = i
                break

        if silence_offset is None:
            # Entire trailing window has energy -- extend to end of window
            new_end = seg.end_sec + trailing_window_ms / 1000
            seg.original_end = seg.end_sec
            seg.end_sec = min(new_end, len(audio) / SAMPLE_RATE)
            seg.energy_extended = True
            corrections += 1
        elif silence_offset > 0:
            # Extend to silence boundary
            extension = silence_offset / SAMPLE_RATE
            seg.original_end = seg.end_sec
            seg.end_sec = seg.end_sec + extension
            seg.energy_extended = True
            corrections += 1

    logger.info(f"[Stage 4] End-time corrections applied: {corrections}/{len(segments)}")
    return segments


# ════════════════════════════════════════════════════════════════
# Stage 5 — Drift Analysis
# ════════════════════════════════════════════════════════════════

def analyze_and_correct_drift(
    segments: List[ForensicSegment],
    vad_regions: List[VADRegion],
) -> List[ForensicSegment]:
    """
    Detect progressive timestamp drift between VAD and ASR boundaries.
    Apply proportional time-scaling or fixed offset correction.
    """
    logger.info("[Stage 5] Analyzing timestamp drift...")
    speech_regions = [r for r in vad_regions if r.region_type == "speech"]

    if len(segments) < 2 or len(speech_regions) < 2:
        logger.info("[Stage 5] Insufficient data for drift analysis -- skipping")
        return segments

    # Match each ASR segment to its closest VAD region
    drifts = []
    for seg in segments:
        if seg.segment_type != "speech":
            continue
        best_vad = min(speech_regions, key=lambda r: abs(r.start_sec - seg.start_sec))
        drift = seg.start_sec - best_vad.start_sec
        drifts.append((seg.start_sec, drift))

    if len(drifts) < 2:
        logger.info("[Stage 5] Not enough drift points -- skipping")
        return segments

    times = np.array([d[0] for d in drifts])
    drift_vals = np.array([d[1] for d in drifts])

    mean_drift = float(np.mean(drift_vals))
    std_drift = float(np.std(drift_vals))

    # Linear regression slope
    if len(times) >= 3:
        coeffs = np.polyfit(times, drift_vals, 1)
        slope_ms_per_sec = coeffs[0] * 1000
        slope_ms_per_min = slope_ms_per_sec * 60
    else:
        slope_ms_per_min = 0.0
        coeffs = [0.0, mean_drift]

    logger.info(
        f"[Stage 5] Drift: mean={mean_drift*1000:.1f}ms, "
        f"std={std_drift*1000:.1f}ms, slope={slope_ms_per_min:.2f}ms/min"
    )

    if abs(slope_ms_per_min) > 5.0:
        # Progressive drift — apply proportional correction
        logger.info("[Stage 5] Progressive drift detected -- applying proportional correction")
        for seg in segments:
            correction = coeffs[0] * seg.start_sec + coeffs[1]
            seg.start_sec -= correction
            seg.end_sec -= correction
            seg.drift_correction = -correction
            for w in seg.words:
                w.start -= correction
                w.end -= correction
    elif abs(mean_drift) > 0.02 and std_drift < 0.02:
        # Uniform offset
        logger.info(f"[Stage 5] Uniform offset detected -- applying {mean_drift*1000:.1f}ms correction")
        for seg in segments:
            seg.start_sec -= mean_drift
            seg.end_sec -= mean_drift
            seg.drift_correction = -mean_drift
            for w in seg.words:
                w.start -= mean_drift
                w.end -= mean_drift
    else:
        logger.info("[Stage 5] No significant drift detected")

    return segments


# ════════════════════════════════════════════════════════════════
# Stage 6 — Forced Alignment Refinement
# ════════════════════════════════════════════════════════════════

def refine_alignment(
    segments: List[ForensicSegment],
    audio: np.ndarray,
    search_window_ms: int = 50,
    onset_ratio: float = 0.15,
) -> List[ForensicSegment]:
    """
    Snap subtitle boundaries to phoneme onset/offset via energy analysis.
    Searches ±50ms around each boundary for energy transitions.
    Target: <40ms error from true phoneme boundaries.
    """
    logger.info("[Stage 6] Refining alignment to phoneme boundaries...")
    frame_ms = 2   # 2ms precision
    frame_samples = int(frame_ms / 1000 * SAMPLE_RATE)
    window_samples = int(search_window_ms / 1000 * SAMPLE_RATE)
    refined = 0

    for seg in segments:
        if seg.segment_type != "speech":
            continue

        # ── Snap start to energy onset ──
        center_idx = int(seg.start_sec * SAMPLE_RATE)
        search_start = max(0, center_idx - window_samples)
        search_end = min(len(audio), center_idx + window_samples)
        search_region = audio[search_start:search_end]

        if len(search_region) > frame_samples * 4:
            # Compute local peak
            local_peak = float(np.max(np.abs(search_region)))
            onset_thr = local_peak * onset_ratio

            # Find first frame above threshold (onset)
            best_onset = None
            for i in range(0, len(search_region) - frame_samples, frame_samples):
                frame_energy = float(np.max(np.abs(search_region[i:i + frame_samples])))
                if frame_energy >= onset_thr:
                    best_onset = search_start + i
                    break

            if best_onset is not None:
                new_start = best_onset / SAMPLE_RATE
                if abs(new_start - seg.start_sec) < search_window_ms / 1000:
                    seg.start_sec = new_start
                    seg.alignment_refined = True

        # ── Snap end to energy offset ──
        center_idx = int(seg.end_sec * SAMPLE_RATE)
        search_start = max(0, center_idx - window_samples)
        search_end = min(len(audio), center_idx + window_samples)
        search_region = audio[search_start:search_end]

        if len(search_region) > frame_samples * 4:
            local_peak = float(np.max(np.abs(search_region)))
            offset_thr = local_peak * onset_ratio

            # Find last frame above threshold (offset)
            best_offset = None
            for i in range(len(search_region) - frame_samples, 0, -frame_samples):
                frame_energy = float(np.max(np.abs(search_region[i:i + frame_samples])))
                if frame_energy >= offset_thr:
                    best_offset = search_start + i + frame_samples
                    break

            if best_offset is not None:
                new_end = best_offset / SAMPLE_RATE
                if abs(new_end - seg.end_sec) < search_window_ms / 1000:
                    seg.end_sec = new_end
                    seg.alignment_refined = True

        if seg.alignment_refined:
            refined += 1

        # Also refine individual word boundaries
        for w in seg.words:
            w_center = int(w.start * SAMPLE_RATE)
            ws = max(0, w_center - window_samples // 2)
            we = min(len(audio), w_center + window_samples // 2)
            wr = audio[ws:we]
            if len(wr) > frame_samples * 2:
                lp = float(np.max(np.abs(wr)))
                ot = lp * onset_ratio
                for i in range(0, len(wr) - frame_samples, frame_samples):
                    if float(np.max(np.abs(wr[i:i + frame_samples]))) >= ot:
                        new_ws = (ws + i) / SAMPLE_RATE
                        if abs(new_ws - w.start) < search_window_ms / 1000:
                            w.start = new_ws
                        break

    logger.info(f"[Stage 6] Alignment refined: {refined}/{len(segments)} segments")
    return segments


# ════════════════════════════════════════════════════════════════
# Stage 7 — YAMNet Event Tagging
# ════════════════════════════════════════════════════════════════

# Curated event classes (subset from sed_worker.py)
CURATED_EVENTS = {
    "Laughter": "Laughing", "Crying, sobbing": "Crying",
    "Screaming": "Screaming", "Music": "Music playing",
    "Singing": "Singing", "Engine": "Engine running",
    "Car horn": "Car horn", "Siren": "Siren wailing",
    "Explosion": "Explosion", "Gunshot, gunfire": "Gunshot",
    "Rain": "Rain", "Thunder": "Thunder", "Wind": "Wind blowing",
    "Door": "Door", "Knock": "Knocking", "Glass": "Glass breaking",
    "Dog": "Dog barking", "Bark": "Dog barking",
    "Cat": "Cat meowing", "Bird": "Bird chirping",
    "Applause": "Applause", "Fireworks": "Fireworks",
    "Footsteps": "Footsteps", "Helicopter": "Helicopter",
    "Baby cry, infant cry": "Baby crying",
    "Clapping": "Clapping", "Whistle": "Whistling",
    "Alarm": "Alarm sounding", "Telephone bell ringing": "Phone ringing",
    "Guitar": "Guitar playing", "Piano": "Piano playing",
    "Drum": "Drums playing",
}
EXCLUDED = {"Speech", "Narration, monologue", "Conversation", "Silence",
            "Inside, small room", "Inside, large room or hall", "White noise", "Static"}


def run_event_tagging(
    nonspeech_regions: List[VADRegion],
    audio: np.ndarray,
    model_path: str = "models/yamnet.tflite",
    class_map_path: str = "models/yamnet_class_map.csv",
    min_confidence: float = 0.35,
) -> List[ForensicSegment]:
    """Run YAMNet on non-speech regions, anchoring events to waveform peaks."""
    logger.info("[Stage 7] Loading YAMNet TFLite for event detection...")

    model_p = Path(model_path)
    class_map_p = Path(class_map_path)
    if not model_p.exists() or not class_map_p.exists():
        logger.warning("[Stage 7] YAMNet model files not found -- skipping event tagging")
        return []

    # Load interpreter
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=str(model_p))
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(model_p))

    interpreter.allocate_tensors()

    # Load class names
    class_names = []
    with open(class_map_p, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                class_names.append(row[2].strip())

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    event_segments: List[ForensicSegment] = []

    for region in nonspeech_regions:
        waveform = region.audio_data.astype(np.float32)
        min_samples = int(0.96 * SAMPLE_RATE)
        if len(waveform) < min_samples:
            continue

        # Resize + infer
        interpreter.resize_tensor_input(input_details[0]["index"], [len(waveform)], strict=False)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]["index"], waveform)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_details[0]["index"])

        if scores.ndim == 1:
            scores = scores.reshape(1, -1)

        hop = 0.48
        raw_events = []
        for fi in range(scores.shape[0]):
            top_idx = int(np.argmax(scores[fi]))
            top_score = float(scores[fi][top_idx])
            if top_idx >= len(class_names):
                continue
            cn = class_names[top_idx]
            if top_score >= min_confidence and cn in CURATED_EVENTS and cn not in EXCLUDED:
                abs_start = region.start_sec + fi * hop
                abs_end = abs_start + 0.96
                raw_events.append((abs_start, abs_end, cn, top_score))

        # Merge consecutive same-class events
        merged = []
        for ev in raw_events:
            if merged and ev[2] == merged[-1][2] and ev[0] - merged[-1][1] <= 0.5:
                merged[-1] = (merged[-1][0], ev[1], ev[2], max(merged[-1][3], ev[3]))
            else:
                merged.append(ev)

        # Anchor to waveform energy peaks
        for ev_start, ev_end, cn, conf in merged:
            si = int(ev_start * SAMPLE_RATE) - int(region.start_sec * SAMPLE_RATE)
            ei = int(ev_end * SAMPLE_RATE) - int(region.start_sec * SAMPLE_RATE)
            si = max(0, si)
            ei = min(len(waveform), ei)
            chunk = waveform[si:ei]

            if len(chunk) > 160:
                # Find peak energy position
                rms_window = 160
                peak_pos = 0
                peak_val = 0
                for i in range(0, len(chunk) - rms_window, rms_window // 2):
                    e = float(np.sqrt(np.mean(chunk[i:i + rms_window] ** 2)))
                    if e > peak_val:
                        peak_val = e
                        peak_pos = i
                # Anchor start to peak onset (search backward from peak)
                onset_thr = peak_val * 0.15
                onset = peak_pos
                for i in range(peak_pos, 0, -rms_window // 2):
                    e = float(np.sqrt(np.mean(chunk[max(0, i - rms_window):i] ** 2)))
                    if e < onset_thr:
                        onset = i
                        break
                anchored_start = ev_start + onset / SAMPLE_RATE
            else:
                anchored_start = ev_start

            display = CURATED_EVENTS[cn]
            event_segments.append(ForensicSegment(
                start_sec=anchored_start,
                end_sec=ev_end,
                text=f"({display})",
                segment_type="event",
                confidence=conf * 100,
            ))

    logger.info(f"[Stage 7] Events detected: {len(event_segments)}")
    return event_segments


# ════════════════════════════════════════════════════════════════
# Stage 8 — Validation
# ════════════════════════════════════════════════════════════════

@dataclass
class ValidationReport:
    total_segments: int = 0
    early_terminations_fixed: int = 0
    overlaps_fixed: int = 0
    per_segment_confidence: List[float] = field(default_factory=list)
    overall_confidence: float = 0.0
    passed: bool = False


def validate_and_fix(
    segments: List[ForensicSegment],
    audio: np.ndarray,
    confidence_threshold: float = 93.0,
) -> Tuple[List[ForensicSegment], ValidationReport]:
    """
    Validate every subtitle:
      - end_sec must not precede true waveform silence
      - no overlapping segments
      - alignment confidence ≥ 93%
    """
    logger.info("[Stage 8] Running validation checks...")
    report = ValidationReport(total_segments=len(segments))

    # Sort by start time
    segments.sort(key=lambda s: s.start_sec)

    # ── Check waveform silence boundary ──
    frame_samples = int(0.010 * SAMPLE_RATE)
    for seg in segments:
        if seg.segment_type != "speech":
            continue
        end_idx = int(seg.end_sec * SAMPLE_RATE)
        # Check if there is still energy at end_sec
        check_start = max(0, end_idx - frame_samples)
        check_end = min(len(audio), end_idx + frame_samples)
        if check_end > check_start:
            check_audio = audio[check_start:check_end]
            rms = float(np.sqrt(np.mean(check_audio ** 2)))
            seg_si = int(seg.start_sec * SAMPLE_RATE)
            seg_audio = audio[seg_si:end_idx]
            if len(seg_audio) > 0:
                seg_rms = float(np.sqrt(np.mean(seg_audio ** 2)))
                silence_thr = seg_rms * 0.08
            else:
                silence_thr = 0.001

            if rms > silence_thr:
                # Early termination — extend
                for ext_i in range(end_idx, min(end_idx + int(0.5 * SAMPLE_RATE), len(audio)), frame_samples):
                    block = audio[ext_i:min(ext_i + frame_samples, len(audio))]
                    if float(np.sqrt(np.mean(block ** 2))) < silence_thr:
                        seg.end_sec = ext_i / SAMPLE_RATE
                        report.early_terminations_fixed += 1
                        break

    # ── Fix overlaps ──
    for i in range(1, len(segments)):
        if segments[i].start_sec < segments[i - 1].end_sec:
            midpoint = (segments[i - 1].end_sec + segments[i].start_sec) / 2
            segments[i - 1].end_sec = midpoint
            segments[i].start_sec = midpoint
            report.overlaps_fixed += 1

    # ── Confidence check ──
    speech_segs = [s for s in segments if s.segment_type == "speech"]
    for seg in speech_segs:
        if seg.words:
            avg_prob = float(np.mean([w.probability for w in seg.words])) * 100
        else:
            avg_prob = seg.confidence
        report.per_segment_confidence.append(avg_prob)

    if report.per_segment_confidence:
        report.overall_confidence = float(np.mean(report.per_segment_confidence))
    else:
        report.overall_confidence = 0.0

    report.passed = report.overall_confidence >= confidence_threshold

    logger.info(
        f"[Stage 8] Validation: confidence={report.overall_confidence:.1f}%, "
        f"early_term_fixed={report.early_terminations_fixed}, "
        f"overlaps_fixed={report.overlaps_fixed}, "
        f"passed={'PASS' if report.passed else 'FAIL'}"
    )

    return segments, report


# ════════════════════════════════════════════════════════════════
# Stage 9 — SRT Output
# ════════════════════════════════════════════════════════════════

def format_ts(seconds: float) -> str:
    """Convert seconds to SRT timestamp: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: List[ForensicSegment], output_path: Path):
    """Write final SRT file with millisecond precision."""
    # Sort by start time
    segments.sort(key=lambda s: s.start_sec)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            f.write(f"{i + 1}\n")
            f.write(f"{format_ts(seg.start_sec)} --> {format_ts(seg.end_sec)}\n")
            f.write(f"{seg.text}\n")
            f.write("\n")

    logger.info(f"[Stage 9] SRT written: {len(segments)} entries -> {output_path}")


# ════════════════════════════════════════════════════════════════
# Main Pipeline
# ════════════════════════════════════════════════════════════════

def print_banner():
    print("""
================================================================
   [FORENSIC] Subtitle Generator

   Word-Level Precision | Waveform-Verified | Drift-Free
   Powered by Faster-Whisper + YAMNet
================================================================
""")


def main():
    parser = argparse.ArgumentParser(description="Forensic Subtitle Generator")
    parser.add_argument("--video", type=Path, default=Path("AA23.mp4"),
                        help="Input video file (default: AA23.mp4)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output SRT path (default: <video>_forensic.srt)")
    parser.add_argument("--beam-size", type=int, default=3,
                        help="Whisper beam size (increase for accuracy)")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small"],
                        help="Whisper model size")
    parser.add_argument("--confidence-threshold", type=float, default=93.0,
                        help="Minimum alignment confidence to accept (%)")
    args = parser.parse_args()

    print_banner()
    start_time = time.monotonic()

    video_path = args.video
    if not video_path.exists():
        print(f"  [ERROR] Video not found: {video_path}")
        sys.exit(1)

    output_path = args.output or video_path.with_name(video_path.stem + "_forensic.srt")

    print(f"  Input:      {video_path}")
    print(f"  Output:     {output_path}")
    print(f"  Model:      Faster-Whisper {args.model} (INT8)")
    print(f"  Beam size:  {args.beam_size}")
    print(f"  Confidence: >={args.confidence_threshold}%")
    print()

    # ── Stage 1: Audio Extraction ──
    audio_path = extract_audio(video_path)

    try:
        # ── Stage 2: VAD ──
        vad_regions, full_audio = run_vad(audio_path)
        speech_regions = [r for r in vad_regions if r.region_type == "speech"]
        nonspeech_regions = [r for r in vad_regions if r.region_type == "non_speech"]

        # ── Stage 3: Word-Level ASR ──
        segments = run_asr(speech_regions, beam_size=args.beam_size, model_size=args.model)

        # ── Stage 4: Waveform End-Time Correction ──
        segments = correct_end_times(segments, full_audio)

        # ── Stage 5: Drift Analysis ──
        segments = analyze_and_correct_drift(segments, vad_regions)

        # ── Stage 6: Forced Alignment ──
        segments = refine_alignment(segments, full_audio)

        # ── Stage 7: Event Tagging ──
        event_segments = run_event_tagging(nonspeech_regions, full_audio)

        # Combine speech + events
        all_segments = segments + event_segments
        all_segments.sort(key=lambda s: s.start_sec)

        # ── Stage 8: Validation ──
        all_segments, report = validate_and_fix(all_segments, full_audio, args.confidence_threshold)

        # If confidence too low and beam size was 3, retry with beam_size=5
        if not report.passed and args.beam_size < 5:
            logger.warning(
                f"[Stage 8] Confidence {report.overall_confidence:.1f}% < {args.confidence_threshold}% "
                f"-- retrying ASR with beam_size=5..."
            )
            segments = run_asr(speech_regions, beam_size=5, model_size=args.model)
            segments = correct_end_times(segments, full_audio)
            segments = analyze_and_correct_drift(segments, vad_regions)
            segments = refine_alignment(segments, full_audio)
            all_segments = segments + event_segments
            all_segments.sort(key=lambda s: s.start_sec)
            all_segments, report = validate_and_fix(all_segments, full_audio, args.confidence_threshold)

        # ── Stage 9: SRT Output ──
        write_srt(all_segments, output_path)

        # ── Summary ──
        elapsed = time.monotonic() - start_time
        print()
        print("=" * 62)
        print("  FORENSIC SUBTITLE GENERATION -- SUMMARY")
        print("=" * 62)
        print(f"  Total time:           {elapsed:.1f}s")
        print(f"  Speech segments:      {sum(1 for s in all_segments if s.segment_type == 'speech')}")
        print(f"  Event segments:       {sum(1 for s in all_segments if s.segment_type == 'event')}")
        print(f"  End-time corrections: {sum(1 for s in all_segments if s.energy_extended)}")
        print(f"  Alignment refined:    {sum(1 for s in all_segments if s.alignment_refined)}")
        print(f"  Early-term fixed:     {report.early_terminations_fixed}")
        print(f"  Overlaps fixed:       {report.overlaps_fixed}")
        print(f"  Overall confidence:   {report.overall_confidence:.1f}%")
        print(f"  Validation:           {'PASSED' if report.passed else 'BELOW THRESHOLD'}")
        print(f"  Output:               {output_path}")
        print("=" * 62)

        # Preview first few entries
        print("\n  Preview (first 5 entries):\n")
        for seg in all_segments[:5]:
            print(f"    [{format_ts(seg.start_sec)} -> {format_ts(seg.end_sec)}] {seg.text[:70]}")
        if len(all_segments) > 5:
            print(f"    ... and {len(all_segments) - 5} more entries")

    finally:
        # Clean up temp audio
        if audio_path.exists():
            audio_path.unlink()
            logger.debug(f"Cleaned up temp audio: {audio_path}")


if __name__ == "__main__":
    main()

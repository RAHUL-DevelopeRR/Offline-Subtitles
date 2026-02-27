"""
Sound Event Detection Worker — YAMNet TFLite-based classifier.

Classifies non-speech audio regions into sound event categories
(e.g., door slam, engine start, crying) using a quantized YAMNet model.
Only reports events from a curated subset of 40 useful classes.
"""

import csv
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


# ============================================================
# Curated sound event classes (from YAMNet's 521 AudioSet classes)
# Maps YAMNet class display_name → human-readable caption text
# ============================================================
CURATED_EVENTS: Dict[str, str] = {
    # Human sounds
    "Laughter":                 "Laughing",
    "Baby laughter":            "Baby laughing",
    "Crying, sobbing":          "Crying",
    "Baby cry, infant cry":     "Baby crying",
    "Screaming":                "Screaming",
    "Whispering":               "Whispering",
    "Clapping":                 "Clapping",
    "Cough":                    "Coughing",
    "Sneeze":                   "Sneezing",
    "Snoring":                  "Snoring",
    "Gasp":                     "Gasping",
    "Yell":                     "Yelling",
    "Groan":                    "Groaning",
    "Sigh":                     "Sighing",
    "Breathing":                "Heavy breathing",

    # Music & instruments
    "Music":                    "Music playing",
    "Singing":                  "Singing",
    "Guitar":                   "Guitar playing",
    "Piano":                    "Piano playing",
    "Drum":                     "Drums playing",
    "Whistle":                  "Whistling",

    # Vehicles & transport
    "Engine":                   "Engine running",
    "Engine starting":          "Engine starting",
    "Car":                      "Car",
    "Car horn":                 "Car horn honking",
    "Motorcycle":               "Motorcycle",
    "Siren":                    "Siren wailing",
    "Helicopter":               "Helicopter overhead",
    "Train":                    "Train",

    # Environment & nature
    "Rain":                     "Rain",
    "Raindrop":                 "Raindrops",
    "Thunder":                  "Thunder",
    "Thunderstorm":             "Thunderstorm",
    "Wind":                     "Wind blowing",
    "Stream":                   "Water flowing",
    "Fire":                     "Fire crackling",

    # Domestic & objects
    "Door":                     "Door",
    "Knock":                    "Knocking",
    "Glass":                    "Glass breaking",
    "Shatter":                  "Shattering",
    "Telephone bell ringing":   "Phone ringing",
    "Alarm":                    "Alarm sounding",
    "Clock alarm":              "Alarm ringing",
    "Doorbell":                 "Doorbell",
    "Typing":                   "Typing",
    "Keys jangling":            "Keys jangling",

    # Animals
    "Dog":                      "Dog barking",
    "Bark":                     "Dog barking",
    "Cat":                      "Cat meowing",
    "Meow":                     "Cat meowing",
    "Purr":                     "Cat purring",
    "Bird":                     "Bird chirping",
    "Bird vocalization, bird call, bird song": "Bird singing",
    "Rooster":                  "Rooster crowing",
    "Crow":                     "Crow cawing",

    # Actions & impacts
    "Gunshot, gunfire":         "Gunshot",
    "Explosion":                "Explosion",
    "Fireworks":                "Fireworks",
    "Footsteps":                "Footsteps",
    "Splash, splashing":        "Splashing",
    "Applause":                 "Applause",
    "Chewing, mastication":     "Chewing",
    "Crushing":                 "Crushing",
    "Slap, smack":              "Slap",
}

# Classes to explicitly EXCLUDE even if they pass confidence threshold
EXCLUDED_CLASSES = {
    "Speech", "Narration, monologue", "Conversation",
    "Silence", "Inside, small room", "Inside, large room or hall",
    "White noise", "Static",
}


@dataclass
class SoundEventResult:
    """A detected sound event with absolute timestamps."""
    start_sec: float
    end_sec: float
    event_label: str        # Original YAMNet class name
    confidence: float
    display_text: str       # e.g., "(Door slams)"

    def __repr__(self):
        return (f"SoundEvent({self.start_sec:.2f}–{self.end_sec:.2f}s, "
                f"{self.display_text}, conf={self.confidence:.2f})")


class SEDWorker:
    """
    Sound Event Detection using YAMNet (TensorFlow Lite, quantized).

    Features:
      - Processes 0.96s frames with 50% overlap
      - Filters to curated event classes only
      - Merges consecutive same-class detections
      - Lazy model loading
    """

    def __init__(self, config):
        self.model_path = getattr(config, "model_path", "models/yamnet.tflite")
        self.class_map_path = getattr(config, "class_map_path", "models/yamnet_class_map.csv")
        self.min_confidence = getattr(config, "min_confidence", 0.4)
        self.frame_duration = getattr(config, "frame_duration", 0.96)
        self.hop_duration = getattr(config, "hop_duration", 0.48)
        self.max_gap_merge = getattr(config, "max_gap_merge", 0.5)

        # Lazy-loaded
        self._interpreter = None
        self._class_names = None
        self._input_details = None
        self._output_details = None

    def _load_model(self):
        """Load the YAMNet TFLite model and class map on first use."""
        if self._interpreter is not None:
            return

        model_path = Path(self.model_path)
        class_map_path = Path(self.class_map_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"YAMNet model not found at {model_path}.\n"
                f"Download it from: "
                f"https://tfhub.dev/google/lite-model/yamnet/tflite/1\n"
                f"Place the .tflite file in the models/ directory."
            )

        logger.info(f"Loading YAMNet TFLite model from {model_path}...")

        try:
            import tflite_runtime.interpreter as tflite
            self._interpreter = tflite.Interpreter(model_path=str(model_path))
        except ImportError:
            # Fallback to full TensorFlow if tflite_runtime not available
            import tensorflow as tf
            self._interpreter = tf.lite.Interpreter(model_path=str(model_path))

        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Load class names
        self._class_names = self._load_class_names(class_map_path)

        logger.info(
            f"YAMNet loaded: {len(self._class_names)} classes, "
            f"{len(CURATED_EVENTS)} curated events"
        )

    def _load_class_names(self, class_map_path: Path) -> List[str]:
        """Load YAMNet class display names from CSV."""
        if not class_map_path.exists():
            raise FileNotFoundError(
                f"YAMNet class map not found at {class_map_path}.\n"
                f"Download yamnet_class_map.csv from: "
                f"https://github.com/tensorflow/models/tree/master/research/audioset/yamnet"
            )

        class_names = []
        with open(class_map_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 3:
                    class_names.append(row[2].strip())  # display_name column
        return class_names

    def detect(self, segment) -> List[SoundEventResult]:
        """
        Detect sound events in a non-speech AudioSegment.

        YAMNet TFLite accepts a variable-length waveform and internally
        frames it into 0.96s windows with 0.48s hops, returning scores
        for each frame. We pass the entire segment and process each
        frame's scores.

        Args:
            segment: AudioSegment with segment_type == "non_speech".

        Returns:
            List of SoundEventResult with absolute timestamps, merged.
        """
        self._load_model()

        audio = segment.audio_data.astype(np.float32)
        sample_rate = 16000

        # Need at least 0.96s of audio for one frame
        min_samples = int(self.frame_duration * sample_rate)
        if len(audio) < min_samples:
            return []

        # Run inference on the whole segment at once
        all_scores = self._infer(audio)

        # all_scores shape: (num_frames, 521) — one row per 0.96s frame
        if all_scores.ndim == 1:
            all_scores = all_scores.reshape(1, -1)

        raw_events = []

        for frame_idx in range(all_scores.shape[0]):
            scores = all_scores[frame_idx]

            # Get top prediction
            top_idx = int(np.argmax(scores))
            top_score = float(scores[top_idx])

            if top_idx >= len(self._class_names):
                continue

            class_name = self._class_names[top_idx]

            # Filter: confidence threshold + curated list + exclusions
            if (top_score >= self.min_confidence
                    and class_name in CURATED_EVENTS
                    and class_name not in EXCLUDED_CLASSES):

                # Each frame covers 0.96s with a 0.48s hop
                abs_start = segment.start_sec + (frame_idx * self.hop_duration)
                abs_end = abs_start + self.frame_duration
                display = CURATED_EVENTS[class_name]

                raw_events.append(SoundEventResult(
                    start_sec=abs_start,
                    end_sec=abs_end,
                    event_label=class_name,
                    confidence=top_score,
                    display_text=f"({display})"
                ))

        # Merge consecutive same-class events
        merged = self._merge_events(raw_events)

        if merged:
            logger.debug(
                f"SED: {len(raw_events)} raw → {len(merged)} merged events "
                f"in {segment.start_sec:.1f}–{segment.end_sec:.1f}s"
            )

        return merged

    def _infer(self, waveform: np.ndarray) -> np.ndarray:
        """
        Run YAMNet inference on a waveform.

        The YAMNet TFLite model accepts a variable-length 1D waveform
        and returns scores for multiple internal frames.

        Args:
            waveform: 1D float32 audio waveform at 16kHz.

        Returns:
            2D array of shape (num_frames, 521) with class scores.
        """
        # Resize input tensor for the variable-length waveform
        self._interpreter.resize_tensor_input(
            self._input_details[0]["index"],
            [len(waveform)],
            strict=False
        )
        self._interpreter.allocate_tensors()

        # Refresh details after resize
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Set input
        self._interpreter.set_tensor(
            self._input_details[0]["index"],
            waveform.astype(np.float32)
        )

        self._interpreter.invoke()

        # Get scores output (first output tensor: shape [num_frames, 521])
        output = self._interpreter.get_tensor(self._output_details[0]["index"])

        return output

    def _merge_events(self, events: List[SoundEventResult]) -> List[SoundEventResult]:
        """
        Merge consecutive events of the same class within max_gap_merge seconds.
        Takes the maximum confidence from merged events.
        """
        if not events:
            return []

        merged = [SoundEventResult(
            start_sec=events[0].start_sec,
            end_sec=events[0].end_sec,
            event_label=events[0].event_label,
            confidence=events[0].confidence,
            display_text=events[0].display_text
        )]

        for e in events[1:]:
            prev = merged[-1]
            if (e.event_label == prev.event_label
                    and e.start_sec - prev.end_sec <= self.max_gap_merge):
                # Extend previous event
                merged[-1] = SoundEventResult(
                    start_sec=prev.start_sec,
                    end_sec=e.end_sec,
                    event_label=prev.event_label,
                    confidence=max(prev.confidence, e.confidence),
                    display_text=prev.display_text
                )
            else:
                merged.append(SoundEventResult(
                    start_sec=e.start_sec,
                    end_sec=e.end_sec,
                    event_label=e.event_label,
                    confidence=e.confidence,
                    display_text=e.display_text
                ))

        return merged

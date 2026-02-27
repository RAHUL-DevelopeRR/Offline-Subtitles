"""
Timeline Merger — Combines ASR transcriptions and SED events into a unified timeline.

Resolves overlaps between speech and sound events, enforces min/max
subtitle durations, and produces ordered SubtitleEntry objects.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SubtitleEntry:
    """A single subtitle entry ready for SRT output."""
    index: int
    start_sec: float
    end_sec: float
    text: str

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    def __repr__(self):
        return (f"Sub#{self.index}({self.start_sec:.2f}–{self.end_sec:.2f}s, "
                f"'{self.text[:50]}')")


class TimelineMerger:
    """
    Merges ASR and SED outputs into a unified subtitle timeline.

    Overlap resolution rules:
    1. Speech always has priority over sound events
    2. Sound events during speech are appended as a new line
    3. Sound events in gaps between speech are standalone entries
    4. Adjacent same-type events within merge_gap are combined
    5. Minimum subtitle duration enforced for readability
    6. Maximum subtitle duration triggers splitting
    """

    def __init__(self, config):
        self.min_duration = getattr(config, "min_duration", 0.8)
        self.max_duration = getattr(config, "max_duration", 7.0)
        self.merge_gap = getattr(config, "merge_gap", 0.5)

    def merge(self, transcriptions, sound_events) -> List[SubtitleEntry]:
        """
        Merge transcription results and sound events into subtitle entries.

        Args:
            transcriptions: List of TranscriptionResult from ASR.
            sound_events: List of SoundEventResult from SED.

        Returns:
            Ordered list of SubtitleEntry objects.
        """
        # Build unified timeline items: (type, start, end, text)
        all_items: List[Tuple[str, float, float, str]] = []

        for t in transcriptions:
            all_items.append(("speech", t.start_sec, t.end_sec, t.text))

        for s in sound_events:
            all_items.append(("sound", s.start_sec, s.end_sec, s.display_text))

        # Sort by start time, speech first on ties
        all_items.sort(key=lambda x: (x[1], 0 if x[0] == "speech" else 1))

        if not all_items:
            logger.warning("No transcriptions or sound events to merge.")
            return []

        # Build subtitle entries with overlap resolution
        entries = self._resolve_overlaps(all_items)

        logger.info(
            f"Merged {len(transcriptions)} transcriptions + "
            f"{len(sound_events)} sound events → {len(entries)} subtitles"
        )

        return entries

    def _resolve_overlaps(self, items: List[Tuple[str, float, float, str]]) -> List[SubtitleEntry]:
        """Resolve overlaps and build final subtitle entries."""
        entries: List[SubtitleEntry] = []
        idx = 1

        for item_type, start, end, text in items:
            # Enforce minimum duration
            if end - start < self.min_duration:
                end = start + self.min_duration

            # Handle long entries by splitting
            if end - start > self.max_duration:
                splits = self._split_long_entry(start, end, text)
                for s_start, s_end, s_text in splits:
                    entries.append(SubtitleEntry(idx, s_start, s_end, s_text))
                    idx += 1
                continue

            # Check overlap with previous entry
            if item_type == "sound" and entries:
                prev = entries[-1]

                # Sound overlaps with previous speech → append as new line
                if (start < prev.end_sec and
                        start >= prev.start_sec and
                        not prev.text.startswith("(")):
                    prev.text += f"\n{text}"
                    continue

                # Sound is very close after speech → also append
                if (start - prev.end_sec < 0.3 and
                        not prev.text.startswith("(")):
                    prev.text += f"\n{text}"
                    prev.end_sec = max(prev.end_sec, end)
                    continue

            # Check if we should merge with previous entry of same type
            if entries:
                prev = entries[-1]
                same_type = (
                    (item_type == "sound" and prev.text.startswith("(")) or
                    (item_type == "speech" and not prev.text.startswith("("))
                )
                if (same_type and
                        start - prev.end_sec < self.merge_gap and
                        prev.duration + (end - start) < self.max_duration):
                    # Merge with previous
                    if item_type == "speech":
                        prev.text += " " + text
                    else:
                        prev.text += "\n" + text
                    prev.end_sec = end
                    continue

            entries.append(SubtitleEntry(idx, start, end, text))
            idx += 1

        return entries

    def _split_long_entry(
        self, start: float, end: float, text: str
    ) -> List[Tuple[float, float, str]]:
        """
        Split a long subtitle entry into segments under max_duration.

        For speech: splits at word boundaries.
        For sound events: just truncates to max_duration.
        """
        if text.startswith("("):
            # Sound event — simply cap duration
            return [(start, start + self.max_duration, text)]

        # Speech — split at word boundaries
        words = text.split()
        if len(words) <= 1:
            return [(start, min(end, start + self.max_duration), text)]

        total_duration = end - start
        splits = []
        num_chunks = max(1, int(total_duration / self.max_duration) + 1)
        words_per_chunk = max(1, len(words) // num_chunks)

        word_offset = 0
        time_per_word = total_duration / len(words)

        while word_offset < len(words):
            chunk_end = min(word_offset + words_per_chunk, len(words))

            # Try to end at a natural break (comma, period)
            for j in range(min(chunk_end + 3, len(words)) - 1, chunk_end - 1, -1):
                if words[j].endswith((",", ".", "!", "?", ";")):
                    chunk_end = j + 1
                    break

            chunk_text = " ".join(words[word_offset:chunk_end])
            chunk_start = start + word_offset * time_per_word
            chunk_end_time = start + chunk_end * time_per_word

            # Enforce min/max duration
            if chunk_end_time - chunk_start < self.min_duration:
                chunk_end_time = chunk_start + self.min_duration

            splits.append((chunk_start, chunk_end_time, chunk_text))
            word_offset = chunk_end

        return splits

"""
SRT Writer — Standard SubRip subtitle file generator.

Converts SubtitleEntry objects into properly formatted .srt files
with sequential indices, HH:MM:SS,mmm timestamps, and UTF-8 encoding.
"""

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class SRTWriter:
    """
    Writes subtitle entries to a standard SRT (SubRip) file.

    SRT format:
        1
        00:00:01,200 --> 00:00:04,800
        Hello everyone, welcome to the show.

        2
        00:00:05,100 --> 00:00:06,300
        (Audience clapping)
    """

    def write(self, entries: List, output_path: Path):
        """
        Write subtitle entries to an SRT file.

        Args:
            entries: List of SubtitleEntry objects (sorted by time).
            output_path: Path for the output .srt file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for i, entry in enumerate(entries):
                # Re-index sequentially (in case of gaps from merging)
                f.write(f"{i + 1}\n")
                f.write(
                    f"{self._format_timestamp(entry.start_sec)} --> "
                    f"{self._format_timestamp(entry.end_sec)}\n"
                )
                f.write(f"{entry.text}\n")
                f.write("\n")  # Blank line separator

        logger.info(
            f"SRT written: {len(entries)} subtitles → {output_path}"
        )

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """
        Convert seconds to SRT timestamp format: HH:MM:SS,mmm

        Args:
            seconds: Time in seconds (e.g., 125.340)

        Returns:
            Formatted timestamp string (e.g., "00:02:05,340")
        """
        if seconds < 0:
            seconds = 0.0

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds % 1) * 1000))

        # Clamp milliseconds (rounding could push to 1000)
        if millis >= 1000:
            millis = 999

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def write_preview(self, entries: List, max_entries: int = 10) -> str:
        """
        Generate a text preview of the subtitle entries.

        Args:
            entries: List of SubtitleEntry objects.
            max_entries: Maximum entries to include in preview.

        Returns:
            Formatted string preview.
        """
        lines = []
        shown = min(len(entries), max_entries)

        for i, entry in enumerate(entries[:shown]):
            ts_start = self._format_timestamp(entry.start_sec)
            ts_end = self._format_timestamp(entry.end_sec)
            text_preview = entry.text[:80]
            if len(entry.text) > 80:
                text_preview += "..."
            lines.append(f"  [{ts_start} → {ts_end}] {text_preview}")

        if len(entries) > shown:
            lines.append(f"  ... and {len(entries) - shown} more entries")

        return "\n".join(lines)

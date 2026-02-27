"""
Audio Extractor — FFmpeg-based audio extraction from video files.

Extracts the audio track from any video format and converts it to
16kHz mono 16-bit PCM WAV suitable for Whisper and YAMNet processing.
"""

import subprocess
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioExtractor:
    """Extracts and downsamples audio from video files using FFmpeg."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._verify_ffmpeg()

    def _verify_ffmpeg(self):
        """Check that FFmpeg is available on the system PATH."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("FFmpeg returned non-zero exit code")
            version_line = result.stdout.split("\n")[0]
            logger.debug(f"FFmpeg found: {version_line}")
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg and add it to PATH.\n"
                "Download: https://ffmpeg.org/download.html"
            )

    def extract(self, video_path: Path) -> Path:
        """
        Extract audio from a video file.

        Args:
            video_path: Path to the input video file.

        Returns:
            Path to the extracted temporary WAV file.

        Raises:
            RuntimeError: If FFmpeg extraction fails.
            FileNotFoundError: If the video file doesn't exist.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create temp file in system temp directory
        output = Path(tempfile.mktemp(suffix=".wav", prefix="offsub_"))

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",                          # No video
            "-acodec", "pcm_s16le",         # 16-bit PCM
            "-ar", str(self.sample_rate),   # Sample rate
            "-ac", str(self.channels),      # Mono
            "-loglevel", "error",           # Suppress verbose output
            "-y",                           # Overwrite
            str(output)
        ]

        logger.info(f"Extracting audio: {video_path.name} → {output.name}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg audio extraction failed:\n{result.stderr}"
            )

        file_size_mb = output.stat().st_size / (1024 * 1024)
        logger.info(f"Audio extracted: {file_size_mb:.1f} MB ({output})")

        return output

    def get_duration(self, video_path: Path) -> float:
        """
        Get the duration of a video/audio file in seconds using ffprobe.

        Args:
            video_path: Path to the media file.

        Returns:
            Duration in seconds.
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        return float(result.stdout.strip())

    @staticmethod
    def cleanup(audio_path: Path):
        """Remove the temporary audio file."""
        audio_path = Path(audio_path)
        if audio_path.exists():
            audio_path.unlink()
            logger.debug(f"Cleaned up temp audio: {audio_path}")

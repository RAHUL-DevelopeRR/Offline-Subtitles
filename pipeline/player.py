"""
Video Player with Subtitles — Playback Utility

Launches the video file with its generated .srt subtitles using
the best available player on the system.

Priority:
  1. VLC (with --sub-file flag for direct subtitle loading)
  2. mpv (with --sub-file flag)
  3. System default player (opens file association, user loads subs manually)
"""

import subprocess
import shutil
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Player detection ────────────────────────────────────────────


def _find_vlc() -> str | None:
    """Find VLC executable path."""
    # Check PATH first
    vlc = shutil.which("vlc")
    if vlc:
        return vlc

    # Common Windows install locations
    if sys.platform == "win32":
        common_paths = [
            Path(r"C:\Program Files\VideoLAN\VLC\vlc.exe"),
            Path(r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe"),
        ]
        for p in common_paths:
            if p.exists():
                return str(p)

    # Common Linux locations
    elif sys.platform.startswith("linux"):
        for p in ["/usr/bin/vlc", "/snap/bin/vlc"]:
            if Path(p).exists():
                return p

    # macOS
    elif sys.platform == "darwin":
        mac_vlc = Path("/Applications/VLC.app/Contents/MacOS/VLC")
        if mac_vlc.exists():
            return str(mac_vlc)

    return None


def _find_mpv() -> str | None:
    """Find mpv executable path."""
    return shutil.which("mpv")


# ── Playback functions ──────────────────────────────────────────


def play_with_vlc(video_path: Path, srt_path: Path, vlc_exe: str) -> None:
    """Launch VLC with subtitle file loaded."""
    cmd = [
        vlc_exe,
        str(video_path),
        f"--sub-file={srt_path}",
        "--no-sub-autodetect-file",  # Use only our subtitle
    ]
    logger.info(f"Launching VLC: {' '.join(cmd)}")
    subprocess.Popen(cmd)


def play_with_mpv(video_path: Path, srt_path: Path, mpv_exe: str) -> None:
    """Launch mpv with subtitle file loaded."""
    cmd = [
        mpv_exe,
        str(video_path),
        f"--sub-file={srt_path}",
    ]
    logger.info(f"Launching mpv: {' '.join(cmd)}")
    subprocess.Popen(cmd)


def play_with_system_default(video_path: Path) -> None:
    """Open video with the OS default application."""
    import os
    logger.info(f"Opening with system default: {video_path}")
    if sys.platform == "win32":
        os.startfile(str(video_path))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(video_path)])
    else:
        subprocess.Popen(["xdg-open", str(video_path)])


# ── Main entry ──────────────────────────────────────────────────


def play_video(video_path: Path, srt_path: Path) -> str:
    """
    Play the video with subtitles using the best available player.

    Returns the name of the player used.
    """
    video_path = Path(video_path).resolve()
    srt_path = Path(srt_path).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not srt_path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {srt_path}")

    # Try VLC first (best subtitle support)
    vlc = _find_vlc()
    if vlc:
        play_with_vlc(video_path, srt_path, vlc)
        return "VLC"

    # Try mpv
    mpv = _find_mpv()
    if mpv:
        play_with_mpv(video_path, srt_path, mpv)
        return "mpv"

    # Fallback to system default (subtitles may not auto-load)
    play_with_system_default(video_path)
    return "system default (place .srt next to video for auto-load)"

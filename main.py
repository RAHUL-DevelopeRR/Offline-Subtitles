"""
Offline Subtitle Generator — CLI Entry Point

Usage:
    python main.py video.mp4
    python main.py video.mp4 -o subtitles.srt
    python main.py video.mp4 --model base --parallel --max-cpu 80
    python main.py video.mp4 --language en
"""

import sys
import argparse
import logging
from pathlib import Path

from config import load_config
from pipeline.orchestrator import SubtitlePipeline
from pipeline.player import play_video


def setup_logging(level: str = "INFO", log_file: str = None):
    """Configure logging for the application."""
    log_format = (
        "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s"
    )
    date_format = "%H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.INFO)


def print_banner():
    """Print the application banner."""
    banner = """
==========================================================
          Offline Subtitle Generator

  Speech Recognition  +  Sound Event Detection
  Powered by Faster-Whisper & YAMNet
  100% Offline  |  CPU Optimized
==========================================================
"""
    print(banner)


def print_progress(message: str, percent: int):
    """Console progress callback with progress bar."""
    bar_width = 30
    filled = int(bar_width * percent / 100)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(f"\r  [{bar}] {percent:3d}%  {message:<50}", end="", flush=True)
    if percent >= 100:
        print()  # Newline at completion


def main():
    parser = argparse.ArgumentParser(
        description="Offline Subtitle Generator — Generate SRT subtitles with "
                    "speech transcription and sound event captions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py movie.mp4                    # Basic usage
  python main.py movie.mp4 -o my_subs.srt     # Custom output path
  python main.py movie.mp4 --model base        # Faster, less accurate
  python main.py movie.mp4 --parallel          # Use parallel processing
  python main.py movie.mp4 --language hi       # Force Hindi language
  python main.py movie.mp4 --max-cpu 80        # Allow higher CPU usage
  python main.py movie.mp4 --play              # Generate subs + play video
  python main.py movie.mp4 --play-only         # Play with existing subs
        """
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Path to the input video file (.mp4, .mkv, .avi, .webm, etc.)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output SRT file path (default: same name as video with .srt extension)"
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        choices=["tiny", "base", "small"],
        help="Whisper model size (default: from config.yaml, usually 'small')"
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Force language code (e.g., 'en', 'hi', 'es'). Default: auto-detect"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run ASR and SED in parallel (faster but higher CPU usage)"
    )
    parser.add_argument(
        "--max-cpu",
        type=int,
        default=None,
        help="Maximum CPU usage percent for throttling (default: 70)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable result caching (re-process even if cached)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to custom config.yaml file"
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play the video with subtitles after generating them"
    )
    parser.add_argument(
        "--play-only",
        action="store_true",
        help="Skip subtitle generation and just play the video with existing .srt"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except the progress bar"
    )

    args = parser.parse_args()

    # ── Validate input ──
    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # ── Determine output path ──
    output_path = args.output or args.video.with_suffix(".srt")

    # ── Load config ──
    config = load_config(args.config)

    # Apply CLI overrides
    config.update_from_args(args)

    if args.no_cache:
        config.cache.enabled = False

    # ── Setup logging ──
    log_level = "DEBUG" if args.verbose else ("WARNING" if args.quiet else config.logging.level)
    setup_logging(level=log_level, log_file=config.logging.file)

    # ── Banner ──
    if not args.quiet:
        print_banner()
        print(f"  Input:    {args.video}")
        print(f"  Output:   {output_path}")
        print(f"  Model:    Faster-Whisper {config.asr.model} ({config.asr.compute_type})")
        print(f"  Mode:     {'Parallel' if config.threading_mode == 'parallel' else 'Sequential'}")
        print(f"  CPU Limit: {config.max_cpu_percent}%")
        if config.asr.language:
            print(f"  Language: {config.asr.language}")
        else:
            print(f"  Language: Auto-detect")
        print()

    # ── Play-only mode (skip generation) ──
    if args.play_only:
        if not output_path.exists():
            print(f"\n  [ERROR] Subtitle file not found: {output_path}")
            print(f"  [TIP] Run without --play-only first to generate subtitles.")
            sys.exit(1)
        if not args.quiet:
            print(f"  [>] Playing video with existing subtitles...")
        try:
            player_used = play_video(args.video, output_path)
            if not args.quiet:
                print(f"  [PLAY] Launched with: {player_used}")
        except Exception as e:
            print(f"\n  [ERROR] Playback error: {e}")
            sys.exit(1)
        return

    # ── Run pipeline ──
    try:
        pipeline = SubtitlePipeline(config)
        progress_fn = print_progress if not args.quiet else None
        entries = pipeline.process(args.video, output_path, progress_cb=progress_fn)

        if not args.quiet:
            print(f"\n  [OK] Subtitles saved to: {output_path}")
            print(f"  [INFO] Total entries: {len(entries)}")

        # ── Auto-play after generation ──
        if args.play:
            if not args.quiet:
                print(f"\n  [>] Launching video player with subtitles...")
            try:
                player_used = play_video(args.video, output_path)
                if not args.quiet:
                    print(f"  [PLAY] Launched with: {player_used}")
            except Exception as e:
                print(f"\n  [WARN] Could not launch player: {e}")

    except KeyboardInterrupt:
        print("\n\n  [WARN] Processing interrupted by user.")
        sys.exit(130)
    except FileNotFoundError as e:
        print(f"\n  [ERROR] File error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n  [ERROR] Runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.exception("Unexpected error")
        print(f"\n  [ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

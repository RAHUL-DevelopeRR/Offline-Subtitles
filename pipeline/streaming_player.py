"""
Streaming Player — Plays video with real-time AI-generated subtitles.

Uses the custom VLC SPU plugin (aisub) for seamless subtitle injection,
or falls back to --sub-file for standard VLC installs.

With the SPU plugin:
  - VLC reads subtitle_feed.json (polled every 500ms by plugin)
  - Python engine updates the JSON file as new chunks are processed
  - Subtitles appear seamlessly — no VLC restart needed

Without the SPU plugin (fallback):
  - Uses --sub-file with periodic VLC restarts at current position
"""

import time
import sys
import os
import logging
import subprocess
import shutil
import threading
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str, int], None]]


class StreamingPlayer:
    """
    Plays video with real-time streaming subtitles.

    Prefers the custom VLC SPU plugin (aisub) for seamless subtitle
    injection. Falls back to --sub-file + restart if plugin is not
    available.
    """

    def __init__(self, engine, vlc_plugin_dir: str = None):
        """
        Args:
            engine: StreamingSubtitleEngine instance.
            vlc_plugin_dir: Path to directory containing libaisub_plugin.
                           If None, auto-detected from the project tree.
        """
        self.engine = engine
        self._player_process = None
        self._player_exe = None
        self._player_name = None
        self._launch_time = 0.0
        self._start_offset = 0.0
        self._last_reload_sub_count = 0
        self._srt_path = None
        self._video_path = None
        self._feed_path = None
        self._vlc_plugin_dir = vlc_plugin_dir
        self._use_spu_plugin = False
        self._final_reload_done = False
        self._dispatcher_stop = False
        self._dispatcher_thread = None
        self._last_dispatched_text = None
        self._pipe_name = (
            r"\\.\pipe\vlc-aisub" if sys.platform == "win32"
            else "/tmp/vlc-aisub"
        )

    def play(self, video_path: str, output_srt: str,
             progress_cb: ProgressCallback = None):
        """
        Start streaming subtitles and play the video.
        Blocks until VLC is closed or Ctrl+C.
        """
        self._video_path = str(Path(video_path).resolve())
        self._srt_path = str(Path(output_srt).resolve())
        self._feed_path = str(
            Path(output_srt).resolve().parent / "subtitle_feed.json"
        )

        self._report(progress_cb, "Starting streaming engine...", 5)

        # Start the engine (background threads)
        self.engine.start(self._video_path, self._srt_path)

        # Wait for first chunk
        self._report(progress_cb, "Processing first 30 seconds...", 10)
        ready = self.engine.wait_for_first_chunk(timeout=120.0)

        if not ready:
            logger.error("First chunk timed out")
            self._report(progress_cb, "Timed out generating subtitles", 0)
            self.engine.stop()
            return

        sub_count = self.engine.total_subtitles
        self._report(
            progress_cb,
            f"First chunk ready ({sub_count} subs) — launching player...",
            30
        )

        # Find player
        self._player_exe, self._player_name = self._find_player()
        if not self._player_exe:
            logger.error("No video player found (VLC or mpv)")
            self._report(progress_cb, "Error: Install VLC or mpv", 0)
            self.engine.stop()
            return

        # Check for SPU plugin availability
        self._use_spu_plugin = self._check_spu_plugin()
        mode = "SPU plugin" if self._use_spu_plugin else "SRT fallback"

        # Write SRT and launch
        self.engine._write_all_subs_to_srt()
        self._launch(start_time=0.0)
        self._last_reload_sub_count = sub_count

        # Start subtitle dispatcher for SPU plugin
        if self._use_spu_plugin:
            self._dispatcher_stop = False
            self._dispatcher_thread = threading.Thread(
                target=self._subtitle_dispatcher,
                daemon=True
            )
            self._dispatcher_thread.start()

        self._report(
            progress_cb,
            f"Playing with {self._player_name} ({mode})!",
            35
        )

        # Poll loop
        try:
            self._poll_loop(progress_cb)
        except KeyboardInterrupt:
            logger.info("User interrupted")
        finally:
            self._cleanup(progress_cb)

    def _poll_loop(self, progress_cb: ProgressCallback = None):
        """Main loop — monitor player and report progress."""
        last_reload_time = time.monotonic()
        reload_interval = 10.0  # min seconds between SRT reloads

        while True:
            # Check if player closed by user
            if self._player_process and self._player_process.poll() is not None:
                logger.info("Player closed by user")
                break

            current_count = self.engine.total_subtitles
            new_since_reload = current_count - self._last_reload_sub_count
            is_complete = self.engine.processing_complete

            # Report progress
            if new_since_reload > 0:
                pct = self._calc_progress()
                if self._use_spu_plugin:
                    # SPU plugin picks up changes automatically
                    self._report(
                        progress_cb,
                        f"Streaming: {current_count} subs | "
                        f"Lead: {self.engine.buffer_lead:.0f}s",
                        pct
                    )
                    self._last_reload_sub_count = current_count
                else:
                    # SRT fallback — reload VLC periodically
                    elapsed = time.monotonic() - last_reload_time
                    if new_since_reload >= 3 and elapsed > reload_interval:
                        # Write updated SRT and restart VLC
                        self.engine._write_all_subs_to_srt()
                        current_pos = self._estimate_playback_position()
                        self._report(
                            progress_cb,
                            f"Updating: {current_count} subs — "
                            f"refreshing player...",
                            pct
                        )
                        self._stop_player()
                        self._launch(start_time=current_pos)
                        self._last_reload_sub_count = current_count
                        last_reload_time = time.monotonic()
                        self._report(
                            progress_cb,
                            f"Resumed at "
                            f"{self._format_time(current_pos)} | "
                            f"{current_count} subs",
                            pct
                        )
                    else:
                        self._report(
                            progress_cb,
                            f"Buffering: {current_count} subs | "
                            f"Lead: {self.engine.buffer_lead:.0f}s",
                            pct
                        )

            # Handle completion
            if is_complete and not self._final_reload_done:
                self._final_reload_done = True
                self.engine._write_all_subs_to_srt()

                if self._use_spu_plugin:
                    self._report(
                        progress_cb,
                        f"Complete! {current_count} subtitles generated.",
                        96
                    )
                else:
                    # Final reload with complete SRT
                    current_pos = self._estimate_playback_position()
                    self._report(
                        progress_cb,
                        f"All {current_count} subs ready — "
                        f"refreshing player...",
                        95
                    )
                    self._stop_player()
                    self._launch(start_time=current_pos)
                    self._last_reload_sub_count = current_count
                    self._report(
                        progress_cb,
                        f"Complete! Resumed at "
                        f"{self._format_time(current_pos)}",
                        96
                    )

                # Wait for player to close
                while (self._player_process and
                       self._player_process.poll() is None):
                    time.sleep(2.0)
                break

            time.sleep(2.0)

    def _launch(self, start_time: float = 0.0):
        """Launch the video player."""
        self._start_offset = start_time
        self._launch_time = time.monotonic()

        if self._player_name == "VLC":
            if self._use_spu_plugin:
                # Use SPU plugin — subtitles injected via named pipe
                cmd = [
                    self._player_exe,
                    self._video_path,
                    f"--sub-source=aisub{{aisub-pipe={self._pipe_name}}}",
                    "--no-sub-autodetect-file",
                ]
                # Add plugin path if custom build
                plugin_dir = self._get_plugin_dir()
                if plugin_dir:
                    cmd.insert(1, f"--plugin-path={plugin_dir}")
            else:
                # Fallback — standard --sub-file
                cmd = [
                    self._player_exe,
                    self._video_path,
                    f"--sub-file={self._srt_path}",
                    "--no-sub-autodetect-file",
                ]

            if start_time > 1.0:
                cmd.append(f"--start-time={start_time:.1f}")

        elif self._player_name == "mpv":
            cmd = [
                self._player_exe,
                self._video_path,
                f"--sub-file={self._srt_path}",
            ]
            if start_time > 1.0:
                cmd.append(f"--start={start_time:.1f}")
        else:
            cmd = [self._player_exe, self._video_path]

        logger.info(f"Launching {self._player_name}: {' '.join(cmd)}")
        self._player_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    def _stop_player(self):
        """Gracefully stop the current player process."""
        if self._player_process and self._player_process.poll() is None:
            try:
                self._player_process.terminate()
                self._player_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._player_process.kill()
            self._player_process = None

    def _estimate_playback_position(self) -> float:
        """Estimate current playback position from elapsed wall time."""
        if self._launch_time <= 0:
            return 0.0
        elapsed = time.monotonic() - self._launch_time
        return self._start_offset + elapsed

    def _check_spu_plugin(self) -> bool:
        """Check if the aisub SPU plugin is installed in VLC."""
        if self._player_name != "VLC":
            return False

        # Check if installed globally in VLC's plugin directory
        vlc_dir = Path(self._player_exe).parent
        global_plugin = vlc_dir / "plugins" / "spu" / "libaisub_plugin.dll"
        if global_plugin.exists():
            logger.info(f"aisub plugin installed globally: {global_plugin}")
            return True

        # 2. Check build directory — try to install
        plugin_dir = self._get_plugin_dir()
        if plugin_dir:
            for ext in [".dll", ".so", ".dylib"]:
                build_path = Path(plugin_dir) / f"libaisub_plugin{ext}"
                if build_path.exists():
                    logger.info(
                        f"aisub plugin found in build dir: {build_path}. "
                        f"Not installed globally — using SRT fallback. "
                        f"To install, run as Admin: "
                        f"Copy-Item \"{build_path}\" "
                        f"\"{global_plugin.parent}\\\" -Force"
                    )
                    # Don't use SPU mode if not installed globally
                    break

        logger.info("aisub plugin not installed — using SRT fallback")
        return False

    def _get_plugin_dir(self) -> Optional[str]:
        """Find the directory containing our custom VLC plugin."""
        if self._vlc_plugin_dir:
            return self._vlc_plugin_dir

        # Look in common locations relative to project
        project_root = Path(__file__).resolve().parent.parent
        candidates = [
        project_root / "vlcl" / "build" / "plugins",
        project_root / "vlcl" / "vlc" / "build" / "modules",
        project_root / "vlcl" / "vlc" / "builddir" / "modules",
        project_root / "vlc_plugins",
    ]

        for d in candidates:
            if d.exists():
                # Search for the plugin file
                for ext in [".dll", ".so", ".dylib"]:
                    matches = list(d.rglob(f"libaisub_plugin{ext}"))
                    if matches:
                        return str(matches[0].parent)

        return None

    def _find_player(self):
        """Find the best available video player."""
        vlc = self._find_vlc()
        if vlc:
            return vlc, "VLC"
        mpv = shutil.which("mpv")
        if mpv:
            return mpv, "mpv"
        return None, None

    def _find_vlc(self) -> Optional[str]:
        """Find VLC executable."""
        vlc = shutil.which("vlc")
        if vlc:
            return vlc
        if sys.platform == "win32":
            for p in [
                Path(r"C:\Program Files\VideoLAN\VLC\vlc.exe"),
                Path(r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe"),
            ]:
                if p.exists():
                    return str(p)
        elif sys.platform == "darwin":
            p = Path("/Applications/VLC.app/Contents/MacOS/VLC")
            if p.exists():
                return str(p)
        elif sys.platform.startswith("linux"):
            for p in ["/usr/bin/vlc", "/snap/bin/vlc"]:
                if Path(p).exists():
                    return p
        return None

    def _calc_progress(self) -> int:
        """Calculate progress percentage."""
        if self.engine._video_duration <= 0:
            return 50
        with self.engine._buffer_lock:
            if self.engine.subtitle_buffer:
                max_time = max(
                    s.end_sec for s in self.engine.subtitle_buffer
                )
                return min(95, int(
                    35 + 60 * max_time / self.engine._video_duration
                ))
        return 40

    def _subtitle_dispatcher(self):
        """Background thread: sends subtitles to pipe at the right time.
        
        Polls every 200ms, estimates playback position from wall clock,
        finds matching subtitle, sends it via pipe. Engine generates
        subs ahead of time; this thread dispatches them on schedule.
        """
        logger.info("Subtitle dispatcher started")
        last_sent_text = None

        while not self._dispatcher_stop:
            time.sleep(0.2)

            # Don't dispatch if player is gone
            if self._player_process and self._player_process.poll() is not None:
                break

            pos = self._estimate_playback_position()

            # Find subtitle for current position
            found = None
            with self.engine._buffer_lock:
                for sub in self.engine.subtitle_buffer:
                    if sub.start_sec <= pos <= sub.end_sec:
                        found = sub
                        break

            if found:
                if found.text != last_sent_text:
                    # New subtitle — send it
                    self.engine._pipe_send_subtitle(found)
                    last_sent_text = found.text
                    logger.debug(
                        f"Dispatched sub at {pos:.1f}s: {found.text[:40]}..."
                    )
            else:
                if last_sent_text is not None:
                    # No subtitle at this position — clear
                    self.engine._pipe_send_clear()
                    last_sent_text = None

        logger.info("Subtitle dispatcher stopped")

    def _cleanup(self, progress_cb: ProgressCallback = None):
        """Stop engine and clean up."""
        self._report(progress_cb, "Finalizing subtitles...", 98)

        # Stop dispatcher
        self._dispatcher_stop = True
        if self._dispatcher_thread and self._dispatcher_thread.is_alive():
            self._dispatcher_thread.join(timeout=2)

        self._stop_player()
        self.engine.stop()
        self.engine._write_all_subs_to_srt()

        total = self.engine.total_subtitles
        srt = Path(self._srt_path).name if self._srt_path else "output"
        self._report(progress_cb, f"Done! {total} subs saved to {srt}", 100)

    @staticmethod
    def _format_time(sec: float) -> str:
        m, s = divmod(int(sec), 60)
        return f"{m}:{s:02d}"

    @staticmethod
    def _report(cb: ProgressCallback, msg: str, pct: int):
        logger.info(msg)
        if cb:
            cb(msg, pct)

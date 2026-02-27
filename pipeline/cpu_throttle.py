"""
CPU Throttle — Voluntary CPU usage limiter.

Monitors CPU usage via psutil and introduces voluntary sleeps
to prevent sustained 100% CPU utilization on mid-range hardware.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CPUThrottle:
    """
    Monitors system CPU usage and introduces voluntary delays
    when usage exceeds the configured threshold.

    This prevents:
      - Thermal throttling on laptops
      - UI freezes when running as background process
      - Fan noise from sustained 100% CPU
    """

    def __init__(self, max_percent: int = 70, check_interval: float = 2.0):
        """
        Args:
            max_percent: Maximum average CPU usage percent before throttling.
            check_interval: Seconds between CPU usage checks.
        """
        self.max_percent = max_percent
        self.check_interval = check_interval
        self._psutil = None
        self._last_check = 0.0
        self._throttle_count = 0

    def _ensure_psutil(self):
        """Lazy-import psutil."""
        if self._psutil is None:
            try:
                import psutil
                self._psutil = psutil
            except ImportError:
                logger.warning(
                    "psutil not installed — CPU throttling disabled. "
                    "Install with: pip install psutil"
                )
                self._psutil = False  # Sentinel: don't retry

    def throttle_if_needed(self):
        """
        Check CPU usage and sleep if it exceeds the threshold.
        Call this between processing chunks.

        Throttling is non-blocking if CPU is within budget.
        """
        self._ensure_psutil()
        if self._psutil is False:
            return  # psutil not available

        now = time.monotonic()
        if now - self._last_check < self.check_interval:
            return  # Too soon since last check

        self._last_check = now
        usage = self._psutil.cpu_percent(interval=0.1)

        if usage > self.max_percent:
            self._throttle_count += 1
            sleep_time = min(2.0, (usage - self.max_percent) / 100.0 + 0.3)

            if self._throttle_count <= 3 or self._throttle_count % 10 == 0:
                logger.debug(
                    f"CPU at {usage:.0f}% (limit: {self.max_percent}%), "
                    f"sleeping {sleep_time:.1f}s (throttle #{self._throttle_count})"
                )

            time.sleep(sleep_time)

    def get_usage(self) -> Optional[float]:
        """Get current CPU usage percentage."""
        self._ensure_psutil()
        if self._psutil is False:
            return None
        return self._psutil.cpu_percent(interval=0.1)

    @property
    def total_throttles(self) -> int:
        """Number of times throttling was triggered."""
        return self._throttle_count

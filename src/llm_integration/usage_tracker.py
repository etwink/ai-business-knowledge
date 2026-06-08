"""Thread-safe LLM usage and cost tracker.

Records real input/output token counts from every API response, computes
approximate cost using LLM_INPUT_PPM / LLM_OUTPUT_PPM from config, and
auto-saves a JSON file after every call so the log survives partial runs.

Usage
-----
    from src.llm_integration.usage_tracker import tracker

    tracker.start_session("my_session")   # call when a session is named
    tracker.record(input_tokens=500, output_tokens=120)
    totals = tracker.get_totals()          # dict for display
    tracker.save()                         # force-write (auto-save happens on every record)
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import config

_LOG_DIR = Path("logs")


class UsageTracker:
    """Accumulates token usage across all LLM calls in the current process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session_name: str = "unknown"
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._reasoning_tokens: int = 0
        self._call_count: int = 0
        self._started_at: str = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_session(self, session_name: str) -> None:
        """Announce the active session name (does NOT reset counters)."""
        with self._lock:
            self._session_name = session_name

    def reset(self, session_name: str | None = None) -> None:
        """Reset all counters, optionally starting a new session name."""
        with self._lock:
            if session_name is not None:
                self._session_name = session_name
            self._input_tokens = 0
            self._output_tokens = 0
            self._reasoning_tokens = 0
            self._call_count = 0
            self._started_at = datetime.now().isoformat()

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int = 0,
    ) -> None:
        """Record token counts from one API response and auto-save."""
        with self._lock:
            self._input_tokens += max(0, input_tokens)
            self._output_tokens += max(0, output_tokens)
            self._reasoning_tokens += max(0, reasoning_tokens)
            self._call_count += 1
        self._auto_save()

    def get_totals(self) -> dict:
        """Return a snapshot dict suitable for JSON serialisation or display."""
        with self._lock:
            input_ppm = config.LLM_INPUT_PPM
            output_ppm = config.LLM_OUTPUT_PPM
            cost = (
                self._input_tokens / 1_000_000 * input_ppm
                + self._output_tokens / 1_000_000 * output_ppm
            )
            return {
                "session_name": self._session_name,
                "model": config.AZURE_OPENAI_DEPLOYMENT_NAME,
                "started_at": self._started_at,
                "last_updated": datetime.now().isoformat(),
                "total_calls": self._call_count,
                "total_input_tokens": self._input_tokens,
                "total_output_tokens": self._output_tokens,
                "total_reasoning_tokens": self._reasoning_tokens,
                "approximate_cost_usd": round(cost, 6),
                "input_ppm": input_ppm,
                "output_ppm": output_ppm,
            }

    def load_session(self, session_name: str) -> bool:
        """Restore tracker state from a previously saved JSON file.

        Looks for ``logs/usage_<session_name>.json``.  If the file exists,
        the tracker's counters and metadata are replaced with the saved values
        and any new calls will continue accumulating from that baseline.
        Returns True on success, False if the file is missing or unreadable.
        """
        path = _LOG_DIR / f"usage_{session_name}.json"
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            # File absent or corrupt — just switch session name, keep counters at zero
            with self._lock:
                self._session_name = session_name
            return False

        with self._lock:
            self._session_name = data.get("session_name", session_name)
            self._input_tokens = int(data.get("total_input_tokens", 0))
            self._output_tokens = int(data.get("total_output_tokens", 0))
            self._reasoning_tokens = int(data.get("total_reasoning_tokens", 0))
            self._call_count = int(data.get("total_calls", 0))
            self._started_at = data.get("started_at", datetime.now().isoformat())
        return True

    def save(self, path: Path | str | None = None) -> Path | None:
        """Explicitly save to *path* (defaults to logs/usage_<session>.json)."""
        try:
            data = self.get_totals()
            target = Path(path) if path else _LOG_DIR / f"usage_{self._session_name}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            return target
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _auto_save(self) -> None:
        try:
            _LOG_DIR.mkdir(exist_ok=True)
            data = self.get_totals()
            path = _LOG_DIR / f"usage_{self._session_name}.json"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception:
            pass


# Module-level singleton — import this everywhere
tracker = UsageTracker()

"""Local single-worker load guard. No automatic retries or restriction bypass."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from starlette.responses import JSONResponse

from deepseek.client import AccountMuted

logger = logging.getLogger("uvicorn.error")


def retry_delay(value: str | None, now: float) -> int:
    """Respect Retry-After seconds or an HTTP date; default to 60 seconds."""
    if value:
        try:
            if value.strip().isascii() and value.strip().isdigit():
                return max(1, int(value))
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return max(1, math.ceil(parsed.timestamp() - now))
        except (ValueError, TypeError, OverflowError, OSError):
            pass
    return 60


class SafetyGate:
    def __init__(self, state_path: Path, max_prompt_chars: int = 80000):
        self.state_path = Path(state_path)
        self.max_prompt_chars = max_prompt_chars
        self._state_lock = threading.Lock()
        self._inflight = threading.Lock()
        self._muted = None
        self._state_error = False
        self._cooldown_until = 0.0
        self._counts = {"admitted": 0, "prompt_rejected": 0,
                        "busy_rejected": 0, "blocked_rejected": 0,
                        "upstream_429": 0}
        try:
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict) or saved.get("code") != "account_muted":
                raise ValueError("invalid safety state")
            self._muted = AccountMuted(saved.get("mute_until"))
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            self._state_error = True
            logger.error("Cannot read account restriction state; requests disabled.")

    def latch_muted(self, exc: AccountMuted):
        with self._state_lock:
            self._muted = exc
            try:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.state_path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"code": "account_muted",
                                           "mute_until": exc.mute_until}),
                               encoding="utf-8")
                tmp.replace(self.state_path)
            except OSError:
                self._state_error = True
                logger.error("Cannot persist account restriction; blocked in this process.")

    def cooldown(self, header: str | None) -> int:
        with self._state_lock:
            now = time.time()
            delay = retry_delay(header, now)
            self._cooldown_until = max(self._cooldown_until, now + delay)
            self._counts["upstream_429"] += 1
            return math.ceil(self._cooldown_until - now)

    def _blocked(self):
        if self._muted is not None:
            exc = self._muted
            return JSONResponse(status_code=403, content={"error": {
                "message": str(exc), "type": "permission_error",
                "code": "account_muted", "mute_until": exc.mute_until,
                "retryable": False}})
        if self._state_error:
            return JSONResponse(status_code=503, content={"error": {
                "message": "Cannot read local restriction state; check server logs.",
                "type": "server_error", "code": "restriction_state_error",
                "retryable": False}})
        delay = math.ceil(self._cooldown_until - time.time())
        if delay > 0:
            return JSONResponse(status_code=429, headers={"Retry-After": str(delay)},
                                content={"error": {
                "message": "DeepSeek requested a pause; no upstream request sent.",
                "type": "rate_limit_error", "code": "upstream_rate_limited"}})
        return None

    def acquire(self):
        """Return an error response or reserve the one completion slot."""
        with self._state_lock:
            blocked = self._blocked()
            if blocked is not None:
                self._counts["blocked_rejected"] += 1
                return blocked
            if not self._inflight.acquire(blocking=False):
                self._counts["busy_rejected"] += 1
                return JSONResponse(status_code=429, headers={"Retry-After": "5"},
                                    content={"error": {
                    "message": "Another completion is still active.",
                    "type": "rate_limit_error", "code": "completion_busy"}})
            return None

    def release(self):
        self._inflight.release()

    def check_prompt(self, chars: int):
        with self._state_lock:
            if chars > self.max_prompt_chars:
                self._counts["prompt_rejected"] += 1
                return JSONResponse(status_code=400, content={"error": {
                    "message": f"Local prompt limit: {chars} characters exceeds "
                               f"{self.max_prompt_chars}; reduce context or tools.",
                    "type": "invalid_request_error", "code": "prompt_too_large",
                    "prompt_chars": chars, "max_prompt_chars": self.max_prompt_chars,
                    "retryable": False}})
            self._counts["admitted"] += 1
        return None

    def snapshot(self):
        with self._state_lock:
            return {"status": "degraded" if self._blocked() is not None else "ok",
                    "account_muted": self._muted is not None,
                    "mute_until": self._muted.mute_until if self._muted else None,
                    "restriction_state_error": self._state_error,
                    "cooldown_seconds": max(0, math.ceil(self._cooldown_until - time.time())),
                    "inflight": int(self._inflight.locked()),
                    "max_prompt_chars": self.max_prompt_chars,
                    "counters": dict(self._counts)}


class SingleCompletionMiddleware:
    """Hold the slot until the ASGI response, including streaming, finishes."""

    def __init__(self, app, gate_provider):
        self.app = app
        self.gate_provider = gate_provider

    async def __call__(self, scope, receive, send):
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path", "").rstrip("/") != "/v1/chat/completions"):
            return await self.app(scope, receive, send)
        gate = self.gate_provider()
        error = gate.acquire()
        if error is not None:
            return await error(scope, receive, send)
        try:
            await self.app(scope, receive, send)
        finally:
            gate.release()

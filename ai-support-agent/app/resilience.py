"""Resilience utilities: async timeouts and circuit breakers for external calls.

Provides:

- :func:`with_timeout`   — hard deadline on any awaitable (tool calls, LLM
  calls, retrieval). Prevents a hung LLM/API from holding a worker forever.
- :class:`CircuitBreaker` — fail-fast breaker per external dependency
  (OpenAI, vector store, …) that opens after N consecutive failures and
  half-opens after a cooldown, so a degraded provider gets traffic again
  instead of the whole request path piling up on timeouts.

Configured via settings (see ``app/config.py``):

    TOOL_TIMEOUT_SECONDS=20.0     # per tool call
    LLM_TIMEOUT_SECONDS=30.0      # per LLM call
    CIRCUIT_BREAKER_ENABLED=true
    CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
    CIRCUIT_BREAKER_RESET_SECONDS=30.0
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

logger = logging.getLogger("support_agent.resilience")

T = TypeVar("T")


class CircuitOpenError(Exception):
    """Raised when a call is attempted while its circuit is open."""

    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(f"Circuit '{name}' is open; retry in {retry_after:.1f}s.")


class CircuitState(str, Enum):
    CLOSED = "closed"      # normal operation
    OPEN = "open"          # failing — reject immediately
    HALF_OPEN = "half_open"  # cooldown elapsed — allow one probe call


class CircuitBreaker:
    """Per-dependency async circuit breaker.

    Usage::

        llm_breaker = CircuitBreaker("openai", failure_threshold=5, reset_timeout=30)

        async with llm_breaker.call():
            result = await llm.ainvoke(...)
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        half_open_successes: int = 1,
        enabled: bool = True,
    ):
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.reset_timeout = max(0.5, reset_timeout)
        self.half_open_successes = max(1, half_open_successes)
        self.enabled = enabled

        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes_since_half_open = 0
        self._opened_at: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and (time.monotonic() - self._opened_at) >= self.reset_timeout:
            self._state = CircuitState.HALF_OPEN
            self._successes_since_half_open = 0
        return self._state

    def _record_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._successes_since_half_open += 1
            if self._successes_since_half_open >= self.half_open_successes:
                logger.info("circuit '%s' closed (recovered)", self.name)
                self._state = CircuitState.CLOSED
                self._failures = 0
        else:
            self._failures = 0

    def _record_failure(self) -> None:
        self._failures += 1
        if self._state == CircuitState.HALF_OPEN or self._failures >= self.failure_threshold:
            if self._state != CircuitState.OPEN:
                logger.warning(
                    "circuit '%s' OPEN after %d consecutive failure(s) — fast-failing for %.0fs",
                    self.name, self._failures, self.reset_timeout,
                )
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    def check(self) -> float:
        """Raise :class:`CircuitOpenError` if calls should be rejected.
        Returns seconds waited if the circuit just half-opened (for logging)."""
        if not self.enabled:
            return 0.0
        state = self.state
        if state == CircuitState.OPEN:
            retry_after = self.reset_timeout - (time.monotonic() - self._opened_at)
            raise CircuitOpenError(self.name, max(0.1, retry_after))
        if state == CircuitState.HALF_OPEN:
            return self.reset_timeout
        return 0.0

    def record_success(self) -> None:
        self._record_success()

    def record_failure(self) -> None:
        self._record_failure()

    # -- context manager form ------------------------------------------------

    class _Call:
        def __init__(self, breaker: "CircuitBreaker"):
            self.breaker = breaker

        async def __aenter__(self) -> "CircuitBreaker._Call":
            self.breaker.check()
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            if exc_type is None:
                self.breaker.record_success()
            elif isinstance(exc, CircuitOpenError):
                return False  # pass through without re-counting
            else:
                self.breaker.record_failure()
            return False  # never swallow the exception

    def call(self) -> "CircuitBreaker._Call":
        return CircuitBreaker._Call(self)

    # -- decorator form --------------------------------------------------------

    def __call__(self, func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.check()
            try:
                result = await func(*args, **kwargs)
            except CircuitOpenError:
                raise
            except Exception:
                self.record_failure()
                raise
            self.record_success()
            return result

        return wrapper


# ---------------------------------------------------------------------------
# Registry — one shared breaker per external dependency
# ---------------------------------------------------------------------------

_BREAKERS: Dict[str, CircuitBreaker] = {}


def get_breaker(
    name: str,
    *,
    failure_threshold: int = 5,
    reset_timeout: float = 30.0,
    enabled: bool = True,
) -> CircuitBreaker:
    """Get (or lazily create) the shared circuit breaker for a dependency."""
    breaker = _BREAKERS.get(name)
    if breaker is None:
        breaker = _BREAKERS[name] = CircuitBreaker(
            name,
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout,
            enabled=enabled,
        )
    return breaker


def breaker_status() -> Dict[str, Dict[str, Any]]:
    """Snapshot of all breakers (for /health or /metrics endpoints)."""
    return {
        name: {"state": b.state.value, "consecutive_failures": b._failures}
        for name, b in _BREAKERS.items()
    }


# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------


async def with_timeout(
    awaitable: Awaitable[T],
    timeout_seconds: float,
    *,
    what: str = "operation",
) -> T:
    """Await *awaitable* with a hard deadline; raise ``TimeoutError`` beyond it."""
    if timeout_seconds <= 0:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        logger.warning("TIMEOUT after %.1fs: %s", timeout_seconds, what)
        raise TimeoutError(f"{what} timed out after {timeout_seconds:.1f}s.") from exc


async def run_tool_with_resilience(
    tool_invoke: Callable[[], Awaitable[Any]],
    *,
    tool_name: str,
    timeout_seconds: float,
    breaker: Optional[CircuitBreaker] = None,
) -> Any:
    """Invoke a tool coroutine with timeout + optional circuit breaking.

    Raises ``TimeoutError`` / ``CircuitOpenError`` — the caller
    (``execute_tools``) converts these into structured per-tool errors so
    one degraded dependency never crashes the whole run.
    """
    if breaker is not None:
        breaker.check()
    try:
        result = await with_timeout(tool_invoke(), timeout_seconds, what=f"tool '{tool_name}'")
    except Exception:
        if breaker is not None:
            breaker.record_failure()
        raise
    if breaker is not None:
        breaker.record_success()
    return result

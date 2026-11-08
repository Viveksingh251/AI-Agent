"""In-memory rate limiting middleware (token bucket, per identity).

Protects the API against brute-force login attempts, chat floods, and
accidental client loops. Uses a sliding-window counter keyed by
``identity`` (authenticated user id, falling back to client IP) plus a
per-keyword scope (``login``, ``chat``, ``default``).

Design notes:
- Zero external dependencies (no Redis required). For multi-worker
  deployments swap ``_buckets`` for a Redis-backed counter — the
  interface (``check(scope, identity) -> (allowed, retry_after)``) is
  kept deliberately simple so the backend can be swapped.
- Fail-open: if the limiter itself errors, requests are allowed
  (availability over strictness), and the error is logged.

Configure via settings (all optional):

    RATE_LIMIT_ENABLED=true
    RATE_LIMIT_DEFAULT="120/minute"
    RATE_LIMIT_LOGIN="10/minute"      # brute-force protection
    RATE_LIMIT_CHAT="30/minute"       # LLM cost protection
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.middleware import get_request_id

logger = logging.getLogger("support_agent.rate_limit")

_REQUEST_ID_HEADER = "X-Request-ID"

# ---------------------------------------------------------------------------
# Limit specification parsing: "N/second" | "N/minute" | "N/hour"
# ---------------------------------------------------------------------------

_LIMIT_RE = re.compile(r"^(\d+)\s*/\s*(second|minute|hour)$", re.IGNORECASE)
_UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600}


@dataclass(frozen=True)
class Limit:
    """A parsed rate limit: *count* events per *period_seconds* window."""

    count: int
    period_seconds: int

    @classmethod
    def parse(cls, spec: str) -> "Limit":
        m = _LIMIT_RE.match(spec.strip())
        if not m:
            raise ValueError(f"Invalid rate limit spec '{spec}' (expected e.g. '30/minute').")
        return cls(count=int(m.group(1)), period_seconds=_UNIT_SECONDS[m.group(2).lower()])


@dataclass
class _Bucket:
    """Sliding-window event log for one (scope, identity) pair."""

    events: List[float] = field(default_factory=list)


class RateLimiter:
    """Sliding-window rate limiter with per-scope limits. Thread-safe enough
    for a single event loop (no awaits inside check())."""

    def __init__(self, default: Limit, scopes: Dict[str, Limit], max_keys: int = 10_000):
        self.default = default
        self.scopes = scopes
        self._buckets: Dict[Tuple[str, str], _Bucket] = {}
        self._max_keys = max_keys

    def check(self, scope: str, identity: str) -> Tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)`` for one event."""
        limit = self.scopes.get(scope, self.default)
        now = time.monotonic()
        key = (scope, identity)
        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= self._max_keys:
                # Evict expired keys to bound memory under key-flood attacks.
                self._evict(now)
            bucket = self._buckets[key] = _Bucket()

        window_start = now - limit.period_seconds
        bucket.events[:] = [t for t in bucket.events if t > window_start]
        if len(bucket.events) >= limit.count:
            # Time until the oldest event in the window ages out.
            retry_after = int(max(1.0, bucket.events[0] + limit.period_seconds - now)) + 1
            return False, retry_after
        bucket.events.append(now)
        return True, 0

    def _evict(self, now: float) -> None:
        max_period = max([l.period_seconds for l in self.scopes.values()] + [self.default.period_seconds])
        stale = [k for k, b in self._buckets.items() if not b.events or now - b.events[-1] > max_period]
        for k in stale:
            self._buckets.pop(k, None)
        if len(self._buckets) >= self._max_keys and stale:
            logger.warning("rate limiter key flood: evicted %d stale buckets", len(stale))


def _identity(request: Request) -> str:
    """Identity for rate limiting: authenticated user id if present,
    otherwise the client IP (behind a proxy, the gateway must set
    X-Forwarded-For — Starlette's request.client is used as fallback)."""
    user = getattr(request.state, "authenticated_user", None)
    if user is not None:
        return f"user:{getattr(user, 'id', '?')}:{getattr(user, 'user_type', '?')}"
    client = request.client
    ip = client.host if client else "unknown"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    return f"ip:{ip}"


def _scope_for(request: Request) -> str:
    path = request.url.path
    if path.endswith("/auth/login") or path.endswith("/auth/token") or path.endswith("/auth/register"):
        return "login"
    if "/chat" in path:
        return "chat"
    return "default"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforces per-identity rate limits and returns RFC-style 429s with
    ``Retry-After`` headers."""

    def __init__(self, app: ASGIApp, limiter: RateLimiter, enabled: bool = True):  # type: ignore[name-defined]
        super().__init__(app)
        self.limiter = limiter
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        if not self.enabled:
            return await call_next(request)
        try:
            allowed, retry_after = self.limiter.check(_scope_for(request), _identity(request))
        except Exception:  # noqa: BLE001 — fail open
            logger.exception("rate limiter failure — allowing request (fail-open)")
            return await call_next(request)
        if not allowed:
            request_id = get_request_id() or ""
            logger.warning(
                "RATE LIMITED | scope=%s identity=%s retry_after=%ss request_id=%s",
                _scope_for(request), _identity(request), retry_after, request_id,
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error_type": "RateLimitExceeded",
                    "message": "Too many requests. Please slow down and try again shortly.",
                    "request_id": request_id,
                    "retry_after_seconds": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    _REQUEST_ID_HEADER: request_id,
                },
            )
        return await call_next(request)


def build_rate_limiter(
    *,
    enabled: bool = True,
    default_spec: str = "120/minute",
    login_spec: str = "10/minute",
    chat_spec: str = "30/minute",
) -> Tuple[RateLimiter, bool]:
    """Build the limiter from spec strings; falls back to safe defaults on bad config."""
    try:
        scopes: Dict[str, Limit] = {
            "login": Limit.parse(login_spec),
            "chat": Limit.parse(chat_spec),
        }
        return RateLimiter(Limit.parse(default_spec), scopes), enabled
    except ValueError as exc:
        logger.error("Invalid rate-limit configuration (%s) — using defaults.", exc)
        scopes = {"login": Limit(10, 60), "chat": Limit(30, 60)}
        return RateLimiter(Limit(120, 60), scopes), enabled


def add_rate_limiting(app: FastAPI, *, enabled: bool, default_spec: str, login_spec: str, chat_spec: str) -> None:
    limiter, enabled = build_rate_limiter(enabled=enabled, default_spec=default_spec, login_spec=login_spec, chat_spec=chat_spec)
    # Middleware order: last added runs first — add limiter AFTER request-id
    # middleware so 429 responses still carry an X-Request-ID.
    app.add_middleware(RateLimitMiddleware, limiter=limiter, enabled=enabled)

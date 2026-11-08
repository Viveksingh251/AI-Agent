# Security & Production-Readiness Review

**Scope:** full `ai-support-agent` codebase — API layer, LangGraph agent, tools, RAG, database, config, logging.
**Status:** the items in §2 are **implemented and tested** (181/181 tests pass); items in §3 are **documented recommendations** for a future iteration, with concrete pointers.

---

## 1. Threat Model Summary

| Surface | Threat |
|---|---|
| Chat / RAG endpoints | Prompt injection, oversized payloads, cost-abuse (LLM flooding) |
| Auth endpoints | Credential brute-force, weak secrets |
| Agent tools | Identity spoofing (LLM-supplied role/id), broken tenancy, IDOR |
| External deps (OpenAI, vector store, DB) | Hangs, outages cascading into request pile-up |
| Logs / audit trail | PII leakage, log forging via control characters |
| Configuration | Default secrets shipped to production, debug mode left on |

---

## 2. Implemented Fixes (verified)

### 2.1 Input validation & prompt-injection protection
**New module: `app/input_sanitizer.py`**

- `sanitize_message()` — rejects empty input, >4000 chars, and control characters (blocks log forging & terminal escapes). Applied at the chat router *before* the graph runs.
- `sanitize_session_id()` — session IDs whitelisted to `[A-Za-z0-9._-]{1,64}` (prevents thread-id manipulation and log injection).
- `detect_prompt_injection()` — 13 heuristic patterns covering instruction override, role hijack, system-prompt extraction, delimiter injection (`<|im_start|>`), template injection, credential fishing, and exfiltration URLs. ≥2 distinct categories ⇒ high-risk.
- `escape_for_prompt()` — high-risk input is **neutralised, not rejected** (no attacker oracle): fake role markers become `[user said]:`, fence tokens are defanged, then the message is processed normally.
- Injection events are logged as `PROMPT INJECTION SUSPECTED` and — see §2.8 — force escalation to human review.
- Double enforcement: `ChatRequest` in `api/schemas.py` now carries `max_length=4000` and a session-id `pattern`, so malformed requests are rejected at validation time (HTTP 422) even before the sanitizer runs.

### 2.2 Rate limiting
**New module: `app/rate_limit.py`** (wired in `app/main.py`)

- Sliding-window limiter, keyed by **authenticated user id** (falls back to client IP / `X-Forwarded-For`), scoped per endpoint class:
  - `login` scope: **10/min** (brute-force protection) — covers `/auth/login`, `/auth/token`, `/auth/register`
  - `chat` scope: **30/min** (LLM cost protection)
  - default: 120/min
- 429 responses include `Retry-After` and the standard `X-Request-ID` envelope.
- Memory-bounded (max 10k buckets with stale eviction) and **fail-open** — a limiter bug can never take the API down.
- Config: `RATE_LIMIT_ENABLED`, `RATE_LIMIT_DEFAULT`, `RATE_LIMIT_LOGIN`, `RATE_LIMIT_CHAT`.
- *Note:* in-process storage; see §3.2 for multi-worker deployment.

### 2.3 Tool authorization hardening
**`agent/tools/base.py` + `agent/tools/registry.py`**

- New `normalize_role()` whitelist: only `customer | support_agent | admin | staff` are valid. **Any other value — hostile, LLM-supplied, or corrupted — silently degrades to `customer`** (deny-by-default). Previously any string in `STAFF_ROLES` granted staff clearance with no validation of where it came from.
- `_system_injected_args()` now normalises the role and no longer contains special-case role logic; identity still comes exclusively from authenticated state, never from the LLM or the request body.

### 2.4 Secrets management
**`app/config.py`** production validator now enforces:

- `SECRET_KEY` ≠ default **and ≥32 chars** (rejects startup otherwise)
- `OPENAI_API_KEY` required when `LLM_PROVIDER != "mock"`
- `DEBUG` force-disabled in production
- JWT algorithm restricted to HS256/HS384/HS512
- `.env` remains the source; see §3.1 for the vault migration path.

### 2.5 SQL injection — verified safe
- All persistence flows through SQLAlchemy ORM models and parameterised Core statements (`select()/update()` with bound values) in `database/crud.py` — **no raw string SQL anywhere** (grep-verified: zero `text(...)` call sites).
- Order/ticket numbers are upper-cased and compared as strings, never interpolated.
- Residual risk to monitor: `Order.items` is a JSON column filled from order data — treat it as untrusted at display time (see §3.6).

### 2.6 PII redaction in logs
**`app/logging_config.py` + `app/input_sanitizer.redact_*`**

- The JSON formatter now **scrubs every log line at the formatter level** — including third-party loggers (uvicorn, sqlalchemy) — so redaction cannot be bypassed by a forgotten direct log call.
- Rules: emails → `jo**@domain`, card numbers → `[card]`, phones → `[phone]`, ≥7-digit runs → `[redacted-number]`, IBANs → `[iban]`.
- `redact_mapping()` applies the same rules recursively to structured `extra` fields (bounded depth 4).
- Audit trail (`agent/tools/base.audit_log`) now logs redacted details only; `execute_tools` logs redacted tool args (previously the raw message — including whatever the customer typed — went to logs at INFO level).
- Console formatter redacts too (dev parity).

### 2.7 Timeouts on LLM and tool calls
**New module: `app/resilience.py`**

- `with_timeout()` — hard `asyncio.wait_for` deadline on every awaitable.
- **LLM calls**: `_get_llm()` in `agent/nodes.py` now returns a proxy that enforces `OPENAI_TIMEOUT_SECONDS` (already configured, now actually enforced at the call level) on every `ainvoke`, in addition to the HTTP client timeout.
- **Tool calls**: `execute_tools` wraps every tool invocation with `run_tool_with_resilience()` using `TOOL_TIMEOUT_SECONDS` (default 20 s). A hung tool becomes a structured per-tool error — the run continues.
- On timeout the tool returns a customer-safe message ("temporarily unavailable") — no stack traces reach the LLM or the user.

### 2.8 Circuit breakers for external services
**`app/resilience.CircuitBreaker`** — per-dependency breakers with closed/open/half-open states:

- Breakers are keyed by dependency, not tool: `database`, `llm`, `vector_store` (`_dependency_for_tool`).
- Opens after `CIRCUIT_BREAKER_FAILURE_THRESHOLD` (default 5) consecutive failures; fast-fails for `CIRCUIT_BREAKER_RESET_SECONDS` (30 s); half-opens with a probe call; closes on success.
- Applied to: all tool invocations and all LLM calls.
- `breaker_status()` snapshot is available for wiring into `/health` (see §3.7).
- Config: `CIRCUIT_BREAKER_ENABLED` (default true).

### 2.9 Safe defaults for escalation
**`agent/nodes.py::intent_classifier`**

- **Triage failure now escalates instead of silently continuing.** Previously an unparseable/failed LLM triage produced `escalation_required=False, confidence=0.5` — i.e. a garbled classification sailed through as low-risk. Now: `urgency=high`, `escalation_required=True`, `confidence=0.0`, reason "Triage classification unavailable — routing to human review (safe default)."
- Detected prompt-injection categories (propagated via `state["metadata"]["injection_categories"]`) **force escalation** with a reason naming the manipulation — attempted manipulation is treated as a security event, not a support question.

### Verification
- `pytest`: **181/181 pass** (one test updated to supply an API key for production-env settings, matching the new validation).
- Dedicated smoke script exercised: rate-limit windows & per-identity isolation, sanitizer rejection/normalisation, injection detection & fencing, PII redaction (flat + nested), timeout firing, circuit open→half-open→closed lifecycle. All passed; script removed after use.

---

## 3. Documented Recommendations (not yet implemented)

### 3.1 Secrets management — move from `.env` to a secrets backend
Current: `.env` + production validators (§2.4). For production deployments:
- Load `SECRET_KEY`, `OPENAI_API_KEY`, `SENTRY_DSN`, DB credentials from **AWS Secrets Manager / GCP Secret Manager / HashiCorp Vault** at startup; pydantic-settings supports custom `settings_customise_sources` for this.
- Rotate `SECRET_KEY` with dual-key verification (accept old + new during rotation window) — `decode_jwt` would need a key list.
- Never bake secrets into Docker images; docker-compose `env_file` is dev-only.

### 3.2 Distributed rate limiting
The in-memory limiter is per-process. With multiple uvicorn/gunicorn workers or replicas, limits are effectively multiplied by worker count. Swap `RateLimiter.check()` for a Redis `INCR`+`EXPIRE` (or `redis-cell` GCRA) implementation — the call site interface is already isolated for this.

### 3.3 Refresh-token revocation & reuse detection
Refresh tokens are stateless JWTs: stolen tokens are valid until expiry. Add a `token_family`/jti table: store hashed jti per user, revoke on use (rotation), and treat reuse of a rotated token as theft (revoke the whole family).

### 3.4 Login timing / enumeration hardening
`authenticate_user` returns 401 with identical detail for unknown-user vs wrong-password (good), but unknown-user logins return faster (no bcrypt verify). Add a dummy bcrypt verify on unknown users to equalise timing.

### 3.5 RBAC matrix drift
`app/security.py::TOOL_PERMISSIONS` uses legacy tool names (`*_tool`) that don't match the registry (`get_order_status`, …) and `check_tool_access` **defaults to allow** for unknown tools. The actual enforcement path (tools/base) is independent and safe, but the matrix should be renamed, kept in sync with `register_all()`, and switched to deny-by-default.

### 3.6 Response-layer output handling
`Order.items` (JSON from order data) and retrieved KB chunks are rendered into prompts/responses. Sanitise on render in the Streamlit frontend (it escapes by default — keep it that way) and strip HTML from ingested documents at ingest time.

### 3.7 Expose breaker/limiter status
Wire `app.resilience.breaker_status()` into the `/health` response (new `components.circuit_breakers`) and add Prometheus counters for `circuit_state` transitions and `rate_limit_rejections_total{scope}` — currently only logged.

### 3.8 Audit log immutability
`agent_action_logs` is a normal table. For compliance-grade trails: append-only DB role (no UPDATE/DELETE grants), or hash-chained entries (`prev_hash + entry -> sha256`).

### 3.9 Content-security for `POST /rag/ingest`
Admin-only today, but ingested documents become agent context. Add per-document size caps, PDF bomb protection (page/character limits in `PyPDFLoader` usage), and an allow-listed source directory (no symlink traversal out of `data/knowledge_base`).

### 3.10 Dependency hygiene
Add `pip-audit` (or `safety`) to `make check`/CI to catch vulnerable pins in `requirements.txt`.

---

## 4. Configuration Reference (new)

| Variable | Default | Purpose |
|---|---|---|
| `RATE_LIMIT_ENABLED` | `true` | Master switch for rate limiting |
| `RATE_LIMIT_DEFAULT` | `120/minute` | Unscoped endpoints |
| `RATE_LIMIT_LOGIN` | `10/minute` | Auth endpoints (brute force) |
| `RATE_LIMIT_CHAT` | `30/minute` | Chat endpoints (LLM cost) |
| `TOOL_TIMEOUT_SECONDS` | `20.0` | Hard deadline per tool call |
| `CIRCUIT_BREAKER_ENABLED` | `true` | Master switch for breakers |
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` | Consecutive failures → open |
| `CIRCUIT_BREAKER_RESET_SECONDS` | `30.0` | Cooldown before half-open probe |
| `MAX_MESSAGE_LENGTH` | `4000` | Chat message character cap |

Add these to `.env.example` when adopting (all have safe defaults, so the app runs unchanged without them).

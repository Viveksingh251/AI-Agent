# REST API Reference

OpenAPI 3.1-compatible REST API. Interactive Swagger docs: **http://localhost:8000/docs** · ReDoc: **/redoc**.

All authenticated endpoints require a Bearer token:

```
Authorization: Bearer <access_token>
```

Roles: `customer`, `support_agent`, `admin`. Every response carries an `X-Request-ID` header (send it back to correlate logs/traces).

---

## 1. General

### `GET /`
API metadata and navigation.

```json
{ "app_name": "AI Customer Support Agent", "version": "0.1.0", "environment": "development", "docs": "/docs", "status": "operational" }
```

### `GET /health`
Health check for Docker/K8s probes. Reports component status and active feature flags.

**Response (200):**
```json
{
  "status": "healthy",
  "app_name": "AI Customer Support Agent",
  "environment": "development",
  "version": "0.1.0",
  "timestamp": "2026-11-05T12:00:00Z",
  "components": {
    "database":    { "status": "healthy",   "details": { "type": "postgresql", "pool_size": 10 } },
    "vector_store":{ "status": "healthy",   "details": { "backend": "faiss", "top_k": 4 } },
    "redis":       { "status": "disabled",  "details": { "enabled": false } },
    "llm":         { "status": "configured","details": { "provider": "openai", "model": "gpt-4o-mini" } }
  },
  "feature_flags": { "enable_escalation": true, "enable_streaming": true, "auto_ticket_creation": true }
}
```

### `GET /metrics`
Prometheus exposition format (`text/plain; version=0.0.4`). No authentication (protect at the gateway in production).

Key metric families:

| Metric | Type | Meaning |
|---|---|---|
| `support_agent_conversations_total{session_id_kind}` | counter | conversations processed (new vs continued) |
| `support_agent_tool_usage_total{tool_name,status}` | counter | per-tool invocation counts |
| `support_agent_escalations_total{reason}` | counter | escalations to humans (→ escalation rate = escalations / conversations) |
| `support_agent_response_time_seconds{endpoint}` | histogram | agent end-to-end latency (→ average response time) |
| `support_agent_tickets_created_total{priority}` | counter | tickets created by the agent (→ ticket creation rate) |
| `support_agent_graph_node_duration_seconds{node}` | histogram | per-node latency |
| `support_agent_tool_duration_seconds{tool_name}` | histogram | per-tool latency |
| `support_agent_http_requests_total{method,endpoint,status_code}` | counter | HTTP traffic |

---

## 2. Authentication — `/api/v1/auth`

### `POST /api/v1/auth/login` (alias: `/api/v1/auth/token`)
OAuth2 password-form login for customers and employees. **No auth required.**

**Form body:** `username` (email), `password`

**Response (200):**
```json
{ "access_token": "eyJ...", "refresh_token": "eyJ...", "token_type": "bearer", "role": "customer" }
```

### `POST /api/v1/auth/refresh`
Exchange a refresh token. **Body:** `{ "refresh_token": "eyJ..." }` → new token pair.

### `POST /api/v1/auth/register`
Self-service customer registration. **No auth required.**
**Body:** `{ "email", "name", "password", "phone?", "company?" }` → `CustomerResponse` (201).

### `GET /api/v1/auth/me`
Profile of the current authenticated user (customer or employee).

---

## 3. Chat & Agent — `/api/v1/chat`

### `POST /api/v1/chat` — Process Chat Turn
Submits a customer message to the LangGraph workflow. **Any authenticated user.**

**Request body:**
```json
{
  "message": "Where is my order ORD-10234?",
  "session_id": "sess_88219a",        // optional; generated when omitted
  "customer_id": "12"                  // optional; staff may impersonate, customers are bound to their own id
}
```

**Response (200):**
```json
{
  "session_id": "sess_88219a",
  "response": "Your order ORD-10234 was dispatched and is currently in transit.",
  "intent": "order_status",
  "sentiment": "neutral",
  "escalation_required": false,
  "ticket_id": null,
  "confidence_score": 0.9,
  "turn_count": 1,
  "timestamp": "2026-11-05T12:00:00Z",
  "actions": {
    "tools_used": ["get_order_status"],
    "documents_retrieved": 0,
    "order_lookup": true
  }
}
```

**Errors:** `502` when the agent execution fails.

### `POST /api/v1/chat/stream` — Server-Sent Events
Streaming variant. Emits `text/event-stream` events in order:

| Event | Payload |
|---|---|
| `meta` | `{ "session_id", "intent", "sentiment", "urgency" }` after triage |
| `status` | `{ "node": "execute_tools" }` per graph node |
| `token` | `{ "text": "word " }` chunks of the final answer |
| `actions` | actions-taken summary |
| `done` | full `ChatResponse` payload |
| `error` | `{ "detail": "..." }` on failure |

### `GET /api/v1/chat/{session_id}/history`
Conversation history from the checkpointer.

```json
{ "session_id": "sess_88219a", "messages": [{ "role": "customer", "content": "Hi" }], "total_turns": 1 }
```

### `DELETE /api/v1/chat/{session_id}`
Clears messages and checkpoints for a session → `{ "session_id": "...", "status": "cleared" }`.

---

## 4. Tickets — `/api/v1/tickets`

### `POST /api/v1/tickets` — Create Ticket (201)
Customers may only create tickets for themselves; staff can create for any customer.

**Body:**
```json
{ "customer_id": 12, "subject": "Refund not received", "description": "...", "priority": "high", "assigned_to": null }
```
**Response:** `TicketResponse` — `{ "id", "ticket_number": "TICK-AB12CD", "status", "priority", "created_at", ... }`

### `GET /api/v1/tickets?status=&priority=&skip=&limit=` — List Tickets
Staff see all tickets; customers see their own.

### `GET /api/v1/tickets/{ticket_number}`
Ticket details by ticket number. Customers: own tickets only (403 otherwise).

### `PATCH /api/v1/tickets/{ticket_id}`
Update status/priority/comment. **Staff only** (`support_agent`, `admin`).

**Body:** `{ "status": "in_progress", "priority": "high", "comment": "Investigating" }`

---

## 5. Customers — `/api/v1/customers`

### `POST /api/v1/customers` — Create Customer Account (201)
**Admin only.** **Body:** `{ "email", "name", "password", "phone?", "company?", "tier" }`

### `GET /api/v1/customers/{customer_id}`
Customer profile. Customers may only read their own profile.

### `GET /api/v1/customers/{customer_id}/tickets`
Tickets belonging to a customer.

### `GET /api/v1/customers/{customer_id}/orders`
Orders belonging to a customer.

---

## 6. Admin & Observability — `/api/v1/admin`

**Both endpoints require** `support_agent` or `admin`.

### `GET /api/v1/admin/performance-summary?window_hours=24&include_report=false`
Aggregated operational metrics over a lookback window (1–720 h):

```json
{
  "window_hours": 24,
  "total_actions": 142,
  "success_rate": 0.93,
  "actions_by_type": { "get_order_status": 51, "create_support_ticket": 18 },
  "tickets_created": 18, "open_tickets": 6, "escalations": 4,
  "narrative_report": "Concise LLM-written summary when include_report=true..."
}
```

### `GET /api/v1/admin/agent-logs?action_type=&customer_id=&success=&skip=&limit=`
Paginated audit trail of every action the AI agent performed.

```json
{ "total": 142, "skip": 0, "limit": 50, "items": [ { "action_type": "...", "customer_id": 12, "success": true, "details": {...}, "created_at": "..." } ] }
```

---

## 7. RAG & Knowledge Base — `/api/v1/rag`

### `POST /api/v1/rag/query` — Query the Knowledge Base
**Any authenticated user.**

**Body:**
```json
{
  "query": "What is the return window?",
  "top_k": 4,
  "doc_type": "policy",       // faq | policy | product | guide | general (optional)
  "category": "returns",       // shipping | returns | payment | product | account (optional)
  "threshold": 0.3,
  "use_hybrid": false
}
```

**Response (200):**
```json
{
  "query": "What is the return window?",
  "results": [ { "content": "...", "score": 0.87, "metadata": { "source": "return_policy.md" } } ],
  "total_found": 3,
  "retriever_type": "faiss",
  "threshold_applied": 0.3
}
```

### `POST /api/v1/rag/ingest` — Trigger Ingestion
**Admin only.**

**Body:** `{ "docs_dir": "./data/knowledge_base", "reset": false, "chunk_size": 1000, "chunk_overlap": 200, "batch_size": 100 }`

**Response (200):**
```json
{ "status": "completed", "documents_loaded": 4, "chunks_created": 37, "chunks_indexed": 37, "elapsed_seconds": 6.2, "error": null }
```

### `GET /api/v1/rag/stats`
**`support_agent` or `admin`.** Index statistics: `{ "backend", "document_count", "index_exists", "retriever_type", "top_k_default", "similarity_threshold" }`.

---

## 8. Error Envelope

All unhandled/domain errors return a uniform JSON body including the request ID:

```json
{
  "error_type": "InternalServerError",
  "message": "An unexpected server error occurred. Please try again later.",
  "request_id": "ea605036d43a4317",
  "path": "/api/v1/chat"
}
```

| Status | Typical cause |
|---|---|
| 401 | Missing/invalid/expired token |
| 403 | Role or ownership check failed |
| 404 | Unknown resource |
| 409 | Duplicate (e.g. customer email) |
| 422 | Validation error (FastAPI detail) |
| 500 | Unexpected server error |
| 502 | Agent/LLM execution failure |

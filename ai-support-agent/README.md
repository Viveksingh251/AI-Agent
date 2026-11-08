# AI Customer Support Agent (LangGraph + FastAPI + RAG)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-teal.svg)](https://fastapi.tiangolo.com/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

A modular, production-grade AI Customer Support Agent platform built with **LangGraph**, **FastAPI**, **RAG (Retrieval-Augmented Generation)**, **PostgreSQL**, **FAISS/pgvector/Qdrant**, and **Streamlit** — with full observability (structured JSON logs, OpenTelemetry traces, Prometheus metrics, LangSmith, Sentry).

---

## 📚 Documentation

| Document | Contents |
|---|---|
| **README.md** (this file) | Setup, running, ingesting documents, adding tools |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | LangGraph flow, tool authorization, state management |
| [docs/API.md](docs/API.md) | OpenAPI-style reference of every endpoint |
| [docs/ADDING_TOOLS.md](docs/ADDING_TOOLS.md) | Step-by-step guide to adding a new agent tool |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute code |
| [docs/SECURITY.md](docs/SECURITY.md) | Security review: implemented hardening + recommendations |
| [docs/deployment.md](docs/deployment.md) | Docker/Kubernetes deployment guide |

---

## 🏛️ Architecture Overview

```mermaid
flowchart TD
    User["Customer / Support Client"]
    UI["Streamlit Portal (frontend/)"]
    Gateway["FastAPI Gateway (app/main.py)"]
    Auth["JWT Auth & Permissions (app/security.py)"]
    ChatRouter["Chat Router (api/routers/chat.py)"]

    subgraph AgentSystem["LangGraph Customer Support Engine (agent/)"]
        Graph["Support StateGraph (agent/graph.py)"]
        TriageNode["Triage & Intent Node"]
        RAGNode["RAG Retrieval Node"]
        ToolsNode["Tool Execution Node"]
        ResponseNode["Response Generator"]
        HumanEscalate["Human Escalation Node"]
        AuditNode["Audit & Log Node"]

        Graph --> TriageNode
        TriageNode -->|FAQ / policy| RAGNode
        TriageNode -->|order / billing / ticket| ToolsNode
        TriageNode -->|angry / complex| HumanEscalate
        RAGNode --> ResponseNode
        ToolsNode -->|escalation needed| HumanEscalate
        ToolsNode --> ResponseNode
        ResponseNode --> AuditNode
        HumanEscalate --> AuditNode
    end

    subgraph Observability["Observability (app/monitoring.py)"]
        Metrics["Prometheus /metrics"]
        Traces["OpenTelemetry + LangSmith"]
        Logs["Structured JSON Logs"]
        Errors["Sentry"]
    end

    subgraph RAGPipeline["RAG Subsystem (rag/)"]
        VectorStore["FAISS / pgvector / Qdrant"]
        Retriever["Retriever (rag/retriever.py)"]
        KB["Knowledge Base docs (data/knowledge_base/)"]
    end

    subgraph DatabaseLayer["Data Persistence (database/)"]
        Postgres[("PostgreSQL")]
        RedisCheckpointer[("Redis / Memory Checkpointer")]
    end

    User <--> UI
    UI <--> Gateway
    Gateway --> Auth
    Auth --> ChatRouter
    ChatRouter <--> Graph
    RAGNode <--> Retriever
    Retriever <--> VectorStore
    ToolsNode <--> Postgres
    Graph <--> RedisCheckpointer
    AgentSystem -.-> Observability
```

---

## 📁 Directory Structure

```text
ai-support-agent/
├── app/                        # Application configuration & FastAPI entry point
│   ├── main.py                 # App factory, lifespan, /metrics, /health
│   ├── config.py               # Pydantic v2 settings (env-driven)
│   ├── monitoring.py           # Observability: logs, OTel traces, Prometheus, LangSmith, Sentry
│   ├── logging_config.py       # JSON log formatter
│   ├── middleware.py           # Request-ID tracing + error envelopes
│   ├── security.py             # JWT creation/verification, RBAC dependencies
│   ├── dependencies.py         # Dependency injection (DB, agent, auth)
│   └── exceptions.py           # Domain exceptions + handlers
├── agent/                      # LangGraph orchestration engine
│   ├── state.py                # SupportAgentState TypedDict + intent taxonomy
│   ├── graph.py                # StateGraph composition + node instrumentation
│   ├── nodes.py                # triage / retrieve / tools / respond / escalate / audit
│   ├── tools/                  # Agent tools
│   │   ├── base.py             # DB session, authorization, audit-log helpers
│   │   ├── registry.py         # Tool registry + execute_tools graph node
│   │   ├── ticket_tools.py     # Ticket CRUD tools
│   │   ├── customer_tools.py   # Customer & order lookup tools
│   │   └── kb_tools.py         # Knowledge-base search tool
│   └── prompts/                # System / triage / response / escalation prompts
├── rag/                        # Retrieval-Augmented Generation pipeline
│   ├── ingest.py               # Document loader, chunker, indexer (CLI + library)
│   ├── retriever.py            # Dense / hybrid retrieval
│   └── vectorstore.py          # FAISS / pgvector / Qdrant adapters
├── database/                   # Persistence layer
│   ├── models.py               # SQLAlchemy ORM models
│   ├── session.py              # Async engine + session maker
│   ├── crud.py                 # CRUD operations
│   └── migrations/             # Alembic environment
├── api/                        # HTTP layer
│   ├── schemas.py              # Pydantic request/response models
│   └── routers/                # chat / tickets / customers / auth / admin / rag
├── frontend/                   # Streamlit UI (chat, agent dashboard, admin)
├── tests/                      # pytest suite (181 tests)
├── scripts/                    # seed_admin.py, seed_db.py, run_ingest.py, evaluate_agent.py
├── docs/                       # ARCHITECTURE.md, API.md, ADDING_TOOLS.md, deployment.md
├── docker-compose.yml          # Full local stack (api, frontend, db, redis, qdrant)
├── Dockerfile
├── Makefile                    # install / run / test / ingest / migrate / docker-up
├── requirements.txt
└── .env.example
```

---

## 🚀 Quickstart

### 1. Prerequisites

- Python 3.11+
- PostgreSQL 14+ (or use Docker)
- Docker & Docker Compose (optional, for the full stack)
- An LLM API key (OpenAI by default) — *the app degrades gracefully to stub responses without one*

### 2. Local Setup

```bash
cd ai-support-agent

# Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# …or: make install
```

### 3. Environment Setup

```bash
cp .env.example .env
# Then edit .env: set OPENAI_API_KEY, DATABASE_URL, SECRET_KEY, etc.
# Every variable is documented inline in .env.example.
```

Minimum viable `.env` for local development:

```dotenv
APP_ENV=development
OPENAI_API_KEY=sk-...              # or LLM_PROVIDER=mock to run without an LLM
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/support_agent_db
SECRET_KEY=change-me-to-a-random-32-plus-char-string
```

### 4. Initialize the Database

```bash
# Apply Alembic migrations (recommended)
make migrate          # alembic upgrade head

# Or let the app create tables at startup (dev only)
# …then seed default accounts:
make seed             # creates admin / support agent / demo customer
```

Default seeded accounts (change in production!):

| Role | Email | Password |
|---|---|---|
| Admin | `admin@support.company.com` | `AdminPassword123!` |
| Support Agent | `agent@support.company.com` | `AgentPassword123!` |
| Customer | `customer@example.com` | `CustomerPassword123!` |

### 5. Ingest Knowledge-Base Documents

Put your documents (`.md`, `.txt`, `.pdf`, `.html`) in `data/knowledge_base/`, then:

```bash
make ingest
# equivalent to:
python -m rag.ingest --docs-dir ./data/knowledge_base

# Options:
python -m rag.ingest --docs-dir ./data/knowledge_base --reset \
    --chunk-size 800 --overlap 150
```

Or via the API (admin only): `POST /api/v1/rag/ingest` — see [docs/API.md](docs/API.md).

Verify with `make ingest-stats` or `GET /api/v1/rag/stats`.

### 6. Run the App

```bash
# Backend (FastAPI + Swagger UI)
make run                    # uvicorn app.main:app --reload --port 8000

# Frontend (Streamlit) — separate terminal
streamlit run frontend/app.py --server.port 8501
```

| URL | What |
|---|---|
| http://localhost:8000/docs | Swagger UI (interactive API docs) |
| http://localhost:8000/health | Health check |
| http://localhost:8000/metrics | Prometheus metrics |
| http://localhost:8501 | Streamlit chat portal |

### 7. Docker Compose (Full Stack)

```bash
make docker-up              # docker compose up --build -d
docker compose logs -f api  # tail API logs
make docker-down            # docker compose down
```

Services: `api` (:8000), `frontend` (:8501), `db` (PostgreSQL :5432), `redis` (:6379), and optional `qdrant` (:6333, profile `infra`).

---

## 🗣️ Talking to the Agent

```bash
# 1. Log in (get a token)
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=customer@example.com&password=CustomerPassword123!"

# 2. Send a chat turn
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"message": "Where is my order ORD-10234?"}'
```

The response includes the agent's answer plus structured `actions` (tools used, ticket created, escalation, retrieved documents).

Streaming: `POST /api/v1/chat/stream` emits SSE events (`meta`, `status`, `token`, `actions`, `done`).

---

## 🔧 How to Add a New Tool

Full walkthrough in [docs/ADDING_TOOLS.md](docs/ADDING_TOOLS.md). Short version:

1. Create the tool in `agent/tools/` with `@tool` from `langchain_core.tools`, returning `{"success": bool, ...}` dicts.
2. Enforce authorization with `is_authorized(...)` / `deny(...)` from `agent.tools.base`.
3. Register it in `agent/tools/registry.py::register_all()`.
4. Add intent routing (optional) in `agent/tools/registry.py::_FALLBACK_MAP` and `agent/graph.py::route_after_triage`.
5. Add tests in `tests/test_tools.py`.

---

## 📊 Observability

Enabled out of the box via `app/monitoring.py`:

- **Structured JSON logs** with `request_id`, `user_id`, `ticket_id`, `session_id` on every line (`LOG_JSON_FORMAT=true`).
- **OpenTelemetry traces** — spans per HTTP request, per graph node, per tool call. Set `OTEL_EXPORTER_OTLP_ENDPOINT` to export.
- **Prometheus metrics** at `/metrics`: conversation count, tool usage, escalation rate, response-time histogram, ticket-creation rate.
- **LangSmith** — set `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY`.
- **Sentry** — set `SENTRY_DSN`.

---

## 🧪 Testing

> **Windows note:** the `Makefile` targets require GNU make (available in WSL, Git Bash, or `choco install make`). On Windows PowerShell, run the underlying commands directly, e.g. `python -m pytest -v`, `python -m rag.ingest --docs-dir ./data/knowledge_base`.

```bash
make test        # pytest -v
pytest --cov=app --cov=agent --cov=rag --cov=api tests/   # with coverage
pytest -m authz  # only authorization tests
```

Lint & format:

```bash
make lint        # ruff + mypy
make format      # black
```

---

## 🛠️ Development Tips

- Prompts live in `agent/prompts/` — tune the persona in `system.py`, routing behavior in `triage.py`.
- Feature flags (escalation, streaming, auto ticket creation, …) are in `.env` under *Feature Flags*.
- The graph is compiled once and cached on `app.state.agent`; the checkpointer defaults to in-memory (swap for Redis/Postgres in multi-worker deployments).
- Production checklist: strong `SECRET_KEY`, `APP_ENV=production` (enforced — DEBUG is forced off), Alembic migrations, real checkpointer, OTLP/Sentry endpoints configured.

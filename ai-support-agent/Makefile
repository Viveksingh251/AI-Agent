# ==========================================================================
# AI Customer Support Agent — common development commands
#
# Usage:
#   make install        install dependencies
#   make run            run FastAPI backend with reload
#   make frontend       run Streamlit UI
#   make test           run the test suite
#   make ingest         ingest knowledge-base documents into the vector store
#   make migrate        apply database migrations (alembic upgrade head)
#   make revision m="msg"   create a new Alembic migration
#   make docker-up      start the full stack with Docker Compose
#   make docker-down    stop the stack
#   make check          run all quality checks (format, lint, types, tests)
#   make help           list available targets
# ==========================================================================

.PHONY: help install install-dev run frontend test test-cov ingest ingest-reset \
        ingest-stats migrate revision seed check lint format typecheck docker-up \
        docker-down docker-logs clean

# Python executable / flags
PYTHON  ?= python
PIP     ?= $(PYTHON) -m pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install: ## Install runtime dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install dev tooling (black, ruff, mypy)
	$(PIP) install black ruff mypy

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run: ## Start the FastAPI backend with hot reload (http://localhost:8000)
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

frontend: ## Start the Streamlit frontend (http://localhost:8501)
	$(PYTHON) -m streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0

# ---------------------------------------------------------------------------
# Tests & quality
# ---------------------------------------------------------------------------

test: ## Run the full test suite
	$(PYTHON) -m pytest -v

test-cov: ## Run tests with a coverage report
	$(PYTHON) -m pytest --cov=app --cov=agent --cov=rag --cov=api --cov=database tests/

lint: ## Lint with ruff
	$(PYTHON) -m ruff check .

typecheck: ## Static type check with mypy
	$(PYTHON) -m mypy app agent api database rag

format: ## Auto-format code with black
	$(PYTHON) -m black .

check: format lint typecheck test ## Run every quality gate (format, lint, types, tests)

# ---------------------------------------------------------------------------
# Knowledge base ingestion (RAG)
# ---------------------------------------------------------------------------

ingest: ## Ingest ./data/knowledge_base into the vector store
	$(PYTHON) -m rag.ingest --docs-dir ./data/knowledge_base

ingest-reset: ## Rebuild the index from scratch (delete + re-ingest)
	$(PYTHON) -m rag.ingest --docs-dir ./data/knowledge_base --reset

ingest-stats: ## Print vector index statistics
	$(PYTHON) -c "from rag.vectorstore import get_vector_store; print(get_vector_store().stats())"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

migrate: ## Apply database migrations (alembic upgrade head)
	$(PYTHON) -m alembic upgrade head

revision: ## Create a new Alembic migration: make revision m="message"
	@if [ -z "$(m)" ]; then echo "Usage: make revision m=\"migration message\""; exit 1; fi
	$(PYTHON) -m alembic revision --autogenerate -m "$(m)"

seed: ## Seed default admin/agent/customer accounts and demo data
	$(PYTHON) -m scripts.seed_admin
	$(PYTHON) -m scripts.seed_db

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-up: ## Build and start the full stack (api, frontend, db, redis)
	docker compose up --build -d
	@echo "API:      http://localhost:8000  (docs: /docs, metrics: /metrics)"
	@echo "Frontend: http://localhost:8501"

docker-down: ## Stop and remove the stack
	docker compose down

docker-logs: ## Tail API container logs
	docker compose logs -f api

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +

#!/usr/bin/env bash
# ==============================================================================
# docker/entrypoint.sh — container bootstrap for the AI Support Agent
# ------------------------------------------------------------------------------
# Responsibilities:
#   1. Wait for Postgres to accept connections (with bounded retries)
#   2. Apply Alembic database migrations (idempotent)
#   3. Seed a default admin user (for demo/local use)
#   4. Hand off to the container CMD via 'exec' (so PID 1 receives signals)
# ==============================================================================
set -Eeuo pipefail

# --- Config -------------------------------------------------------------------
DB_HOST="${POSTGRES_HOST:-db}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_USER="${POSTGRES_USER:-postgres}"
DB_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
DB_NAME="${POSTGRES_DB:-support_agent_db}"
MAX_DB_WAIT_RETRIES="${MAX_DB_WAIT_RETRIES:-30}"
SEED_ADMIN_ON_STARTUP="${SEED_ADMIN_ON_STARTUP:-true}"

# --- Wait for Postgres readiness ----------------------------------------------
echo "[entrypoint] Waiting for database at ${DB_HOST}:${DB_PORT} (db=${DB_NAME})..."
attempt=0
until python -c 'import os,socket,sys; s=socket.socket(); s.settimeout(3); sys.exit(0 if s.connect_ex((os.environ.get("DB_HOST","db"),int(os.environ.get("DB_PORT","5432"))))==0 else 1)'
do
  attempt=$((attempt + 1))
  if [ "${attempt}" -ge "${MAX_DB_WAIT_RETRIES}" ]; then
    echo "[entrypoint] ERROR: database not reachable after ${MAX_DB_WAIT_RETRIES} attempts — aborting." >&2
    exit 1
  fi
  echo "[entrypoint] attempt ${attempt}/${MAX_DB_WAIT_RETRIES}: db not ready — retrying in 2s..."
  sleep 2
done
echo "[entrypoint] Database is up."

# --- Alembic migrations --------------------------------------------------------
if [ -f "alembic.ini" ]; then
  echo "[entrypoint] Applying Alembic migrations..."
  alembic upgrade head
else
  echo "[entrypoint] No alembic.ini found — skipping migrations."
fi

# --- Optional admin seed -------------------------------------------------------
if [ "${SEED_ADMIN_ON_STARTUP}" = "true" ]; then
  echo "[entrypoint] Seeding default admin user (if missing)..."
  python scripts/seed_admin.py || echo "[entrypoint] WARN: admin seeding failed (non-fatal)"
fi

# --- Hand off to CMD ------------------------------------------------------------
# 'exec' replaces this shell so SIGTERM/SIGINT reach the app process directly,
# enabling fast, correct container shutdown.
echo "[entrypoint] Starting: $*"
exec "$@"

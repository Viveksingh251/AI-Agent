#!/bin/bash
set -e

# ==============================================================================
# Container Entrypoint Script for AI Customer Support Agent
# ==============================================================================

# TODO: Add database wait-for-it checks if needed
echo "Waiting for database and vector store connectivity..."

# TODO: Run alembic migrations automatically on startup
# alembic upgrade head

# Execute main process command passed to container
exec "$@"


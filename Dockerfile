# Use Python 3.11 as the base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Configure Poetry to create virtual environment in the project
RUN poetry config virtualenvs.create false

# Install dependencies without installing the project
RUN poetry install --no-interaction --no-ansi --no-root --without dev

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p db logs data .vector_store

# Make scripts executable
RUN chmod +x scripts/run_migrations.py 2>/dev/null || true

# Create startup script
RUN { \
    echo '#!/bin/bash'; \
    echo 'set -e'; \
    echo ''; \
    echo 'echo "Starting Agentic Workflow Builder..."'; \
    echo ''; \
    echo 'if [ -z "$DATABASE_URL" ]; then'; \
    echo '    echo "ERROR: DATABASE_URL environment variable is not set"'; \
    echo '    exit 1'; \
    echo 'fi'; \
    echo ''; \
    echo 'PORT=${PORT:-8000}'; \
    echo 'WORKERS=${WORKERS:-2}'; \
    echo ''; \
    echo 'echo "Running database migrations..."'; \
    echo 'if [ -f "scripts/run_migrations.py" ]; then'; \
    echo '    python scripts/run_migrations.py'; \
    echo 'else'; \
    echo '    echo "scripts/run_migrations.py not found; falling back to db/init_db.py"'; \
    echo '    python -m db.init_db'; \
    echo 'fi'; \
    echo ''; \
    echo 'echo "Starting application on port ${PORT} with ${WORKERS} workers..."'; \
    echo 'exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}'; \
} > /app/start.sh && chmod +x /app/start.sh

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["/app/start.sh"]

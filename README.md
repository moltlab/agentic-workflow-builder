# Agentic Workflow Builder

A production-grade framework for building and orchestrating multi-agent AI systems. It exposes a FastAPI backend that manages agent lifecycle, multi-agent workflow execution, MCP (Model Context Protocol) tool integration, session-based memory, and multi-modal inputs. Agents are defined in a database and executed via LangGraph state machines.

## Features

- **Multi-agent orchestration** — hierarchical and config-driven workflows via LangGraph
- **MCP tool integration** — 37+ tools synced from an MCP server into LangChain-compatible objects
- **Session-based memory** — per-agent, per-session chat history with configurable limits
- **Datasource management** — encrypted credentials for SQL databases, vector stores, and document corpora
- **Multi-tenancy** — every resource is scoped by `user_id` + `entity_id`
- **Multi-modal inputs** — image and document support via media routes
- **Multiple LLM providers** — OpenAI, DeepSeek, Qwen, Ollama, and local endpoints
- **Observability** — optional OpenTelemetry tracing with ClickHouse backend; LangTrace, OpenLit, and Opik integrations

## Architecture

```
HTTP Request
    └── api/routes/
            └── agents/langgraph_agent.py
                    ├── MCP Tools (agents/shared/)
                    ├── LLM (agents/shared/llm_factory.py)
                    ├── Memory (agents/memory/)
                    └── db/crud/
```


| Layer  | Location         | Description                                                             |
| ------ | ---------------- | ----------------------------------------------------------------------- |
| API    | `api/routes/`    | Six routers: agents, workflows, media, datasources, MCP servers, auth   |
| Agents | `agents/`        | All agents extend `BaseAgent`; LangGraph drives state machine execution |
| Memory | `agents/memory/` | `ChatMemory` keyed by `agent_id + session_id`                           |
| Tools  | `agents/shared/` | MCP tool sync, LLM factory, datasource scope injection                  |
| Data   | `db/`            | SQLAlchemy 2.0; `db/models.py` is the canonical schema                  |
| Config | `config/`        | `config_loader.py` merges `config.yaml` + env vars                      |


---

## Local Setup

### Prerequisites

- Python 3.10–3.12
- [Poetry](https://python-poetry.org/docs/#installation)
- Docker & Docker Compose
- An OpenAI API key (or compatible LLM endpoint)

---

### Option A — Docker Compose (recommended)

This starts Postgres and the app together.

**1. Clone the repo**

```bash
git clone <repo-url>
cd agentic-workflow-builder
```

**2. Create your `.env` file**

```bash
cp .env.example .env
```

Edit `.env` and fill in the required values:

```env
OPENAI_API_KEY=sk-...
MCP_SERVER_URL=http://localhost:8001/mcp
DATABASE_URL=postgresql://postgres:postgres@db:5432/agentic_ai_db
DATASOURCE_ENCRYPTION_KEY=change-this-to-a-strong-secret
```

**3. Start the stack**

```bash
docker-compose up
```

The API will be available at `http://localhost:8000`.
Swagger docs: `http://localhost:8000/docs`

---

### Option B — Local Python (without Docker)

**1. Install dependencies**

```bash
poetry install
```

**2. Start Postgres**

You need a running Postgres instance. The simplest way:

```bash
docker run -d \
  --name agentic-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=agentic_ai_db \
  -p 5432:5432 \
  postgres:16-alpine
```

**3. Create your `.env` file**

```bash
cp .env.example .env
```

Update `DATABASE_URL` to point to your local Postgres:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/agentic_ai_db
OPENAI_API_KEY=sk-...
DATASOURCE_ENCRYPTION_KEY=change-this-to-a-strong-secret
```

**4. Run database migrations**

```bash
alembic upgrade head
```

**5. Start the server**

```bash
poetry run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Environment Variables


| Variable                    | Required | Description                                                    |
| --------------------------- | -------- | -------------------------------------------------------------- |
| `OPENAI_API_KEY`            | Yes      | OpenAI (or compatible) API key                                 |
| `DATABASE_URL`              | Yes      | PostgreSQL connection string                                   |
| `MCP_SERVER_URL`            | Yes      | URL of the MCP tool server                                     |
| `DATASOURCE_ENCRYPTION_KEY` | Yes      | Fernet key or passphrase for encrypting datasource credentials |
| `APP_PORT`                  | No       | Port to bind (default: `8000`)                                 |
| `CORS_ORIGINS`              | No       | Comma-separated allowed origins                                |
| `DEEPSEEK_URL`              | No       | DeepSeek-compatible LLM endpoint                               |
| `QWEN_URL`                  | No       | Qwen-compatible LLM endpoint                                   |
| `OTLP_ENDPOINT`             | No       | OpenTelemetry collector endpoint                               |
| `GCS_BUCKET_NAME`           | No       | Google Cloud Storage bucket for media                          |


---

## Testing

```bash
# Install test dependencies
make install-test-deps

# 30-second smoke test
make test-health

# 2-3 minute API validation
make test-quick

# Full pytest suite with coverage
make test-full

# Single test
poetry run pytest tests/<file>.py::test_name -v
```

---

## Key Configuration Files


| File                 | Purpose                                          |
| -------------------- | ------------------------------------------------ |
| `.env.example`       | Template for all environment variables           |
| `config/config.yaml` | LLM model defaults, MCP server URL, RAG settings |
| `db/models.py`       | Source of truth for the database schema          |
| `pyproject.toml`     | Python dependencies                              |
| `alembic.ini`        | Database migration config                        |



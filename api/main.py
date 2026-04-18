# CRITICAL: Set OpenTelemetry environment before ANY imports
import os
os.environ.pop('OTEL_SDK_DISABLED', None)
os.environ['OTEL_SDK_DISABLED'] = 'true'

import sys
import time
import requests
from contextlib import asynccontextmanager
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from fastapi import FastAPI, HTTPException
from api.routes.agent_routes import agent_router
from api.routes.workflow_routes import workflow_router
from api.routes.media_routes import media_router
from api.routes.datasource_routes import datasource_router
from api.routes.mcp_server_routes import mcp_server_router
import yaml
try:
    from scripts.sync_mcp_tools import check_mcp_sse_reachable, sync_all_registered_mcp_servers
except ImportError:
    def check_mcp_sse_reachable(url: str, *args, **kwargs) -> bool:
        _ = (url, args, kwargs)
        return True

    def sync_all_registered_mcp_servers(*args, **kwargs) -> None:
        _ = (args, kwargs)
        return None
from db.crud.mcp_server import ensure_default_mcp_server_from_env, backfill_tools_mcp_server_id, get_default_server
from db.config import SessionLocal, engine
from utils.logging_utils import get_logger, init_logging
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace.sampling import ALWAYS_ON
except ImportError:
    class _DummySpan:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def set_attribute(self, *args, **kwargs):
            return None

        class _Ctx:
            trace_id = 0
            span_id = 0

        def get_span_context(self):
            return self._Ctx()

        def is_recording(self):
            return False

    class _DummyTracer:
        def start_as_current_span(self, *args, **kwargs):
            _ = (args, kwargs)
            return _DummySpan()

    class _DummyTraceModule:
        def __init__(self):
            self._provider = None

        def set_tracer_provider(self, provider):
            self._provider = provider

        def get_tracer_provider(self):
            return self._provider

        def get_tracer(self, name):
            _ = name
            return _DummyTracer()

    class Resource:
        @staticmethod
        def create(data):
            _ = data
            return {}

    class TracerProvider:
        def __init__(self, resource=None, sampler=None):
            _ = (resource, sampler)

        def add_span_processor(self, processor):
            _ = processor
            return None

    class BatchSpanProcessor:
        def __init__(self, exporter):
            _ = exporter

    class ConsoleSpanExporter:
        pass

    class OTLPSpanExporter:
        def __init__(self, endpoint=None):
            _ = endpoint

    class FastAPIInstrumentor:
        @staticmethod
        def instrument_app(app):
            _ = app
            return None

    ALWAYS_ON = object()
    trace = _DummyTraceModule()
import asyncio
from time import perf_counter
from fastapi.middleware.cors import CORSMiddleware

def validate_environment():
    """
    Validate mandatory environment variables are set.
    Loads from .env file for local development, expects them from Kubernetes in production.
    """
    # Define mandatory environment variables
    MANDATORY_VARS = [
        'MCP_SERVER_URL',
        'DATABASE_URL',
        'OPENAI_API_KEY',
    ]
    
    # Check if running in Kubernetes
    if os.path.exists('/var/run/secrets/kubernetes.io'):
        print("🔧 Running in Kubernetes environment - using secrets")
        # Kubernetes secrets are automatically mounted as environment variables
        # No additional loading needed
        is_kubernetes = True
    else:
        print("🏠 Running in local environment - loading from .env file")
        # Load from .env file for local development
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
            print("✅ Loaded environment from .env file")
        except ImportError:
            print("⚠️  python-dotenv not installed, trying without .env file")
        except Exception as e:
            print(f"❌ Error loading .env file: {e}")
            sys.exit(1)
        is_kubernetes = False
    
    # Validate all mandatory variables are present
    missing_vars = []
    for var in MANDATORY_VARS:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        env_source = "Kubernetes secrets" if is_kubernetes else ".env file"
        print(f"❌ Missing mandatory environment variables in {env_source}:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease ensure all required variables are set before starting the application.")
        sys.exit(1)
    
    print(f"✅ All mandatory environment variables validated")
    return True

# Validate environment before proceeding
validate_environment()

logger = get_logger('api')

# Load minimal logging configuration (dedicated file)
_config_dir = os.path.join(project_root, "config")
_logging_config_path = os.path.join(_config_dir, "logging_config.yaml")
with open(_logging_config_path, "r") as _f:
    _logging_config = yaml.safe_load(_f)
init_logging(_logging_config)

# Disable uvicorn access logs to prevent duplicate logging
import logging
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.disabled = True

def check_database_connection(max_retries=5, retry_delay=2):
    """Check database connection with retries"""
    for attempt in range(max_retries):
        try:
            # Try to execute a simple query
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")
            return True
        except SQLAlchemyError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database connection attempt {attempt + 1} failed: {str(e)}")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to database after {max_retries} attempts: {str(e)}")
                return False

def check_mcp_server(url, max_retries=5, retry_delay=5):
    """Check MCP server SSE connectivity with retries."""
    for attempt in range(max_retries):
        try:
            if check_mcp_sse_reachable(url):
                logger.info("MCP server connection established successfully")
                return True
            raise RuntimeError("SSE unreachable")
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"MCP server connection attempt {attempt + 1} failed: {str(e)}")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to MCP server after {max_retries} attempts: {str(e)}")
                return False

# --- Manual OpenTelemetry Setup (Working) ---
logger.info("Initializing OpenTelemetry tracing manually...")

# Create resource
resource = Resource.create({
    "service.name": "agentic-ai-framework",
    "service.version": "1.0.0"
})

# Create tracer provider
provider = TracerProvider(resource=resource, sampler=ALWAYS_ON)

# # Console exporter disabled to reduce log noise
# provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

# Add OTLP exporter if configured
otlp_endpoint = os.getenv('OTLP_ENDPOINT')
if otlp_endpoint:
    try:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info("OTLP exporter added successfully")
    except Exception as e:
        logger.warning(f"OTLP exporter failed (non-fatal): {e}")
else:
    logger.info("OTLP_ENDPOINT not set, skipping trace export")

# Set the global tracer provider
trace.set_tracer_provider(provider)

# Get tracer
tracer = trace.get_tracer("app.routes")
logger.info(f"Tracer initialized: {tracer}")
logger.info(f"Tracer type: {type(tracer)}")
logger.info(f"Current tracer provider: {trace.get_tracer_provider()}")

# # Test span creation during startup
# logger.info("Testing span creation during startup...")
# with tracer.start_as_current_span("startup-test-span") as span:
#     span.set_attribute("test.startup", "true")
#     span.set_attribute("gen_ai.agent.name", "agentic-ai-framework")
#     logger.info(f"Startup span created: {span.get_span_context()}")
#     logger.info(f"Startup trace ID: {hex(span.get_span_context().trace_id)}")
#     logger.info(f"Startup span ID: {hex(span.get_span_context().span_id)}")
#     logger.info(f"Startup span is recording: {span.is_recording()}")
# logger.info("Startup span test completed")

# Create FastAPI app
# root_path makes Swagger docs and OpenAPI schema work behind Nginx at /agentic-workflow-builder
app = FastAPI(
    title="Agentic Workflow Builder",
    description="API for agents, workflows, and media.",
    root_path=os.getenv("API_ROOT_PATH", ""),
)

# CORS: allow frontend origins to call the API
_cors_origins = os.getenv("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
FastAPIInstrumentor.instrument_app(app)

# Include routers
app.include_router(agent_router)
app.include_router(workflow_router, prefix="/api")
app.include_router(media_router)
app.include_router(datasource_router)
app.include_router(mcp_server_router)

# Helper function to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def simulated_work(ms: int):
    with tracer.start_as_current_span("simulated-work") as span:
        span.set_attribute("sleep.ms", ms)
        span.set_attribute("gen_ai.agent.name", "agentic-ai-framework")
        logger.info(f"Simulated work span created: ID={hex(span.get_span_context().span_id)}, Trace={hex(span.get_span_context().trace_id)}, Recording={span.is_recording()}")
        await asyncio.sleep(ms / 1000)

@app.get("/trace-test")
async def trace_test():
    logger.info("=== Starting trace-test endpoint ===")
    logger.info(f"Current tracer: {tracer}")
    logger.info(f"Current tracer type: {type(tracer)}")
    
    with tracer.start_as_current_span("main-span") as span:
        logger.info(f"Main span created: {span.get_span_context()}")
        logger.info(f"Main trace ID: {hex(span.get_span_context().trace_id)}")
        logger.info(f"Main span ID: {hex(span.get_span_context().span_id)}")
        logger.info(f"Main span is recording: {span.is_recording()}")
        
        span.set_attribute("endpoint.name", "trace-test")
        span.set_attribute("gen_ai.agent.name", "agentic-ai-framework")
        
        t0 = perf_counter()
        await simulated_work(100)
        await simulated_work(50)
        elapsed_ms = int((perf_counter() - t0) * 1000)
        span.set_attribute("elapsed.ms", elapsed_ms)
        
        logger.info(f"Trace test completed: elapsed={elapsed_ms}ms")
    
    logger.info("=== Finished trace-test endpoint ===")
    return {
        "status": "span created",
        "message": "Check logs for detailed span information"
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint that verifies:
    1. Database connection
    2. MCP server connection (if enabled)
    """
    health_status = {
        "status": "healthy",
        "database": "healthy",
        "mcp": "healthy"
    }

    # Check database connection
    if not check_database_connection():
        health_status["status"] = "unhealthy"
        health_status["database"] = "unhealthy"
        raise HTTPException(status_code=503, detail="Database connection failed")

    # Always check MCP server connection (URL from mandatory MCP_SERVER_URL env)
    mcp_server_url = os.getenv("MCP_SERVER_URL")
    if not mcp_server_url:
        health_status["mcp"] = "misconfigured"
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=503, detail="MCP server URL not configured")
    if not check_mcp_server(mcp_server_url):
        health_status["mcp"] = "unhealthy"
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=503, detail="MCP server connection failed")
    health_status["mcp"] = "healthy"

    return health_status

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    # Check database connection
    if not check_database_connection():
        raise Exception("Failed to establish database connection")

    # Always check MCP server connection and sync tools (URL from mandatory MCP_SERVER_URL env)
    mcp_server_url = os.getenv("MCP_SERVER_URL")
    if not mcp_server_url:
        raise Exception("MCP server URL not configured")
    if not check_mcp_server(mcp_server_url):
        raise Exception("Failed to establish MCP server connection")
    try:
        db = next(get_db())
        try:
            ensure_default_mcp_server_from_env(db)
            default_srv = get_default_server(db)
            if default_srv:
                backfill_tools_mcp_server_id(db, default_srv)
            sync_all_registered_mcp_servers(db)
            logger.info("Successfully synchronized MCP tools with database")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error synchronizing MCP tools: {str(e)}")
        raise

    yield

# Set lifespan
app.router.lifespan_context = lifespan
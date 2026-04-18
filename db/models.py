# db/models.py
from sqlalchemy import Column, String, Integer, Float, Text, JSON, TIMESTAMP, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from db.base import Base

class Session(Base):
    __tablename__ = "sessions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=True, server_default="system")
    entity_id = Column(String, nullable=True, server_default="default_entity")
    start_time = Column(TIMESTAMP)
    end_time = Column(TIMESTAMP)
    last_active = Column(TIMESTAMP)  # Last time the session was active
    status = Column(String)
    ip_address = Column(String)      # Optional: IP address of the client
    user_agent = Column(Text)        # Optional: User agent string
    session_metadata = Column(JSON)  # instead of metadata = Column(JSON)

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"))
    user_id = Column(String, nullable=True, server_default="system")
    entity_id = Column(String, nullable=True, server_default="default_entity")
    task_description = Column(String)
    status = Column(String)
    start_time = Column(TIMESTAMP)
    end_time = Column(TIMESTAMP)
    total_duration_ms = Column(Integer)
    input_data = Column(JSON)  # includes query, agent, media_ids / media_refs
    final_output = Column(JSON)


class Media(Base):
    """Stores metadata for uploaded media (images/videos) used in multi-modal agent inputs."""
    __tablename__ = "media"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=True)
    transaction_id = Column(UUID(as_uuid=True), ForeignKey("transactions.id"), nullable=True)
    user_id = Column(String, nullable=True, server_default="system")
    entity_id = Column(String, nullable=True, server_default="default_entity")
    filename = Column(String, nullable=False)
    media_type = Column(String, nullable=False)  # e.g. image/png, video/mp4
    cloud_path = Column(String, nullable=False)  # path/key in cloud storage (GCS, S3, Azure, etc.)
    file_size = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())

class Agent(Base):
    __tablename__ = "agents"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String)
    type = Column(String)
    description = Column(String)
    created_by_user_id = Column(String, nullable=True, server_default="system")
    entity_id = Column(String, nullable=True, server_default="default_entity")
    llm_used = Column(String)
    prompt_template = Column(String)
    config = Column(JSON)
    output_format = Column(String, default="")
    is_active = Column(Boolean, default=True)


class MCPServer(Base):
    """Registered MCP tool servers (framework control plane)."""

    __tablename__ = "mcp_servers"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    base_url = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    entity_id = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    config = Column(JSON, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())


class Datasource(Base):
    """Datasource definitions owned by the framework (non-secret metadata)."""

    __tablename__ = "datasources"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    datasource_type = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    connection_metadata = Column(JSON, nullable=True)
    credential_type = Column(String, nullable=False, server_default="direct")
    credential_ref = Column(String, nullable=True)
    entity_id = Column(String, nullable=True, server_default="default_entity")
    created_by_user_id = Column(String, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True)


class DatasourceCredential(Base):
    """Encrypted or sealed credential payload for a datasource (one row per datasource)."""

    __tablename__ = "datasource_credentials"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    datasource_id = Column(
        UUID(as_uuid=True),
        ForeignKey("datasources.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    payload_ciphertext = Column(Text, nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())


class Tool(Base):
    __tablename__ = "tools"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String)
    description = Column(Text)
    endpoint = Column(String)
    config = Column(JSON)
    created_at = Column(TIMESTAMP)
    created_by_user_id = Column(String, nullable=True, server_default="system")
    entity_id = Column(String, nullable=True, server_default="default_entity")
    mcp_server_id = Column(UUID(as_uuid=True), ForeignKey("mcp_servers.id", ondelete="SET NULL"), nullable=True)
    accepts_datasource_types = Column(JSON, nullable=True)
    risk_level = Column(String, nullable=True)
  
class MemoryStore(Base):
    __tablename__ = "memory_stores"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String)
    type = Column(String)
    config = Column(JSON)

class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id = Column(UUID(as_uuid=True), ForeignKey("transactions.id"))
    component_type = Column(String)
    component_id = Column(UUID(as_uuid=True))
    metric_name = Column(String)
    metric_value = Column(Float)
    unit = Column(String)
    timestamp = Column(TIMESTAMP)


# ... existing code ...
# AI Suggested
class BaseTransaction(Base):
    __abstract__ = True  # This makes it a base class that won't create a table
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id = Column(UUID(as_uuid=True), ForeignKey("transactions.id"))
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"))
    user_id = Column(String, nullable=True, server_default="system")
    entity_id = Column(String, nullable=True, server_default="default_entity")
    start_time = Column(TIMESTAMP)
    end_time = Column(TIMESTAMP)
    duration_ms = Column(Integer)
    input_data = Column(JSON)
    output_data = Column(JSON)
    error_message = Column(Text)

class AgentExecutionLog(BaseTransaction):
    __tablename__ = "agent_execution_log"
    # Inherits all fields from BaseTransaction
    # Add any agent-specific fields here if needed

class ToolExecutionLog(BaseTransaction):
    __tablename__ = "tool_execution_log"
    tool_id = Column(UUID(as_uuid=True), ForeignKey("tools.id"))
    # Inherits all fields from BaseTransaction

class MemoryTransactionLog(BaseTransaction):
    __tablename__ = "memory_transaction_log"
    memory_id = Column(UUID(as_uuid=True), ForeignKey("memory_stores.id"))
    operation = Column(String)
    key = Column(String)
    value_before = Column(JSON)
    value_after = Column(JSON)
    # Inherits all fields from BaseTransaction

class RAGExecutionLog(BaseTransaction):
    __tablename__ = "rag_execution_log"
    retriever_config = Column(JSON)
    retrieved_chunks = Column(JSON)
    context_used = Column(JSON)
    vector_store = Column(String)
    embedding_model = Column(String)
    # Inherits all fields from BaseTransaction

class LLMUsageLog(BaseTransaction):
    __tablename__ = "llm_usage_log"
    model_name = Column(String)
    model_provider = Column(String)
    temperature = Column(Float)
    max_tokens = Column(Integer)
    total_tokens_used = Column(Integer)
    response_latency_ms = Column(Integer)
    prompt = Column(Text)
    response = Column(Text)
    # Inherits all fields from BaseTransaction

class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    user_id = Column(String, nullable=True, server_default="system")
    entity_id = Column(String, nullable=True, server_default="default_entity")
    sender = Column(String)  # "user" or "assistant"
    message = Column(Text)  # The chat message (now Text type)
    timestamp = Column(TIMESTAMP)
    message_metadata = Column(JSON)  # Renamed from metadata
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), nullable=True)
    attachments = Column(JSON, nullable=True)  # [{"media_id": "...", "media_type": "...", "cloud_path": "..."}]

# ... existing code ...
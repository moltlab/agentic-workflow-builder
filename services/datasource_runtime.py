"""Datasource runtime helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from db.models import Datasource, DatasourceCredential
from services.datasource_crypto import decrypt_secrets_blob


def new_document_corpus_metadata() -> Dict[str, Any]:
    return {
        "provider": "qdrant",
        "corpus_files": [],
        "qdrant_url": "http://localhost:6333",
        "collection_name": "agentic_documents",
    }


def raise_if_connection_failed(result: Dict[str, Any]) -> None:
    if (result or {}).get("status") != "ok":
        raise ValueError((result or {}).get("message", "Connection test failed"))


def skip_sql_connection_test_for_secret_manager(payload: Dict[str, Any]) -> bool:
    return (payload or {}).get("credential_type") == "secret_manager"


def split_sql_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    p = dict(payload or {})
    meta: Dict[str, Any] = {
        "provider": p.get("provider"),
        "host": p.get("host"),
        "port": p.get("port"),
        "database": p.get("database"),
        "username": p.get("username"),
        "file_path": p.get("file_path"),
    }
    secrets: Dict[str, Any] = {}
    password = p.get("password")
    if password is not None and str(password) != "":
        secrets["password"] = password
    return ({k: v for k, v in meta.items() if v is not None}, secrets)


def split_vector_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    p = dict(payload or {})
    cfg = p.get("connection_config") if isinstance(p.get("connection_config"), dict) else {}
    provider = p.get("provider") or cfg.get("provider") or "qdrant"
    meta = {"provider": provider, **{k: v for k, v in cfg.items() if k not in {"api_key", "token", "secret"}}}
    secrets: Dict[str, Any] = {}
    for k in ("api_key", "token", "secret"):
        if k in cfg and cfg[k]:
            secrets[k] = cfg[k]
    return meta, secrets


def _sql_url(provider: str, cfg: Dict[str, Any], password: Optional[str]) -> str:
    p = (provider or "").lower()
    if p == "sqlite":
        file_path = cfg.get("file_path")
        if not file_path:
            raise ValueError("SQLite requires file_path")
        return f"sqlite:///{file_path}"

    host = cfg.get("host")
    port = cfg.get("port")
    database = cfg.get("database")
    username = cfg.get("username")
    if not (host and database and username):
        raise ValueError("SQL datasource missing host/database/username")
    pwd = password or ""
    if p in {"postgres", "postgresql"}:
        return f"postgresql+psycopg2://{username}:{pwd}@{host}:{port or 5432}/{database}"
    if p in {"mysql", "mariadb"}:
        return f"mysql+pymysql://{username}:{pwd}@{host}:{port or 3306}/{database}"
    raise ValueError(f"Unsupported SQL provider: {provider}")


def _check_sql_connection(provider: str, cfg: Dict[str, Any], password: Optional[str]) -> Dict[str, Any]:
    try:
        url = _sql_url(provider, cfg, password)
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok", "message": "Connection validated"}
    except (ValueError, SQLAlchemyError) as exc:
        return {"status": "error", "message": str(exc)}


def test_inline_connection(payload: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(payload or {})
    dtype = (p.get("datasource_type") or "").lower()
    provider = p.get("provider") or ""
    if dtype == "sql_database":
        return _check_sql_connection(provider, p, p.get("password"))
    if dtype == "vector_store":
        # Keep conservative behavior: validate shape only.
        cfg = p.get("connection_config") if isinstance(p.get("connection_config"), dict) else {}
        if not cfg:
            return {"status": "error", "message": "connection_config is required for vector_store"}
        if not (cfg.get("url") or cfg.get("host")):
            return {"status": "error", "message": "vector_store requires url or host"}
        return {"status": "ok", "message": "Vector store config looks valid"}
    return {"status": "error", "message": f"Unsupported datasource_type: {dtype or 'unknown'}"}


def test_document_corpus_qdrant_reachability(meta: Dict[str, Any], secrets: Dict[str, Any]) -> Dict[str, Any]:
    _ = (meta, secrets)
    # Keep lightweight in framework layer; deep connectivity is validated at ingest/runtime calls.
    return {"status": "ok", "message": "Document corpus configuration accepted"}


def test_datasource_connection(db: Session, datasource_id: Any) -> Dict[str, Any]:
    ds = db.query(Datasource).filter(Datasource.id == datasource_id, Datasource.is_active.is_(True)).first()
    if not ds:
        return {"status": "error", "message": "Datasource not found"}

    if ds.datasource_type == "sql_database":
        cfg = dict(ds.connection_metadata or {})
        password: Optional[str] = None
        row = (
            db.query(DatasourceCredential)
            .filter(DatasourceCredential.datasource_id == ds.id)
            .first()
        )
        if row:
            try:
                secret_payload = decrypt_secrets_blob(row.payload_ciphertext)
                password = secret_payload.get("password")
            except Exception:
                return {"status": "error", "message": "Unable to decrypt datasource credentials"}
        return _check_sql_connection(ds.provider or "postgresql", cfg, password)

    if ds.datasource_type in {"vector_store", "document_corpus"}:
        return {"status": "ok", "message": "Datasource config accepted"}

    return {"status": "error", "message": f"Unsupported datasource_type: {ds.datasource_type}"}


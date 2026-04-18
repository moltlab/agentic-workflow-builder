"""Datasource persistence (framework control plane)."""

from __future__ import annotations

import copy
import os
import uuid as uuid_mod
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from db.models import Datasource, DatasourceCredential
from services.datasource_crypto import encrypt_secrets_blob
from services.datasource_runtime import (
    new_document_corpus_metadata,
    raise_if_connection_failed,
    skip_sql_connection_test_for_secret_manager,
    split_sql_payload,
    split_vector_payload,
    test_datasource_connection,
    test_document_corpus_qdrant_reachability,
    test_inline_connection,
)
from utils.logging_utils import get_logger

logger = get_logger("datasource_crud")

_SENSITIVE_KEYS = frozenset(
    k.lower()
    for k in ("password", "api_key", "secret", "token", "openai_api_key", "qdrant_api_key")
)


def _redact(obj: Any) -> Any:
    if obj is None:
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kl = str(k).lower()
            if any(s in kl for s in _SENSITIVE_KEYS):
                out[k] = "••••••••" if v else ""
            elif isinstance(v, (dict, list)):
                out[k] = _redact(v)
            else:
                out[k] = v
        return out
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    return obj


def list_datasources_for_entity(db: Session, entity_id: str) -> List[Dict[str, Any]]:
    q = db.query(Datasource).filter(Datasource.is_active.is_(True))
    if entity_id:
        q = q.filter(Datasource.entity_id == entity_id)
    rows = q.order_by(Datasource.created_at.desc()).all()
    return [_to_list_item(ds) for ds in rows]


def _to_list_item(ds: Datasource) -> Dict[str, Any]:
    return {
        "id": str(ds.id),
        "name": ds.name,
        "datasource_type": ds.datasource_type,
        "provider": ds.provider,
        "description": ds.description,
        "created_at": ds.created_at.isoformat() if ds.created_at else None,
        "created_by_user_id": ds.created_by_user_id,
        "entity_id": ds.entity_id,
    }


def get_datasource(db: Session, ds_id: uuid_mod.UUID) -> Optional[Datasource]:
    return db.query(Datasource).filter(Datasource.id == ds_id, Datasource.is_active.is_(True)).first()


def serialize_datasource_for_get(ds: Datasource) -> Dict[str, Any]:
    """Single-datasource JSON for list row + edit form (no secrets)."""
    out = _to_list_item(ds)
    out["connection_metadata"] = copy.deepcopy(ds.connection_metadata or {})
    return out


def build_detail_payload(db: Session, ds: Datasource) -> Dict[str, Any]:
    meta = copy.deepcopy(ds.connection_metadata or {})
    secrets_preview: Dict[str, Any] = {}
    cred = (
        db.query(DatasourceCredential)
        .filter(DatasourceCredential.datasource_id == ds.id)
        .first()
    )
    if cred and ds.credential_type != "secret_manager":
        from services.datasource_crypto import decrypt_secrets_blob

        try:
            secrets_preview = decrypt_secrets_blob(cred.payload_ciphertext)
        except Exception:
            secrets_preview = {}
    connection_config = _redact({**meta, **secrets_preview})
    corpus_files = meta.get("corpus_files") if ds.datasource_type == "document_corpus" else []
    vector_identifiers: List[str] = []
    if ds.datasource_type == "vector_store":
        cn = meta.get("collection_name")
        if cn:
            vector_identifiers.append(f"collection: {cn}")
        vsid = meta.get("vector_store_id")
        if vsid:
            vector_identifiers.append(f"openai_vector_store: {vsid}")

    return {
        "id": str(ds.id),
        "name": ds.name,
        "datasource_type": ds.datasource_type,
        "provider": ds.provider,
        "description": ds.description,
        "created_at": ds.created_at.isoformat() if ds.created_at else None,
        "created_by_user_id": ds.created_by_user_id,
        "entity_id": ds.entity_id,
        "connection_config": connection_config,
        "corpus_files": corpus_files or [],
        "vector_identifiers": vector_identifiers,
        "corpus_files_error": None,
    }


def create_datasource_from_body(
    db: Session,
    body: Dict[str, Any],
    entity_id: str,
    user_id: str,
) -> Datasource:
    dtype = body.get("datasource_type") or "sql_database"
    name = (body.get("name") or "").strip()
    if not name:
        raise ValueError("name is required")

    if dtype == "sql_database":
        prov = body.get("provider") or "postgresql"
        if not skip_sql_connection_test_for_secret_manager(
            {**body, "datasource_type": "sql_database", "provider": prov}
        ):
            raise_if_connection_failed(
                test_inline_connection({**body, "datasource_type": "sql_database", "provider": prov})
            )
        meta, secrets = split_sql_payload({**body, "provider": prov})
        ds = Datasource(
            name=name,
            datasource_type="sql_database",
            provider=prov,
            description=body.get("description"),
            connection_metadata=meta,
            credential_type=body.get("credential_type") or "direct",
            credential_ref=body.get("credential_ref"),
            entity_id=entity_id,
            created_by_user_id=user_id,
        )
        db.add(ds)
        db.flush()
        if ds.credential_type == "direct" and secrets:
            db.add(
                DatasourceCredential(
                    datasource_id=ds.id,
                    payload_ciphertext=encrypt_secrets_blob(secrets),
                )
            )
        db.commit()
        db.refresh(ds)
        return ds

    if dtype == "document_corpus":
        meta = new_document_corpus_metadata()
        secrets: Dict[str, Any] = {}
        qk = os.getenv("CORPUS_QDRANT_API_KEY")
        if qk:
            secrets["api_key"] = qk
        raise_if_connection_failed(test_document_corpus_qdrant_reachability(meta, secrets))
        ds = Datasource(
            name=name,
            datasource_type="document_corpus",
            provider=body.get("provider") or "qdrant",
            description=body.get("description"),
            connection_metadata=meta,
            credential_type="direct",
            entity_id=entity_id,
            created_by_user_id=user_id,
        )
        db.add(ds)
        db.flush()
        if secrets:
            db.add(
                DatasourceCredential(
                    datasource_id=ds.id,
                    payload_ciphertext=encrypt_secrets_blob(secrets),
                )
            )
        db.commit()
        db.refresh(ds)
        return ds

    if dtype == "vector_store":
        prov = body.get("provider") or "qdrant"
        if (body.get("credential_type") or "direct") != "secret_manager":
            raise_if_connection_failed(
                test_inline_connection({**body, "datasource_type": "vector_store", "provider": prov})
            )
        meta, secrets = split_vector_payload(body)
        ds = Datasource(
            name=name,
            datasource_type="vector_store",
            provider=prov,
            description=body.get("description"),
            connection_metadata=meta,
            credential_type=body.get("credential_type") or "direct",
            credential_ref=body.get("credential_ref"),
            entity_id=entity_id,
            created_by_user_id=user_id,
        )
        db.add(ds)
        db.flush()
        db.add(
            DatasourceCredential(
                datasource_id=ds.id,
                payload_ciphertext=encrypt_secrets_blob(secrets),
            )
        )
        db.commit()
        db.refresh(ds)
        return ds

    raise ValueError(f"Unsupported datasource_type: {dtype}")


def update_datasource_from_body(
    db: Session,
    ds: Datasource,
    body: Dict[str, Any],
) -> Datasource:
    if "name" in body and body["name"]:
        ds.name = str(body["name"]).strip()
    if "description" in body:
        ds.description = body.get("description")

    dtype = ds.datasource_type
    if dtype == "sql_database":
        cur = dict(ds.connection_metadata or {})
        prov = body.get("provider") or ds.provider
        if "host" in body and body["host"]:
            cur["host"] = body["host"]
        if "port" in body and body["port"] is not None:
            cur["port"] = body["port"]
        if "database" in body and body["database"]:
            cur["database"] = body["database"]
        if "username" in body and body["username"]:
            cur["username"] = body["username"]
        if "file_path" in body and body["file_path"]:
            cur["file_path"] = body["file_path"]
        cur["provider"] = prov
        ds.connection_metadata = cur
        ds.provider = prov
        pwd = body.get("password")
        if pwd is not None and str(pwd) != "":
            from services.datasource_crypto import decrypt_secrets_blob

            row = (
                db.query(DatasourceCredential)
                .filter(DatasourceCredential.datasource_id == ds.id)
                .first()
            )
            prev: Dict[str, Any] = {}
            if row:
                try:
                    prev = decrypt_secrets_blob(row.payload_ciphertext)
                except Exception:
                    prev = {}
            prev["password"] = pwd
            blob = encrypt_secrets_blob(prev)
            if row:
                row.payload_ciphertext = blob
            else:
                db.add(DatasourceCredential(datasource_id=ds.id, payload_ciphertext=blob))
    elif dtype == "vector_store":
        cfg_in = body.get("connection_config")
        if isinstance(cfg_in, dict) and cfg_in:
            merged_cfg = {**(ds.connection_metadata or {}), **cfg_in}
            meta, secrets = split_vector_payload(
                {"provider": body.get("provider") or ds.provider, "connection_config": merged_cfg}
            )
            ds.connection_metadata = {**(ds.connection_metadata or {}), **meta}
            if body.get("provider"):
                ds.provider = body["provider"]
            if secrets:
                row = (
                    db.query(DatasourceCredential)
                    .filter(DatasourceCredential.datasource_id == ds.id)
                    .first()
                )
                from services.datasource_crypto import decrypt_secrets_blob

                prev = {}
                if row:
                    try:
                        prev = decrypt_secrets_blob(row.payload_ciphertext)
                    except Exception:
                        prev = {}
                prev.update(secrets)
                blob = encrypt_secrets_blob(prev)
                if row:
                    row.payload_ciphertext = blob
                else:
                    db.add(DatasourceCredential(datasource_id=ds.id, payload_ciphertext=blob))
        elif body.get("provider"):
            ds.provider = body["provider"]
    elif dtype == "document_corpus":
        pass

    db.add(ds)
    db.flush()
    try:
        if not (
            ds.datasource_type == "sql_database"
            and ds.credential_type == "secret_manager"
        ):
            raise_if_connection_failed(test_datasource_connection(db, ds.id))
        db.commit()
    except ValueError:
        db.rollback()
        raise
    db.refresh(ds)
    return ds


def soft_delete_datasource(db: Session, ds: Datasource) -> None:
    ds.is_active = False
    db.add(ds)
    db.commit()

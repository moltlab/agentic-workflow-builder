"""Minimal file ingest handlers for local compatibility."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from sqlalchemy.orm import Session

from db.models import Datasource


def ingest_files_into_corpus(
    db: Session, datasource_id: Any, files: Sequence[Tuple[str, bytes]]
) -> Dict[str, Any]:
    ds = db.query(Datasource).filter(Datasource.id == datasource_id, Datasource.is_active.is_(True)).first()
    if not ds:
        raise ValueError("Datasource not found")
    if ds.datasource_type != "document_corpus":
        raise ValueError("Ingest is only supported for document_corpus datasources")
    meta = dict(ds.connection_metadata or {})
    existing: List[Dict[str, Any]] = list(meta.get("corpus_files") or [])
    for name, content in files:
        existing.append({"name": name, "size": len(content)})
    meta["corpus_files"] = existing
    ds.connection_metadata = meta
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return {"status": "ok", "files_count": len(files), "corpus_files": existing}


def remove_corpus_file(db: Session, datasource_id: Any, source_basename: str) -> Dict[str, Any]:
    ds = db.query(Datasource).filter(Datasource.id == datasource_id, Datasource.is_active.is_(True)).first()
    if not ds:
        raise ValueError("Datasource not found")
    meta = dict(ds.connection_metadata or {})
    existing: List[Dict[str, Any]] = list(meta.get("corpus_files") or [])
    filtered = [f for f in existing if str(f.get("name")) != source_basename]
    removed = len(existing) - len(filtered)
    meta["corpus_files"] = filtered
    ds.connection_metadata = meta
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return {"status": "ok", "removed": removed, "corpus_files": filtered}


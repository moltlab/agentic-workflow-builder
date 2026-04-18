"""Datasource CRUD — framework owns storage and credentials (control plane)."""

from __future__ import annotations

import uuid as uuid_mod
from urllib.parse import unquote

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from db.config import get_db
from db.crud.datasource import (
    build_detail_payload,
    create_datasource_from_body,
    get_datasource,
    list_datasources_for_entity,
    serialize_datasource_for_get,
    soft_delete_datasource,
    update_datasource_from_body,
)
from services.datasource_ingest import ingest_files_into_corpus, remove_corpus_file
from services.datasource_runtime import test_datasource_connection, test_inline_connection
from utils.logging_utils import get_logger

logger = get_logger("datasource_routes")

datasource_router = APIRouter(tags=["datasources"])

_FRIENDLY = "Unable to complete this request right now. Please try again later or contact support."


def _can_modify(ds, user_id: str) -> bool:
    owner = ds.created_by_user_id
    if owner is None or str(owner).strip() == "":
        return True
    return str(owner) == str(user_id)


@datasource_router.post("/datasources")
async def create_datasource(
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        body = await request.json()
        ds = create_datasource_from_body(
            db,
            body,
            entity_id="default_entity",
            user_id="system",
        )
        return JSONResponse(content=serialize_datasource_for_get(ds), status_code=201)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        logger.error("create_datasource: %s", e)
        raise HTTPException(status_code=500, detail=_FRIENDLY) from e
    except Exception as e:
        logger.error("create_datasource failed: %s", e)
        raise HTTPException(status_code=500, detail=_FRIENDLY) from e


@datasource_router.get("/datasources")
async def list_datasources(
    db: Session = Depends(get_db),
):
    rows = list_datasources_for_entity(db, "default_entity")
    return JSONResponse(content=rows)


@datasource_router.get("/datasources/{datasource_id}")
async def get_datasource_by_id(
    datasource_id: str,
    db: Session = Depends(get_db),
):
    try:
        uid = uuid_mod.UUID(datasource_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found") from None
    ds = get_datasource(db, uid)
    if not ds or ds.entity_id != "default_entity":
        raise HTTPException(status_code=404, detail="Not found")
    return JSONResponse(content=serialize_datasource_for_get(ds))


@datasource_router.get("/datasources/{datasource_id}/detail")
async def get_datasource_detail(
    datasource_id: str,
    db: Session = Depends(get_db),
):
    try:
        uid = uuid_mod.UUID(datasource_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found") from None
    ds = get_datasource(db, uid)
    if not ds or ds.entity_id != "default_entity":
        raise HTTPException(status_code=404, detail="Not found")
    return JSONResponse(content=build_detail_payload(db, ds))


@datasource_router.put("/datasources/{datasource_id}")
async def update_datasource(
    datasource_id: str,
    request: Request,
    db: Session = Depends(get_db),
):
    try:
        uid = uuid_mod.UUID(datasource_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found") from None
    ds = get_datasource(db, uid)
    if not ds or ds.entity_id != "default_entity":
        raise HTTPException(status_code=404, detail="Not found")
    if not _can_modify(ds, "system"):
        raise HTTPException(status_code=403, detail="Only the datasource creator can update this datasource")
    body = await request.json()
    try:
        ds = update_datasource_from_body(db, ds, body)
        return JSONResponse(content=serialize_datasource_for_get(ds))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("update_datasource: %s", e)
        raise HTTPException(status_code=500, detail=_FRIENDLY) from e


@datasource_router.delete("/datasources/{datasource_id}")
async def delete_datasource(
    datasource_id: str,
    db: Session = Depends(get_db),
):
    try:
        uid = uuid_mod.UUID(datasource_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found") from None
    ds = get_datasource(db, uid)
    if not ds or ds.entity_id != "default_entity":
        raise HTTPException(status_code=404, detail="Not found")
    if not _can_modify(ds, "system"):
        raise HTTPException(status_code=403, detail="Only the datasource creator can delete this datasource")
    soft_delete_datasource(db, ds)
    return JSONResponse(content={"status": "ok"})


@datasource_router.delete("/datasources/{datasource_id}/files")
async def delete_datasource_corpus_file(
    datasource_id: str,
    source_basename: str,
    db: Session = Depends(get_db),
):
    try:
        uid = uuid_mod.UUID(datasource_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found") from None
    ds = get_datasource(db, uid)
    if not ds or ds.entity_id != "default_entity":
        raise HTTPException(status_code=404, detail="Not found")
    if not _can_modify(ds, "system"):
        raise HTTPException(status_code=403, detail="Forbidden")
    name = unquote(source_basename)
    try:
        out = remove_corpus_file(db, uid, name)
        return JSONResponse(content=out)
    except Exception as e:
        logger.error("remove corpus file: %s", e)
        raise HTTPException(status_code=500, detail=_FRIENDLY) from e


@datasource_router.post("/datasources/{datasource_id}/test")
async def test_datasource(
    datasource_id: str,
    db: Session = Depends(get_db),
):
    try:
        uid = uuid_mod.UUID(datasource_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found") from None
    ds = get_datasource(db, uid)
    if not ds or ds.entity_id != "default_entity":
        raise HTTPException(status_code=404, detail="Not found")
    out = test_datasource_connection(db, uid)
    status_code = 200 if out.get("status") == "ok" else 400
    return JSONResponse(content=out, status_code=status_code)


@datasource_router.post("/datasources/test-connection")
async def test_connection_inline(
    request: Request,
):
    try:
        body = await request.json()
        out = test_inline_connection(body)
    except Exception as e:
        logger.debug("test_connection_inline unexpected: %s: %s", type(e).__name__, e)
        return JSONResponse(
            content={
                "status": "error",
                "message": "Connection test could not be completed. Check your inputs and try again.",
            },
            status_code=400,
        )
    status_code = 200 if out.get("status") == "ok" else 400
    return JSONResponse(content=out, status_code=status_code)


@datasource_router.post("/datasources/{datasource_id}/ingest")
async def ingest_datasource_files(
    datasource_id: str,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    try:
        uid = uuid_mod.UUID(datasource_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Not found") from None
    ds = get_datasource(db, uid)
    if not ds or ds.entity_id != "default_entity":
        raise HTTPException(status_code=404, detail="Not found")
    if not _can_modify(ds, "system"):
        raise HTTPException(status_code=403, detail="Forbidden")
    if ds.datasource_type != "document_corpus":
        raise HTTPException(status_code=400, detail="Ingest is only for document_corpus datasources")
    pairs = []
    for f in files:
        content = await f.read()
        pairs.append((f.filename or "upload.bin", content))
    try:
        out = ingest_files_into_corpus(db, uid, pairs)
        return JSONResponse(content=out)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("ingest: %s", e)
        raise HTTPException(status_code=500, detail=_FRIENDLY) from e

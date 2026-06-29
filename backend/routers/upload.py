"""
Upload router — thin delegation layer.

Endpoints
---------
  POST   /upload                — upload and ingest a document
  GET    /upload                — list all uploaded documents
  DELETE /upload/{filename}     — delete all chunks for a file
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from db import get_supabase
from schemas import UploadResponse
from services.ingestion import ingest_document
from services.cache import cache
from services.security import check_rate_limit
from utils.logger import logger

router = APIRouter(prefix="/upload", tags=["Upload"])


class DeleteResponse(BaseModel):
    filename: str
    chunks_deleted: int
    message: str


# ── POST /upload ─────────────────────────────────────────────────────


@router.post("", response_model=UploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(..., description="PDF, TXT, or CSV file (max 20 MB)"),
    request: Request = None,
) -> UploadResponse:
    """
    Upload and ingest a document.

    **Pipeline**: validate → extract → sanitise → clean → split → deduplicate → embed → store.
    """
    check_rate_limit(request, "upload")

    file_bytes = await file.read()
    filename = file.filename or "unknown"
    content_type = file.content_type or "application/octet-stream"

    final_detail: dict = {}
    async for event in ingest_document(file_bytes, filename, content_type):
        if event.stage in ("validating", "splitting", "embedding", "done"):
            logger.info(f"[upload] {event.stage}: {event.message}")

        if event.detail:
            final_detail = event.detail

        if event.stage == "error":
            raise HTTPException(
                status_code=422 if "extract" in event.message.lower() else 502,
                detail=event.message,
            )

    if not final_detail:
        raise HTTPException(status_code=500, detail="Ingestion produced no result.")

    return UploadResponse(
        filename=filename,
        chunks_created=final_detail.get("chunks_created", 0),
        duplicates_skipped=final_detail.get("duplicates_skipped", 0),
        message=(
            f"Document ingested: {final_detail.get('chunks_created', 0)} chunks created, "
            f"{final_detail.get('duplicates_skipped', 0)} duplicates skipped."
        ),
    )


# ── DELETE /upload/{filename} ────────────────────────────────────────


@router.delete("/{filename:path}", response_model=DeleteResponse)
async def delete_document(filename: str, request: Request):
    """
    Delete all chunks belonging to a specific filename.
    Also clears the retrieval cache so old results aren't served.
    """
    check_rate_limit(request, "upload")

    client = get_supabase()

    count_result = (
        client.table("documents")
        .select("id", count="exact")
        .eq("filename", filename)
        .execute()
    )
    chunk_count = count_result.count or 0

    if chunk_count == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for '{filename}'.",
        )

    client.table("documents").delete().eq("filename", filename).execute()
    await cache.clear()

    logger.info(
        f"[upload] deleted '{filename}' — {chunk_count} chunks removed."
    )

    return DeleteResponse(
        filename=filename,
        chunks_deleted=chunk_count,
        message=f"Deleted {chunk_count} chunks from '{filename}'.",
    )


# ── GET /upload ──────────────────────────────────────────────────────


@router.get("", response_model=list[dict])
async def list_documents(request: Request):
    """List all unique document filenames with chunk counts."""
    check_rate_limit(request, "upload")

    client = get_supabase()
    result = (
        client.table("documents")
        .select("filename, created_at")
        .order("created_at", desc=True)
        .execute()
    )

    rows = result.data or []
    docs: dict[str, dict] = {}
    for row in rows:
        fn = row["filename"]
        if fn not in docs:
            docs[fn] = {
                "filename": fn,
                "chunks": 0,
                "uploaded_at": row.get("created_at", ""),
            }
        docs[fn]["chunks"] += 1

    return sorted(
        docs.values(), key=lambda d: d.get("uploaded_at", ""), reverse=True
    )

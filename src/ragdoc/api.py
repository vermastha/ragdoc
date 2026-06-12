"""REST API (optional extra: pip install "ragdoc[api]").

    uvicorn ragdoc.api:app --reload

Endpoints:
    POST /ingest   multipart file upload -> index it
    POST /ask      {"question": "..."}   -> answer + sources
    GET  /health   index stats

The pipeline lives in app.state and the index persists to disk after each
ingest, so restarts don't lose data. Single-process only by design — the
README discusses what changes for multi-worker deployments.
"""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from ragdoc.config import Settings
from ragdoc.ingestion.loader import SUPPORTED_EXTENSIONS
from ragdoc.pipeline import RagPipeline


class AskRequest(BaseModel):
    question: str


class SourceOut(BaseModel):
    source: str
    chunk_id: int
    text: str
    found_by: list[str]


class AskResponse(BaseModel):
    answer: str
    backend: str
    sources: list[SourceOut]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    pipeline = RagPipeline(settings=settings)
    try:
        pipeline.load_index()
    except FileNotFoundError:
        pass  # fresh start; index is created on first ingest
    app.state.pipeline = pipeline
    yield


app = FastAPI(title="ragdoc", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    pipeline: RagPipeline = app.state.pipeline
    return {"status": "ok", "indexed_chunks": len(pipeline.retriever)}


@app.post("/ingest")
async def ingest(file: UploadFile):
    pipeline: RagPipeline = app.state.pipeline
    suffix = Path(file.filename or "upload.txt").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type {suffix!r}")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        count = pipeline.ingest([tmp_path])
        pipeline.save_index()
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"indexed_chunks": count, "total_chunks": len(pipeline.retriever)}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    pipeline: RagPipeline = app.state.pipeline
    if len(pipeline.retriever) == 0:
        raise HTTPException(409, "Index is empty — ingest documents first.")
    answer = pipeline.ask(request.question)
    return AskResponse(
        answer=answer.text,
        backend=answer.backend,
        sources=[
            SourceOut(
                source=r.chunk.source,
                chunk_id=r.chunk.chunk_id,
                text=r.chunk.text,
                found_by=list(r.found_by),
            )
            for r in answer.sources
        ],
    )

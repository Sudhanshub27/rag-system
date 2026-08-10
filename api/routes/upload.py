import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile
from sse_starlette.sse import EventSourceResponse

from api.deps import get_pipeline, get_tenant_id

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload")
async def upload_files(
    request: Request,
    response: Response,
    files: list[UploadFile] = File(...),
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Upload document files and run the ingestion pipeline.
    Yields real-time Server-Sent Events (SSE) detailing multi-stage progress
    (receiving → chunking → embedding → indexing → complete).
    """
    pipeline = get_pipeline(tenant_id)
    upload_dir = Path(f"./tmp_uploads/{tenant_id}")
    upload_dir.mkdir(parents=True, exist_ok=True)

    async def event_generator():
        for file in files:
            file_path = upload_dir / file.filename
            contents = await file.read()
            file_path.write_bytes(contents)

            # Stage 1: Receiving
            yield {
                "event": "progress",
                "data": json.dumps(
                    {
                        "filename": file.filename,
                        "stage": "chunking",
                        "message": f"Extracting text & splitting {file.filename} into semantic chunks...",
                        "progress": 25,
                    }
                ),
            }
            await asyncio.sleep(0.2)

            # Stage 2: Chunking & Ingestion
            try:
                # Load docs
                docs = pipeline._ingestion.ingest(str(file_path), tenant_id=tenant_id)
                yield {
                    "event": "progress",
                    "data": json.dumps(
                        {
                            "filename": file.filename,
                            "stage": "embedding",
                            "message": f"Generating vector embeddings for {len(docs)} page document...",
                            "progress": 60,
                        }
                    ),
                }
                await asyncio.sleep(0.2)

                # Chunk & embed
                chunks = pipeline._chunker.chunk(docs, tenant_id=tenant_id)
                embeddings = pipeline._embedder.embed_chunks(chunks)

                yield {
                    "event": "progress",
                    "data": json.dumps(
                        {
                            "filename": file.filename,
                            "stage": "indexing",
                            "message": f"Storing {len(chunks)} chunks in ChromaDB vector store...",
                            "progress": 85,
                        }
                    ),
                }
                await asyncio.sleep(0.2)

                # Store & index
                pipeline._vector_store.add_chunks(chunks, embeddings)
                pipeline._all_chunks.extend(chunks)
                pipeline._bm25.build(pipeline._all_chunks)

                yield {
                    "event": "complete",
                    "data": json.dumps(
                        {
                            "filename": file.filename,
                            "stage": "complete",
                            "chunks_added": len(chunks),
                            "message": f"Successfully indexed {len(chunks)} chunks from {file.filename}.",
                            "progress": 100,
                        }
                    ),
                }
            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps(
                        {
                            "filename": file.filename,
                            "error": str(e),
                            "message": f"Failed to ingest {file.filename}: {str(e)}",
                        }
                    ),
                }

    return EventSourceResponse(event_generator())

import asyncio
import json

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from api.deps import check_rate_limit, get_pipeline, get_tenant_id

router = APIRouter(prefix="/api", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    use_hyde: bool = False
    use_multi_query: bool = False


@router.post("/query")
async def query_pipeline(
    req: QueryRequest,
    request: Request,
    response: Response,
    tenant_id: str = Depends(get_tenant_id),
):
    """
    Query the RAG pipeline. Streams the generated answer token-by-token via SSE,
    followed by a final event containing citations, source chunks, and Self-RAG scores.
    """
    check_rate_limit(tenant_id)
    pipeline = get_pipeline(tenant_id)

    async def event_generator():
        try:
            # 1. Execute query on existing RAG pipeline facade
            rag_res = pipeline.query(
                question=req.question,
                use_hyde=req.use_hyde,
                use_multi_query=req.use_multi_query,
            )

            # 2. Stream answer text in realistic token chunks
            full_text = rag_res.answer
            words = full_text.split(" ")

            for i, word in enumerate(words):
                chunk_text = word + (" " if i < len(words) - 1 else "")
                yield {
                    "event": "token",
                    "data": json.dumps({"token": chunk_text}),
                }
                await asyncio.sleep(0.02)  # Smooth typewriter streaming effect

            # 3. Final metadata event with citations & source chunks
            citations_data = []
            for idx, rc in enumerate(rag_res.retrieved_chunks):
                citations_data.append(
                    {
                        "id": idx + 1,
                        "chunk_id": rc.chunk.chunk_id,
                        "source": rc.chunk.source,
                        "page": rc.chunk.page,
                        "text": rc.chunk.text,
                        "score": round(rc.score, 4),
                    }
                )

            yield {
                "event": "final",
                "data": json.dumps(
                    {
                        "answer": rag_res.answer,
                        "citations": rag_res.citations,
                        "retrieved_chunks": citations_data,
                        "is_fallback": rag_res.is_fallback,
                        "faithfulness_score": rag_res.faithfulness_score,
                        "relevance_score": rag_res.relevance_score,
                    }
                ),
            }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(event_generator())

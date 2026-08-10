import time
import uuid
from typing import Annotated

from fastapi import HTTPException, Request, Response, status

from pipeline import RAGPipeline

# In-memory pipeline instances per tenant_id
_tenant_pipelines: dict[str, RAGPipeline] = {}

COOKIE_NAME = "tenant_id"
COOKIE_MAX_AGE = 60 * 60 * 24 * 730  # 2 years


def get_tenant_id(request: Request, response: Response) -> str:
    """
    Retrieve anonymous tenant_id from HttpOnly cookie.
    If no cookie exists, generate a new UUID4 tenant_id and set the HttpOnly cookie.
    """
    tenant_id = request.cookies.get(COOKIE_NAME)
    if not tenant_id:
        tenant_id = str(uuid.uuid4())
        response.set_cookie(
            key=COOKIE_NAME,
            value=tenant_id,
            max_age=COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=False,  # Set to True in production HTTPS
        )
    return tenant_id


def get_pipeline(tenant_id: Annotated[str, get_tenant_id]) -> RAGPipeline:
    """
    Dependency returning an initialized RAGPipeline bound to the tenant_id.
    """
    if tenant_id not in _tenant_pipelines:
        _tenant_pipelines[tenant_id] = RAGPipeline(tenant_id=tenant_id)
    return _tenant_pipelines[tenant_id]


# Basic rate limiter: 20 queries/hour per tenant

_rate_limit_store: dict[str, list[float]] = {}
RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 3600


def check_rate_limit(tenant_id: str):
    now = time.time()
    history = _rate_limit_store.setdefault(tenant_id, [])
    # Filter out requests older than window
    history = [t for t in history if now - t < RATE_LIMIT_WINDOW_SECONDS]
    _rate_limit_store[tenant_id] = history

    if len(history) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Maximum {RATE_LIMIT_MAX_REQUESTS} requests per hour allowed. Please try again later.",
            },
        )
    history.append(now)

from fastapi import APIRouter, Depends, HTTPException, status

from api.deps import get_pipeline, get_tenant_id

router = APIRouter(prefix="/api", tags=["documents"])


@router.get("/documents")
async def list_documents(tenant_id: str = Depends(get_tenant_id)):
    """
    List all ingested documents and chunk counts for the current tenant.
    """
    pipeline = get_pipeline(tenant_id)
    all_chunks = pipeline.get_all_chunks()

    doc_map: dict[str, dict] = {}
    for c in all_chunks:
        src = c.source
        if src not in doc_map:
            doc_map[src] = {"filename": src, "chunk_count": 0}
        doc_map[src]["chunk_count"] += 1

    return {"tenant_id": tenant_id, "documents": list(doc_map.values())}


@router.delete("/documents/{filename:path}")
async def delete_document(filename: str, tenant_id: str = Depends(get_tenant_id)):
    """
    Delete a specific document and all its indexed vector chunks.
    """
    pipeline = get_pipeline(tenant_id)
    deleted_count = pipeline.delete_document(filename)
    if deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{filename}' not found for tenant '{tenant_id}'",
        )
    return {
        "status": "success",
        "message": f"Deleted {deleted_count} chunks for '{filename}'",
        "deleted_chunks": deleted_count,
    }

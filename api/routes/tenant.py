from fastapi import APIRouter, Depends

from api.deps import get_pipeline, get_tenant_id

router = APIRouter(prefix="/api", tags=["tenant"])


@router.get("/stats")
async def get_stats(tenant_id: str = Depends(get_tenant_id)):
    """
    Get vector store and knowledge base statistics for the current tenant.
    """
    pipeline = get_pipeline(tenant_id)
    stats = pipeline.get_stats()
    all_chunks = pipeline.get_all_chunks()
    unique_sources = len(set(c.source for c in all_chunks))

    return {
        "tenant_id": tenant_id,
        "total_chunks": stats.get("total_chunks_in_vector_store", 0),
        "total_documents": unique_sources,
        "embedding_model": stats.get("embedding_model", ""),
    }


@router.delete("/tenant")
async def wipe_tenant_data(current_tenant: str = Depends(get_tenant_id)):
    """
    Permanently purge all uploaded files, vector collections, and indices for the tenant.
    """
    pipeline = get_pipeline(current_tenant)
    pipeline.delete_all_tenant_data()

    return {
        "status": "success",
        "message": f"All data for tenant '{current_tenant}' has been permanently purged.",
    }

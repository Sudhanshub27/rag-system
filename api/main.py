import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import documents, followup, query, tenant, upload

app = FastAPI(
    title="Ask My Documents RAG API",
    description="Production-grade Multi-Tenant RAG Backend API with SSE Streaming",
    version="2.0.0",
)

# CORS Configuration for local React (Vite) dev server & Vercel deployments
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler to ensure JSON error payloads across all endpoints
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error", "message": str(exc)},
    )


# Include API Routers
app.include_router(upload.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(tenant.router)
app.include_router(followup.router)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Ask My Documents RAG API"}


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

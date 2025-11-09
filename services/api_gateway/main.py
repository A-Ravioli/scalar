"""API Gateway FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from libs.common.logging import setup_logging

from routers import jobs, endpoints, auth, orderbook

logger = setup_logging("api_gateway")

app = FastAPI(
    title="Scalar API Gateway",
    description="Serverless compute platform API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(endpoints.router, prefix="/endpoints", tags=["endpoints"])
app.include_router(orderbook.router, prefix="/orderbook", tags=["orderbook"])


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


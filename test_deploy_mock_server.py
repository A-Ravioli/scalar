"""Simple mock server for testing deployment with demo container."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

app = FastAPI(title="Mock Deployment API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DeployRequest(BaseModel):
    name: str
    tier: str
    gpu_count: int
    vram_per_gpu_gb: int
    cpu_cores: int
    ram_gb: int
    image: str
    command: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/jobs")
async def create_job(request: DeployRequest):
    """Mock job creation endpoint."""
    return {
        "id": "job-demo-12345",
        "name": request.name,
        "tier": request.tier,
        "status": "pending",
        "gpu_count": request.gpu_count,
        "image": request.image,
        "command": request.command or [],
        "env": request.env or {},
        "created_at": "2025-11-09T12:00:00Z",
        "message": "Demo container deployment submitted successfully!"
    }


@app.get("/jobs")
async def list_jobs():
    """Mock job listing endpoint."""
    return [
        {
            "id": "job-demo-12345",
            "name": "demo-pytorch-test",
            "tier": "FAST",
            "status": "running",
            "gpu_count": 1,
            "image": "demo/pytorch-gpu:latest",
            "created_at": "2025-11-09T12:00:00Z"
        }
    ]


if __name__ == "__main__":
    print("🚀 Starting Mock Deployment API on http://localhost:8000")
    print("   - Health: http://localhost:8000/health")
    print("   - Deploy: POST http://localhost:8000/jobs")
    uvicorn.run(app, host="0.0.0.0", port=8000)


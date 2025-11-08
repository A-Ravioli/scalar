"""Runtime Agent FastAPI application."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import docker
from libs.common.logging import setup_logging

logger = setup_logging("runtime_agent")

app = FastAPI(
    title="Scalar Runtime Agent",
    description="Docker-based runtime for job execution",
    version="1.0.0",
)

# Docker client
docker_client = docker.from_env()


class PlaceRequest(BaseModel):
    """Job placement request."""

    job_id: str
    node_id: str
    gpu_indices: List[int]
    image: str
    command: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    gpu_count: int
    cpu_cores: int
    ram_gb: float


@app.post("/place")
async def place_job(request: PlaceRequest):
    """Place a job container."""
    try:
        # Build device requests for GPUs
        device_requests = []
        for gpu_idx in request.gpu_indices:
            device_requests.append(
                docker.types.DeviceRequest(
                    device_ids=[str(gpu_idx)],
                    capabilities=[["gpu"]],
                )
            )

        # Create container
        container = docker_client.containers.run(
            image=request.image,
            command=request.command,
            environment=request.env or {},
            device_requests=device_requests,
            mem_limit=f"{int(request.ram_gb * 1024 * 1024 * 1024)}b",
            cpu_count=request.cpu_cores,
            detach=True,
            name=f"scalar-job-{request.job_id}",
        )

        logger.info(f"Placed job {request.job_id} in container {container.id}")

        return {
            "status": "placed",
            "container_id": container.id,
            "job_id": request.job_id,
        }

    except Exception as e:
        logger.error(f"Failed to place job {request.job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get job container status."""
    try:
        container_name = f"scalar-job-{job_id}"
        container = docker_client.containers.get(container_name)

        return {
            "status": container.status,
            "exit_code": container.attrs.get("State", {}).get("ExitCode"),
        }

    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail="Container not found")


@app.get("/jobs/{job_id}/logs")
async def get_job_logs(job_id: str, tail: int = 100):
    """Get job container logs."""
    try:
        container_name = f"scalar-job-{job_id}"
        container = docker_client.containers.get(container_name)
        logs = container.logs(tail=tail).decode("utf-8")

        return {"logs": logs}

    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail="Container not found")


@app.delete("/jobs/{job_id}")
async def stop_job(job_id: str):
    """Stop and remove a job container."""
    try:
        container_name = f"scalar-job-{job_id}"
        container = docker_client.containers.get(container_name)
        container.stop()
        container.remove()

        return {"status": "stopped"}

    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail="Container not found")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)


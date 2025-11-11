"""Job management routes."""

import json
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
import redis
from libs.common.config import config
from libs.common.db import get_supabase_client
from libs.common.logging import get_logger
from libs.common.types import JobStatus, Tier
from .auth import verify_api_key

logger = get_logger(__name__)
router = APIRouter()


class JobCreate(BaseModel):
    """Job creation request."""

    name: str
    tier: Tier
    gpu_count: int
    vram_per_gpu_gb: float
    cpu_cores: int
    ram_gb: float
    expected_duration_sec: int = 3600
    priority: int = 0
    image: str
    command: Optional[List[str]] = None
    env: Optional[dict] = None


class JobResponse(BaseModel):
    """Job response model."""

    id: str
    user_id: str
    tier: str
    status: str
    gpu_count: int
    vram_per_gpu_gb: float
    cpu_cores: int
    ram_gb: float
    image: str
    created_at: str
    updated_at: str


def get_redis_client():
    """Get Redis client for job queue."""
    if config.redis_url:
        return redis.from_url(config.redis_url)
    return None


@router.post("", response_model=JobResponse)
async def create_job(
    job: JobCreate,
    user_id: UUID = Depends(verify_api_key),
):
    """Submit a new job."""
    job_id = uuid4()
    now = datetime.utcnow()

    supabase = get_supabase_client()
    result = supabase.table("jobs").insert({
        "id": str(job_id),
        "user_id": str(user_id),
        "name": job.name,
        "tier": job.tier.value,
        "gpu_count": job.gpu_count,
        "vram_per_gpu_gb": job.vram_per_gpu_gb,
        "cpu_cores": job.cpu_cores,
        "ram_gb": job.ram_gb,
        "expected_duration_sec": job.expected_duration_sec,
        "priority": job.priority,
        "status": JobStatus.PENDING.value,
        "image": job.image,
        "command": json.dumps(job.command) if job.command else None,
        "env": json.dumps(job.env) if job.env else None,
    }).execute()

    # Publish to job queue
    redis_client = get_redis_client()
    if redis_client:
        redis_client.lpush("job_queue", str(job_id))
    else:
        # Fallback: mark job as queued (scheduler will poll)
        supabase.table("jobs").update({
            "status": JobStatus.QUEUED.value,
        }).eq("id", str(job_id)).execute()

    logger.info(f"Created job {job_id} for user {user_id}")

    return JobResponse(**result.data[0])


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: UUID,
    user_id: UUID = Depends(verify_api_key),
):
    """Get job status."""
    supabase = get_supabase_client()
    result = supabase.table("jobs").select("*").eq("id", str(job_id)).eq("user_id", str(user_id)).single().execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(**result.data)


@router.get("", response_model=List[JobResponse])
async def list_jobs(
    status: Optional[str] = Query(None),
    tier: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    user_id: UUID = Depends(verify_api_key),
):
    """List user's jobs."""
    supabase = get_supabase_client()
    query = supabase.table("jobs").select("*").eq("user_id", str(user_id)).limit(limit)

    if status:
        query = query.eq("status", status)
    if tier:
        query = query.eq("tier", tier)

    result = query.order("created_at", desc=True).execute()
    return [JobResponse(**job) for job in result.data]


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: UUID,
    user_id: UUID = Depends(verify_api_key),
):
    """Cancel a job."""
    supabase = get_supabase_client()
    result = supabase.table("jobs").update({
        "status": JobStatus.CANCELLED.value,
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("id", str(job_id)).eq("user_id", str(user_id)).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"status": "cancelled"}


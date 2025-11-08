"""Capacity Manager FastAPI endpoints."""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from libs.common.db import get_supabase_client
from libs.common.logging import get_logger
from libs.common.types import CapacitySnapshot, Reservation, Tier
from .capacity_state import CapacityState

logger = get_logger(__name__)
router = APIRouter()

# Global capacity state (would be injected in production)
capacity_state = CapacityState()


class ReserveRequest(BaseModel):
    """Reservation request."""

    job_id: str
    node_id: str
    gpu_indices: List[int]
    expires_in_sec: int = 300


class ReleaseRequest(BaseModel):
    """Release request."""

    reservation_id: str


@router.get("/capacity_snapshot", response_model=CapacitySnapshot)
async def get_capacity_snapshot(tier: Optional[str] = None):
    """Get current capacity snapshot."""
    await capacity_state.sync_from_db()
    tier_enum = Tier(tier) if tier else None
    return capacity_state.get_capacity_snapshot(tier_enum)


@router.post("/reserve", response_model=Reservation)
async def reserve(request: ReserveRequest):
    """Reserve capacity for a job."""
    # Verify node exists and has capacity
    await capacity_state.sync_from_db()
    snapshot = capacity_state.get_capacity_snapshot()
    node = capacity_state.nodes.get(UUID(request.node_id))

    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    # Verify GPUs are available
    for gpu_idx in request.gpu_indices:
        if gpu_idx >= len(node.gpu_slots):
            raise HTTPException(status_code=400, detail=f"Invalid GPU index {gpu_idx}")
        gpu = node.gpu_slots[gpu_idx]
        if gpu.job_id is not None:
            raise HTTPException(status_code=409, detail=f"GPU {gpu_idx} already allocated")

    # Create reservation
    reservation_id = uuid4()
    expires_at = datetime.utcnow() + timedelta(seconds=request.expires_in_sec)

    supabase = get_supabase_client()
    result = supabase.table("reservations").insert({
        "id": str(reservation_id),
        "job_id": request.job_id,
        "node_id": request.node_id,
        "gpu_indices": request.gpu_indices,
        "expires_at": expires_at.isoformat(),
    }).execute()

    return Reservation(
        id=reservation_id,
        job_id=UUID(request.job_id),
        node_id=UUID(request.node_id),
        gpu_indices=request.gpu_indices,
        expires_at=expires_at,
        created_at=datetime.utcnow(),
    )


@router.post("/release")
async def release(request: ReleaseRequest):
    """Release a reservation."""
    supabase = get_supabase_client()
    supabase.table("reservations").delete().eq("id", request.reservation_id).execute()
    return {"status": "released"}


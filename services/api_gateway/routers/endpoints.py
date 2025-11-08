"""Serverless endpoint routes."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from libs.common.db import get_supabase_client
from libs.common.logging import get_logger
from routers.auth import verify_api_key

logger = get_logger(__name__)
router = APIRouter()


class EndpointCreate(BaseModel):
    """Endpoint creation request."""

    name: str
    image: str
    gpu_count: int = 1
    vram_per_gpu_gb: float = 40.0
    cpu_cores: int = 4
    ram_gb: float = 16.0
    env: Optional[dict] = None


class EndpointResponse(BaseModel):
    """Endpoint response model."""

    id: str
    name: str
    url: str
    status: str


@router.post("", response_model=EndpointResponse)
async def create_endpoint(
    endpoint: EndpointCreate,
    user_id: UUID = Depends(verify_api_key),
):
    """Create a serverless endpoint."""
    # TODO: Implement endpoint creation
    # This would create a persistent job that stays running
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.get("", response_model=List[EndpointResponse])
async def list_endpoints(user_id: UUID = Depends(verify_api_key)):
    """List user's endpoints."""
    # TODO: Implement endpoint listing
    return []


@router.delete("/{endpoint_id}")
async def delete_endpoint(
    endpoint_id: UUID,
    user_id: UUID = Depends(verify_api_key),
):
    """Delete an endpoint."""
    # TODO: Implement endpoint deletion
    raise HTTPException(status_code=501, detail="Not implemented yet")


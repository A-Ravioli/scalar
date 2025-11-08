"""Client for communicating with Capacity Manager."""

import httpx
from typing import List
from libs.common.types import Node, CapacitySnapshot, Reservation
from libs.common.config import config
from libs.common.logging import get_logger

logger = get_logger(__name__)


class CapacityClient:
    """Client for Capacity Manager API."""

    def __init__(self):
        self.base_url = config.capacity_manager_url

    async def get_capacity_snapshot(self, tier: str = None) -> CapacitySnapshot:
        """Get current capacity snapshot."""
        async with httpx.AsyncClient() as client:
            url = f"{self.base_url}/capacity_snapshot"
            params = {"tier": tier} if tier else {}
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return CapacitySnapshot(**data)

    async def reserve(
        self, job_id: str, node_id: str, gpu_indices: List[int], expires_in_sec: int = 300
    ) -> Reservation:
        """Reserve capacity for a job."""
        async with httpx.AsyncClient() as client:
            url = f"{self.base_url}/reserve"
            response = await client.post(
                url,
                json={
                    "job_id": job_id,
                    "node_id": node_id,
                    "gpu_indices": gpu_indices,
                    "expires_in_sec": expires_in_sec,
                },
            )
            response.raise_for_status()
            data = response.json()
            return Reservation(**data)

    async def release(self, reservation_id: str):
        """Release a reservation."""
        async with httpx.AsyncClient() as client:
            url = f"{self.base_url}/release"
            response = await client.post(url, json={"reservation_id": reservation_id})
            response.raise_for_status()


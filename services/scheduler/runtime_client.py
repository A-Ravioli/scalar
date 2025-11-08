"""Client for communicating with Runtime Agents."""

import httpx
from typing import List, Optional
from libs.common.types import Job
from libs.common.logging import get_logger

logger = get_logger(__name__)


class RuntimeClient:
    """Client for Runtime Agent API."""

    def __init__(self, runtime_agent_url: Optional[str] = None):
        self.runtime_agent_url = runtime_agent_url

    async def place_job(
        self, job: Job, node_id: str, gpu_indices: List[int]
    ) -> dict:
        """
        Place a job on a node.

        Returns placement result with container_id, etc.
        """
        if not self.runtime_agent_url:
            # Fallback: direct database update
            logger.warning("No runtime agent URL configured, skipping placement")
            return {"status": "skipped"}

        async with httpx.AsyncClient() as client:
            url = f"{self.runtime_agent_url}/place"
            response = await client.post(
                url,
                json={
                    "job_id": str(job.id),
                    "node_id": node_id,
                    "gpu_indices": gpu_indices,
                    "image": job.image,
                    "command": job.command,
                    "env": job.env,
                    "gpu_count": job.gpu_count,
                    "cpu_cores": job.cpu_cores,
                    "ram_gb": job.ram_gb,
                },
            )
            response.raise_for_status()
            return response.json()


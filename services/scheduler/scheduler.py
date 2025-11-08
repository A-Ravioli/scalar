"""Core scheduler loop."""

import asyncio
import json
from datetime import datetime
from typing import Optional
from uuid import UUID
import redis
from libs.common.config import config
from libs.common.db import get_supabase_client
from libs.common.logging import get_logger
from libs.common.types import Job, JobStatus, PlacementDecision
from .bin_packer import schedule_job
from .capacity_client import CapacityClient
from .runtime_client import RuntimeClient

logger = get_logger(__name__)


class Scheduler:
    """Job scheduler with bin-packing."""

    def __init__(self):
        self.capacity_client = CapacityClient()
        self.runtime_client = RuntimeClient(config.runtime_agent_url)
        self.supabase = get_supabase_client()
        self.running = False

    def get_redis_client(self):
        """Get Redis client."""
        if config.redis_url:
            return redis.from_url(config.redis_url)
        return None

    async def load_job(self, job_id: UUID) -> Optional[Job]:
        """Load job from database."""
        result = (
            self.supabase.table("jobs")
            .select("*")
            .eq("id", str(job_id))
            .single()
            .execute()
        )

        if not result.data:
            return None

        data = result.data
        return Job(
            id=UUID(data["id"]),
            user_id=UUID(data["user_id"]),
            tier=data["tier"],
            gpu_count=data["gpu_count"],
            vram_per_gpu_gb=data["vram_per_gpu_gb"],
            cpu_cores=data["cpu_cores"],
            ram_gb=data["ram_gb"],
            expected_duration_sec=data["expected_duration_sec"],
            priority=data["priority"],
            status=data["status"],
            image=data["image"],
            command=json.loads(data["command"]) if data.get("command") else None,
            env=json.loads(data["env"]) if data.get("env") else None,
            node_id=UUID(data["node_id"]) if data.get("node_id") else None,
            gpu_indices=data.get("gpu_indices"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    async def process_job(self, job_id: UUID):
        """Process a single job."""
        logger.info(f"Processing job {job_id}")

        # Load job
        job = await self.load_job(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return

        # Skip if already scheduled/running
        if job.status in [JobStatus.SCHEDULED, JobStatus.RUNNING]:
            return

        # Get capacity snapshot
        try:
            snapshot = await self.capacity_client.get_capacity_snapshot(job.tier.value)
        except Exception as e:
            logger.error(f"Failed to get capacity snapshot: {e}")
            return

        # Run bin-packer
        decision = schedule_job(job, snapshot.nodes)

        if decision.kind == "EXISTING_NODE":
            # Reserve capacity
            try:
                reservation = await self.capacity_client.reserve(
                    str(job.id),
                    str(decision.node_id),
                    decision.gpu_indices,
                )
            except Exception as e:
                logger.error(f"Failed to reserve capacity: {e}")
                # Update job status to queued for retry
                self.supabase.table("jobs").update({
                    "status": JobStatus.QUEUED.value,
                }).eq("id", str(job.id)).execute()
                return

            # Place job on runtime
            try:
                placement_result = await self.runtime_client.place_job(
                    job, str(decision.node_id), decision.gpu_indices
                )
            except Exception as e:
                logger.error(f"Failed to place job: {e}")
                # Release reservation
                await self.capacity_client.release(str(reservation.id))
                return

            # Update job status
            self.supabase.table("jobs").update({
                "status": JobStatus.SCHEDULED.value,
                "node_id": str(decision.node_id),
                "gpu_indices": decision.gpu_indices,
                "updated_at": datetime.utcnow().isoformat(),
            }).eq("id", str(job.id)).execute()

            logger.info(f"Successfully scheduled job {job_id} on node {decision.node_id}")

        elif decision.kind == "REQUEST_MORE_CAPACITY":
            # Request more capacity from capacity manager
            logger.info(f"Requesting more capacity for FAST tier job {job_id}")
            # Capacity manager will handle this via autoscaler
            self.supabase.table("jobs").update({
                "status": JobStatus.QUEUED.value,
            }).eq("id", str(job.id)).execute()

        elif decision.kind == "QUEUE_FOR_FLEX":
            # Queue for Flex tier
            logger.info(f"Queueing FLEX tier job {job_id}")
            self.supabase.table("jobs").update({
                "status": JobStatus.QUEUED.value,
            }).eq("id", str(job.id)).execute()

    async def run(self):
        """Main scheduler loop."""
        self.running = True
        logger.info("Scheduler started")

        redis_client = self.get_redis_client()

        while self.running:
            try:
                # Get next job from queue
                if redis_client:
                    job_id_str = redis_client.brpop("job_queue", timeout=5)
                    if job_id_str:
                        job_id = UUID(job_id_str[1].decode())
                        await self.process_job(job_id)
                else:
                    # Fallback: poll database for pending jobs
                    result = (
                        self.supabase.table("jobs")
                        .select("id")
                        .eq("status", JobStatus.PENDING.value)
                        .order("priority", desc=True)
                        .order("created_at")
                        .limit(10)
                        .execute()
                    )

                    for job_data in result.data:
                        job_id = UUID(job_data["id"])
                        await self.process_job(job_id)

                    await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    def stop(self):
        """Stop scheduler."""
        self.running = False


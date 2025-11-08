"""Shared Pydantic models for Scalar services."""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class Tier(str, Enum):
    """Job tier."""

    FAST = "FAST"
    FLEX = "FLEX"


class BlockStatus(str, Enum):
    """Block status."""

    PENDING = "pending"
    ACTIVE = "active"
    SELLING = "selling"
    SOLD = "sold"


class JobStatus(str, Enum):
    """Job status."""

    PENDING = "pending"
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Block(BaseModel):
    """SFCompute block contract."""

    id: UUID
    provider: str = "sfcompute"
    instance_type: str = Field(..., description="e.g. '8xH100'")
    gpus_per_node: int
    vram_per_gpu_gb: float
    cpu_per_node: int
    ram_per_node_gb: float
    start_time: datetime
    end_time: datetime
    cost_per_hour: float
    status: BlockStatus
    tier: Optional[Tier] = None  # Which tier this block serves
    created_at: datetime
    updated_at: datetime


class GPUAssignment(BaseModel):
    """GPU slot assignment on a node."""

    index: int = Field(..., description="GPU index on the node")
    vram_total_gb: float
    vram_used_gb: float = 0.0
    job_id: Optional[UUID] = None


class Node(BaseModel):
    """Compute node spawned from a block."""

    id: UUID
    block_id: UUID
    gpu_slots: List[GPUAssignment]
    cpu_used: float = 0.0
    ram_used_gb: float = 0.0
    tier: Tier
    created_at: datetime
    updated_at: datetime

    def free_cpu(self) -> float:
        """Get free CPU cores."""
        block = None  # Would be loaded from DB
        # For now, assume 64 cores per node
        return 64.0 - self.cpu_used

    def free_ram_gb(self) -> float:
        """Get free RAM in GB."""
        block = None  # Would be loaded from DB
        # For now, assume 512 GB per node
        return 512.0 - self.ram_used_gb

    def time_to_expiry(self) -> float:
        """Get seconds until expiry."""
        block = None  # Would be loaded from DB
        # Would calculate from block.end_time
        return 3600.0  # Placeholder


class Job(BaseModel):
    """Job resource requirements."""

    id: UUID
    user_id: UUID
    tier: Tier
    gpu_count: int
    vram_per_gpu_gb: float
    cpu_cores: int
    ram_gb: float
    expected_duration_sec: int = Field(
        default=3600, description="User hint or estimate"
    )
    priority: int = Field(default=0, description="Higher = more priority")
    status: JobStatus = JobStatus.PENDING
    image: str
    command: Optional[List[str]] = None
    env: Optional[dict] = None
    node_id: Optional[UUID] = None
    gpu_indices: Optional[List[int]] = None
    created_at: datetime
    updated_at: datetime


class CapacitySnapshot(BaseModel):
    """Snapshot of available capacity."""

    nodes: List[Node]
    timestamp: datetime


class PlacementDecision(BaseModel):
    """Scheduler placement decision."""

    kind: str = Field(
        ...,
        description="EXISTING_NODE, REQUEST_MORE_CAPACITY, QUEUE_FOR_FLEX",
    )
    node_id: Optional[UUID] = None
    gpu_indices: Optional[List[int]] = None
    tier: Optional[Tier] = None
    job_id: UUID


class Reservation(BaseModel):
    """Capacity reservation."""

    id: UUID
    job_id: UUID
    node_id: UUID
    gpu_indices: List[int]
    expires_at: datetime
    created_at: datetime


class UsageEvent(BaseModel):
    """Usage event for billing."""

    id: UUID
    job_id: UUID
    user_id: UUID
    node_id: UUID
    gpu_hours: float
    cpu_hours: float
    start_time: datetime
    end_time: datetime
    cost_usd: float
    created_at: datetime


"""Tests for tier selection logic - FAST vs FLEX from idea.md."""

import pytest
from datetime import datetime
from uuid import uuid4

from libs.common.types import Job, Tier, JobStatus
from services.scheduler.bin_packer import schedule_job


class TestTierSelection:
    """Test tier selection and separation from idea.md section 1."""

    def test_fast_tier_job_only_schedules_on_fast_nodes(self):
        """Test that FAST tier jobs only schedule on FAST tier nodes."""
        from libs.common.types import Node, Block, GPUAssignment, BlockStatus
        from datetime import timedelta

        now = datetime.utcnow()
        
        # Create FAST block and node
        fast_block = Block(
            id=uuid4(),
            instance_type="8xH100",
            gpus_per_node=8,
            vram_per_gpu_gb=80.0,
            cpu_per_node=64,
            ram_per_node_gb=512.0,
            start_time=now,
            end_time=now + timedelta(hours=24),
            cost_per_hour=10.0,
            status=BlockStatus.ACTIVE,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )
        
        fast_gpus = [
            GPUAssignment(index=i, vram_total_gb=80.0, vram_used_gb=0.0)
            for i in range(8)
        ]
        fast_node = Node(
            id=uuid4(),
            block_id=fast_block.id,
            gpu_slots=fast_gpus,
            cpu_used=0.0,
            ram_used_gb=0.0,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )
        fast_node.set_block(fast_block)

        # Create FLEX block and node
        flex_block = Block(
            id=uuid4(),
            instance_type="8xH100",
            gpus_per_node=8,
            vram_per_gpu_gb=80.0,
            cpu_per_node=64,
            ram_per_node_gb=512.0,
            start_time=now,
            end_time=now + timedelta(hours=24),
            cost_per_hour=8.0,
            status=BlockStatus.ACTIVE,
            tier=Tier.FLEX,
            created_at=now,
            updated_at=now,
        )
        
        flex_gpus = [
            GPUAssignment(index=i, vram_total_gb=80.0, vram_used_gb=0.0)
            for i in range(8)
        ]
        flex_node = Node(
            id=uuid4(),
            block_id=flex_block.id,
            gpu_slots=flex_gpus,
            cpu_used=0.0,
            ram_used_gb=0.0,
            tier=Tier.FLEX,
            created_at=now,
            updated_at=now,
        )
        flex_node.set_block(flex_block)

        # Create FAST tier job
        fast_job = Job(
            id=uuid4(),
            user_id=uuid4(),
            tier=Tier.FAST,
            gpu_count=2,
            vram_per_gpu_gb=40.0,
            cpu_cores=8,
            ram_gb=64.0,
            expected_duration_sec=3600,
            priority=0,
            status=JobStatus.PENDING,
            image="test-image:latest",
            created_at=now,
            updated_at=now,
        )

        # When only FLEX node is available, should request more capacity
        decision = schedule_job(fast_job, [flex_node])
        assert decision.kind == "REQUEST_MORE_CAPACITY"
        assert decision.tier == Tier.FAST

        # When FAST node is available, should schedule on it
        decision = schedule_job(fast_job, [fast_node])
        assert decision.kind == "EXISTING_NODE"
        assert decision.node_id == fast_node.id

    def test_flex_tier_job_only_schedules_on_flex_nodes(self):
        """Test that FLEX tier jobs only schedule on FLEX tier nodes."""
        from libs.common.types import Node, Block, GPUAssignment, BlockStatus
        from datetime import timedelta

        now = datetime.utcnow()
        
        # Create FAST block and node
        fast_block = Block(
            id=uuid4(),
            instance_type="8xH100",
            gpus_per_node=8,
            vram_per_gpu_gb=80.0,
            cpu_per_node=64,
            ram_per_node_gb=512.0,
            start_time=now,
            end_time=now + timedelta(hours=24),
            cost_per_hour=10.0,
            status=BlockStatus.ACTIVE,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )
        
        fast_gpus = [
            GPUAssignment(index=i, vram_total_gb=80.0, vram_used_gb=0.0)
            for i in range(8)
        ]
        fast_node = Node(
            id=uuid4(),
            block_id=fast_block.id,
            gpu_slots=fast_gpus,
            cpu_used=0.0,
            ram_used_gb=0.0,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )
        fast_node.set_block(fast_block)

        # Create FLEX block and node
        flex_block = Block(
            id=uuid4(),
            instance_type="8xH100",
            gpus_per_node=8,
            vram_per_gpu_gb=80.0,
            cpu_per_node=64,
            ram_per_node_gb=512.0,
            start_time=now,
            end_time=now + timedelta(hours=24),
            cost_per_hour=8.0,
            status=BlockStatus.ACTIVE,
            tier=Tier.FLEX,
            created_at=now,
            updated_at=now,
        )
        
        flex_gpus = [
            GPUAssignment(index=i, vram_total_gb=80.0, vram_used_gb=0.0)
            for i in range(8)
        ]
        flex_node = Node(
            id=uuid4(),
            block_id=flex_block.id,
            gpu_slots=flex_gpus,
            cpu_used=0.0,
            ram_used_gb=0.0,
            tier=Tier.FLEX,
            created_at=now,
            updated_at=now,
        )
        flex_node.set_block(flex_block)

        # Create FLEX tier job
        flex_job = Job(
            id=uuid4(),
            user_id=uuid4(),
            tier=Tier.FLEX,
            gpu_count=2,
            vram_per_gpu_gb=40.0,
            cpu_cores=8,
            ram_gb=64.0,
            expected_duration_sec=7200,
            priority=0,
            status=JobStatus.PENDING,
            image="test-image:latest",
            created_at=now,
            updated_at=now,
        )

        # When only FAST node is available, should queue for FLEX
        decision = schedule_job(flex_job, [fast_node])
        assert decision.kind == "QUEUE_FOR_FLEX"

        # When FLEX node is available, should schedule on it
        decision = schedule_job(flex_job, [flex_node])
        assert decision.kind == "EXISTING_NODE"
        assert decision.node_id == flex_node.id


"""Tests for capacity state management."""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from libs.common.types import (
    Node,
    Block,
    GPUAssignment,
    CapacitySnapshot,
    Tier,
    BlockStatus,
)
from services.capacity_manager.capacity_state import CapacityState


@pytest.fixture
def sample_block():
    """Create a sample block."""
    now = datetime.utcnow()
    return Block(
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


class TestCapacitySnapshot:
    """Test capacity snapshot functionality."""

    def test_get_capacity_snapshot_all_tiers(self, sample_block):
        """Test getting capacity snapshot for all tiers."""
        state = CapacityState()
        
        # Manually add nodes to state (simulating DB sync)
        now = datetime.utcnow()
        gpu_slots = [
            GPUAssignment(index=i, vram_total_gb=80.0, vram_used_gb=0.0)
            for i in range(8)
        ]
        
        fast_node = Node(
            id=uuid4(),
            block_id=sample_block.id,
            gpu_slots=gpu_slots,
            cpu_used=0.0,
            ram_used_gb=0.0,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )
        fast_node.set_block(sample_block)
        
        state.nodes[fast_node.id] = fast_node
        state.blocks[sample_block.id] = sample_block

        snapshot = state.get_capacity_snapshot()
        assert len(snapshot.nodes) == 1
        assert snapshot.nodes[0].tier == Tier.FAST

    def test_get_capacity_snapshot_filtered_by_tier(self, sample_block):
        """Test getting capacity snapshot filtered by tier."""
        state = CapacityState()
        
        now = datetime.utcnow()
        gpu_slots = [
            GPUAssignment(index=i, vram_total_gb=80.0, vram_used_gb=0.0)
            for i in range(8)
        ]
        
        fast_node = Node(
            id=uuid4(),
            block_id=sample_block.id,
            gpu_slots=gpu_slots,
            cpu_used=0.0,
            ram_used_gb=0.0,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )
        fast_node.set_block(sample_block)
        
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
        
        flex_node = Node(
            id=uuid4(),
            block_id=flex_block.id,
            gpu_slots=gpu_slots,
            cpu_used=0.0,
            ram_used_gb=0.0,
            tier=Tier.FLEX,
            created_at=now,
            updated_at=now,
        )
        flex_node.set_block(flex_block)
        
        state.nodes[fast_node.id] = fast_node
        state.nodes[flex_node.id] = flex_node
        state.blocks[sample_block.id] = sample_block
        state.blocks[flex_block.id] = flex_block

        fast_snapshot = state.get_capacity_snapshot(Tier.FAST)
        assert len(fast_snapshot.nodes) == 1
        assert fast_snapshot.nodes[0].tier == Tier.FAST

        flex_snapshot = state.get_capacity_snapshot(Tier.FLEX)
        assert len(flex_snapshot.nodes) == 1
        assert flex_snapshot.nodes[0].tier == Tier.FLEX


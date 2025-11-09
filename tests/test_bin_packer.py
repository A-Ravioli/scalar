"""Tests for bin-packing algorithm - core functionality from idea.md."""

import pytest
import sys
from unittest.mock import MagicMock
from datetime import datetime, timedelta
from uuid import uuid4, UUID

# Mock config before importing bin_packer
if 'libs.common.config' not in sys.modules:
    mock_config = MagicMock()
    mock_config.bin_pack_safety_margin_sec = 300
    sys.modules['libs.common.config'] = MagicMock()
    sys.modules['libs.common.config'].config = mock_config

from libs.common.types import (
    Job,
    Node,
    Block,
    GPUAssignment,
    PlacementDecision,
    Tier,
    JobStatus,
    BlockStatus,
)
from services.scheduler.bin_packer import schedule_job, compute_score


@pytest.fixture
def sample_block_fast():
    """Create a sample FAST tier block."""
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


@pytest.fixture
def sample_block_flex():
    """Create a sample FLEX tier block."""
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
        cost_per_hour=8.0,
        status=BlockStatus.ACTIVE,
        tier=Tier.FLEX,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def empty_node_fast(sample_block_fast):
    """Create an empty FAST tier node."""
    now = datetime.utcnow()
    gpu_slots = [
        GPUAssignment(index=i, vram_total_gb=80.0, vram_used_gb=0.0)
        for i in range(8)
    ]
    node = Node(
        id=uuid4(),
        block_id=sample_block_fast.id,
        gpu_slots=gpu_slots,
        cpu_used=0.0,
        ram_used_gb=0.0,
        tier=Tier.FAST,
        created_at=now,
        updated_at=now,
    )
    node.set_block(sample_block_fast)
    return node


@pytest.fixture
def empty_node_flex(sample_block_flex):
    """Create an empty FLEX tier node."""
    now = datetime.utcnow()
    gpu_slots = [
        GPUAssignment(index=i, vram_total_gb=80.0, vram_used_gb=0.0)
        for i in range(8)
    ]
    node = Node(
        id=uuid4(),
        block_id=sample_block_flex.id,
        gpu_slots=gpu_slots,
        cpu_used=0.0,
        ram_used_gb=0.0,
        tier=Tier.FLEX,
        created_at=now,
        updated_at=now,
    )
    node.set_block(sample_block_flex)
    return node


@pytest.fixture
def partially_used_node_fast(sample_block_fast):
    """Create a partially used FAST tier node."""
    now = datetime.utcnow()
    gpu_slots = [
        GPUAssignment(index=i, vram_total_gb=80.0, vram_used_gb=40.0 if i < 4 else 0.0)
        for i in range(8)
    ]
    node = Node(
        id=uuid4(),
        block_id=sample_block_fast.id,
        gpu_slots=gpu_slots,
        cpu_used=32.0,
        ram_used_gb=256.0,
        tier=Tier.FAST,
        created_at=now,
        updated_at=now,
    )
    node.set_block(sample_block_fast)
    return node


@pytest.fixture
def sample_job_fast():
    """Create a sample FAST tier job."""
    now = datetime.utcnow()
    return Job(
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


@pytest.fixture
def sample_job_flex():
    """Create a sample FLEX tier job."""
    now = datetime.utcnow()
    return Job(
        id=uuid4(),
        user_id=uuid4(),
        tier=Tier.FLEX,
        gpu_count=4,
        vram_per_gpu_gb=40.0,
        cpu_cores=16,
        ram_gb=128.0,
        expected_duration_sec=7200,
        priority=0,
        status=JobStatus.PENDING,
        image="test-image:latest",
        created_at=now,
        updated_at=now,
    )


class TestBinPackingCore:
    """Test core bin-packing functionality from idea.md section 3."""

    def test_schedule_job_on_empty_node_fast(self, sample_job_fast, empty_node_fast):
        """Test scheduling a FAST job on an empty FAST node."""
        decision = schedule_job(sample_job_fast, [empty_node_fast])

        assert decision.kind == "EXISTING_NODE"
        assert decision.node_id == empty_node_fast.id
        assert decision.job_id == sample_job_fast.id
        assert len(decision.gpu_indices) == 2
        assert all(idx in range(8) for idx in decision.gpu_indices)

    def test_schedule_job_on_partially_used_node(self, sample_job_fast, partially_used_node_fast):
        """Test scheduling a job on a partially used node."""
        decision = schedule_job(sample_job_fast, [partially_used_node_fast])

        assert decision.kind == "EXISTING_NODE"
        assert decision.node_id == partially_used_node_fast.id
        # Should use GPUs that have enough free VRAM (GPUs 4-7 have 80GB free, GPUs 0-3 have 40GB free)
        # The algorithm will select GPUs with >= 40GB free VRAM
        assert len(decision.gpu_indices) == 2
        assert all(idx in range(8) for idx in decision.gpu_indices)

    def test_tier_mismatch_rejected(self, sample_job_fast, empty_node_flex):
        """Test that jobs are rejected if tier doesn't match."""
        decision = schedule_job(sample_job_fast, [empty_node_flex])

        # Should request more capacity for FAST tier
        assert decision.kind == "REQUEST_MORE_CAPACITY"
        assert decision.tier == Tier.FAST

    def test_insufficient_gpu_count(self, sample_job_fast, empty_node_fast):
        """Test job requiring more GPUs than available."""
        # Create job requiring 10 GPUs, but node only has 8
        large_job = Job(
            id=uuid4(),
            user_id=uuid4(),
            tier=Tier.FAST,
            gpu_count=10,
            vram_per_gpu_gb=40.0,
            cpu_cores=8,
            ram_gb=64.0,
            expected_duration_sec=3600,
            priority=0,
            status=JobStatus.PENDING,
            image="test-image:latest",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        decision = schedule_job(large_job, [empty_node_fast])
        assert decision.kind == "REQUEST_MORE_CAPACITY"

    def test_insufficient_vram(self, empty_node_fast):
        """Test job requiring more VRAM than available."""
        # Create job requiring 100GB VRAM per GPU, but GPUs only have 80GB
        large_vram_job = Job(
            id=uuid4(),
            user_id=uuid4(),
            tier=Tier.FAST,
            gpu_count=2,
            vram_per_gpu_gb=100.0,
            cpu_cores=8,
            ram_gb=64.0,
            expected_duration_sec=3600,
            priority=0,
            status=JobStatus.PENDING,
            image="test-image:latest",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        decision = schedule_job(large_vram_job, [empty_node_fast])
        assert decision.kind == "REQUEST_MORE_CAPACITY"

    def test_insufficient_cpu(self, empty_node_fast):
        """Test job requiring more CPU than available."""
        # Create job requiring 100 CPU cores, but node only has 64
        large_cpu_job = Job(
            id=uuid4(),
            user_id=uuid4(),
            tier=Tier.FAST,
            gpu_count=2,
            vram_per_gpu_gb=40.0,
            cpu_cores=100,
            ram_gb=64.0,
            expected_duration_sec=3600,
            priority=0,
            status=JobStatus.PENDING,
            image="test-image:latest",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        decision = schedule_job(large_cpu_job, [empty_node_fast])
        assert decision.kind == "REQUEST_MORE_CAPACITY"

    def test_insufficient_ram(self, empty_node_fast):
        """Test job requiring more RAM than available."""
        # Create job requiring 600GB RAM, but node only has 512GB
        large_ram_job = Job(
            id=uuid4(),
            user_id=uuid4(),
            tier=Tier.FAST,
            gpu_count=2,
            vram_per_gpu_gb=40.0,
            cpu_cores=8,
            ram_gb=600.0,
            expected_duration_sec=3600,
            priority=0,
            status=JobStatus.PENDING,
            image="test-image:latest",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        decision = schedule_job(large_ram_job, [empty_node_fast])
        assert decision.kind == "REQUEST_MORE_CAPACITY"

    def test_node_expiring_too_soon(self, sample_job_fast, sample_block_fast):
        """Test that jobs are rejected if node expires too soon."""
        # Create a block that expires in 5 minutes
        now = datetime.utcnow()
        expiring_block = Block(
            id=uuid4(),
            instance_type="8xH100",
            gpus_per_node=8,
            vram_per_gpu_gb=80.0,
            cpu_per_node=64,
            ram_per_node_gb=512.0,
            start_time=now,
            end_time=now + timedelta(minutes=5),  # Expires soon
            cost_per_hour=10.0,
            status=BlockStatus.ACTIVE,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )

        gpu_slots = [
            GPUAssignment(index=i, vram_total_gb=80.0, vram_used_gb=0.0)
            for i in range(8)
        ]
        expiring_node = Node(
            id=uuid4(),
            block_id=expiring_block.id,
            gpu_slots=gpu_slots,
            cpu_used=0.0,
            ram_used_gb=0.0,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )
        expiring_node.set_block(expiring_block)

        # Job expects to run for 1 hour, but node expires in 5 minutes
        decision = schedule_job(sample_job_fast, [expiring_node])
        assert decision.kind == "REQUEST_MORE_CAPACITY"

    def test_flex_job_queued_when_no_capacity(self, sample_job_flex):
        """Test that FLEX jobs are queued when no capacity available."""
        decision = schedule_job(sample_job_flex, [])  # No nodes available

        assert decision.kind == "QUEUE_FOR_FLEX"
        assert decision.job_id == sample_job_flex.id

    def test_multiple_nodes_selects_best(self, sample_job_fast, sample_block_fast):
        """Test that scheduler selects best node from multiple candidates."""
        # Create two nodes: one empty, one partially used
        now = datetime.utcnow()
        
        empty_gpus = [
            GPUAssignment(index=i, vram_total_gb=80.0, vram_used_gb=0.0)
            for i in range(8)
        ]
        empty_node = Node(
            id=uuid4(),
            block_id=sample_block_fast.id,
            gpu_slots=empty_gpus,
            cpu_used=0.0,
            ram_used_gb=0.0,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )
        empty_node.set_block(sample_block_fast)

        used_gpus = [
            GPUAssignment(index=i, vram_total_gb=80.0, vram_used_gb=60.0)
            for i in range(8)
        ]
        used_node = Node(
            id=uuid4(),
            block_id=sample_block_fast.id,
            gpu_slots=used_gpus,
            cpu_used=50.0,
            ram_used_gb=400.0,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )
        used_node.set_block(sample_block_fast)

        decision = schedule_job(sample_job_fast, [used_node, empty_node])
        
        # Should prefer empty node (better utilization after placement)
        assert decision.kind == "EXISTING_NODE"
        assert decision.node_id == empty_node.id


class TestScoringHeuristic:
    """Test scoring heuristic from idea.md section 3.3."""

    def test_score_prefers_higher_utilization(self, sample_job_fast, sample_block_fast):
        """Test that scoring prefers nodes with higher utilization after placement."""
        now = datetime.utcnow()

        # Empty node
        empty_gpus = [
            GPUAssignment(index=i, vram_total_gb=80.0, vram_used_gb=0.0)
            for i in range(8)
        ]
        empty_node = Node(
            id=uuid4(),
            block_id=sample_block_fast.id,
            gpu_slots=empty_gpus,
            cpu_used=0.0,
            ram_used_gb=0.0,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )
        empty_node.set_block(sample_block_fast)

        # Partially used node
        partial_gpus = [
            GPUAssignment(index=i, vram_total_gb=80.0, vram_used_gb=40.0)
            for i in range(8)
        ]
        partial_node = Node(
            id=uuid4(),
            block_id=sample_block_fast.id,
            gpu_slots=partial_gpus,
            cpu_used=32.0,
            ram_used_gb=256.0,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )
        partial_node.set_block(sample_block_fast)

        empty_possible = [g for g in empty_node.gpu_slots if g.vram_total_gb - g.vram_used_gb >= sample_job_fast.vram_per_gpu_gb]
        partial_possible = [g for g in partial_node.gpu_slots if g.vram_total_gb - g.vram_used_gb >= sample_job_fast.vram_per_gpu_gb]

        empty_score = compute_score(empty_node, sample_job_fast, empty_possible)
        partial_score = compute_score(partial_node, sample_job_fast, partial_possible)

        # Both nodes should be scoreable (the scoring function considers utilization after placement)
        # Empty node starts at 0% utilization, partial node starts at 50% utilization
        # After placing job: empty goes to ~12.5% (2 GPUs * 40GB / 8 GPUs * 80GB), partial goes to ~62.5%
        # The scoring prefers higher utilization, so partial node gets higher score
        # This is actually correct behavior - we want to pack jobs efficiently
        assert isinstance(empty_score, float)
        assert isinstance(partial_score, float)
        # Partial node should have higher score because it achieves better utilization
        assert partial_score > empty_score

    def test_score_penalizes_fragmentation(self, sample_job_fast, sample_block_fast):
        """Test that scoring penalizes non-contiguous GPU selection."""
        now = datetime.utcnow()

        # Node with GPUs 0, 2, 4, 6 free (fragmented)
        fragmented_gpus = [
            GPUAssignment(index=i, vram_total_gb=80.0, vram_used_gb=80.0 if i % 2 == 1 else 0.0)
            for i in range(8)
        ]
        fragmented_node = Node(
            id=uuid4(),
            block_id=sample_block_fast.id,
            gpu_slots=fragmented_gpus,
            cpu_used=0.0,
            ram_used_gb=0.0,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )
        fragmented_node.set_block(sample_block_fast)

        # Node with GPUs 0, 1, 2, 3 free (contiguous)
        contiguous_gpus = [
            GPUAssignment(index=i, vram_total_gb=80.0, vram_used_gb=80.0 if i >= 4 else 0.0)
            for i in range(8)
        ]
        contiguous_node = Node(
            id=uuid4(),
            block_id=sample_block_fast.id,
            gpu_slots=contiguous_gpus,
            cpu_used=0.0,
            ram_used_gb=0.0,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )
        contiguous_node.set_block(sample_block_fast)

        fragmented_possible = [g for g in fragmented_node.gpu_slots if g.vram_total_gb - g.vram_used_gb >= sample_job_fast.vram_per_gpu_gb]
        contiguous_possible = [g for g in contiguous_node.gpu_slots if g.vram_total_gb - g.vram_used_gb >= sample_job_fast.vram_per_gpu_gb]

        fragmented_score = compute_score(fragmented_node, sample_job_fast, fragmented_possible)
        contiguous_score = compute_score(contiguous_node, sample_job_fast, contiguous_possible)

        # Contiguous node should have higher score
        assert contiguous_score > fragmented_score


class TestResourceModel:
    """Test resource model from idea.md section 3.1."""

    def test_node_free_cpu_calculation(self, sample_block_fast):
        """Test Node.free_cpu() calculation."""
        now = datetime.utcnow()
        node = Node(
            id=uuid4(),
            block_id=sample_block_fast.id,
            gpu_slots=[],
            cpu_used=32.0,
            ram_used_gb=0.0,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )
        node.set_block(sample_block_fast)

        assert node.free_cpu() == 32.0  # 64 - 32

    def test_node_free_ram_calculation(self, sample_block_fast):
        """Test Node.free_ram_gb() calculation."""
        now = datetime.utcnow()
        node = Node(
            id=uuid4(),
            block_id=sample_block_fast.id,
            gpu_slots=[],
            cpu_used=0.0,
            ram_used_gb=256.0,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )
        node.set_block(sample_block_fast)

        assert node.free_ram_gb() == 256.0  # 512 - 256

    def test_node_time_to_expiry(self, sample_block_fast):
        """Test Node.time_to_expiry() calculation."""
        now = datetime.utcnow()
        # Block expires in 2 hours
        block = Block(
            id=uuid4(),
            instance_type="8xH100",
            gpus_per_node=8,
            vram_per_gpu_gb=80.0,
            cpu_per_node=64,
            ram_per_node_gb=512.0,
            start_time=now,
            end_time=now + timedelta(hours=2),
            cost_per_hour=10.0,
            status=BlockStatus.ACTIVE,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )

        node = Node(
            id=uuid4(),
            block_id=block.id,
            gpu_slots=[],
            cpu_used=0.0,
            ram_used_gb=0.0,
            tier=Tier.FAST,
            created_at=now,
            updated_at=now,
        )
        node.set_block(block)

        expiry = node.time_to_expiry()
        # Should be approximately 2 hours (7200 seconds), allow some tolerance
        assert 7000 < expiry < 7300


class TestPlacementDecisions:
    """Test placement decision logic from idea.md section 3.3."""

    def test_existing_node_decision(self, sample_job_fast, empty_node_fast):
        """Test EXISTING_NODE placement decision."""
        decision = schedule_job(sample_job_fast, [empty_node_fast])

        assert decision.kind == "EXISTING_NODE"
        assert decision.node_id is not None
        assert decision.gpu_indices is not None
        assert len(decision.gpu_indices) == sample_job_fast.gpu_count

    def test_request_more_capacity_fast(self, sample_job_fast):
        """Test REQUEST_MORE_CAPACITY for FAST tier."""
        decision = schedule_job(sample_job_fast, [])

        assert decision.kind == "REQUEST_MORE_CAPACITY"
        assert decision.tier == Tier.FAST
        assert decision.job_id == sample_job_fast.id

    def test_queue_for_flex(self, sample_job_flex):
        """Test QUEUE_FOR_FLEX for FLEX tier."""
        decision = schedule_job(sample_job_flex, [])

        assert decision.kind == "QUEUE_FOR_FLEX"
        assert decision.job_id == sample_job_flex.id


"""In-memory capacity state management."""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID
from libs.common.db import get_supabase_client
from libs.common.types import Node, Block, GPUAssignment, CapacitySnapshot, Tier
from libs.common.logging import get_logger

logger = get_logger(__name__)


class CapacityState:
    """Manages in-memory capacity state synced with database."""

    def __init__(self):
        self.supabase = get_supabase_client()
        self.nodes: Dict[UUID, Node] = {}
        self.blocks: Dict[UUID, Block] = {}
        self.last_sync: Optional[datetime] = None

    async def sync_from_db(self):
        """Sync state from database."""
        logger.info("Syncing capacity state from database")

        # Load active blocks
        blocks_result = (
            self.supabase.table("blocks")
            .select("*")
            .in_("status", ["pending", "active"])
            .execute()
        )

        self.blocks = {}
        for block_data in blocks_result.data:
            # Parse drain_deadline if present
            drain_deadline = None
            if block_data.get("drain_deadline"):
                drain_deadline = datetime.fromisoformat(block_data["drain_deadline"])

            block = Block(
                id=UUID(block_data["id"]),
                provider=block_data.get("provider", "sfcompute"),
                instance_type=block_data["instance_type"],
                gpus_per_node=block_data["gpus_per_node"],
                vram_per_gpu_gb=block_data["vram_per_gpu_gb"],
                cpu_per_node=block_data["cpu_per_node"],
                ram_per_node_gb=block_data["ram_per_node_gb"],
                start_time=datetime.fromisoformat(block_data["start_time"]),
                end_time=datetime.fromisoformat(block_data["end_time"]),
                cost_per_hour=block_data["cost_per_hour"],
                status=block_data["status"],
                tier=Tier(block_data["tier"]) if block_data.get("tier") else None,
                provider_instance_id=block_data.get("provider_instance_id"),
                region=block_data.get("region"),
                preemptible=block_data.get("preemptible", False),
                drain_deadline=drain_deadline,
                created_at=datetime.fromisoformat(block_data["created_at"]),
                updated_at=datetime.fromisoformat(block_data["updated_at"]),
            )
            self.blocks[block.id] = block

        # Load nodes with GPU assignments
        nodes_result = (
            self.supabase.table("nodes")
            .select("*, gpu_assignments(*)")
            .execute()
        )

        self.nodes = {}
        for node_data in nodes_result.data:
            block_id = UUID(node_data["block_id"])
            block = self.blocks.get(block_id)

            # Load GPU assignments
            gpu_assignments = []
            for ga_data in node_data.get("gpu_assignments", []):
                gpu_assignments.append(
                    GPUAssignment(
                        index=ga_data["gpu_index"],
                        vram_total_gb=ga_data["vram_total_gb"],
                        vram_used_gb=ga_data["vram_used_gb"],
                        job_id=UUID(ga_data["job_id"]) if ga_data.get("job_id") else None,
                    )
                )

            # If no GPU assignments exist, create them from block spec
            if not gpu_assignments and block:
                for i in range(block.gpus_per_node):
                    gpu_assignments.append(
                        GPUAssignment(
                            index=i,
                            vram_total_gb=block.vram_per_gpu_gb,
                            vram_used_gb=0.0,
                        )
                    )

            node = Node(
                id=UUID(node_data["id"]),
                block_id=block_id,
                gpu_slots=gpu_assignments,
                cpu_used=node_data["cpu_used"],
                ram_used_gb=node_data["ram_used_gb"],
                tier=Tier(node_data["tier"]),
                created_at=datetime.fromisoformat(node_data["created_at"]),
                updated_at=datetime.fromisoformat(node_data["updated_at"]),
            )
            # Attach block reference for richer calculations
            if block:
                node.set_block(block)
            self.nodes[node.id] = node

        self.last_sync = datetime.utcnow()
        logger.info(f"Synced {len(self.blocks)} blocks and {len(self.nodes)} nodes")

    def get_capacity_snapshot(self, tier: Optional[Tier] = None) -> CapacitySnapshot:
        """Get current capacity snapshot."""
        # Only expose nodes whose blocks are active to the scheduler
        active_block_ids = {
            block_id for block_id, block in self.blocks.items() if block.status.value == "active"
        }
        nodes = [n for n in self.nodes.values() if n.block_id in active_block_ids]
        if tier:
            nodes = [n for n in nodes if n.tier == tier]
        return CapacitySnapshot(nodes=nodes, timestamp=datetime.utcnow())


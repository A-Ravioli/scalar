"""Autoscaling logic for Fast and Flex tiers."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
from libs.common.config import config
from libs.common.db import get_supabase_client
from libs.common.logging import get_logger
from libs.common.types import Tier, BlockStatus
from .sfcompute_client import SFComputeClient
from .capacity_state import CapacityState

logger = get_logger(__name__)


class Autoscaler:
    """Autoscaler for Fast and Flex tiers."""

    def __init__(self, capacity_state: CapacityState):
        self.capacity_state = capacity_state
        self.sfcompute = SFComputeClient()
        self.supabase = get_supabase_client()
        self.running = False

    async def autoscale_fast_tier(self):
        """Autoscale Fast tier based on demand."""
        logger.info("Running Fast tier autoscaler")

        # Get current utilization
        snapshot = self.capacity_state.get_capacity_snapshot(Tier.FAST)
        total_gpus = sum(len(node.gpu_slots) for node in snapshot.nodes)
        used_gpus = sum(
            sum(1 for g in node.gpu_slots if g.job_id is not None)
            for node in snapshot.nodes
        )
        utilization = used_gpus / total_gpus if total_gpus > 0 else 0.0

        # Get queue length
        queue_result = (
            self.supabase.table("jobs")
            .select("id", count="exact")
            .eq("tier", Tier.FAST.value)
            .in_("status", ["pending", "queued"])
            .execute()
        )
        queue_length = queue_result.count or 0

        # Simple demand estimate: queue length + current utilization
        target_gpus = int(
            (queue_length + used_gpus) * config.fast_tier_safety_factor
        )

        if target_gpus > total_gpus:
            # Need more capacity
            gpus_needed = target_gpus - total_gpus
            logger.info(f"Fast tier needs {gpus_needed} more GPUs")

            # Buy blocks (simplified: assume 8 GPUs per node)
            nodes_needed = (gpus_needed + 7) // 8
            instance_type = "8xH100"  # Default instance type

            try:
                contract = await self.sfcompute.buy_block(
                    instance_type=instance_type,
                    duration_hours=24,
                )

                # Create block record
                block_id = contract.get("contract_id")
                if block_id:
                    self.supabase.table("blocks").insert({
                        "id": block_id,
                        "instance_type": instance_type,
                        "gpus_per_node": 8,
                        "vram_per_gpu_gb": 80.0,
                        "cpu_per_node": 64,
                        "ram_per_node_gb": 512.0,
                        "start_time": datetime.utcnow().isoformat(),
                        "end_time": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
                        "cost_per_hour": contract.get("price_per_hour", 0.0),
                        "status": BlockStatus.PENDING.value,
                        "tier": Tier.FAST.value,
                        "sfcompute_contract_id": block_id,
                    }).execute()

                    # Provision nodes
                    await self.sfcompute.provision_nodes(block_id, nodes_needed)

                    logger.info(f"Bought Fast tier block {block_id}")

            except Exception as e:
                logger.error(f"Failed to buy Fast tier block: {e}")

        elif utilization < 0.3 and total_gpus > 10:
            # Over-capacity, mark some blocks for non-renewal
            logger.info("Fast tier over-capacity, marking blocks for non-renewal")
            # TODO: Implement block non-renewal logic

    async def autoscale_flex_tier(self):
        """Autoscale Flex tier based on queue and prices."""
        logger.info("Running Flex tier autoscaler")

        # Get Flex queue length
        queue_result = (
            self.supabase.table("jobs")
            .select("id", count="exact")
            .eq("tier", Tier.FLEX.value)
            .in_("status", ["pending", "queued"])
            .execute()
        )
        queue_length = queue_result.count or 0

        if queue_length == 0:
            return

        # Check if we have capacity
        snapshot = self.capacity_state.get_capacity_snapshot(Tier.FLEX)
        total_gpus = sum(len(node.gpu_slots) for node in snapshot.nodes)
        used_gpus = sum(
            sum(1 for g in node.gpu_slots if g.job_id is not None)
            for node in snapshot.nodes
        )
        free_gpus = total_gpus - used_gpus

        if free_gpus < queue_length:
            # Need more capacity, check prices
            instance_type = "8xH100"
            try:
                orderbook = await self.sfcompute.get_orderbook(instance_type)
                # Simple: buy if price is reasonable
                # TODO: Implement price threshold logic

                contract = await self.sfcompute.buy_block(
                    instance_type=instance_type,
                    duration_hours=24,
                    max_price_per_hour=10.0,  # Example threshold
                )

                block_id = contract.get("contract_id")
                if block_id:
                    self.supabase.table("blocks").insert({
                        "id": block_id,
                        "instance_type": instance_type,
                        "gpus_per_node": 8,
                        "vram_per_gpu_gb": 80.0,
                        "cpu_per_node": 64,
                        "ram_per_node_gb": 512.0,
                        "start_time": datetime.utcnow().isoformat(),
                        "end_time": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
                        "cost_per_hour": contract.get("price_per_hour", 0.0),
                        "status": BlockStatus.PENDING.value,
                        "tier": Tier.FLEX.value,
                        "sfcompute_contract_id": block_id,
                    }).execute()

                    await self.sfcompute.provision_nodes(block_id, 1)

                    logger.info(f"Bought Flex tier block {block_id}")

            except Exception as e:
                logger.error(f"Failed to buy Flex tier block: {e}")

    async def run(self):
        """Main autoscaler loop."""
        self.running = True
        logger.info("Autoscaler started")

        while self.running:
            try:
                # Sync state
                await self.capacity_state.sync_from_db()

                # Run autoscaling for both tiers
                await self.autoscale_fast_tier()
                await self.autoscale_flex_tier()

                await asyncio.sleep(config.autoscaler_interval_sec)

            except Exception as e:
                logger.error(f"Error in autoscaler loop: {e}", exc_info=True)
                await asyncio.sleep(config.autoscaler_interval_sec)

    def stop(self):
        """Stop autoscaler."""
        self.running = False


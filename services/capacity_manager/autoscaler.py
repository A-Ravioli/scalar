"""Autoscaling logic for Fast and Flex tiers with hybrid provider support."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4
from libs.common.config import config
from libs.common.db import get_supabase_client
from libs.common.logging import get_logger
from libs.common.types import Tier, BlockStatus
from .capacity_state import CapacityState
from .providers import SFComputeProvider, PrimeIntellectProvider, ComputeProvider

logger = get_logger(__name__)


class Autoscaler:
    """Autoscaler for Fast and Flex tiers with hybrid provider support."""

    def __init__(self, capacity_state: CapacityState):
        self.capacity_state = capacity_state
        self.supabase = get_supabase_client()
        self.running = False
        self.last_fast_downscale_at: Optional[datetime] = None
        self.last_consolidation_check: Optional[datetime] = None
        self.sustained_demand_gpus: int = 0  # Track sustained demand for consolidation

        # Initialize providers
        self.providers: Dict[str, ComputeProvider] = {}
        self.providers["sfcompute"] = SFComputeProvider()
        if config.enable_prime and config.prime_api_key:
            self.providers["prime"] = PrimeIntellectProvider()
        else:
            logger.warning("Prime Intellect disabled or API key not configured")

    async def select_provider(
        self,
        gpu_count: int,
        vram_per_gpu_gb: float,
        cpu_cores: int,
        ram_gb: float,
        tier: Tier,
    ) -> Tuple[Optional[ComputeProvider], Optional[str]]:
        """
        Select best provider for given GPU count and requirements.

        Returns (provider, instance_type) or (None, None) if no provider available.
        """
        # Prime only supports 1-8 GPUs
        if gpu_count <= config.prime_max_gpus and "prime" in self.providers:
            # Get Prime quotes
            try:
                prime_quotes = await self.providers["prime"].get_price_quotes(
                    gpu_count=gpu_count,
                    vram_per_gpu_gb=vram_per_gpu_gb,
                    cpu_cores=cpu_cores,
                    ram_gb=ram_gb,
                )

                if prime_quotes:
                    best_prime = prime_quotes[0]
                    # Check price cap
                    if best_prime.price_per_gpu_hour <= config.max_prime_price_per_gpu_hour:
                        # Check startup SLA
                        if best_prime.estimated_startup_sec <= config.prime_startup_sla_sec:
                            logger.info(
                                f"Selected Prime for {gpu_count} GPUs at ${best_prime.price_per_gpu_hour:.2f}/GPU-hr"
                            )
                            return self.providers["prime"], best_prime.instance_type

        # For >= 8 GPUs or if Prime unavailable/too expensive, use SFCompute
        if gpu_count >= 8 and "sfcompute" in self.providers:
            logger.info(f"Selected SFCompute for {gpu_count} GPUs")
            return self.providers["sfcompute"], "8xH100"

        # Fallback: if we need < 8 GPUs but Prime is unavailable, round up to SFCompute
        if gpu_count < 8 and "sfcompute" in self.providers:
            logger.warning(
                f"Prime unavailable for {gpu_count} GPUs, rounding up to SFCompute 8x"
            )
            return self.providers["sfcompute"], "8xH100"

        return None, None

    async def provision_capacity(
        self,
        gpu_count: int,
        tier: Tier,
        duration_hours: int = 24,
        max_price_per_hour: Optional[float] = None,
    ) -> Optional[UUID]:
        """
        Provision capacity using appropriate provider.

        Returns block_id if successful, None otherwise.
        """
        # Select provider
        provider, instance_type = await self.select_provider(
            gpu_count=gpu_count,
            vram_per_gpu_gb=80.0,  # Default H100 VRAM
            cpu_cores=8,
            ram_gb=64.0,
            tier=tier,
        )

        if not provider:
            logger.error(f"No provider available for {gpu_count} GPUs")
            return None

        try:
            # Provision via provider
            result = await provider.provision(
                instance_type=instance_type,
                gpu_count=gpu_count,
                duration_hours=duration_hours,
                tier=tier.value,
                max_price_per_hour=max_price_per_hour,
            )

            # Create block record in database
            block_id = result.block_id
            block_data = {
                "id": str(block_id),
                "provider": provider.name,
                "instance_type": result.instance_type,
                "gpus_per_node": result.gpu_count,  # For Prime, this is total GPUs
                "vram_per_gpu_gb": 80.0,  # Default H100
                "cpu_per_node": 64,  # Default
                "ram_per_node_gb": 512.0,  # Default
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "cost_per_hour": result.cost_per_hour,
                "status": BlockStatus.PENDING.value,
                "tier": tier.value,
                "provider_instance_id": result.provider_instance_id,
                "region": result.metadata.get("region") if result.metadata else None,
                "preemptible": provider.name == "prime",  # Prime pods are preemptible
            }

            self.supabase.table("blocks").insert(block_data).execute()

            # Create node records in database
            # For SFCompute, create nodes based on node_count from result
            if provider.name == "sfcompute":
                for node_idx in range(result.node_count):
                    node_id = uuid4()
                    self.supabase.table("nodes").insert({
                        "id": str(node_id),
                        "block_id": str(block_id),
                        "tier": tier.value,
                        "cpu_used": 0.0,
                        "ram_used_gb": 0.0,
                    }).execute()

                    # Create GPU assignments (8 GPUs per SFCompute node)
                    for gpu_idx in range(8):
                        self.supabase.table("gpu_assignments").insert({
                            "node_id": str(node_id),
                            "gpu_index": gpu_idx,
                            "vram_total_gb": 80.0,
                            "vram_used_gb": 0.0,
                        }).execute()

            # For Prime, create a single node record (Prime pods are single-node)
            elif provider.name == "prime":
                # Create node record
                node_id = uuid4()
                self.supabase.table("nodes").insert({
                    "id": str(node_id),
                    "block_id": str(block_id),
                    "tier": tier.value,
                    "cpu_used": 0.0,
                    "ram_used_gb": 0.0,
                }).execute()

                # Create GPU assignments
                for i in range(result.gpu_count):
                    self.supabase.table("gpu_assignments").insert({
                        "node_id": str(node_id),
                        "gpu_index": i,
                        "vram_total_gb": 80.0,
                        "vram_used_gb": 0.0,
                    }).execute()

            logger.info(
                f"Provisioned {result.gpu_count} GPUs via {provider.name} (block {block_id})"
            )
            return block_id

        except Exception as e:
            logger.error(f"Failed to provision capacity via {provider.name}: {e}", exc_info=True)
            return None

    async def autoscale_fast_tier(self):
        """Autoscale Fast tier with hybrid provider support."""
        logger.info("Running Fast tier autoscaler")

        # Get current utilization
        snapshot = self.capacity_state.get_capacity_snapshot(Tier.FAST)
        total_gpus = sum(len(node.gpu_slots) for node in snapshot.nodes)
        used_gpus = sum(
            sum(1 for g in node.gpu_slots if g.job_id is not None)
            for node in snapshot.nodes
        )
        utilization = used_gpus / total_gpus if total_gpus > 0 else 0.0

        # Get queue with GPU requirements
        queue_result = (
            self.supabase.table("jobs")
            .select("id, gpu_count")
            .eq("tier", Tier.FAST.value)
            .in_("status", ["pending", "queued"])
            .execute()
        )

        queue_jobs = queue_result.data if queue_result.data else []
        queue_gpus = sum(job.get("gpu_count", 1) for job in queue_jobs)

        # Simple demand estimate
        target_gpus = int((queue_gpus + used_gpus) * config.fast_tier_safety_factor)

        # Check for consolidation opportunity
        now = datetime.utcnow()
        if self.last_consolidation_check is None:
            self.last_consolidation_check = now
            self.sustained_demand_gpus = queue_gpus

        consolidation_window_sec = config.consolidation_window_min * 60
        time_since_check = (now - self.last_consolidation_check).total_seconds()

        # Track sustained demand
        if queue_gpus >= 8:
            if time_since_check >= consolidation_window_sec:
                # Check if demand has been sustained
                if self.sustained_demand_gpus >= 8:
                    logger.info(
                        f"Sustained demand of {queue_gpus} GPUs detected, considering SFCompute consolidation"
                    )
                    # Prefer SFCompute for sustained demand
                    gpus_needed = max(0, target_gpus - total_gpus)
                    if gpus_needed > 0:
                        # Round up to 8x for SFCompute
                        sfcompute_gpus = ((gpus_needed + 7) // 8) * 8
                        await self.provision_capacity(
                            gpu_count=sfcompute_gpus, tier=Tier.FAST
                        )
                self.sustained_demand_gpus = queue_gpus
                self.last_consolidation_check = now
        else:
            self.sustained_demand_gpus = queue_gpus

        if target_gpus > total_gpus:
            # Need more capacity
            gpus_needed = target_gpus - total_gpus

            # For small bursts (< 8 GPUs), use Prime
            if gpus_needed < 8 and "prime" in self.providers:
                logger.info(f"Fast tier needs {gpus_needed} GPUs, provisioning via Prime")
                await self.provision_capacity(gpu_count=gpus_needed, tier=Tier.FAST)
            else:
                # For larger needs, use SFCompute (round up to 8x)
                sfcompute_gpus = ((gpus_needed + 7) // 8) * 8
                logger.info(
                    f"Fast tier needs {gpus_needed} GPUs, provisioning {sfcompute_gpus} via SFCompute"
                )
                await self.provision_capacity(gpu_count=sfcompute_gpus, tier=Tier.FAST)

        else:
            # Consider downscaling
            await self._downscale_fast_tier(
                snapshot, total_gpus, used_gpus, target_gpus, now
            )

    async def _downscale_fast_tier(
        self,
        snapshot,
        total_gpus: int,
        used_gpus: int,
        target_gpus: int,
        now: datetime,
    ):
        """Downscale Fast tier capacity."""
        threshold = config.downscale_utilization_threshold
        warm_reserve = max(0, int(config.min_fast_tier_warm_gpus))
        cooldown_sec = max(0, int(config.downscale_cooldown_sec))

        if used_gpus / total_gpus if total_gpus > 0 else 0.0 < threshold and total_gpus > warm_reserve:
            # Enforce cooldown
            if (
                self.last_fast_downscale_at is not None
                and (now - self.last_fast_downscale_at).total_seconds() < cooldown_sec
            ):
                logger.info("Downscale cooldown active; skipping this cycle")
                return

            # Compute surplus
            target_floor = max(target_gpus, warm_reserve)
            surplus_gpus = max(0, total_gpus - target_floor)
            if surplus_gpus <= 0:
                return

            # Group nodes by block
            block_to_nodes: Dict[str, List] = {}
            for node in snapshot.nodes:
                bid = str(node.block_id)
                block_to_nodes.setdefault(bid, []).append(node)

            candidates = []
            for block_id_str, nodes in block_to_nodes.items():
                used = sum(
                    sum(1 for g in n.gpu_slots if g.job_id is not None) for n in nodes
                )
                total = sum(len(n.gpu_slots) for n in nodes)
                if used == 0 and total > 0:
                    try:
                        block_obj = self.capacity_state.blocks.get(UUID(block_id_str))
                    except Exception:
                        block_obj = None
                    if not block_obj:
                        continue
                    if block_obj.status == BlockStatus.ACTIVE and block_obj.tier == Tier.FAST:
                        # Prefer terminating Prime pods (shorter cooldown)
                        cooldown = (
                            config.prime_idle_cooldown_sec
                            if block_obj.provider == "prime"
                            else config.sfcompute_idle_cooldown_sec
                        )
                        remaining_secs = (block_obj.end_time - now).total_seconds()
                        candidates.append(
                            (cooldown, remaining_secs, block_id_str, total, block_obj.provider)
                        )
                    except Exception:
                        continue

            if not candidates:
                logger.info("No idle FAST blocks available to terminate")
                return

            # Sort by cooldown (Prime first), then by remaining time
            candidates.sort(key=lambda x: (x[0], -x[1]))

            gpus_to_terminate = surplus_gpus
            terminated_any = False
            for cooldown, _, block_id_str, block_gpu_capacity, provider_name in candidates:
                if gpus_to_terminate <= 0:
                    break

                # Check if block has been idle long enough
                block_obj = self.capacity_state.blocks.get(UUID(block_id_str))
                if not block_obj:
                    continue

                idle_time = (now - block_obj.updated_at).total_seconds()
                if idle_time < cooldown:
                    continue  # Not idle long enough

                try:
                    logger.info(f"Terminating FAST {provider_name} block {block_id_str}")
                    provider = self.providers.get(provider_name)
                    if provider and block_obj.provider_instance_id:
                        await provider.terminate(block_obj.provider_instance_id)

                    # Mark block as sold/terminated
                    self.supabase.table("blocks").update({
                        "status": BlockStatus.SOLD.value
                    }).eq("id", block_id_str).execute()

                    gpus_to_terminate -= block_gpu_capacity
                    terminated_any = True
                except Exception as e:
                    logger.error(f"Failed to terminate FAST block {block_id_str}: {e}")

            if terminated_any:
                self.last_fast_downscale_at = now
                logger.info("Completed FAST downscale cycle")

    async def autoscale_flex_tier(self):
        """Autoscale Flex tier with hybrid provider support."""
        logger.info("Running Flex tier autoscaler")

        # Get Flex queue
        queue_result = (
            self.supabase.table("jobs")
            .select("id, gpu_count")
            .eq("tier", Tier.FLEX.value)
            .in_("status", ["pending", "queued"])
            .execute()
        )

        queue_jobs = queue_result.data if queue_result.data else []
        queue_gpus = sum(job.get("gpu_count", 1) for job in queue_jobs)

        if queue_gpus == 0:
            return

        # Check current capacity
        snapshot = self.capacity_state.get_capacity_snapshot(Tier.FLEX)
        total_gpus = sum(len(node.gpu_slots) for node in snapshot.nodes)
        used_gpus = sum(
            sum(1 for g in node.gpu_slots if g.job_id is not None)
            for node in snapshot.nodes
        )
        free_gpus = total_gpus - used_gpus

        if free_gpus < queue_gpus:
            # Need more capacity
            gpus_needed = queue_gpus - free_gpus

            # Use cheapest provider (Prime for < 8, SFCompute for >= 8)
            if gpus_needed < 8 and "prime" in self.providers:
                logger.info(f"Flex tier needs {gpus_needed} GPUs, provisioning via Prime")
                await self.provision_capacity(
                    gpu_count=gpus_needed, tier=Tier.FLEX, max_price_per_hour=10.0
                )
            else:
                # Round up to 8x for SFCompute
                sfcompute_gpus = ((gpus_needed + 7) // 8) * 8
                logger.info(
                    f"Flex tier needs {gpus_needed} GPUs, provisioning {sfcompute_gpus} via SFCompute"
                )
                await self.provision_capacity(
                    gpu_count=sfcompute_gpus,
                    tier=Tier.FLEX,
                    max_price_per_hour=10.0,
                )

    async def run(self):
        """Main autoscaler loop."""
        self.running = True
        logger.info("Autoscaler started with hybrid provider support")

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

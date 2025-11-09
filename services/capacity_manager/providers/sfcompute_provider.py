"""SFCompute provider implementation."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from libs.common.logging import get_logger
from .base import ComputeProvider, ProvisionResult, PriceQuote
from ..sfcompute_client import SFComputeClient

logger = get_logger(__name__)


class SFComputeProvider(ComputeProvider):
    """SFCompute provider for 8+ GPU nodes."""

    def __init__(self):
        self.client = SFComputeClient()

    @property
    def name(self) -> str:
        return "sfcompute"

    async def get_price_quotes(
        self,
        gpu_count: int,
        vram_per_gpu_gb: float,
        cpu_cores: int,
        ram_gb: float,
        duration_hours: int = 1,
    ) -> List[PriceQuote]:
        """
        Get price quotes from SFCompute.

        SFCompute only supports 8x GPU nodes, so we round up.
        """
        # SFCompute only supports 8x GPU increments
        if gpu_count < 8:
            return []  # Use Prime for < 8 GPUs

        # Round up to nearest 8
        nodes_needed = (gpu_count + 7) // 8
        instance_type = "8xH100"  # Default instance type

        try:
            quote_data = await self.client.quote_price(instance_type, duration_hours)
            price_per_hour = quote_data.get("price_per_hour", 0.0)
            price_per_gpu_hour = price_per_hour / 8.0  # 8 GPUs per node

            return [
                PriceQuote(
                    provider=self.name,
                    instance_type=instance_type,
                    gpu_count=nodes_needed * 8,  # Total GPUs (may be more than requested)
                    price_per_hour=price_per_hour * nodes_needed,
                    price_per_gpu_hour=price_per_gpu_hour,
                    vram_per_gpu_gb=80.0,  # H100 has 80GB
                    cpu_per_node=64,
                    ram_per_node_gb=512.0,
                    available=True,
                    estimated_startup_sec=120,  # SFCompute startup time
                )
            ]
        except Exception as e:
            logger.error(f"Failed to get SFCompute price quote: {e}")
            return []

    async def provision(
        self,
        instance_type: str,
        gpu_count: int,
        duration_hours: int,
        tier: str,
        max_price_per_hour: Optional[float] = None,
    ) -> ProvisionResult:
        """Provision SFCompute block."""
        # Calculate nodes needed (8 GPUs per node)
        nodes_needed = (gpu_count + 7) // 8

        try:
            contract = await self.client.buy_block(
                instance_type=instance_type,
                duration_hours=duration_hours,
                max_price_per_hour=max_price_per_hour,
            )

            contract_id = contract.get("contract_id")
            if not contract_id:
                raise ValueError("No contract_id in SFCompute response")

            # Provision nodes
            await self.client.provision_nodes(contract_id, nodes_needed)

            start_time = datetime.utcnow()
            end_time = start_time + timedelta(hours=duration_hours)

            return ProvisionResult(
                provider=self.name,
                block_id=UUID(contract_id),
                provider_instance_id=contract_id,
                instance_type=instance_type,
                gpu_count=nodes_needed * 8,
                node_count=nodes_needed,
                cost_per_hour=contract.get("price_per_hour", 0.0),
                start_time=start_time,
                end_time=end_time,
                status="pending",
                metadata={"contract_id": contract_id},
            )
        except Exception as e:
            logger.error(f"Failed to provision SFCompute capacity: {e}")
            raise

    async def terminate(self, provider_instance_id: str) -> bool:
        """Terminate SFCompute block."""
        try:
            await self.client.terminate_nodes(provider_instance_id)
            await self.client.sell_block(provider_instance_id)
            return True
        except Exception as e:
            logger.error(f"Failed to terminate SFCompute capacity: {e}")
            return False

    async def get_status(self, provider_instance_id: str) -> Dict:
        """Get SFCompute block status."""
        # SFCompute client doesn't have status endpoint yet
        # Return basic info
        return {
            "provider": self.name,
            "instance_id": provider_instance_id,
            "status": "active",  # Assume active if we have the ID
        }

    async def list_capacity(self, filter_active: bool = True) -> List[Dict]:
        """List SFCompute capacity."""
        # Would need to query SFCompute API for this
        # For now, return empty list
        return []


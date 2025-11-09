"""Base provider abstraction for compute capacity."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID


@dataclass
class PriceQuote:
    """Price quote for compute capacity."""

    provider: str
    instance_type: str
    gpu_count: int
    price_per_hour: float
    price_per_gpu_hour: float
    vram_per_gpu_gb: float
    cpu_per_node: int
    ram_per_node_gb: float
    region: Optional[str] = None
    available: bool = True
    estimated_startup_sec: int = 60  # Estimated time to provision


@dataclass
class ProvisionResult:
    """Result of provisioning capacity."""

    provider: str
    block_id: UUID
    provider_instance_id: str  # Provider-specific instance ID
    instance_type: str
    gpu_count: int
    node_count: int
    cost_per_hour: float
    start_time: datetime
    end_time: datetime
    status: str  # "pending", "active", "failed"
    metadata: Dict = None  # Provider-specific metadata


class ComputeProvider(ABC):
    """Abstract base class for compute providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    async def get_price_quotes(
        self,
        gpu_count: int,
        vram_per_gpu_gb: float,
        cpu_cores: int,
        ram_gb: float,
        duration_hours: int = 1,
    ) -> List[PriceQuote]:
        """
        Get price quotes for capacity matching requirements.

        Returns list of quotes sorted by price (cheapest first).
        """
        pass

    @abstractmethod
    async def provision(
        self,
        instance_type: str,
        gpu_count: int,
        duration_hours: int,
        tier: str,
        max_price_per_hour: Optional[float] = None,
    ) -> ProvisionResult:
        """
        Provision compute capacity.

        Returns ProvisionResult with block details.
        """
        pass

    @abstractmethod
    async def terminate(self, provider_instance_id: str) -> bool:
        """
        Terminate provisioned capacity.

        Returns True if successful.
        """
        pass

    @abstractmethod
    async def get_status(self, provider_instance_id: str) -> Dict:
        """
        Get status of provisioned capacity.

        Returns dict with status information.
        """
        pass

    @abstractmethod
    async def list_capacity(self, filter_active: bool = True) -> List[Dict]:
        """
        List all capacity provisioned through this provider.

        Returns list of capacity records.
        """
        pass


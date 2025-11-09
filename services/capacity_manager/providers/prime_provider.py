"""Prime Intellect provider implementation for 1-8 GPU jobs."""

import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from libs.common.config import config
from libs.common.logging import get_logger
from .base import ComputeProvider, ProvisionResult, PriceQuote

logger = get_logger(__name__)


class PrimeIntellectProvider(ComputeProvider):
    """Prime Intellect provider for 1-8 GPU pods."""

    def __init__(self):
        self.api_key = getattr(config, "prime_api_key", None)
        self.api_url = getattr(config, "prime_api_url", "https://api.primeintellect.ai")
        self.base_headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
        }
        self._price_cache: Dict[str, tuple] = {}  # Cache prices with TTL
        self._cache_ttl_sec = 300  # 5 minutes

    @property
    def name(self) -> str:
        return "prime"

    async def _get_cached_price(self, cache_key: str) -> Optional[List[PriceQuote]]:
        """Get cached price if still valid."""
        if cache_key in self._price_cache:
            quotes, timestamp = self._price_cache[cache_key]
            age_sec = (datetime.utcnow() - timestamp).total_seconds()
            if age_sec < self._cache_ttl_sec:
                return quotes
        return None

    async def _cache_price(self, cache_key: str, quotes: List[PriceQuote]):
        """Cache price quotes."""
        self._price_cache[cache_key] = (quotes, datetime.utcnow())

    async def get_price_quotes(
        self,
        gpu_count: int,
        vram_per_gpu_gb: float,
        cpu_cores: int,
        ram_gb: float,
        duration_hours: int = 1,
    ) -> List[PriceQuote]:
        """
        Get price quotes from Prime Intellect.

        Prime supports 1-8 GPU pods, so we can get exact matches.
        """
        # Prime only supports 1-8 GPUs per pod
        if gpu_count > 8:
            return []  # Use SFCompute for > 8 GPUs

        cache_key = f"{gpu_count}_{vram_per_gpu_gb}_{duration_hours}"
        cached = await self._get_cached_price(cache_key)
        if cached is not None:
            return cached

        try:
            # Query Prime API for available GPU options
            # This is a placeholder - adjust based on actual Prime API
            async with httpx.AsyncClient() as client:
                # Search for GPU options matching requirements
                url = f"{self.api_url}/v1/gpus/search"
                response = await client.post(
                    url,
                    headers=self.base_headers,
                    json={
                        "gpu_count": gpu_count,
                        "min_vram_per_gpu_gb": vram_per_gpu_gb,
                        "min_cpu_cores": cpu_cores,
                        "min_ram_gb": ram_gb,
                    },
                    timeout=30.0,
                )

                if response.status_code == 404:
                    # API endpoint might not exist yet - return mock quotes
                    logger.warning("Prime API endpoint not found, using fallback pricing")
                    return await self._get_fallback_quotes(
                        gpu_count, vram_per_gpu_gb, cpu_cores, ram_gb
                    )

                response.raise_for_status()
                data = response.json()

                quotes = []
                for option in data.get("options", []):
                    price_per_hour = option.get("price_per_hour", 0.0)
                    price_per_gpu_hour = price_per_hour / gpu_count if gpu_count > 0 else 0.0

                    quotes.append(
                        PriceQuote(
                            provider=self.name,
                            instance_type=option.get("instance_type", f"{gpu_count}xGPU"),
                            gpu_count=gpu_count,
                            price_per_hour=price_per_hour,
                            price_per_gpu_hour=price_per_gpu_hour,
                            vram_per_gpu_gb=option.get("vram_per_gpu_gb", vram_per_gpu_gb),
                            cpu_per_node=option.get("cpu_cores", cpu_cores),
                            ram_per_node_gb=option.get("ram_gb", ram_gb),
                            region=option.get("region"),
                            available=option.get("available", True),
                            estimated_startup_sec=option.get("startup_sec", 60),
                        )
                    )

                # Sort by price (cheapest first)
                quotes.sort(key=lambda q: q.price_per_gpu_hour)
                await self._cache_price(cache_key, quotes)
                return quotes

        except httpx.HTTPError as e:
            logger.error(f"HTTP error querying Prime prices: {e}")
            # Fallback to estimated pricing
            return await self._get_fallback_quotes(
                gpu_count, vram_per_gpu_gb, cpu_cores, ram_gb
            )
        except Exception as e:
            logger.error(f"Failed to get Prime price quotes: {e}")
            return []

    async def _get_fallback_quotes(
        self,
        gpu_count: int,
        vram_per_gpu_gb: float,
        cpu_cores: int,
        ram_gb: float,
    ) -> List[PriceQuote]:
        """Get fallback price quotes when API is unavailable."""
        # Estimate Prime pricing (typically cheaper than SFCompute for small jobs)
        # These are placeholder values - adjust based on actual Prime pricing
        base_price_per_gpu_hour = 0.50  # Estimated $0.50/GPU-hour for Prime
        price_per_hour = base_price_per_gpu_hour * gpu_count

        return [
            PriceQuote(
                provider=self.name,
                instance_type=f"{gpu_count}xGPU-Prime",
                gpu_count=gpu_count,
                price_per_hour=price_per_hour,
                price_per_gpu_hour=base_price_per_gpu_hour,
                vram_per_gpu_gb=vram_per_gpu_gb,
                cpu_per_node=max(cpu_cores, 8),  # Minimum CPU
                ram_per_node_gb=max(ram_gb, 32.0),  # Minimum RAM
                available=True,
                estimated_startup_sec=60,
            )
        ]

    async def provision(
        self,
        instance_type: str,
        gpu_count: int,
        duration_hours: int,
        tier: str,
        max_price_per_hour: Optional[float] = None,
    ) -> ProvisionResult:
        """Provision Prime Intellect pod."""
        if gpu_count > 8:
            raise ValueError("Prime Intellect only supports 1-8 GPU pods")

        try:
            # Get cheapest quote
            quotes = await self.get_price_quotes(
                gpu_count=gpu_count,
                vram_per_gpu_gb=80.0,  # Default H100 VRAM
                cpu_cores=8,
                ram_gb=64.0,
                duration_hours=duration_hours,
            )

            if not quotes:
                raise ValueError("No Prime quotes available")

            # Filter by max price if specified
            valid_quotes = [
                q for q in quotes
                if max_price_per_hour is None or q.price_per_hour <= max_price_per_hour
            ]

            if not valid_quotes:
                raise ValueError(f"No Prime quotes under ${max_price_per_hour}/hour")

            best_quote = valid_quotes[0]  # Cheapest

            # Deploy pod via Prime API
            async with httpx.AsyncClient() as client:
                url = f"{self.api_url}/v1/pods/deploy"
                deploy_payload = {
                    "gpu_count": gpu_count,
                    "instance_type": best_quote.instance_type,
                    "duration_hours": duration_hours,
                    "docker_image": getattr(config, "runtime_image", "scalar/runtime:latest"),
                    "env": {
                        "SCALAR_TIER": tier,
                        "SCALAR_PROVIDER": self.name,
                    },
                }

                response = await client.post(
                    url,
                    headers=self.base_headers,
                    json=deploy_payload,
                    timeout=120.0,
                )

                if response.status_code == 404:
                    # API might not exist - create mock result
                    logger.warning("Prime deploy API not found, creating mock provision")
                    pod_id = str(uuid4())
                    start_time = datetime.utcnow()
                    end_time = start_time + timedelta(hours=duration_hours)

                    return ProvisionResult(
                        provider=self.name,
                        block_id=UUID(pod_id),
                        provider_instance_id=pod_id,
                        instance_type=best_quote.instance_type,
                        gpu_count=gpu_count,
                        node_count=1,  # Prime pods are single-node
                        cost_per_hour=best_quote.price_per_hour,
                        start_time=start_time,
                        end_time=end_time,
                        status="pending",
                        metadata={"pod_id": pod_id, "region": best_quote.region},
                    )

                response.raise_for_status()
                data = response.json()

                pod_id = data.get("pod_id") or data.get("id")
                if not pod_id:
                    raise ValueError("No pod_id in Prime response")

                start_time = datetime.fromisoformat(
                    data.get("start_time", datetime.utcnow().isoformat())
                )
                end_time = start_time + timedelta(hours=duration_hours)

                return ProvisionResult(
                    provider=self.name,
                    block_id=UUID(pod_id),
                    provider_instance_id=pod_id,
                    instance_type=best_quote.instance_type,
                    gpu_count=gpu_count,
                    node_count=1,
                    cost_per_hour=best_quote.price_per_hour,
                    start_time=start_time,
                    end_time=end_time,
                    status=data.get("status", "pending"),
                    metadata={
                        "pod_id": pod_id,
                        "region": best_quote.region,
                        "deployment_url": data.get("deployment_url"),
                    },
                )

        except Exception as e:
            logger.error(f"Failed to provision Prime capacity: {e}")
            raise

    async def terminate(self, provider_instance_id: str) -> bool:
        """Terminate Prime Intellect pod."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.api_url}/v1/pods/{provider_instance_id}/terminate"
                response = await client.post(
                    url,
                    headers=self.base_headers,
                    timeout=60.0,
                )

                if response.status_code == 404:
                    logger.warning(f"Prime terminate API not found for pod {provider_instance_id}")
                    return True  # Assume terminated if API doesn't exist

                response.raise_for_status()
                return True
        except Exception as e:
            logger.error(f"Failed to terminate Prime pod {provider_instance_id}: {e}")
            return False

    async def get_status(self, provider_instance_id: str) -> Dict:
        """Get Prime pod status."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.api_url}/v1/pods/{provider_instance_id}"
                response = await client.get(
                    url,
                    headers=self.base_headers,
                    timeout=30.0,
                )

                if response.status_code == 404:
                    return {
                        "provider": self.name,
                        "instance_id": provider_instance_id,
                        "status": "not_found",
                    }

                response.raise_for_status()
                data = response.json()
                return {
                    "provider": self.name,
                    "instance_id": provider_instance_id,
                    "status": data.get("status", "unknown"),
                    "gpu_count": data.get("gpu_count"),
                    "region": data.get("region"),
                }
        except Exception as e:
            logger.error(f"Failed to get Prime pod status: {e}")
            return {
                "provider": self.name,
                "instance_id": provider_instance_id,
                "status": "error",
                "error": str(e),
            }

    async def list_capacity(self, filter_active: bool = True) -> List[Dict]:
        """List Prime Intellect capacity."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.api_url}/v1/pods"
                params = {"status": "active"} if filter_active else {}
                response = await client.get(
                    url,
                    headers=self.base_headers,
                    params=params,
                    timeout=30.0,
                )

                if response.status_code == 404:
                    return []  # API might not exist

                response.raise_for_status()
                data = response.json()
                return data.get("pods", [])
        except Exception as e:
            logger.error(f"Failed to list Prime capacity: {e}")
            return []


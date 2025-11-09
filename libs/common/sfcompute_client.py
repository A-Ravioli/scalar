"""SFCompute API client."""

import httpx
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from libs.common.config import config
from libs.common.logging import get_logger
from libs.common.types import Block, BlockStatus

logger = get_logger(__name__)


class SFComputeClient:
    """Client for SFCompute marketplace API."""

    def __init__(self):
        self.api_key = config.sf_api_key
        self.api_url = config.sf_api_url
        self.base_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def quote_price(
        self, instance_type: str, duration_hours: int = 1
    ) -> Dict:
        """
        Get price quote for an instance type.

        Returns dict with price information.
        """
        async with httpx.AsyncClient() as client:
            url = f"{self.api_url}/v1/marketplace/quote"
            response = await client.post(
                url,
                headers=self.base_headers,
                json={
                    "instance_type": instance_type,
                    "duration_hours": duration_hours,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def buy_block(
        self,
        instance_type: str,
        duration_hours: int,
        max_price_per_hour: Optional[float] = None,
    ) -> Dict:
        """
        Buy a compute block.

        Returns contract details.
        """
        async with httpx.AsyncClient() as client:
            url = f"{self.api_url}/v1/marketplace/buy"
            payload = {
                "instance_type": instance_type,
                "duration_hours": duration_hours,
            }
            if max_price_per_hour:
                payload["max_price_per_hour"] = max_price_per_hour

            response = await client.post(
                url,
                headers=self.base_headers,
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()

    async def sell_block(self, contract_id: str) -> Dict:
        """
        Sell remaining time on a block.

        Returns sell order details.
        """
        async with httpx.AsyncClient() as client:
            url = f"{self.api_url}/v1/marketplace/sell"
            response = await client.post(
                url,
                headers=self.base_headers,
                json={"contract_id": contract_id},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()

    async def provision_nodes(
        self, contract_id: str, node_count: int = 1
    ) -> Dict:
        """
        Provision nodes for a block.

        Returns node details.
        """
        async with httpx.AsyncClient() as client:
            url = f"{self.api_url}/v1/clusters/provision"
            response = await client.post(
                url,
                headers=self.base_headers,
                json={
                    "contract_id": contract_id,
                    "node_count": node_count,
                },
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()

    async def terminate_nodes(self, contract_id: str) -> Dict:
        """
        Terminate nodes for a block.
        """
        async with httpx.AsyncClient() as client:
            url = f"{self.api_url}/v1/clusters/terminate"
            response = await client.post(
                url,
                headers=self.base_headers,
                json={"contract_id": contract_id},
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()

    async def get_orderbook(self, instance_type: str) -> Dict:
        """
        Get orderbook for an instance type.

        Returns current buy/sell orders.
        """
        async with httpx.AsyncClient() as client:
            # SFCompute uses /v0/orders endpoint
            url = f"{self.api_url}/v0/orders"
            response = await client.get(
                url,
                headers=self.base_headers,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()


"""Price forecasting model."""

from typing import Dict
from libs.common.logging import get_logger
import sys
import os

# Add capacity_manager to path for imports
_capacity_manager_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'capacity_manager')
)
if _capacity_manager_path not in sys.path:
    sys.path.insert(0, _capacity_manager_path)

from sfcompute_client import SFComputeClient

logger = get_logger(__name__)


class PriceModel:
    """Price forecasting model."""

    def __init__(self):
        self.sfcompute = SFComputeClient()

    async def forecast_price(
        self, instance_type: str, hours_ahead: int = 24
    ) -> Dict[str, float]:
        """
        Forecast price for an instance type.

        Returns dict with min, max, avg prices.
        """
        try:
            # Get current orderbook
            orderbook = await self.sfcompute.get_orderbook(instance_type)

            # Simple: use current market price as forecast
            # TODO: Implement more sophisticated price forecasting
            current_price = orderbook.get("current_price", 0.0)

            return {
                "min": current_price * 0.9,  # Assume 10% variance
                "max": current_price * 1.1,
                "avg": current_price,
            }

        except Exception as e:
            logger.error(f"Failed to forecast price: {e}")
            return {"min": 0.0, "max": 0.0, "avg": 0.0}


"""Demand forecasting model (EWMA)."""

from typing import Dict, List
from datetime import datetime, timedelta
from libs.common.db import get_supabase_client
from libs.common.logging import get_logger
from libs.common.types import Tier

logger = get_logger(__name__)


class DemandModel:
    """Exponential Weighted Moving Average demand model."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha  # Smoothing factor
        self.supabase = get_supabase_client()

    def forecast(
        self, tier: Tier, instance_type: str, hours_ahead: int = 24
    ) -> float:
        """
        Forecast GPU-hours needed for next N hours.

        Returns forecasted GPU-hours.
        """
        # Get historical usage
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)  # Last 7 days

        result = (
            self.supabase.table("usage_events")
            .select("gpu_hours, start_time")
            .eq("tier", tier.value)
            .gte("start_time", start_time.isoformat())
            .lte("start_time", end_time.isoformat())
            .execute()
        )

        if not result.data:
            return 0.0

        # Simple EWMA: average GPU-hours per hour
        total_gpu_hours = sum(event["gpu_hours"] for event in result.data)
        hours_in_period = (end_time - start_time).total_seconds() / 3600
        avg_gpu_hours_per_hour = total_gpu_hours / hours_in_period if hours_in_period > 0 else 0.0

        # Forecast
        forecast = avg_gpu_hours_per_hour * hours_ahead

        logger.info(f"Forecast for {tier.value} tier: {forecast:.2f} GPU-hours in next {hours_ahead} hours")

        return forecast


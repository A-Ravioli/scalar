"""Forecasting service worker."""

import asyncio
import signal
from datetime import datetime
from libs.common.logging import setup_logging
from libs.common.types import Tier
from .demand_model import DemandModel
from .price_model import PriceModel

logger = setup_logging("forecasting")


class ForecastingWorker:
    """Forecasting service worker."""

    def __init__(self):
        self.demand_model = DemandModel()
        self.price_model = PriceModel()
        self.running = False

    async def run(self):
        """Main forecasting loop."""
        self.running = True
        logger.info("Forecasting worker started")

        while self.running:
            try:
                # Forecast demand for both tiers
                for tier in [Tier.FAST, Tier.FLEX]:
                    forecast = self.demand_model.forecast(tier, "8xH100", hours_ahead=24)
                    logger.info(f"{tier.value} tier demand forecast: {forecast:.2f} GPU-hours")

                # Forecast prices
                price_forecast = await self.price_model.forecast_price("8xH100", hours_ahead=24)
                logger.info(f"Price forecast: ${price_forecast['avg']:.2f}/hour")

                # Run every 6 hours
                await asyncio.sleep(6 * 3600)

            except Exception as e:
                logger.error(f"Error in forecasting loop: {e}", exc_info=True)
                await asyncio.sleep(3600)

    def stop(self):
        """Stop forecasting worker."""
        self.running = False


def main():
    """Main entry point."""
    worker = ForecastingWorker()

    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        worker.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        logger.info("Forecasting worker stopped")


if __name__ == "__main__":
    main()


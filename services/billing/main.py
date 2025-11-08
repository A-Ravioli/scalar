"""Billing service worker."""

import asyncio
import signal
from datetime import datetime, timedelta
from libs.common.logging import setup_logging
from .usage_collector import UsageCollector
from .rating_engine import RatingEngine
from .stripe_client import StripeClient

logger = setup_logging("billing")


class BillingWorker:
    """Billing service worker."""

    def __init__(self):
        self.usage_collector = UsageCollector()
        self.rating_engine = RatingEngine()
        self.stripe_client = StripeClient()
        self.running = False

    async def run(self):
        """Main billing loop."""
        self.running = True
        logger.info("Billing worker started")

        while self.running:
            try:
                # Collect usage events
                usage_events = self.usage_collector.collect_from_completed_jobs()
                if usage_events:
                    self.usage_collector.save_usage_events(usage_events)

                # Generate monthly invoices (run once per day)
                now = datetime.utcnow()
                if now.hour == 0:  # Run at midnight
                    await self.generate_monthly_invoices()

                await asyncio.sleep(3600)  # Run every hour

            except Exception as e:
                logger.error(f"Error in billing loop: {e}", exc_info=True)
                await asyncio.sleep(3600)

    async def generate_monthly_invoices(self):
        """Generate monthly invoices for all users."""
        logger.info("Generating monthly invoices")

        # Get all users with usage in the last month
        period_end = datetime.utcnow()
        period_start = period_end - timedelta(days=30)

        # TODO: Get distinct users from usage_events
        # For now, this is a placeholder
        logger.info(f"Would generate invoices for period {period_start} to {period_end}")


def main():
    """Main entry point."""
    worker = BillingWorker()

    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        worker.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        logger.info("Billing worker stopped")


if __name__ == "__main__":
    main()


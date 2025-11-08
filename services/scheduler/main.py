"""Scheduler service entry point."""

import asyncio
import signal
from libs.common.logging import setup_logging
from .scheduler import Scheduler

logger = setup_logging("scheduler")


def main():
    """Main entry point."""
    scheduler = Scheduler()

    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        scheduler.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(scheduler.run())
    except KeyboardInterrupt:
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()


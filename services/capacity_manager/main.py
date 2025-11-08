"""Capacity Manager FastAPI application."""

import asyncio
import signal
from contextlib import asynccontextmanager
from fastapi import FastAPI
from libs.common.logging import setup_logging
from .capacity_state import CapacityState
from .autoscaler import Autoscaler
from .api import router, capacity_state

logger = setup_logging("capacity_manager")

autoscaler_task = None
autoscaler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global autoscaler_task, autoscaler

    # Startup
    logger.info("Starting Capacity Manager")
    autoscaler = Autoscaler(capacity_state)
    autoscaler_task = asyncio.create_task(autoscaler.run())

    yield

    # Shutdown
    logger.info("Shutting down Capacity Manager")
    if autoscaler:
        autoscaler.stop()
    if autoscaler_task:
        autoscaler_task.cancel()
        try:
            await autoscaler_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Scalar Capacity Manager",
    description="Capacity planning and autoscaling service",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)


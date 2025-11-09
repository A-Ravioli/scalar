#!/usr/bin/env python3
"""Simple mock server for testing the orderbook frontend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/orderbook")
async def get_orderbook(instance_type: str = "8xH100", node_count: int = 1):
    """Return mock orderbook data."""
    
    gpus_per_node = 8 if "8x" in instance_type else 1
    required_gpus = node_count * gpus_per_node
    
    # Mock orderbook data
    asks = [
        {"price": 24.00, "quantity_gpus": 32, "duration_hours": 168, "cumulative_quantity": None},
        {"price": 24.50, "quantity_gpus": 64, "duration_hours": 168, "cumulative_quantity": None},
        {"price": 25.00, "quantity_gpus": 128, "duration_hours": 168, "cumulative_quantity": None},
        {"price": 25.50, "quantity_gpus": 48, "duration_hours": 168, "cumulative_quantity": None},
        {"price": 26.00, "quantity_gpus": 96, "duration_hours": 720, "cumulative_quantity": None},
        {"price": 27.00, "quantity_gpus": 64, "duration_hours": 720, "cumulative_quantity": None},
        {"price": 28.50, "quantity_gpus": 32, "duration_hours": 168, "cumulative_quantity": None},
    ]
    
    bids = [
        {"price": 23.00, "quantity_gpus": 16, "duration_hours": 24, "cumulative_quantity": None},
        {"price": 22.50, "quantity_gpus": 24, "duration_hours": 168, "cumulative_quantity": None},
        {"price": 22.00, "quantity_gpus": 48, "duration_hours": 168, "cumulative_quantity": None},
        {"price": 21.00, "quantity_gpus": 32, "duration_hours": 720, "cumulative_quantity": None},
    ]
    
    # Calculate optimal price
    sorted_asks = sorted(asks, key=lambda x: x["price"])
    cumulative = 0
    optimal_idx = None
    for i, ask in enumerate(sorted_asks):
        cumulative += ask["quantity_gpus"]
        if cumulative >= required_gpus and optimal_idx is None:
            optimal_idx = i
    
    optimal_price = sorted_asks[optimal_idx]["price"] if optimal_idx is not None else asks[0]["price"]
    
    return {
        "instance_type": instance_type,
        "asks": asks,
        "bids": bids,
        "optimal_price": optimal_price,
        "optimal_index": optimal_idx,
        "spread": 24.00 - 23.00,
        "total_ask_liquidity": sum(a["quantity_gpus"] for a in asks),
        "total_bid_liquidity": sum(b["quantity_gpus"] for b in bids),
        "last_updated": "2025-11-09T12:00:00Z",
        "metadata": {
            "required_gpus": required_gpus,
            "node_count": node_count,
            "gpus_per_node": gpus_per_node,
            "mock_data": True,
        }
    }

if __name__ == "__main__":
    print("Starting mock orderbook API server on http://localhost:8000")
    print("Frontend can connect to: http://localhost:8000/orderbook")
    uvicorn.run(app, host="0.0.0.0", port=8000)


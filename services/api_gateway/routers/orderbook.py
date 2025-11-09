"""Orderbook routes for SFCompute marketplace visualization."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from libs.common.logging import get_logger
from routers.auth import verify_api_key
import sys
import os

# Add capacity_manager to path for SFComputeClient import
_capacity_manager_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'capacity_manager')
)
if _capacity_manager_path not in sys.path:
    sys.path.insert(0, _capacity_manager_path)

from sfcompute_client import SFComputeClient

logger = get_logger(__name__)
router = APIRouter()


class OrderbookEntry(BaseModel):
    """Single orderbook entry (bid or ask)."""
    price: float
    quantity_gpus: int
    duration_hours: int
    cumulative_quantity: Optional[int] = None


class OrderbookResponse(BaseModel):
    """Orderbook response with optimal price recommendation."""
    instance_type: str
    asks: List[OrderbookEntry]
    bids: List[OrderbookEntry]
    optimal_price: Optional[float] = None
    optimal_index: Optional[int] = None  # Index in asks list
    spread: Optional[float] = None
    total_ask_liquidity: int = 0
    total_bid_liquidity: int = 0
    last_updated: str
    metadata: dict = {}


def calculate_optimal_price(
    asks: List[OrderbookEntry],
    required_gpus: int
) -> tuple[Optional[float], Optional[int]]:
    """
    Calculate optimal price to place a buy order.
    
    Strategy:
    1. Find orders that can fulfill the requirement
    2. Balance between best price and execution certainty (depth)
    3. Prefer price levels with good liquidity to minimize slippage
    
    Returns: (optimal_price, optimal_index_in_asks)
    """
    if not asks or required_gpus <= 0:
        return None, None
    
    # Sort asks by price (ascending - lowest first)
    sorted_asks = sorted(asks, key=lambda x: x.price)
    
    # Calculate cumulative quantities
    cumulative = 0
    for i, ask in enumerate(sorted_asks):
        cumulative += ask.quantity_gpus
        sorted_asks[i].cumulative_quantity = cumulative
    
    # Find first price level that has sufficient cumulative liquidity
    min_viable_idx = None
    for i, ask in enumerate(sorted_asks):
        if ask.cumulative_quantity >= required_gpus:
            min_viable_idx = i
            break
    
    if min_viable_idx is None:
        # Not enough liquidity in orderbook
        # Return the best available price
        return sorted_asks[0].price, 0
    
    # Score each viable price level
    # Score = depth_score * price_score * liquidity_score
    best_score = -1
    best_idx = min_viable_idx
    
    for i in range(min_viable_idx, len(sorted_asks)):
        ask = sorted_asks[i]
        
        # Depth score: how much liquidity is available at this level
        depth_score = min(ask.cumulative_quantity / required_gpus, 2.0)
        
        # Price score: prefer lower prices (inverse relationship)
        # Normalize against the minimum viable price
        min_price = sorted_asks[min_viable_idx].price
        if min_price > 0:
            price_score = min_price / ask.price
        else:
            price_score = 1.0
        
        # Liquidity at this specific level (not cumulative)
        level_liquidity = ask.quantity_gpus
        liquidity_score = min(level_liquidity / required_gpus, 1.5)
        
        # Combined score (weighted)
        score = (depth_score * 0.4) + (price_score * 0.4) + (liquidity_score * 0.2)
        
        if score > best_score:
            best_score = score
            best_idx = i
        
        # Don't look too far up the orderbook (diminishing returns)
        if i > min_viable_idx + 5:
            break
    
    return sorted_asks[best_idx].price, best_idx


@router.get("", response_model=OrderbookResponse)
async def get_orderbook(
    instance_type: str = Query(..., description="Instance type (e.g., '8xH100')"),
    node_count: int = Query(1, ge=1, description="Number of nodes needed"),
    user_id: UUID = Depends(verify_api_key),
):
    """
    Get orderbook for an instance type with optimal price recommendation.
    
    The optimal price is calculated based on:
    - Order depth (sufficient liquidity to fulfill order)
    - Price (lower is better)
    - Liquidity concentration (avoid thin markets)
    """
    try:
        # Calculate required GPUs (assuming 8 GPUs per node for 8xH100)
        gpus_per_node = 8 if "8x" in instance_type else 1
        required_gpus = node_count * gpus_per_node
        
        # Get orderbook from SFCompute
        sf_client = SFComputeClient()
        orderbook_data = await sf_client.get_orderbook(instance_type)
        
        # Parse orderbook data from SFCompute's /v0/orders endpoint
        # Real format: {"object": "list", "data": [...], "has_more": false}
        orders_list = orderbook_data.get("data", [])
        
        # Separate into asks (sell orders) and bids (buy orders)
        asks_raw = []
        bids_raw = []
        
        for order in orders_list:
            # Determine if this is a buy or sell order
            order_type = order.get("type", "")
            price = order.get("price", 0.0)
            quantity = order.get("quantity", 0)
            duration = order.get("duration_hours", 24)
            
            order_data = {
                "price": price,
                "quantity": quantity,
                "duration": duration
            }
            
            if order_type == "sell":
                asks_raw.append(order_data)
            elif order_type == "buy":
                bids_raw.append(order_data)
        
        # Convert to our model
        asks = [
            OrderbookEntry(
                price=ask.get("price", 0.0),
                quantity_gpus=ask.get("quantity", 0),
                duration_hours=ask.get("duration", 24),
            )
            for ask in asks_raw
        ]
        
        bids = [
            OrderbookEntry(
                price=bid.get("price", 0.0),
                quantity_gpus=bid.get("quantity", 0),
                duration_hours=bid.get("duration", 24),
            )
            for bid in bids_raw
        ]
        
        # Calculate optimal price
        optimal_price, optimal_idx = calculate_optimal_price(asks, required_gpus)
        
        # Calculate spread
        spread = None
        if asks and bids:
            lowest_ask = min(ask.price for ask in asks)
            highest_bid = max(bid.price for bid in bids)
            spread = lowest_ask - highest_bid
        
        # Calculate total liquidity
        total_ask_liquidity = sum(ask.quantity_gpus for ask in asks)
        total_bid_liquidity = sum(bid.quantity_gpus for bid in bids)
        
        # If orderbook is empty, use mock data for demonstration
        if not asks and not bids:
            logger.warning("Orderbook is empty, using mock data for demonstration")
            return _get_mock_orderbook(instance_type, node_count)
        
        return OrderbookResponse(
            instance_type=instance_type,
            asks=asks,
            bids=bids,
            optimal_price=optimal_price,
            optimal_index=optimal_idx,
            spread=spread,
            total_ask_liquidity=total_ask_liquidity,
            total_bid_liquidity=total_bid_liquidity,
            last_updated=orderbook_data.get("timestamp", ""),
            metadata={
                "required_gpus": required_gpus,
                "node_count": node_count,
                "gpus_per_node": gpus_per_node,
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get orderbook: {e}", exc_info=True)
        
        # If SFCompute API is unavailable, return mock data for development
        if "404" in str(e) or "Not Found" in str(e):
            logger.warning("SFCompute orderbook API returned 404, using mock data")
            return _get_mock_orderbook(instance_type, node_count)
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch orderbook: {str(e)}"
        )


def _get_mock_orderbook(instance_type: str, node_count: int) -> OrderbookResponse:
    """Generate mock orderbook data for development/testing."""
    gpus_per_node = 8 if "8x" in instance_type else 1
    required_gpus = node_count * gpus_per_node
    
    # Mock asks (sell orders) - people offering compute
    asks = [
        OrderbookEntry(price=24.00, quantity_gpus=32, duration_hours=168),
        OrderbookEntry(price=24.50, quantity_gpus=64, duration_hours=168),
        OrderbookEntry(price=25.00, quantity_gpus=128, duration_hours=168),
        OrderbookEntry(price=25.50, quantity_gpus=48, duration_hours=168),
        OrderbookEntry(price=26.00, quantity_gpus=96, duration_hours=720),
        OrderbookEntry(price=27.00, quantity_gpus=64, duration_hours=720),
        OrderbookEntry(price=28.50, quantity_gpus=32, duration_hours=168),
    ]
    
    # Mock bids (buy orders) - people wanting compute
    bids = [
        OrderbookEntry(price=23.00, quantity_gpus=16, duration_hours=24),
        OrderbookEntry(price=22.50, quantity_gpus=24, duration_hours=168),
        OrderbookEntry(price=22.00, quantity_gpus=48, duration_hours=168),
        OrderbookEntry(price=21.00, quantity_gpus=32, duration_hours=720),
    ]
    
    # Calculate optimal price
    optimal_price, optimal_idx = calculate_optimal_price(asks, required_gpus)
    
    # Calculate spread
    lowest_ask = min(ask.price for ask in asks)
    highest_bid = max(bid.price for bid in bids)
    spread = lowest_ask - highest_bid
    
    return OrderbookResponse(
        instance_type=instance_type,
        asks=asks,
        bids=bids,
        optimal_price=optimal_price,
        optimal_index=optimal_idx,
        spread=spread,
        total_ask_liquidity=sum(ask.quantity_gpus for ask in asks),
        total_bid_liquidity=sum(bid.quantity_gpus for bid in bids),
        last_updated="2025-11-09T12:00:00Z",
        metadata={
            "required_gpus": required_gpus,
            "node_count": node_count,
            "gpus_per_node": gpus_per_node,
            "mock_data": True,
        }
    )


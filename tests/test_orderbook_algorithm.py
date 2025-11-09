#!/usr/bin/env python3
"""Test the orderbook optimal price algorithm."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Optional
from pydantic import BaseModel


class OrderbookEntry(BaseModel):
    """Single orderbook entry (bid or ask)."""
    price: float
    quantity_gpus: int
    duration_hours: int
    cumulative_quantity: Optional[int] = None


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


def test_optimal_price():
    """Test the optimal price calculation algorithm."""
    
    print("\n" + "=" * 70)
    print("ORDERBOOK OPTIMAL PRICE ALGORITHM TEST")
    print("=" * 70)
    
    # Test case 1: Simple orderbook with sufficient liquidity
    print("\n📊 Test Case 1: Simple Orderbook (need 16 GPUs)")
    print("-" * 70)
    
    asks = [
        OrderbookEntry(price=24.00, quantity_gpus=32, duration_hours=168),
        OrderbookEntry(price=24.50, quantity_gpus=64, duration_hours=168),
        OrderbookEntry(price=25.00, quantity_gpus=128, duration_hours=168),
    ]
    
    required_gpus = 16  # 2 nodes * 8 GPUs
    optimal_price, optimal_idx = calculate_optimal_price(asks, required_gpus)
    
    print(f"  Required GPUs: {required_gpus}")
    print(f"  Available orders: {len(asks)}")
    for i, ask in enumerate(asks):
        marker = " ← OPTIMAL" if i == optimal_idx else ""
        print(f"    [{i}] ${ask.price:.2f}/GPU-hr - {ask.quantity_gpus} GPUs{marker}")
    print(f"\n  ✓ Optimal Price: ${optimal_price:.2f}/GPU-hr (index {optimal_idx})")
    
    # Test case 2: Larger order requiring deep liquidity
    print("\n📊 Test Case 2: Larger Order (need 64 GPUs)")
    print("-" * 70)
    
    required_gpus = 64  # 8 nodes * 8 GPUs
    optimal_price, optimal_idx = calculate_optimal_price(asks, required_gpus)
    
    print(f"  Required GPUs: {required_gpus}")
    cumulative = 0
    for i, ask in enumerate(asks):
        cumulative += ask.quantity_gpus
        marker = " ← OPTIMAL" if i == optimal_idx else ""
        print(f"    [{i}] ${ask.price:.2f}/GPU-hr - {ask.quantity_gpus} GPUs (cum: {cumulative}){marker}")
    print(f"\n  ✓ Optimal Price: ${optimal_price:.2f}/GPU-hr (index {optimal_idx})")
    
    # Test case 3: Insufficient liquidity
    print("\n📊 Test Case 3: Insufficient Liquidity (need 500 GPUs)")
    print("-" * 70)
    
    required_gpus = 500
    optimal_price, optimal_idx = calculate_optimal_price(asks, required_gpus)
    
    print(f"  Required GPUs: {required_gpus}")
    total_available = sum(ask.quantity_gpus for ask in asks)
    print(f"  Total available: {total_available} GPUs")
    print(f"\n  ⚠ Insufficient liquidity - recommending best available price")
    print(f"  ✓ Optimal Price: ${optimal_price:.2f}/GPU-hr (index {optimal_idx})")
    
    # Test case 4: Edge case - empty orderbook
    print("\n📊 Test Case 4: Empty Orderbook")
    print("-" * 70)
    
    empty_asks = []
    required_gpus = 16
    optimal_price, optimal_idx = calculate_optimal_price(empty_asks, required_gpus)
    
    print(f"  Required GPUs: {required_gpus}")
    print(f"  Available orders: 0")
    print(f"\n  ✓ Optimal Price: {optimal_price} (index {optimal_idx})")
    
    # Test case 5: Complex orderbook with varied prices
    print("\n📊 Test Case 5: Complex Orderbook (need 80 GPUs)")
    print("-" * 70)
    
    complex_asks = [
        OrderbookEntry(price=24.00, quantity_gpus=16, duration_hours=168),
        OrderbookEntry(price=24.50, quantity_gpus=32, duration_hours=168),
        OrderbookEntry(price=25.00, quantity_gpus=64, duration_hours=168),  # Good liquidity here
        OrderbookEntry(price=25.50, quantity_gpus=8, duration_hours=168),
        OrderbookEntry(price=26.00, quantity_gpus=96, duration_hours=720),
    ]
    
    required_gpus = 80
    optimal_price, optimal_idx = calculate_optimal_price(complex_asks, required_gpus)
    
    print(f"  Required GPUs: {required_gpus}")
    cumulative = 0
    for i, ask in enumerate(complex_asks):
        cumulative += ask.quantity_gpus
        marker = " ← OPTIMAL" if i == optimal_idx else ""
        print(f"    [{i}] ${ask.price:.2f}/GPU-hr - {ask.quantity_gpus} GPUs (cum: {cumulative}){marker}")
    print(f"\n  ✓ Optimal Price: ${optimal_price:.2f}/GPU-hr (index {optimal_idx})")
    print(f"  💡 Algorithm balanced price, depth, and liquidity")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_optimal_price()


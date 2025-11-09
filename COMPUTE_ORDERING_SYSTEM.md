# Compute Ordering Algorithm & System Documentation

## Overview

The Scalar compute platform implements a sophisticated orderbook-based system for procuring GPU compute capacity from multiple providers (SFCompute and Prime Intellect). The system uses a marketplace-style orderbook to visualize supply and demand, and an intelligent algorithm to determine optimal pricing for compute purchases.

---

## System Architecture

```
┌─────────────────┐
│   Frontend UI   │
│  (Orderbook     │
│   Visualization)│
└────────┬────────┘
         │
         │ HTTP GET /orderbook
         │
┌────────▼─────────────────────────────────────┐
│         API Gateway                          │
│  ┌──────────────────────────────────────┐   │
│  │  /orderbook endpoint                 │   │
│  │  - Fetches from SFCompute            │   │
│  │  - Calculates optimal price          │   │
│  │  - Returns formatted orderbook        │   │
│  └──────────────────────────────────────┘   │
└────────┬─────────────────────────────────────┘
         │
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────────────┐
│SFCompute│ │ Prime Intellect│
│Marketplace│ │   (Optional)   │
│          │ │                │
│/v0/orders│ │                │
└──────────┘ └────────────────┘
```

---

## Core Components

### 1. Orderbook Data Structure

The orderbook contains two types of orders:

- **Asks (Sell Orders)**: Providers/users offering compute for sale
- **Bids (Buy Orders)**: Users looking to purchase compute

Each order contains:
- `price`: Price per GPU-hour (in USD)
- `quantity_gpus`: Number of GPUs available/requested
- `duration_hours`: Duration of the order
- `cumulative_quantity`: Running total of liquidity (calculated)

### 2. Optimal Price Calculation Algorithm

The algorithm (`calculate_optimal_price`) determines the best price to place a buy order by balancing three factors:

#### Algorithm Flow

```
┌─────────────────────────────────────────┐
│  Input: asks[], required_gpus           │
└──────────────┬──────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Sort asks by price   │
    │ (ascending)          │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Calculate cumulative │
    │ quantities           │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Find minimum viable  │
    │ index (sufficient    │
    │ cumulative liquidity)│
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Score each viable    │
    │ price level:         │
    │                      │
    │ • Depth Score        │
    │ • Price Score        │
    │ • Liquidity Score    │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Return best scoring  │
    │ price & index        │
    └──────────────────────┘
```

#### Scoring Formula

For each viable price level (from `min_viable_idx` to `min_viable_idx + 5`):

```python
# Depth Score: How much liquidity is available at this level
depth_score = min(cumulative_quantity / required_gpus, 2.0)

# Price Score: Prefer lower prices (inverse relationship)
price_score = min_viable_price / current_price

# Liquidity Score: Liquidity at this specific level
liquidity_score = min(level_liquidity / required_gpus, 1.5)

# Combined Score (weighted)
score = (depth_score * 0.4) + (price_score * 0.4) + (liquidity_score * 0.2)
```

**Weighting Rationale:**
- **40% Depth**: Ensures sufficient cumulative liquidity to fulfill the order
- **40% Price**: Prioritizes lower prices for cost efficiency
- **20% Liquidity**: Prefers price levels with concentrated liquidity to minimize slippage

#### Example Calculation

Given:
- Required GPUs: 64
- Orderbook asks:
  - $24.00: 32 GPUs (cum: 32)
  - $24.50: 64 GPUs (cum: 96) ← Minimum viable
  - $25.00: 128 GPUs (cum: 224)
  - $25.50: 48 GPUs (cum: 272)

**At $24.50 (index 1):**
- Depth score: min(96/64, 2.0) = 1.5
- Price score: 24.50/24.50 = 1.0
- Liquidity score: min(64/64, 1.5) = 1.0
- **Total: (1.5 × 0.4) + (1.0 × 0.4) + (1.0 × 0.2) = 1.2**

**At $25.00 (index 2):**
- Depth score: min(224/64, 2.0) = 2.0
- Price score: 24.50/25.00 = 0.98
- Liquidity score: min(128/64, 1.5) = 1.5
- **Total: (2.0 × 0.4) + (0.98 × 0.4) + (1.5 × 0.2) = 1.492**

**Result:** $25.00 wins due to superior depth and liquidity scores, despite slightly higher price.

---

## System Flow

### 1. Orderbook Request Flow

```
User Request
    │
    ▼
Frontend: api.getOrderbook(instanceType, nodeCount)
    │
    ▼
API Gateway: GET /orderbook?instance_type=8xH100&node_count=1
    │
    ├─► Calculate required_gpus (node_count × gpus_per_node)
    │
    ├─► SFComputeClient.get_orderbook(instance_type)
    │   │
    │   └─► GET /v0/orders from SFCompute API
    │
    ├─► Parse orders into asks[] and bids[]
    │
    ├─► calculate_optimal_price(asks, required_gpus)
    │   │
    │   └─► Returns (optimal_price, optimal_index)
    │
    ├─► Calculate spread (lowest_ask - highest_bid)
    │
    └─► Return OrderbookResponse
```

### 2. Autoscaler Integration

The autoscaler uses the orderbook system indirectly through provider selection:

```
Autoscaler Loop
    │
    ├─► Check queue for pending jobs
    │
    ├─► Calculate required capacity
    │
    ├─► select_provider(gpu_count, ...)
    │   │
    │   ├─► If gpu_count ≤ 8: Try Prime Intellect
    │   │   │
    │   │   └─► Check price quotes & startup SLA
    │   │
    │   └─► If gpu_count ≥ 8: Use SFCompute
    │
    └─► provision_capacity()
        │
        └─► Provider.buy_block() or Provider.provision()
            │
            └─► Uses marketplace orderbook to match orders
```

---

## Visual Orderbook Representation

### Example Orderbook State

```
┌─────────────────────────────────────────────────────────────┐
│                    ORDERBOOK: 8xH100                        │
│              Required: 64 GPUs (8 nodes × 8 GPUs)          │
└─────────────────────────────────────────────────────────────┘

ASKS (Sellers) ────────────────────────────  BIDS (Buyers)
────────────────────────────────────────────  ────────────────────────
Price    Qty    Cum    Duration              Price    Qty    Duration
────────────────────────────────────────────  ────────────────────────
$24.00   32     32     1w                     $23.00   16     1d
$24.50   64     96     1w  ← OPTIMAL          $22.50   24     1w
$25.00  128    224     1w                     $22.00   48     1w
$25.50   48    272     1w                     $21.00   32     1mo
$26.00   96    368     1mo
$27.00   64    432     1mo
────────────────────────────────────────────  ────────────────────────
Total: 432 GPUs                               Total: 120 GPUs
Spread: $1.00 ($24.00 - $23.00)
Optimal Price: $24.50/GPU-hr
```

### Key Metrics

- **Spread**: Difference between lowest ask and highest bid
- **Total Ask Liquidity**: Sum of all GPUs available for sale
- **Total Bid Liquidity**: Sum of all GPUs requested
- **Optimal Price**: Recommended price based on algorithm
- **Optimal Index**: Position in asks array for optimal price

---

## Algorithm Edge Cases

### 1. Insufficient Liquidity

If cumulative liquidity never reaches `required_gpus`:
- Returns the best available price (lowest ask)
- Index set to 0
- Frontend can display warning about insufficient liquidity

### 2. Empty Orderbook

If no orders exist:
- Returns `None` for optimal price
- Falls back to mock data for development/demo
- Logs warning for monitoring

### 3. Single Large Order

If one order has sufficient liquidity:
- Algorithm still evaluates multiple price levels
- May recommend a slightly higher price if it offers better depth/liquidity scores

---

## Integration with Autoscaler

The autoscaler uses provider selection logic that considers:

1. **GPU Count Thresholds**:
   - < 8 GPUs: Prefer Prime Intellect (if available and within price cap)
   - ≥ 8 GPUs: Use SFCompute (8xH100 instances)

2. **Consolidation Logic**:
   - Tracks sustained demand over time window
   - Prefers SFCompute for sustained demand ≥ 8 GPUs
   - Helps reduce fragmentation

3. **Price Constraints**:
   - Flex tier: `max_price_per_hour = $10.0`
   - Fast tier: No explicit cap (prioritizes speed)

---

## API Endpoints

### GET /orderbook

**Query Parameters:**
- `instance_type` (required): e.g., "8xH100"
- `node_count` (optional, default=1): Number of nodes needed

**Response:**
```json
{
  "instance_type": "8xH100",
  "asks": [
    {
      "price": 24.00,
      "quantity_gpus": 32,
      "duration_hours": 168,
      "cumulative_quantity": 32
    }
  ],
  "bids": [...],
  "optimal_price": 24.50,
  "optimal_index": 1,
  "spread": 1.00,
  "total_ask_liquidity": 432,
  "total_bid_liquidity": 120,
  "last_updated": "2025-01-09T12:00:00Z",
  "metadata": {
    "required_gpus": 64,
    "node_count": 8,
    "gpus_per_node": 8
  }
}
```

---

## Frontend Visualization

The frontend (`/orderbook` page) displays:

1. **Input Controls**:
   - Instance type selector
   - Node count input

2. **Orderbook Tables**:
   - Asks table (red theme) with optimal price highlighted
   - Bids table (green theme)

3. **Metrics Dashboard**:
   - Optimal price recommendation
   - Spread visualization
   - Total liquidity indicators
   - Last updated timestamp

4. **Visual Indicators**:
   - Optimal price row highlighted
   - Cumulative quantity bars
   - Price trend indicators

---

## Algorithm Performance Characteristics

### Time Complexity
- **Sorting**: O(n log n) where n = number of asks
- **Cumulative calculation**: O(n)
- **Scoring loop**: O(min(n, 6)) - limited to 5 levels beyond minimum viable
- **Overall**: O(n log n)

### Space Complexity
- **O(n)**: Stores sorted asks array with cumulative quantities

### Optimization Notes
- Algorithm limits search to 5 levels beyond minimum viable to prevent over-optimization
- Early termination if insufficient liquidity
- Efficient scoring with pre-calculated cumulative quantities

---

## Future Enhancements

Potential improvements to the system:

1. **Dynamic Weighting**: Adjust scoring weights based on market conditions
2. **Historical Analysis**: Track optimal price accuracy over time
3. **Multi-Provider Aggregation**: Combine orderbooks from multiple providers
4. **Price Prediction**: ML-based price forecasting
5. **Slippage Modeling**: Better estimate of execution cost at different price levels
6. **Order Placement**: Direct integration to place orders at optimal price

---

## Testing

The algorithm is tested with multiple scenarios:

1. **Simple orderbook** with sufficient liquidity
2. **Large orders** requiring deep liquidity
3. **Insufficient liquidity** edge case
4. **Empty orderbook** edge case
5. **Complex orderbook** with varied prices and liquidity

See `tests/test_orderbook_algorithm.py` for detailed test cases.

---

## Summary

The compute ordering system provides:

✅ **Marketplace Visualization**: Real-time orderbook from SFCompute  
✅ **Intelligent Pricing**: Multi-factor algorithm for optimal price selection  
✅ **Provider Integration**: Seamless integration with SFCompute and Prime Intellect  
✅ **Autoscaling Support**: Helps autoscaler make cost-effective provisioning decisions  
✅ **User-Friendly UI**: Clear visualization of market conditions and recommendations  

The system balances cost efficiency, execution certainty, and liquidity concentration to provide optimal pricing recommendations for GPU compute purchases.


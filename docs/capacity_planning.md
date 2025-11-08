# Capacity Planning Documentation

## Overview

The Capacity Manager maintains capacity state and autoscales both Fast and Flex tiers based on demand and pricing.

## Capacity State

Maintains in-memory view of:
- Active blocks (SFCompute contracts)
- Nodes spawned from blocks
- GPU assignments and usage
- Reservations

Synced with database periodically.

## Autoscaling

### Fast Tier

**Goal**: Keep warm capacity for low latency

**Strategy**:
1. Monitor utilization and queue length
2. Forecast demand (simple: queue + current usage)
3. Buy blocks if capacity shortfall
4. Mark blocks for non-renewal if over-capacity

**Configuration**:
- `FAST_TIER_SAFETY_FACTOR`: Multiplier for target capacity (default: 1.5)
- `AUTOSCALER_INTERVAL_SEC`: How often to run (default: 60s)

### Flex Tier

**Goal**: Minimize cost, accept delays

**Strategy**:
1. Monitor queue length
2. Buy blocks only when:
   - Queue is non-empty
   - Price is below threshold
3. Backfill idle Fast tier capacity

## SFCompute Integration

The SFCompute client handles:
- Price quotes
- Buying blocks
- Selling blocks
- Node provisioning
- Node termination
- Orderbook queries

## Block Lifecycle

1. **Buy**: Purchase contract from SFCompute
2. **Provision**: Spawn nodes for block
3. **Active**: Nodes available for jobs
4. **Selling**: Marked for sale (optional)
5. **Sold/Terminated**: Block ended

## Capacity API

### Get Snapshot

```http
GET /capacity_snapshot?tier=FAST
```

Returns current capacity snapshot.

### Reserve

```http
POST /reserve
{
  "job_id": "uuid",
  "node_id": "uuid",
  "gpu_indices": [0, 1],
  "expires_in_sec": 300
}
```

Reserves capacity for a job.

### Release

```http
POST /release
{
  "reservation_id": "uuid"
}
```

Releases a reservation.


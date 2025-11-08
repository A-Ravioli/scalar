# Scheduler Documentation

## Overview

The scheduler is responsible for placing jobs onto compute nodes using a bin-packing algorithm. It consumes jobs from a queue and makes placement decisions based on available capacity.

## Architecture

```
Job Queue → Scheduler → Bin-Packer → Capacity Manager → Runtime Agent
```

## Bin-Packing Algorithm

### Resource Model

- **Blocks**: SFCompute contracts
- **Nodes**: Compute instances spawned from blocks
- **GPUs**: GPU slots on nodes with VRAM
- **Jobs**: Tasks with resource requirements

### Algorithm Steps

1. **Filter Candidates**: Filter nodes by:
   - Tier match
   - Sufficient time remaining
   - Sufficient CPU/RAM
   - Sufficient GPU VRAM

2. **Score Nodes**: Calculate score using:
   ```
   score = w1 * utilization + w2 * (-fragmentation) + w3 * (-expiry_penalty)
   ```

3. **Select Best**: Choose highest-scoring node

4. **Reserve**: Lock capacity via Capacity Manager

5. **Place**: Send placement request to Runtime Agent

### Scoring Heuristics

- **Utilization**: Prefer nodes that will have higher utilization after placement
- **Fragmentation**: Penalize non-contiguous GPU selection
- **Expiry**: Penalize nodes expiring soon

## Capacity Escalation

If no node fits:

- **FAST tier**: Request more capacity from Capacity Manager
- **FLEX tier**: Queue job for later

## Configuration

- `BIN_PACK_SAFETY_MARGIN_SEC`: Safety margin for node expiry (default: 300s)

## Future Improvements

- Multi-GPU graph-aware packing
- MIG (Multi-Instance GPU) support
- Image-based co-location for caching
- Short vs long job co-location


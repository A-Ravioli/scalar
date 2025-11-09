# Bin Packing Algorithm & System Documentation

## Overview

The Scalar platform uses a **multi-dimensional bin-packing algorithm** to efficiently place GPU compute jobs onto available nodes. This is the core value proposition: transforming large, lumpy compute blocks (like 8xH100 instances) into smooth, serverless slices that can be allocated to individual jobs.

---

## The Bin Packing Problem

### Problem Statement

We need to pack **jobs** (items) into **nodes** (bins) with multiple resource constraints:

- **Bins** = Compute nodes (each with multiple GPUs)
- **Items** = Jobs requiring GPU compute
- **Dimensions**:
  - GPU count (how many GPUs the job needs)
  - VRAM per GPU (memory requirement per GPU)
  - CPU cores (node-level resource)
  - RAM (node-level resource)
  - Time (node must not expire before job completes)

### Goal

**Maximize utilization** while:
- Respecting all resource constraints
- Minimizing fragmentation (prefer contiguous GPU allocations)
- Avoiding premature node expiry
- Optimizing for provider characteristics (SFCompute vs Prime)

---

## Resource Model

### Hierarchical Structure

```
Block (SFCompute Contract)
  └─ Node (Compute Instance)
      └─ GPU Slots (8 GPUs per node for SFCompute)
          └─ VRAM Allocation per GPU
```

### Data Structures

#### Block
```python
Block:
  - id: UUID
  - provider: "sfcompute" | "prime"
  - instance_type: "8xH100"
  - gpus_per_node: 8
  - vram_per_gpu_gb: 80.0
  - cpu_per_node: 64
  - ram_per_node_gb: 512.0
  - start_time: datetime
  - end_time: datetime  # Contract expiry
  - cost_per_hour: float
  - tier: "FAST" | "FLEX"
```

#### Node
```python
Node:
  - id: UUID
  - block_id: UUID
  - tier: "FAST" | "FLEX"
  - gpu_slots: List[GPUAssignment]
  - cpu_used: float
  - ram_used_gb: float
```

#### GPU Assignment
```python
GPUAssignment:
  - index: int  # GPU index on node (0-7)
  - vram_total_gb: float  # Total VRAM (e.g., 80GB)
  - vram_used_gb: float   # Currently used VRAM
  - job_id: UUID | None   # Which job is using this GPU
```

#### Job
```python
Job:
  - id: UUID
  - tier: "FAST" | "FLEX"
  - gpu_count: int        # How many GPUs needed
  - vram_per_gpu_gb: float  # VRAM requirement per GPU
  - cpu_cores: int        # CPU requirement
  - ram_gb: float         # RAM requirement
  - expected_duration_sec: int  # How long job will run
  - priority: int         # Higher = more important
```

---

## Algorithm Flow

### High-Level Process

```
┌─────────────────┐
│  Job Queue      │
│  (Pending Job)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Scheduler.process_job() │
└────────┬────────────────┘
         │
         ├─► Load job from DB
         │
         ├─► Get capacity snapshot
         │   (all available nodes)
         │
         └─► schedule_job(job, nodes)
             │
             ├─► Filter candidate nodes
             │   ├─ Tier match?
             │   ├─ Enough time remaining?
             │   ├─ Enough CPU/RAM?
             │   └─ Enough GPU VRAM?
             │
             ├─► Score each candidate
             │   └─ compute_score(node, job, gpus)
             │
             ├─► Select best node
             │
             └─► Return PlacementDecision
                 │
                 ├─► EXISTING_NODE → Reserve & place
                 ├─► REQUEST_MORE_CAPACITY (FAST tier)
                 └─► QUEUE_FOR_FLEX (FLEX tier)
```

### Detailed Algorithm Steps

#### Step 1: Filter Candidate Nodes

For each node in the capacity snapshot:

```python
# Must match tier
if node.tier != job.tier:
    continue

# Must have enough time remaining
if node.time_to_expiry() < job.expected_duration_sec + SAFETY_MARGIN_SEC:
    continue  # Node expires too soon

# Must have enough CPU/RAM
if node.free_cpu() < job.cpu_cores:
    continue
if node.free_ram_gb() < job.ram_gb:
    continue

# Must have enough GPUs with sufficient VRAM
possible_gpus = [
    g for g in node.gpu_slots
    if (g.vram_total_gb - g.vram_used_gb >= job.vram_per_gpu_gb
        and g.job_id is None)  # GPU not already allocated
]

if len(possible_gpus) >= job.gpu_count:
    candidate_nodes.append((node, possible_gpus))
```

**Visual Example:**

```
Node A (8 GPUs, 80GB each):
  GPU 0: 80GB free ✓
  GPU 1: 80GB free ✓
  GPU 2: 40GB used, 40GB free ✗ (job needs 60GB)
  GPU 3: 80GB free ✓
  GPU 4: 80GB free ✓
  GPU 5: 80GB free ✓
  GPU 6: 80GB free ✓
  GPU 7: 80GB free ✓

Job needs: 4 GPUs × 60GB each
Result: 6 GPUs available (0,1,3,4,5,6,7) → Candidate ✓
```

#### Step 2: Score Each Candidate

The scoring function balances multiple factors:

```python
def compute_score(node, job, possible_gpus):
    # Weight factors
    w1 = 1.0   # Utilization weight
    w2 = -0.5  # Fragmentation penalty
    w3 = -0.3  # Time to expiry penalty
    w4 = 0.2   # Provider affinity bonus
    
    # 1. Utilization Score
    total_vram = sum(g.vram_total_gb for g in node.gpu_slots)
    used_vram_before = sum(g.vram_used_gb for g in node.gpu_slots)
    used_vram_after = used_vram_before + (job.vram_per_gpu_gb * job.gpu_count)
    utilization = min(used_vram_after / total_vram, 1.0)
    
    # 2. Fragmentation Score (penalty)
    chosen_indices = [g.index for g in possible_gpus[:job.gpu_count]]
    fragmentation = 0.0
    if len(chosen_indices) > 1:
        sorted_indices = sorted(chosen_indices)
        gaps = [sorted_indices[i+1] - sorted_indices[i] - 1 
                for i in range(len(sorted_indices) - 1)]
        fragmentation = sum(gaps) / len(chosen_indices)
    
    # 3. Expiry Penalty
    time_to_expiry = node.time_to_expiry()
    expiry_penalty = 0.0
    if time_to_expiry < job.expected_duration_sec * 2:
        expiry_penalty = 1.0 - (time_to_expiry / (job.expected_duration_sec * 2))
    
    # 4. Provider Affinity Bonus
    provider = get_node_provider(node)
    provider_bonus = 0.0
    if job.gpu_count >= 8 and provider == "sfcompute":
        provider_bonus = 0.2  # Large jobs fit better on SFCompute 8x nodes
    elif job.gpu_count < 8 and provider == "prime":
        provider_bonus = 0.1  # Small jobs fit better on Prime pods
    
    # Combined score
    score = (w1 * utilization) + (w2 * fragmentation) + (w3 * expiry_penalty) + (w4 * provider_bonus)
    return score
```

#### Step 3: Select Best Node

```python
# Sort by score (descending)
node_scores.sort(reverse=True, key=lambda x: x[0])

# Pick best candidate
_, best_node, gpu_slots = node_scores[0]
chosen_gpus = gpu_slots[:job.gpu_count]
chosen_indices = [g.index for g in chosen_gpus]

return PlacementDecision(
    kind="EXISTING_NODE",
    node_id=best_node.id,
    gpu_indices=chosen_indices,
    job_id=job.id
)
```

---

## Scoring Heuristics Explained

### 1. Utilization Score (Weight: 1.0)

**Goal**: Maximize GPU utilization to get better ROI from purchased blocks.

**Calculation**:
```
utilization = (used_vram_after / total_vram)
```

**Example**:
- Node has 8 GPUs × 80GB = 640GB total VRAM
- Currently using: 160GB (2 GPUs at 80GB each)
- Job needs: 4 GPUs × 40GB = 160GB
- After placement: 320GB used
- **Utilization: 320/640 = 0.5 (50%)**

**Preference**: Higher utilization is better (more efficient use of resources)

### 2. Fragmentation Penalty (Weight: -0.5)

**Goal**: Prefer contiguous GPU allocations to avoid fragmentation.

**Why it matters**: 
- Contiguous GPUs are better for multi-GPU communication (NVLink)
- Reduces fragmentation that could prevent future large jobs

**Calculation**:
```
fragmentation = average_gap_between_gpus
```

**Example**:
- **Contiguous**: GPUs [0, 1, 2, 3] → gaps = [0, 0, 0] → fragmentation = 0.0
- **Fragmented**: GPUs [0, 2, 4, 6] → gaps = [1, 1, 1] → fragmentation = 1.0

**Visual**:
```
Contiguous (better):
[GPU0][GPU1][GPU2][GPU3][GPU4][GPU5][GPU6][GPU7]
  ✓     ✓     ✓     ✓
  └─────────────┘
  No gaps

Fragmented (worse):
[GPU0][GPU1][GPU2][GPU3][GPU4][GPU5][GPU6][GPU7]
  ✓           ✓           ✓           ✓
  └─┘         └─┘         └─┘         └─┘
  Gaps between GPUs
```

### 3. Expiry Penalty (Weight: -0.3)

**Goal**: Avoid placing jobs on nodes that expire soon.

**Why it matters**:
- If node expires before job completes, job fails
- Safety margin ensures buffer time

**Calculation**:
```
if time_to_expiry < job.expected_duration_sec * 2:
    expiry_penalty = 1.0 - (time_to_expiry / (job.expected_duration_sec * 2))
```

**Example**:
- Job duration: 1 hour (3600 sec)
- Node expires in: 30 minutes (1800 sec)
- Threshold: 2 × 3600 = 7200 sec
- **Penalty: 1.0 - (1800/7200) = 0.75** (high penalty)

### 4. Provider Affinity Bonus (Weight: 0.2)

**Goal**: Match job size to optimal provider.

**Logic**:
- **Large jobs (≥8 GPUs)**: Prefer SFCompute (8xH100 nodes are perfect fit)
- **Small jobs (<8 GPUs)**: Prefer Prime (exact match, no waste)

**Example**:
- Job needs 8 GPUs → SFCompute node gets +0.2 bonus
- Job needs 2 GPUs → Prime node gets +0.1 bonus

---

## Visual Examples

### Example 1: Simple Placement

**Scenario**: Job needs 2 GPUs × 40GB each

```
Node State:
┌─────────────────────────────────────────┐
│ Node A (FAST tier, expires in 10 hours) │
├─────────────────────────────────────────┤
│ GPU 0: 80GB free                        │
│ GPU 1: 80GB free                        │
│ GPU 2: 80GB free                        │
│ GPU 3: 80GB free                        │
│ GPU 4: 80GB free                        │
│ GPU 5: 80GB free                        │
│ GPU 6: 80GB free                        │
│ GPU 7: 80GB free                        │
│                                         │
│ CPU: 0/64 used                          │
│ RAM: 0/512GB used                       │
└─────────────────────────────────────────┘

Scoring:
- Utilization: 80GB / 640GB = 0.125 (12.5%)
- Fragmentation: 0.0 (contiguous GPUs 0,1)
- Expiry: 0.0 (plenty of time)
- Provider: +0.1 (Prime, small job)
- Score: 1.0 × 0.125 + (-0.5) × 0.0 + (-0.3) × 0.0 + 0.2 × 0.1
       = 0.125 + 0.02 = 0.145

Result: Job placed on GPUs [0, 1]
```

### Example 2: Fragmentation Consideration

**Scenario**: Two nodes, same utilization, different fragmentation

```
Node A (Fragmented):
[GPU0: free][GPU1: used][GPU2: free][GPU3: used][GPU4: free][GPU5: used][GPU6: free][GPU7: used]
  ✓                      ✓                      ✓                      ✓

Node B (Contiguous):
[GPU0: free][GPU1: free][GPU2: free][GPU3: free][GPU4: used][GPU5: used][GPU6: used][GPU7: used]
  ✓           ✓           ✓           ✓

Job needs: 4 GPUs × 40GB

Node A Score:
- Utilization: 0.5
- Fragmentation: 1.0 (gaps: [1, 1, 1])
- Score: 1.0 × 0.5 + (-0.5) × 1.0 = 0.0

Node B Score:
- Utilization: 0.5
- Fragmentation: 0.0 (contiguous)
- Score: 1.0 × 0.5 + (-0.5) × 0.0 = 0.5

Result: Node B selected (better score)
```

### Example 3: Multi-Node Selection

**Scenario**: Job needs 4 GPUs, multiple nodes available

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Node A        │  │   Node B        │  │   Node C        │
│   (Empty)       │  │   (50% used)    │  │   (75% used)    │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ Util: 12.5%     │  │ Util: 62.5%     │  │ Util: 87.5%     │
│ Frag: 0.0       │  │ Frag: 0.0       │  │ Frag: 0.0       │
│ Expiry: 0.0     │  │ Expiry: 0.0     │  │ Expiry: 0.0     │
│ Provider: +0.1  │  │ Provider: +0.1  │  │ Provider: +0.1  │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ Score: 0.225    │  │ Score: 0.725    │  │ Score: 0.975    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                                                      ↑
                                                  SELECTED
```

**Why Node C?**
- Higher utilization after placement (87.5% vs 12.5%)
- Better resource efficiency
- All other factors equal

---

## Placement Decisions

The algorithm returns one of three decision types:

### 1. EXISTING_NODE

**When**: A suitable node is found

```python
PlacementDecision(
    kind="EXISTING_NODE",
    node_id=UUID("..."),
    gpu_indices=[0, 1, 2, 3],
    job_id=UUID("...")
)
```

**Next Steps**:
1. Reserve capacity via Capacity Manager
2. Place job on Runtime Agent
3. Update job status to SCHEDULED

### 2. REQUEST_MORE_CAPACITY

**When**: No suitable node found, job is FAST tier

```python
PlacementDecision(
    kind="REQUEST_MORE_CAPACITY",
    tier=Tier.FAST,
    job_id=UUID("..."),
    provider="sfcompute" | "prime"
)
```

**Next Steps**:
1. Autoscaler provisions new capacity
2. Job queued for retry when capacity arrives
3. Scheduler re-runs bin-packing

### 3. QUEUE_FOR_FLEX

**When**: No suitable node found, job is FLEX tier

```python
PlacementDecision(
    kind="QUEUE_FOR_FLEX",
    job_id=UUID("..."),
    provider="sfcompute" | "prime"
)
```

**Next Steps**:
1. Job status set to QUEUED
2. Autoscaler provisions capacity when queue builds up
3. Scheduler processes when capacity available

---

## Integration with Scheduler

### Scheduler Loop

```python
while running:
    # Get next job from queue
    job_id = get_next_job()
    
    # Load job details
    job = load_job(job_id)
    
    # Get current capacity
    snapshot = get_capacity_snapshot(job.tier)
    
    # Run bin-packer
    decision = schedule_job(job, snapshot.nodes)
    
    if decision.kind == "EXISTING_NODE":
        # Reserve capacity
        reservation = reserve_capacity(decision.node_id, decision.gpu_indices)
        
        # Place job
        place_job(job, decision.node_id, decision.gpu_indices)
        
        # Update status
        update_job_status(job.id, "SCHEDULED")
        
    elif decision.kind == "REQUEST_MORE_CAPACITY":
        # Trigger autoscaler
        update_job_status(job.id, "QUEUED")
        
    elif decision.kind == "QUEUE_FOR_FLEX":
        # Queue for later
        update_job_status(job.id, "QUEUED")
```

---

## Edge Cases & Constraints

### 1. Insufficient GPU Count

**Scenario**: Job needs 10 GPUs, node only has 8

```
Decision: REQUEST_MORE_CAPACITY (FAST) or QUEUE_FOR_FLEX (FLEX)
```

### 2. Insufficient VRAM

**Scenario**: Job needs 100GB per GPU, GPUs only have 80GB

```
Decision: REQUEST_MORE_CAPACITY or QUEUE_FOR_FLEX
```

### 3. Node Expiring Soon

**Scenario**: Job needs 2 hours, node expires in 30 minutes

```
Filtered out in candidate selection (safety margin check)
```

### 4. Tier Mismatch

**Scenario**: FAST job, only FLEX nodes available

```
Decision: REQUEST_MORE_CAPACITY (FAST tier)
```

### 5. CPU/RAM Constraints

**Scenario**: Job needs 100 CPU cores, node only has 64

```
Filtered out in candidate selection
```

---

## Performance Characteristics

### Time Complexity

- **Filtering**: O(n) where n = number of nodes
- **Scoring**: O(n × m) where m = average GPUs per node
- **Sorting**: O(n log n)
- **Overall**: O(n log n + n × m)

### Space Complexity

- **O(n)**: Store candidate nodes and scores

### Optimization Notes

- Early termination: Stop if perfect match found
- Caching: Capacity snapshots cached briefly
- Batch processing: Process multiple jobs in one cycle

---

## Configuration

### Safety Margin

```python
BIN_PACK_SAFETY_MARGIN_SEC = 300  # 5 minutes
```

Ensures nodes don't expire during job execution.

### Scoring Weights

Current weights (tunable):
- `w1 = 1.0` (utilization)
- `w2 = -0.5` (fragmentation penalty)
- `w3 = -0.3` (expiry penalty)
- `w4 = 0.2` (provider affinity)

---

## Future Enhancements

1. **Multi-GPU Graph Awareness**: Consider NVLink topology for multi-GPU jobs
2. **MIG Support**: Support NVIDIA Multi-Instance GPU slices
3. **Image Co-location**: Pack jobs with same Docker image for cache efficiency
4. **Short vs Long Job Co-location**: Separate short and long jobs to reduce preemption
5. **Predictive Packing**: Use ML to predict job durations for better packing
6. **Dynamic Weights**: Adjust scoring weights based on cluster state
7. **Backfilling**: Pack lower-priority jobs into gaps

---

## Testing

The bin-packing algorithm is thoroughly tested:

- ✅ Empty node placement
- ✅ Partially used node placement
- ✅ Tier mismatch rejection
- ✅ Insufficient resource rejection
- ✅ Expiry time validation
- ✅ Multi-node selection
- ✅ Fragmentation scoring
- ✅ Utilization scoring

See `tests/test_bin_packer.py` for comprehensive test suite.

---

## Summary

The bin-packing algorithm is the **core intelligence** of the Scalar platform:

✅ **Multi-dimensional**: Considers GPU count, VRAM, CPU, RAM, and time  
✅ **Optimization-focused**: Maximizes utilization while minimizing fragmentation  
✅ **Provider-aware**: Matches job size to optimal provider  
✅ **Constraint-respecting**: Ensures all resource and time constraints are met  
✅ **Efficient**: O(n log n) complexity with early termination  
✅ **Robust**: Handles edge cases and escalates appropriately  

This transforms expensive, lumpy compute blocks into efficient, granular allocations that maximize ROI while meeting user SLOs.


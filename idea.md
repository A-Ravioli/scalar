# Scalar

## 1. Product shape: two-tier Scalar

We’ll stick with the **two-tier** model that balances UX and risk:

### Tier A – **Fast (Serverless-ish)**

* Promise: low startup latency (e.g. seconds–tens of seconds).
* Backed by: **warm capacity** (pre-bought SFCompute blocks → running VMs/K8s nodes).
* Pricing: higher $/GPU-hr, includes risk premium.
* Ideal workloads: online inference, interactive tools, short fine-tune jobs.

### Tier B – **Flex (Cheapest-possible)**

* Promise: “We will start within N hours at the best market price we can get.”
* Backed by: **just-in-time capacity** (buy blocks only when price is good; can also backfill idle Fast capacity).
* Pricing: very close to SFCompute cost, maybe small fee.
* Ideal workloads: large training jobs, non-urgent experiments.

Both tiers share the same **core scheduler/bin-packer**; they just have different **SLOs and pricing knobs**.

---

## 2. System architecture

Think of Scalar in three planes:

* **Control plane:** API, scheduler, capacity manager, billing, forecasting.
* **Data plane:** SFCompute-backed clusters where jobs actually run.
* **Market plane:** integration with SFCompute’s order book & cluster APIs.

### Core components

1. **API Gateway / Frontend**

   * REST/gRPC + CLI:

     * `POST /jobs` (submit job)
     * `GET /jobs/{id}` (status/logs)
     * `POST /endpoints` (create serverless endpoint)
   * Auth (keys/Tokens).
   * Writes to `jobs` table & pushes a message to `job_queue`.

2. **Job Orchestrator + Scheduler (with Bin-Packer)**

   * Consumes `job_queue`.
   * Decides:

     * Which *tier* and *cluster* a job runs on.
     * On which **node(s)/GPU(s)** inside that cluster.
   * Talks to:

     * **Capacity Manager** for free capacity snapshot.
     * **Runtime Agent** (in-cluster) to actually create Pods/containers.

3. **Capacity Manager**

   * Maintains in-memory + DB view of:

     * **Blocks** (bought from SFCompute).
     * **Nodes** spawned per block.
     * **Current usage** per node: GPU, VRAM, CPU, RAM.
   * Exposes an internal API:

     * `GET /capacity_snapshot`
     * `POST /reserve` (optimistic: scheduler calls to “hold” capacity)
     * `POST /release`
   * Runs **autoscaling loop** (see section 4).

4. **SFCompute Market Adapter**

   * Wrapper around SFCompute:

     * Quote prices.
     * Place buy/sell orders.
     * Create/destroy clusters or VMs for blocks.
   * Exposes internal API:

     * `POST /buy_block`
     * `POST /sell_block`
     * `POST /provision_nodes_for_block`
     * `POST /terminate_nodes_for_block`

5. **Runtime Agent (per SFCompute cluster)**

   * Runs inside each K8s cluster/VM group.
   * Receives “place job” requests with:

     * Image, command, env, resource requests.
   * Creates workloads (Pods / containers).
   * Reports back:

     * Start time, completion, exit code, resource usage.

6. **Metrics & Forecasting**

   * Ingests:

     * Historical usage per tier.
     * SFCompute price history.
   * Produces:

     * Demand forecast (GPU-hrs per tier per instance type).
     * Recommended pre-buy amounts.

7. **Billing & Accounting**

   * Records:

     * Per-job GPU/CPU time used.
     * Underlying cost from SFCompute.
   * Computes:

     * Margin per job / customer.
     * Bills customers (Stripe, etc.).

---

## 3. Bin-packing: how we actually pack jobs

This is the core value: **turn lumpy SFCompute blocks into smooth serverless slices**.

### 3.1 Resource model

We need a clear model of:

* **Blocks**: contracts bought from SFCompute.
* **Nodes**: actual machines/instances running inside those blocks.
* **Jobs**: tasks with resource demands.

#### Block

```ts
type Block = {
  id: string
  provider: 'sfcompute'
  instance_type: string           // e.g. "8xH100"
  gpus_per_node: number           // e.g. 8
  vram_per_gpu_gb: number         // e.g. 80
  cpu_per_node: number            // in cores
  ram_per_node_gb: number
  start_time: Instant
  end_time: Instant
  cost_per_hour: number
  status: 'pending' | 'active' | 'selling' | 'sold'
}
```

#### Node

```ts
type Node = {
  id: string
  block_id: string
  gpu_slots: GPUAssignment[]
  cpu_used: number
  ram_used_gb: number
}

type GPUAssignment = {
  index: number          // GPU index on the node
  vram_total_gb: number
  vram_used_gb: number
  // later: "slices" for MIG, etc.
}
```

#### Job

```ts
type Job = {
  id: string
  tier: 'FAST' | 'FLEX'
  gpu_count: number
  vram_per_gpu_gb: number
  cpu_cores: number
  ram_gb: number
  expected_duration_sec: number  // user hint or estimate model
  priority: number
}
```

### 3.2 Bin-packing problem

We’re basically doing **multi-dimensional bin-packing**:

* Bins = GPUs (or nodes).
* Dimensions:

  * **GPU count**
  * **VRAM per GPU**
  * Node-level CPU/RAM.

Goal: **maximize utilization** while respecting SLOs and avoiding fragmentation that forces new blocks.

Full optimal packing is NP-hard, so we use greedy + heuristics.

### 3.3 Bin-packing algorithm (v0)

A solid v0 for each job:

1. **Filter candidate nodes**

   From Capacity Manager:

   * Only nodes in clusters appropriate for the **job’s tier**.
   * Node must:

     * Not be expiring too soon:

       * `node.block.end_time - now > expected_duration_sec + safety_margin`
     * Have enough free CPU/RAM.

2. **Find candidate GPU combinations**

   For each candidate node:

   * Count GPUs where:

     * `free_vram >= vram_per_gpu_gb`
     * GPU not already fully allocated.
   * If `count >= job.gpu_count`, candidate node works.

3. **Score nodes using heuristic**

   Examples:

   ```text
   score(node) =
       w1 * utilization_after_job(node)
     + w2 * (-fragmentation_after_job(node))
     + w3 * (time_to_expiry_penalty(node))
   ```

   Intuition:

   * Prefer nodes where adding this job **increases utilization** nicely.
   * Avoid leaving “weird leftover slices” (e.g. tiny stranded VRAM pockets).
   * Avoid nodes that expire too soon for long jobs.

4. **Select best node & allocate**

   * Sort nodes by score descending.
   * Pick top candidate, allocate job to particular GPU indices.
   * Write a **reservation** via Capacity Manager (to avoid races).
   * If reservation succeeds, send placement to Runtime Agent.

5. **If no node fits**

   * Ask Capacity Manager to:

     * For **FAST tier**:

       * See if a *warm pool scale-up* is allowed → buy new block if yes.
     * For **FLEX tier**:

       * Either:

         * Queue job until next capacity refresh, or
         * Trigger a buy at a max price threshold.

   * When new capacity arrives, scheduler re-runs bin-packing.

#### Pseudocode sketch (scheduler)

```python
def schedule_job(job: Job, capacity_snapshot: CapacitySnapshot) -> PlacementDecision:
    candidate_nodes = [
        node for node in capacity_snapshot.nodes
        if node.tier == job.tier
        and node.time_to_expiry() > job.expected_duration_sec + SAFTY_MARGIN
        and node.free_cpu() >= job.cpu_cores
        and node.free_ram_gb() >= job.ram_gb
    ]

    node_scores = []
    for node in candidate_nodes:
        possible_gpus = [g for g in node.gpu_slots if g.free_vram_gb >= job.vram_per_gpu_gb]
        if len(possible_gpus) < job.gpu_count:
            continue

        score = compute_score(node, job, possible_gpus)
        node_scores.append((score, node, possible_gpus))

    if node_scores:
        node_scores.sort(reverse=True, key=lambda x: x[0])
        _, node, gpu_slots = node_scores[0]
        chosen_gpus = gpu_slots[:job.gpu_count]
        return PlacementDecision(kind="EXISTING_NODE", node_id=node.id,
                                 gpu_indices=[g.index for g in chosen_gpus])

    # No fit found, escalate:
    if job.tier == "FAST":
        return PlacementDecision(kind="REQUEST_MORE_CAPACITY", tier="FAST", job_id=job.id)
    else:
        return PlacementDecision(kind="QUEUE_FOR_FLEX", job_id=job.id)
```

Later you can add more sophistication:

* Distinguish *short* vs *long* jobs and co-locate them differently.
* Group by image to exploit caching.

---

## 4. Autoscaling & capacity planning loop

Bin-packing + autoscaling are tightly coupled.

### 4.1 Fast tier (serverless)

**Goal:** keep enough warm capacity to hit latency SLO with acceptable utilization.

Loop (every N seconds):

1. **Collect usage signals**

   * Current GPU utilization per block.
   * Queue length of FAST jobs.
   * Recent arrival rate & job duration.

2. **Compute target capacity**

   * Estimate next hour’s demand (simple v0: exponential moving average).
   * Target GPU-hours = `k * expected_load`, where `k > 1` is safety factor.

3. **Compare to current booked blocks**

   * If capacity shortfall:

     * Call SFCompute adapter to buy new blocks:

       * Choose block shapes that match average job size.
       * Place limit orders at acceptable price (or market if urgent).
   * If persistent over-capacity:

     * Mark some blocks as “not renewed” when they expire.
     * Optionally place sell orders if SFCompute supports selling remaining time.

4. **Inform scheduler**

   * Capacity Manager updates node states as new blocks become ready.
   * Scheduler immediately sees more nodes to bin-pack into.

### 4.2 Flex tier (cheapest)

**Goal:** spend minimal premium above SFCompute, accept delays.

Loop:

1. If Flex queue is non-empty and there’s no capacity:

   * Look at SFCompute orderbook for desired instance types.
   * If price ≤ `target_price`:

     * Buy blocks.
   * Else:

     * Wait; maybe notify users if delay exceeds some threshold.

2. When Fast tier has idle capacity (low utilization):

   * Allow Flex jobs to **backfill**:

     * As long as they’re preemptible or bounded-length so they don’t block future Fast jobs.

This backfilling is also a bin-packing problem; you just assign Flex jobs with **lower priority** and preemption flags.

---

## 5. Repo scaffold (binary ready)

Here’s a starter monorepo layout structured for you + a coding agent:

```text
scalar/
  README.md
  docs/
    architecture.md
    api.md
    scheduler.md
    capacity_planning.md

  infra/
    terraform/
      sfcompute_clusters.tf
      networking.tf
    k8s/
      base_manifests/
      fast-tier/
      flex-tier/

  services/
    api_gateway/
      main.py
      routers/
        jobs.py
        endpoints.py
      models.py
      auth.py

    scheduler/
      main.py
      scheduler.py        # core loop
      bin_packer.py       # bin-packing heuristics & scoring
      models.py           # Job, Node, Block, etc.
      capacity_client.py  # talks to capacity manager
      runtime_client.py   # talks to runtime agents

    capacity_manager/
      main.py
      capacity_state.py   # in-memory + DB sync
      autoscaler.py       # fast & flex tier logic
      sfcompute_client.py # wrapper around SFCompute API
      api.py              # /capacity_snapshot, /reserve, /release

    runtime_agent/
      main.py
      k8s_adapter.py      # or docker_adapter.py
      metrics_exporter.py

    billing/
      main.py
      usage_collector.py
      rating_engine.py
      stripe_client.py

    forecasting/
      main.py
      demand_model.py     # simple EWMA initially
      price_model.py

  libs/
    common/
      config.py
      logging.py
      db.py
      types.py

  migrations/
    001_init.sql          # jobs, blocks, nodes, usage tables
```

You could write `docs/architecture.md` roughly as:

* Intro to Scalar.
* Component diagrams (API, Scheduler, Capacity Manager, SFCompute).
* Detailed section for `bin_packer.md` describing:

  * Resource model.
  * Heuristics.
  * Future improvements (MIG, multi-GPU graph-aware packing, etc.).

---

## Where bin-packing really earns you money

Bin-packing is what actually turns this into more than “just a nicer SFCompute client”:

* **Higher utilization** → lower effective cost per GPU-hr → you can:

  * Undercut other platforms, or
  * Keep a better margin at the same price.
* With **two tiers**, you can:

  * Use Fast capacity for latency-sensitive workloads.
  * Backfill with Flex jobs, smoothing out utilization spikes and dips.

If you want, I can next write the **actual `bin_packer.py` skeleton** (classes + function signatures) or a minimal `scheduler.main` loop you can plug into this repo.

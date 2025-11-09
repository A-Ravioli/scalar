"""Bin-packing algorithm for job placement with provider awareness."""

from typing import List, Optional, Tuple
from libs.common.types import Job, Node, GPUAssignment, PlacementDecision, Tier
from libs.common.config import config

SAFETY_MARGIN_SEC = config.bin_pack_safety_margin_sec


def get_node_provider(node: Node) -> str:
    """Get provider name for a node from its block."""
    if node._block:
        return node._block.provider
    return "sfcompute"  # Default


def compute_score(
    node: Node, job: Job, possible_gpus: List[GPUAssignment]
) -> float:
    """
    Score a node for placing a job.

    Higher score = better fit.
    Considers provider characteristics to optimize placement.
    """
    # Weight factors
    w1 = 1.0  # Utilization weight
    w2 = -0.5  # Fragmentation penalty
    w3 = -0.3  # Time to expiry penalty
    w4 = 0.2  # Provider affinity bonus

    # Calculate utilization after placing job
    total_vram = sum(g.vram_total_gb for g in node.gpu_slots)
    used_vram_before = sum(g.vram_used_gb for g in node.gpu_slots)
    used_vram_after = used_vram_before + (job.vram_per_gpu_gb * job.gpu_count)
    utilization = min(used_vram_after / total_vram, 1.0) if total_vram > 0 else 0.0

    # Calculate fragmentation (prefer contiguous GPUs)
    chosen_indices = [g.index for g in possible_gpus[: job.gpu_count]]
    fragmentation = 0.0
    if len(chosen_indices) > 1:
        # Penalize non-contiguous GPU selection
        sorted_indices = sorted(chosen_indices)
        gaps = [
            sorted_indices[i + 1] - sorted_indices[i] - 1
            for i in range(len(sorted_indices) - 1)
        ]
        fragmentation = sum(gaps) / len(chosen_indices)

    # Time to expiry penalty (prefer nodes with more time remaining)
    time_to_expiry = node.time_to_expiry()
    expiry_penalty = 0.0
    if time_to_expiry < job.expected_duration_sec * 2:
        expiry_penalty = 1.0 - (time_to_expiry / (job.expected_duration_sec * 2))

    # Provider affinity: prefer SFCompute for large jobs (>= 8 GPUs) to avoid fragmentation
    # Prefer Prime for small jobs (< 8 GPUs) if available
    provider = get_node_provider(node)
    provider_bonus = 0.0
    if job.gpu_count >= 8 and provider == "sfcompute":
        # Large jobs fit better on SFCompute 8x nodes
        provider_bonus = 0.2
    elif job.gpu_count < 8 and provider == "prime":
        # Small jobs fit better on Prime pods (exact match)
        provider_bonus = 0.1

    score = (
        w1 * utilization
        + w2 * fragmentation
        + w3 * expiry_penalty
        + w4 * provider_bonus
    )

    return score


def schedule_job(
    job: Job, capacity_snapshot: List[Node]
) -> PlacementDecision:
    """
    Schedule a job using bin-packing algorithm.

    Returns a PlacementDecision indicating where to place the job.
    """
    # Filter candidate nodes
    candidate_nodes = []
    for node in capacity_snapshot:
        # Must match tier
        if node.tier != job.tier:
            continue

        # Must have enough time remaining
        if node.time_to_expiry() < job.expected_duration_sec + SAFETY_MARGIN_SEC:
            continue

        # Must have enough CPU/RAM
        if node.free_cpu() < job.cpu_cores:
            continue
        if node.free_ram_gb() < job.ram_gb:
            continue

        # Find GPUs that can fit this job
        possible_gpus = [
            g
            for g in node.gpu_slots
            if g.vram_total_gb - g.vram_used_gb >= job.vram_per_gpu_gb
            and g.job_id is None
        ]

        if len(possible_gpus) >= job.gpu_count:
            candidate_nodes.append((node, possible_gpus))

    if not candidate_nodes:
        # No fit found, escalate with provider recommendation
        preferred_provider = None
        if job.gpu_count <= config.prime_max_gpus:
            preferred_provider = "prime"
        else:
            preferred_provider = "sfcompute"

        if job.tier == Tier.FAST:
            return PlacementDecision(
                kind="REQUEST_MORE_CAPACITY",
                tier=Tier.FAST,
                job_id=job.id,
                provider=preferred_provider,
            )
        else:
            return PlacementDecision(
                kind="QUEUE_FOR_FLEX", job_id=job.id, provider=preferred_provider
            )

    # Score all candidates
    node_scores = []
    for node, possible_gpus in candidate_nodes:
        score = compute_score(node, job, possible_gpus)
        node_scores.append((score, node, possible_gpus))

    # Sort by score descending
    node_scores.sort(reverse=True, key=lambda x: x[0])

    # Pick best candidate
    _, best_node, gpu_slots = node_scores[0]
    chosen_gpus = gpu_slots[: job.gpu_count]
    chosen_indices = [g.index for g in chosen_gpus]

    return PlacementDecision(
        kind="EXISTING_NODE",
        node_id=best_node.id,
        gpu_indices=chosen_indices,
        job_id=job.id,
    )


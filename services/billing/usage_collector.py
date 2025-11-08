"""Usage event collector."""

from datetime import datetime, timedelta
from typing import List
from libs.common.db import get_supabase_client
from libs.common.logging import get_logger
from libs.common.types import UsageEvent

logger = get_logger(__name__)


class UsageCollector:
    """Collects usage events from running jobs."""

    def __init__(self):
        self.supabase = get_supabase_client()

    def collect_from_completed_jobs(self) -> List[UsageEvent]:
        """Collect usage from recently completed jobs."""
        # Get completed jobs without usage events
        result = (
            self.supabase.table("jobs")
            .select("*, nodes!inner(blocks(*))")
            .eq("status", "completed")
            .is_("usage_collected", "null")
            .execute()
        )

        usage_events = []
        for job_data in result.data:
            if not job_data.get("started_at") or not job_data.get("completed_at"):
                continue

            start_time = datetime.fromisoformat(job_data["started_at"])
            end_time = datetime.fromisoformat(job_data["completed_at"])
            duration_hours = (end_time - start_time).total_seconds() / 3600

            # Calculate GPU and CPU hours
            gpu_hours = duration_hours * job_data["gpu_count"]
            cpu_hours = duration_hours * job_data["cpu_cores"]

            # Get block cost
            block = job_data.get("nodes", {}).get("blocks", {})
            cost_per_hour = block.get("cost_per_hour", 0.0)
            sfcompute_cost = cost_per_hour * duration_hours

            # Calculate customer cost (with margin)
            # Fast tier: 2x markup, Flex tier: 1.1x markup
            tier = job_data["tier"]
            markup = 2.0 if tier == "FAST" else 1.1
            customer_cost = sfcompute_cost * markup
            margin = customer_cost - sfcompute_cost

            usage_event = {
                "job_id": job_data["id"],
                "user_id": job_data["user_id"],
                "node_id": job_data["node_id"],
                "gpu_hours": gpu_hours,
                "cpu_hours": cpu_hours,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "cost_usd": customer_cost,
                "sfcompute_cost_usd": sfcompute_cost,
                "margin_usd": margin,
            }

            usage_events.append(usage_event)

        return usage_events

    def save_usage_events(self, events: List[dict]):
        """Save usage events to database."""
        if not events:
            return

        self.supabase.table("usage_events").insert(events).execute()

        # Mark jobs as usage collected
        job_ids = [e["job_id"] for e in events]
        self.supabase.table("jobs").update({
            "usage_collected": True,
        }).in_("id", job_ids).execute()

        logger.info(f"Saved {len(events)} usage events")


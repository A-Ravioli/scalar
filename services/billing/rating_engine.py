"""Billing rating engine."""

from datetime import datetime, timedelta
from typing import List
from libs.common.db import get_supabase_client
from libs.common.logging import get_logger

logger = get_logger(__name__)


class RatingEngine:
    """Calculates costs and generates invoices."""

    def __init__(self):
        self.supabase = get_supabase_client()

    def calculate_invoice(
        self, user_id: str, period_start: datetime, period_end: datetime
    ) -> dict:
        """Calculate invoice for a user for a time period."""
        # Get usage events in period
        result = (
            self.supabase.table("usage_events")
            .select("*")
            .eq("user_id", user_id)
            .gte("start_time", period_start.isoformat())
            .lte("end_time", period_end.isoformat())
            .execute()
        )

        total_cost = sum(event["cost_usd"] for event in result.data)

        return {
            "user_id": user_id,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "total_usd": total_cost,
            "usage_events": result.data,
        }

    def generate_invoice(self, invoice_data: dict) -> str:
        """Generate and save invoice."""
        result = self.supabase.table("invoices").insert(invoice_data).execute()
        invoice_id = result.data[0]["id"]
        logger.info(f"Generated invoice {invoice_id} for user {invoice_data['user_id']}")
        return invoice_id


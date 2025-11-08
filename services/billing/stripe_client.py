"""Stripe integration client."""

from typing import Optional
from libs.common.config import config
from libs.common.logging import get_logger

logger = get_logger(__name__)


class StripeClient:
    """Stripe API client."""

    def __init__(self):
        self.api_key = config.stripe_api_key
        # Only initialize Stripe if key is provided
        if self.api_key:
            try:
                import stripe
                stripe.api_key = self.api_key
                self.stripe = stripe
            except ImportError:
                logger.warning("stripe package not installed")
                self.stripe = None
        else:
            self.stripe = None
            logger.warning("Stripe API key not configured, using stub mode")

    def create_invoice(self, invoice_id: str, amount_cents: int, customer_email: str) -> Optional[dict]:
        """Create Stripe invoice."""
        if not self.stripe:
            logger.info(f"Stub: Would create Stripe invoice for {invoice_id}, amount ${amount_cents/100}")
            return {"id": f"stub_{invoice_id}", "status": "draft"}

        try:
            # Create or get customer
            customers = self.stripe.Customer.list(email=customer_email, limit=1)
            if customers.data:
                customer_id = customers.data[0].id
            else:
                customer = self.stripe.Customer.create(email=customer_email)
                customer_id = customer.id

            # Create invoice
            invoice = self.stripe.Invoice.create(
                customer=customer_id,
                auto_advance=True,
            )

            # Add line item
            self.stripe.InvoiceItem.create(
                customer=customer_id,
                amount=amount_cents,
                currency="usd",
                invoice=invoice.id,
            )

            # Finalize invoice
            invoice.finalize_invoice()

            return {"id": invoice.id, "status": invoice.status}

        except Exception as e:
            logger.error(f"Failed to create Stripe invoice: {e}")
            return None


"""Unified configuration and secrets loader."""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Application configuration."""

    # Supabase
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_service_key: str = Field(..., env="SUPABASE_SERVICE_KEY")
    supabase_anon_key: Optional[str] = Field(None, env="SUPABASE_ANON_KEY")

    # SFCompute
    sf_api_key: str = Field(..., env="SF_API_KEY")
    sf_api_url: str = Field(
        default="https://api.sfcompute.com", env="SF_API_URL"
    )

    # Redis (for job queue)
    redis_url: Optional[str] = Field(None, env="REDIS_URL")

    # Capacity Manager
    capacity_manager_url: str = Field(
        default="http://localhost:8001", env="CAPACITY_MANAGER_URL"
    )

    # Runtime Agent
    runtime_agent_url: Optional[str] = Field(None, env="RUNTIME_AGENT_URL")

    # Billing
    stripe_api_key: Optional[str] = Field(None, env="STRIPE_API_KEY")
    stripe_webhook_secret: Optional[str] = Field(
        None, env="STRIPE_WEBHOOK_SECRET"
    )

    # Autoscaling
    fast_tier_safety_factor: float = Field(default=1.5, env="FAST_TIER_SAFETY_FACTOR")
    autoscaler_interval_sec: int = Field(default=60, env="AUTOSCALER_INTERVAL_SEC")

    # Bin-packing
    bin_pack_safety_margin_sec: int = Field(
        default=300, env="BIN_PACK_SAFETY_MARGIN_SEC"
    )

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    otlp_endpoint: Optional[str] = Field(None, env="OTLP_ENDPOINT")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global config instance
config = Config()


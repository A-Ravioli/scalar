"""Database connection and utilities."""

from typing import Optional

from postgrest import SyncPostgrestClient
from supabase import create_client, Client
from libs.common.config import config


def get_supabase_client() -> Client:
    """Get Supabase client."""
    return create_client(config.supabase_url, config.supabase_service_key)


def get_postgrest_client() -> SyncPostgrestClient:
    """Get PostgREST client for direct queries."""
    return SyncPostgrestClient(
        f"{config.supabase_url}/rest/v1",
        headers={
            "apikey": config.supabase_service_key,
            "Authorization": f"Bearer {config.supabase_service_key}",
        },
    )


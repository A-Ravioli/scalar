"""Authentication routes."""

import hashlib
import secrets
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Header
from libs.common.db import get_supabase_client
from libs.common.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


def hash_api_key(key: str) -> str:
    """Hash an API key."""
    return hashlib.sha256(key.encode()).hexdigest()


async def verify_api_key(
    authorization: Optional[str] = Header(None),
) -> UUID:
    """Verify API key and return user ID."""
    # For development: allow unauthenticated requests with a default user ID
    if not authorization or not authorization.startswith("Bearer "):
        # Return a default development user ID (you should create this user in your database)
        # TODO: In production, uncomment the line below to enforce authentication
        # raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
        logger.warning("Using default development user ID - authentication is disabled!")
        return UUID("00000000-0000-0000-0000-000000000001")

    token = authorization.replace("Bearer ", "")
    
    # Skip validation for development API key
    if token == "sk_your_api_key":
        logger.warning("Using development API key - authentication is disabled!")
        return UUID("00000000-0000-0000-0000-000000000001")
    
    key_hash = hash_api_key(token)

    supabase = get_supabase_client()
    result = supabase.table("api_keys").select("user_id").eq("key_hash", key_hash).single().execute()

    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Update last_used_at
    supabase.table("api_keys").update({"last_used_at": "now()"}).eq("key_hash", key_hash).execute()

    return UUID(result.data["user_id"])


@router.post("/api-keys")
async def create_api_key(
    name: str,
    user_id: UUID = Depends(verify_api_key),
):
    """Create a new API key."""
    # Generate new key
    api_key = f"sk_{secrets.token_urlsafe(32)}"
    key_hash = hash_api_key(api_key)

    supabase = get_supabase_client()
    result = supabase.table("api_keys").insert({
        "user_id": str(user_id),
        "key_hash": key_hash,
        "name": name,
    }).execute()

    # Return the key only once
    return {"api_key": api_key, "id": result.data[0]["id"]}


@router.get("/api-keys")
async def list_api_keys(user_id: UUID = Depends(verify_api_key)):
    """List user's API keys."""
    supabase = get_supabase_client()
    result = supabase.table("api_keys").select("*").eq("user_id", str(user_id)).execute()
    # Don't return key_hash
    keys = [{k: v for k, v in key.items() if k != "key_hash"} for key in result.data]
    return {"keys": keys}


#!/usr/bin/env python3
"""Test the correct SFCompute API endpoint."""

import asyncio
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

SF_API_KEY = os.getenv("SF_API_KEY")
SF_API_URL = os.getenv("SF_API_URL", "https://api.sfcompute.com")

async def test_orders_endpoint():
    """Test the /v0/orders endpoint."""
    
    print(f"API URL: {SF_API_URL}")
    print(f"API Key: {SF_API_KEY[:20]}...")
    print()
    
    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {SF_API_KEY}",
            "Content-Type": "application/json",
        }
        
        # Test the correct endpoint
        print("=" * 70)
        print("Testing: GET /v0/orders")
        print("=" * 70)
        try:
            response = await client.get(
                f"{SF_API_URL}/v0/orders",
                headers=headers,
                timeout=30.0
            )
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print("\n✅ SUCCESS! Got order data:")
                print(json.dumps(data, indent=2))
            else:
                print(f"\nResponse: {response.text}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_orders_endpoint())


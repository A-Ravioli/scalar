#!/usr/bin/env python3
"""Test the real SFCompute API to understand the orderbook endpoint."""

import asyncio
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

SF_API_KEY = os.getenv("SF_API_KEY")
SF_API_URL = os.getenv("SF_API_URL", "https://api.sfcompute.com")

async def test_sfcompute_api():
    """Test various SFCompute API endpoints to find orderbook."""
    
    print(f"API URL: {SF_API_URL}")
    print(f"API Key: {SF_API_KEY[:20]}...")
    print()
    
    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {SF_API_KEY}",
            "Content-Type": "application/json",
        }
        
        # Test 1: Get API root/docs
        print("=" * 70)
        print("Test 1: GET /")
        print("=" * 70)
        try:
            response = await client.get(f"{SF_API_URL}/", headers=headers, timeout=30.0)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                print(response.text[:500])
        except Exception as e:
            print(f"Error: {e}")
        
        # Test 2: Try to list marketplace endpoints
        print("\n" + "=" * 70)
        print("Test 2: GET /v1/marketplace")
        print("=" * 70)
        try:
            response = await client.get(f"{SF_API_URL}/v1/marketplace", headers=headers, timeout=30.0)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(json.dumps(data, indent=2)[:1000])
            else:
                print(response.text[:500])
        except Exception as e:
            print(f"Error: {e}")
        
        # Test 3: Try orderbook endpoint with various instance types
        for instance_type in ["8xH100", "h100", "H100", "8xh100"]:
            print("\n" + "=" * 70)
            print(f"Test: GET /v1/marketplace/orderbook?instance_type={instance_type}")
            print("=" * 70)
            try:
                response = await client.get(
                    f"{SF_API_URL}/v1/marketplace/orderbook",
                    headers=headers,
                    params={"instance_type": instance_type},
                    timeout=30.0
                )
                print(f"Status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    print("✅ SUCCESS! Got orderbook data:")
                    print(json.dumps(data, indent=2))
                    return  # Found working endpoint
                else:
                    print(f"Response: {response.text[:200]}")
            except Exception as e:
                print(f"Error: {e}")
        
        # Test 4: Try getting available instance types
        print("\n" + "=" * 70)
        print("Test: GET /v1/instances")
        print("=" * 70)
        try:
            response = await client.get(f"{SF_API_URL}/v1/instances", headers=headers, timeout=30.0)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(json.dumps(data, indent=2)[:1000])
            else:
                print(response.text[:500])
        except Exception as e:
            print(f"Error: {e}")
        
        # Test 5: Try listings
        print("\n" + "=" * 70)
        print("Test: GET /v1/marketplace/listings")
        print("=" * 70)
        try:
            response = await client.get(f"{SF_API_URL}/v1/marketplace/listings", headers=headers, timeout=30.0)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(json.dumps(data, indent=2)[:1000])
            else:
                print(response.text[:500])
        except Exception as e:
            print(f"Error: {e}")
        
        # Test 6: Try offers
        print("\n" + "=" * 70)
        print("Test: GET /v1/marketplace/offers")
        print("=" * 70)
        try:
            response = await client.get(f"{SF_API_URL}/v1/marketplace/offers", headers=headers, timeout=30.0)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(json.dumps(data, indent=2)[:1000])
            else:
                print(response.text[:500])
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_sfcompute_api())


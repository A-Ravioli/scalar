#!/usr/bin/env python3
"""Explore SFCompute API endpoints."""

import asyncio
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

SF_API_KEY = os.getenv("SF_API_KEY")
SF_API_URL = os.getenv("SF_API_URL", "https://api.sfcompute.com")

async def explore_api():
    """Explore various API endpoints."""
    
    print(f"API URL: {SF_API_URL}")
    print()
    
    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {SF_API_KEY}",
            "Content-Type": "application/json",
        }
        
        endpoints = [
            "/v0/orders",
            "/v0/marketplace",
            "/v0/marketplace/orders",
            "/v0/marketplace/orderbook",
            "/v0/listings",
            "/v0/offers",
            "/v0/instances",
            "/v0/compute/instances",
            "/v0/compute/listings",
        ]
        
        for endpoint in endpoints:
            print("=" * 70)
            print(f"Testing: GET {endpoint}")
            print("=" * 70)
            try:
                response = await client.get(
                    f"{SF_API_URL}{endpoint}",
                    headers=headers,
                    timeout=30.0
                )
                print(f"Status: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        print(f"✅ Response:")
                        print(json.dumps(data, indent=2)[:1000])
                    except:
                        print(response.text[:500])
                elif response.status_code != 404:
                    print(f"Response: {response.text[:300]}")
                else:
                    print("404 Not Found")
                    
            except Exception as e:
                print(f"Error: {e}")
            print()

if __name__ == "__main__":
    asyncio.run(explore_api())


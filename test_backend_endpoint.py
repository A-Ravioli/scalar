#!/usr/bin/env python3
"""Test the backend orderbook endpoint."""

import asyncio
import httpx
import json

async def test_backend():
    """Test the orderbook endpoint."""
    
    print("Testing backend orderbook endpoint...")
    print("=" * 70)
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "http://localhost:8000/orderbook",
                params={
                    "instance_type": "8xH100",
                    "node_count": 2
                },
                timeout=30.0
            )
            
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print("\n✅ SUCCESS! Got orderbook data:")
                print(json.dumps(data, indent=2))
            else:
                print(f"\nError response:")
                print(response.text)
                
        except httpx.ConnectError:
            print("\n❌ Could not connect to backend server at http://localhost:8000")
            print("Make sure the API gateway is running!")
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_backend())


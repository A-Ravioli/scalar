#!/usr/bin/env python3
"""
Historical pricing data collector for SF Compute.

This script fetches orderbook data from SF Compute API and stores it
for time series analysis and ML model training.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SF_API_KEY = os.getenv("SF_API_KEY")
SF_API_URL = os.getenv("SF_API_URL", "https://api.sfcompute.com")


class SFComputeDataCollector:
    """Collects historical pricing data from SF Compute."""

    def __init__(self, data_dir: str = "trading/data"):
        self.api_key = SF_API_KEY
        self.api_url = SF_API_URL
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def fetch_orderbook(self, instance_type: str = "8xH100") -> Dict:
        """Fetch current orderbook from SF Compute."""
        async with httpx.AsyncClient() as client:
            try:
                url = f"{self.api_url}/v0/orders"
                response = await client.get(
                    url,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                print(f"HTTP error fetching orderbook: {e}")
                return None
            except Exception as e:
                print(f"Error fetching orderbook: {e}")
                return None

    def parse_orderbook_data(self, raw_data: Dict, timestamp: datetime) -> pd.DataFrame:
        """Parse raw orderbook data into structured DataFrame."""
        if not raw_data or "data" not in raw_data:
            return pd.DataFrame()

        orders = raw_data.get("data", [])
        
        parsed_orders = []
        for order in orders:
            parsed_orders.append({
                "timestamp": timestamp,
                "order_id": order.get("id", ""),
                "order_type": order.get("type", ""),  # "buy" or "sell"
                "instance_type": order.get("instance_type", ""),
                "price": float(order.get("price", 0.0)),
                "quantity_gpus": int(order.get("quantity_gpus", 0)),
                "duration_hours": int(order.get("duration_hours", 0)),
                "status": order.get("status", ""),
            })
        
        return pd.DataFrame(parsed_orders)

    def aggregate_orderbook_snapshot(self, df: pd.DataFrame) -> Dict:
        """Aggregate orderbook data into price statistics."""
        if df.empty:
            return {}

        timestamp = df["timestamp"].iloc[0]
        
        # Separate buy (bid) and sell (ask) orders
        bids = df[df["order_type"] == "buy"]
        asks = df[df["order_type"] == "sell"]
        
        stats = {
            "timestamp": timestamp,
            "best_bid": float(bids["price"].max()) if not bids.empty else None,
            "best_ask": float(asks["price"].min()) if not asks.empty else None,
            "mid_price": None,
            "spread": None,
            "bid_volume": int(bids["quantity_gpus"].sum()) if not bids.empty else 0,
            "ask_volume": int(asks["quantity_gpus"].sum()) if not asks.empty else 0,
            "total_orders": len(df),
            "num_bids": len(bids),
            "num_asks": len(asks),
        }
        
        # Calculate mid price and spread
        if stats["best_bid"] and stats["best_ask"]:
            stats["mid_price"] = (stats["best_bid"] + stats["best_ask"]) / 2
            stats["spread"] = stats["best_ask"] - stats["best_bid"]
        
        return stats

    async def collect_snapshot(self, instance_type: str = "8xH100") -> Dict:
        """Collect a single snapshot of orderbook data."""
        timestamp = datetime.now()
        raw_data = await self.fetch_orderbook(instance_type)
        
        if not raw_data:
            return None
        
        df = self.parse_orderbook_data(raw_data, timestamp)
        stats = self.aggregate_orderbook_snapshot(df)
        
        return stats

    async def collect_historical_data(
        self,
        duration_hours: int = 24,
        interval_seconds: int = 300,  # 5 minutes
        instance_type: str = "8xH100"
    ):
        """Collect historical data over a period of time."""
        print(f"Starting data collection for {duration_hours} hours...")
        print(f"Collecting snapshots every {interval_seconds} seconds")
        
        snapshots = []
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        while datetime.now() < end_time:
            snapshot = await self.collect_snapshot(instance_type)
            
            if snapshot:
                snapshots.append(snapshot)
                print(f"Collected snapshot at {snapshot['timestamp']}: "
                      f"bid={snapshot['best_bid']:.2f}, "
                      f"ask={snapshot['best_ask']:.2f}, "
                      f"spread={snapshot.get('spread', 0):.4f}")
            else:
                print(f"Failed to collect snapshot at {datetime.now()}")
            
            # Save intermediate results
            if len(snapshots) % 12 == 0:  # Every hour (if 5min intervals)
                self.save_snapshots(snapshots, instance_type)
            
            await asyncio.sleep(interval_seconds)
        
        self.save_snapshots(snapshots, instance_type)
        print(f"Data collection complete. Collected {len(snapshots)} snapshots.")
        return snapshots

    def save_snapshots(self, snapshots: List[Dict], instance_type: str):
        """Save collected snapshots to CSV file."""
        if not snapshots:
            return
        
        df = pd.DataFrame(snapshots)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.data_dir / f"orderbook_snapshots_{instance_type}_{timestamp_str}.csv"
        
        df.to_csv(filepath, index=False)
        print(f"Saved {len(snapshots)} snapshots to {filepath}")

    def load_historical_data(self, pattern: str = "orderbook_snapshots_*.csv") -> pd.DataFrame:
        """Load all historical data from CSV files."""
        csv_files = list(self.data_dir.glob(pattern))
        
        if not csv_files:
            print(f"No historical data files found matching {pattern}")
            return pd.DataFrame()
        
        dfs = []
        for filepath in csv_files:
            df = pd.read_csv(filepath)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values("timestamp")
        combined_df = combined_df.drop_duplicates(subset=["timestamp"])
        
        print(f"Loaded {len(combined_df)} historical snapshots from {len(csv_files)} files")
        return combined_df

    def generate_synthetic_historical_data(
        self,
        days: int = 30,
        interval_minutes: int = 5
    ) -> pd.DataFrame:
        """
        Generate synthetic historical data for training when real data is not available.
        
        This simulates realistic orderbook dynamics based on:
        - Price trends and volatility
        - Bid-ask spread dynamics
        - Volume patterns
        - Time-of-day effects
        """
        import numpy as np
        
        print(f"Generating {days} days of synthetic historical data...")
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = pd.date_range(start_time, end_time, freq=f"{interval_minutes}min")
        
        n_samples = len(timestamps)
        
        # Base price around $1.45/GPU-hour (realistic for H100)
        base_price = 1.45
        
        # Generate price with trend and noise
        trend = np.linspace(0, 0.1, n_samples)  # Slight upward trend
        
        # Add multiple time scales of variation
        daily_cycle = 0.05 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 60 / interval_minutes))
        weekly_cycle = 0.03 * np.sin(2 * np.pi * np.arange(n_samples) / (7 * 24 * 60 / interval_minutes))
        
        # Random walk component
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(n_samples) * 0.01)
        
        # High frequency noise
        noise = np.random.randn(n_samples) * 0.02
        
        # Combine components
        mid_price = base_price + trend + daily_cycle + weekly_cycle + random_walk + noise
        
        # Generate spread (varies with volatility)
        spread_base = 0.03
        spread_volatility = np.abs(np.diff(mid_price, prepend=mid_price[0]))
        spread = spread_base + spread_volatility * 2
        spread = np.clip(spread, 0.01, 0.1)  # Keep spread reasonable
        
        # Calculate bid and ask
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2
        
        # Generate volumes (with time-of-day patterns)
        hour_of_day = np.array([ts.hour for ts in timestamps])
        # Higher volume during business hours (9-17)
        volume_multiplier = 1 + 0.5 * np.maximum(0, 1 - np.abs(hour_of_day - 13) / 8)
        base_volume = 100
        bid_volume = (base_volume * volume_multiplier * (1 + np.random.rand(n_samples) * 0.5)).astype(int)
        ask_volume = (base_volume * volume_multiplier * (1 + np.random.rand(n_samples) * 0.5)).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": timestamps,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread": spread,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "total_orders": np.random.randint(10, 50, n_samples),
            "num_bids": np.random.randint(5, 25, n_samples),
            "num_asks": np.random.randint(5, 25, n_samples),
        })
        
        # Save synthetic data
        filepath = self.data_dir / "synthetic_historical_data.csv"
        df.to_csv(filepath, index=False)
        print(f"Generated and saved {len(df)} synthetic data points to {filepath}")
        
        return df


async def main():
    """Main entry point for data collection."""
    collector = SFComputeDataCollector()
    
    # Option 1: Generate synthetic historical data for training
    print("=" * 70)
    print("Generating synthetic historical data...")
    print("=" * 70)
    synthetic_df = collector.generate_synthetic_historical_data(days=30, interval_minutes=5)
    print(f"\nSynthetic data shape: {synthetic_df.shape}")
    print(f"\nFirst few rows:")
    print(synthetic_df.head())
    print(f"\nData statistics:")
    print(synthetic_df.describe())
    
    # Option 2: Collect a single snapshot of real data
    print("\n" + "=" * 70)
    print("Fetching real-time snapshot from SF Compute...")
    print("=" * 70)
    snapshot = await collector.collect_snapshot()
    if snapshot:
        print(f"Current market data:")
        print(json.dumps(snapshot, indent=2, default=str))
    else:
        print("Unable to fetch real-time data (API may be unavailable)")
    
    # Option 3: Start continuous collection (uncomment to run)
    # print("\n" + "=" * 70)
    # print("Starting continuous data collection...")
    # print("=" * 70)
    # await collector.collect_historical_data(
    #     duration_hours=24,
    #     interval_seconds=300,
    #     instance_type="8xH100"
    # )


if __name__ == "__main__":
    asyncio.run(main())


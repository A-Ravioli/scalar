#!/usr/bin/env python3
"""
Inference script for price prediction models.
"""

import torch
import numpy as np
import pandas as pd
import argparse
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List
import json

from models import get_model
from data_preprocessing import PriceDataPreprocessor
from data_collector import SFComputeDataCollector


class PricePredictor:
    """Make predictions using trained models."""
    
    def __init__(
        self,
        model_path: str,
        preprocessor_path: str,
        model_type: str = 'lstm',
        hidden_size: int = 64,
        num_layers: int = 2,
        device: torch.device = None
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model checkpoint
            preprocessor_path: Path to saved preprocessor
            model_type: Type of model ('rnn', 'gru', 'lstm')
            hidden_size: Hidden layer size used in training
            num_layers: Number of layers used in training
            device: Device to run inference on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load preprocessor
        self.preprocessor = PriceDataPreprocessor.load(preprocessor_path)
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model with correct architecture
        input_size = len(self.preprocessor.feature_columns)
        self.model = get_model(
            model_type=model_type,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded {model_type.upper()} model from {model_path}")
        print(f"Using device: {self.device}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on data.
        
        Args:
            df: DataFrame with historical price data
            
        Returns:
            Array of predictions
        """
        # Preprocess data
        sequences, _ = self.preprocessor.transform(df)
        
        # Convert to tensor
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(sequences_tensor)
        
        return predictions.cpu().numpy()
    
    def predict_next(self, df: pd.DataFrame, steps: int = 1) -> List[float]:
        """
        Predict next N steps into the future.
        
        Args:
            df: DataFrame with historical price data
            steps: Number of steps to predict
            
        Returns:
            List of predicted prices
        """
        predictions = []
        current_df = df.copy()
        
        for _ in range(steps):
            # Predict next step
            pred = self.predict(current_df)
            next_price = float(pred[-1])
            predictions.append(next_price)
            
            # Append prediction to dataframe for next iteration
            # (This is a simplified approach; in practice, you'd need to
            # update all features properly)
            last_row = current_df.iloc[-1].copy()
            last_row['mid_price'] = next_price
            last_row['timestamp'] = pd.to_datetime(last_row['timestamp']) + timedelta(minutes=5)
            
            current_df = pd.concat([current_df, pd.DataFrame([last_row])], ignore_index=True)
        
        return predictions


async def predict_realtime():
    """Make real-time predictions using live data."""
    print("Real-time prediction mode")
    print("=" * 70)
    
    # Collect recent data
    collector = SFComputeDataCollector()
    snapshot = await collector.collect_snapshot()
    
    if not snapshot:
        print("Failed to fetch real-time data. Using synthetic data for demo.")
        df = collector.generate_synthetic_historical_data(days=1, interval_minutes=5)
    else:
        print(f"Collected snapshot: {json.dumps(snapshot, indent=2, default=str)}")
        print("\nNote: Single snapshot is not enough for prediction.")
        print("Using synthetic data for demonstration...")
        df = collector.generate_synthetic_historical_data(days=1, interval_minutes=5)
    
    # Load predictor
    predictor = PricePredictor(
        model_path="trading/checkpoints/best_PriceLSTM.pt",
        preprocessor_path="trading/checkpoints/preprocessor.pkl",
        model_type='lstm'
    )
    
    # Make predictions
    predictions = predictor.predict(df)
    
    print("\nRecent predictions:")
    for i, pred in enumerate(predictions[-10:]):
        print(f"  Step {i+1}: ${pred[0]:.4f}")
    
    # Predict next 5 steps
    print("\nForecasting next 5 time steps:")
    future_predictions = predictor.predict_next(df, steps=5)
    for i, pred in enumerate(future_predictions):
        print(f"  +{i+1} step: ${pred:.4f}")


def predict_from_file(data_file: str, model_path: str, preprocessor_path: str, model_type: str = 'lstm'):
    """Make predictions from historical data file."""
    print(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    
    # Load predictor
    predictor = PricePredictor(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        model_type=model_type
    )
    
    # Make predictions
    predictions = predictor.predict(df)
    
    # Show statistics
    print("\nPrediction Statistics:")
    print(f"  Min:  ${predictions.min():.4f}")
    print(f"  Max:  ${predictions.max():.4f}")
    print(f"  Mean: ${predictions.mean():.4f}")
    print(f"  Std:  ${predictions.std():.4f}")
    
    # Show recent predictions
    print("\nLast 10 predictions:")
    for i, pred in enumerate(predictions[-10:]):
        print(f"  {i+1}. ${pred[0]:.4f}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Make price predictions')
    parser.add_argument('--mode', type=str, default='realtime',
                        choices=['realtime', 'file'],
                        help='Prediction mode')
    parser.add_argument('--data-file', type=str,
                        default='trading/data/synthetic_historical_data.csv',
                        help='Path to historical data CSV (for file mode)')
    parser.add_argument('--model', type=str,
                        default='trading/checkpoints/best_PriceLSTM.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='lstm',
                        choices=['rnn', 'gru', 'lstm'],
                        help='Model type (rnn, gru, or lstm)')
    parser.add_argument('--preprocessor', type=str,
                        default='trading/checkpoints/preprocessor.pkl',
                        help='Path to preprocessor')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        print("Please train a model first using train.py")
        return
    
    if args.mode == 'realtime':
        asyncio.run(predict_realtime())
    else:
        predict_from_file(args.data_file, args.model, args.preprocessor, args.model_type)


if __name__ == "__main__":
    main()


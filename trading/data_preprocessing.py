"""
Data preprocessing utilities for time series price prediction.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, List
import pickle
from pathlib import Path


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    
    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray
    ):
        """
        Initialize dataset.
        
        Args:
            sequences: Input sequences of shape (n_samples, sequence_length, n_features)
            targets: Target values of shape (n_samples, 1)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


class PriceDataPreprocessor:
    """Preprocess pricing data for time series models."""
    
    def __init__(
        self,
        sequence_length: int = 12,  # Look back 12 time steps (1 hour if 5min intervals)
        forecast_horizon: int = 1,   # Predict 1 step ahead
        scaler_type: str = "standard"  # "standard" or "minmax"
    ):
        """
        Initialize preprocessor.
        
        Args:
            sequence_length: Number of past time steps to use as input
            forecast_horizon: Number of steps ahead to predict
            scaler_type: Type of scaling to use
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.feature_columns = None
        self.target_column = "mid_price"
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from raw data.
        
        Args:
            df: DataFrame with columns: timestamp, best_bid, best_ask, mid_price, spread, etc.
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Basic features
        features = pd.DataFrame()
        features['timestamp'] = df['timestamp']
        features['mid_price'] = df['mid_price']
        features['spread'] = df['spread']
        features['bid_volume'] = df['bid_volume']
        features['ask_volume'] = df['ask_volume']
        
        # Volume imbalance
        features['volume_imbalance'] = (
            (df['bid_volume'] - df['ask_volume']) / 
            (df['bid_volume'] + df['ask_volume'] + 1e-10)
        )
        
        # Price momentum features
        features['price_change_1'] = df['mid_price'].diff(1)
        features['price_change_3'] = df['mid_price'].diff(3)
        features['price_change_12'] = df['mid_price'].diff(12)
        
        # Moving averages
        features['ma_5'] = df['mid_price'].rolling(window=5, min_periods=1).mean()
        features['ma_12'] = df['mid_price'].rolling(window=12, min_periods=1).mean()
        features['ma_24'] = df['mid_price'].rolling(window=24, min_periods=1).mean()
        
        # Volatility features
        features['volatility_5'] = df['mid_price'].rolling(window=5, min_periods=1).std()
        features['volatility_12'] = df['mid_price'].rolling(window=12, min_periods=1).std()
        
        # Spread dynamics
        features['spread_ma_5'] = df['spread'].rolling(window=5, min_periods=1).mean()
        features['spread_vol_5'] = df['spread'].rolling(window=5, min_periods=1).std()
        
        # Time-based features
        features['hour'] = df['timestamp'].dt.hour
        features['day_of_week'] = df['timestamp'].dt.dayofweek
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # Fill NaN values (from diff and rolling operations)
        features = features.bfill().ffill()
        
        return features
    
    def create_sequences(
        self,
        data: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data: Input data of shape (n_samples, n_features)
            target: Target data of shape (n_samples,)
            
        Returns:
            Tuple of (sequences, targets)
            - sequences: shape (n_sequences, sequence_length, n_features)
            - targets: shape (n_sequences, 1)
        """
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length - self.forecast_horizon + 1):
            # Input sequence
            seq = data[i:i + self.sequence_length]
            
            # Target (future value)
            target_idx = i + self.sequence_length + self.forecast_horizon - 1
            target_val = target[target_idx]
            
            sequences.append(seq)
            targets.append(target_val)
        
        return np.array(sequences), np.array(targets).reshape(-1, 1)
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scaler and transform data.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names to use
            
        Returns:
            Tuple of (sequences, targets)
        """
        # Prepare features
        features_df = self.prepare_features(df)
        
        # Select feature columns
        if feature_cols is None:
            # Use all numeric columns except timestamp and target
            feature_cols = [
                col for col in features_df.columns 
                if col not in ['timestamp', self.target_column] 
                and pd.api.types.is_numeric_dtype(features_df[col])
            ]
        
        self.feature_columns = feature_cols
        
        # Extract feature matrix and target
        X = features_df[feature_cols].values
        y = features_df[self.target_column].values
        
        # Fit and transform features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        sequences, targets = self.create_sequences(X_scaled, y)
        
        return sequences, targets
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted scaler.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Tuple of (sequences, targets)
        """
        if self.feature_columns is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Prepare features
        features_df = self.prepare_features(df)
        
        # Extract feature matrix and target
        X = features_df[self.feature_columns].values
        y = features_df[self.target_column].values
        
        # Transform features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        sequences, targets = self.create_sequences(X_scaled, y)
        
        return sequences, targets
    
    def save(self, filepath: str):
        """Save preprocessor state."""
        state = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'target_column': self.target_column
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved preprocessor to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PriceDataPreprocessor':
        """Load preprocessor state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(
            sequence_length=state['sequence_length'],
            forecast_horizon=state['forecast_horizon']
        )
        preprocessor.scaler = state['scaler']
        preprocessor.feature_columns = state['feature_columns']
        preprocessor.target_column = state['target_column']
        
        print(f"Loaded preprocessor from {filepath}")
        return preprocessor


def split_data(
    sequences: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    """
    Split data into train, validation, and test sets.
    
    Uses temporal split (no shuffling) to preserve time order.
    
    Args:
        sequences: Input sequences
        targets: Target values
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        
    Returns:
        Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n_samples = len(sequences)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    X_train = sequences[:train_end]
    y_train = targets[:train_end]
    
    X_val = sequences[train_end:val_end]
    y_val = targets[train_end:val_end]
    
    X_test = sequences[val_end:]
    y_test = targets[val_end:]
    
    print(f"Data split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_dataloaders(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders.
    
    Args:
        train_data: (X_train, y_train)
        val_data: (X_val, y_val)
        test_data: (X_test, y_test)
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = TimeSeriesDataset(*train_data)
    val_dataset = TimeSeriesDataset(*val_data)
    test_dataset = TimeSeriesDataset(*test_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle within temporal batches is okay
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


#!/usr/bin/env python3
"""
Training script for RNN, GRU, and LSTM price prediction models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import argparse

from models import get_model, count_parameters
from data_preprocessing import (
    PriceDataPreprocessor,
    split_data,
    create_dataloaders
)
from data_collector import SFComputeDataCollector


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether to minimize or maximize the metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """Trainer for time series models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        scheduler_type: str = "plateau"
    ):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            scheduler_type: Learning rate scheduler ('plateau', 'cosine', or None)
        """
        self.model = model.to(device)
        self.device = device
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        if scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=50,
                eta_min=1e-6
            )
        else:
            self.scheduler = None
        
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for sequences, targets in train_loader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    def validate(self, val_loader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 100,
        early_stopping_patience: int = 15,
        checkpoint_dir: str = "trading/checkpoints"
    ) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Dictionary with training history
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Model has {count_parameters(self.model):,} trainable parameters")
        print("=" * 70)
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_name = self.model.__class__.__name__
                checkpoint_file = checkpoint_path / f"best_{model_name}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_file)
                print(f"  → Saved best model (val_loss: {val_loss:.6f})")
            
            # Early stopping check
            if early_stopping(val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        print("=" * 70)
        print(f"Training complete!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': best_val_loss
        }
    
    def evaluate(self, test_loader) -> Dict:
        """Evaluate model on test set."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(sequences)
                
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Calculate metrics
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        mape = np.mean(np.abs((predictions - targets) / (targets + 1e-10))) * 100
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2)
        }
        
        print("\nTest Set Evaluation:")
        print("=" * 50)
        print(f"MSE:   {mse:.6f}")
        print(f"RMSE:  {rmse:.6f}")
        print(f"MAE:   {mae:.6f}")
        print(f"MAPE:  {mape:.2f}%")
        print(f"R²:    {r2:.4f}")
        print("=" * 50)
        
        return metrics, predictions, targets


def plot_training_history(history: Dict, save_path: str):
    """Plot training and validation losses."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_losses'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_losses'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate plot
    axes[1].plot(history['learning_rates'], linewidth=2, color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training history plot to {save_path}")
    plt.close()


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str,
    n_samples: int = 200
):
    """Plot predictions vs actual values."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Limit to n_samples for clarity
    pred_plot = predictions[:n_samples].flatten()
    target_plot = targets[:n_samples].flatten()
    
    # Time series plot
    axes[0].plot(target_plot, label='Actual', linewidth=2, alpha=0.7)
    axes[0].plot(pred_plot, label='Predicted', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Price')
    axes[0].set_title('Actual vs Predicted Prices')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(target_plot, pred_plot, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(target_plot.min(), pred_plot.min())
    max_val = max(target_plot.max(), pred_plot.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    axes[1].set_xlabel('Actual Price')
    axes[1].set_ylabel('Predicted Price')
    axes[1].set_title('Prediction Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved predictions plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train price prediction models')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['rnn', 'gru', 'lstm', 'all'],
                        help='Model type to train')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Hidden layer size')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of recurrent layers')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--sequence-length', type=int, default=12,
                        help='Sequence length (lookback window)')
    parser.add_argument('--data-file', type=str, 
                        default='trading/data/synthetic_historical_data.csv',
                        help='Path to historical data CSV')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    Path("trading/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("trading/results").mkdir(parents=True, exist_ok=True)
    Path("trading/plots").mkdir(parents=True, exist_ok=True)
    
    # Load or generate data
    print("\n" + "=" * 70)
    print("Loading data...")
    print("=" * 70)
    
    collector = SFComputeDataCollector()
    
    if not Path(args.data_file).exists():
        print(f"Data file not found: {args.data_file}")
        print("Generating synthetic data...")
        df = collector.generate_synthetic_historical_data(days=30, interval_minutes=5)
    else:
        df = pd.read_csv(args.data_file)
        print(f"Loaded {len(df)} samples from {args.data_file}")
    
    # Preprocess data
    print("\n" + "=" * 70)
    print("Preprocessing data...")
    print("=" * 70)
    
    preprocessor = PriceDataPreprocessor(
        sequence_length=args.sequence_length,
        forecast_horizon=1
    )
    
    sequences, targets = preprocessor.fit_transform(df)
    print(f"Created {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Save preprocessor
    preprocessor.save("trading/checkpoints/preprocessor.pkl")
    
    # Split data
    train_data, val_data, test_data = split_data(
        sequences, targets,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=args.batch_size
    )
    
    input_size = sequences.shape[2]
    
    # Train models
    models_to_train = ['rnn', 'gru', 'lstm'] if args.model == 'all' else [args.model]
    
    all_results = {}
    
    for model_type in models_to_train:
        print("\n" + "=" * 70)
        print(f"Training {model_type.upper()} model")
        print("=" * 70)
        
        # Create model
        model = get_model(
            model_type=model_type,
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=1,
            dropout=0.2
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=args.lr,
            weight_decay=1e-5,
            scheduler_type="plateau"
        )
        
        # Train
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            early_stopping_patience=15
        )
        
        # Evaluate
        metrics, predictions, test_targets = trainer.evaluate(test_loader)
        
        # Save results
        results = {
            'model_type': model_type,
            'history': history,
            'metrics': metrics,
            'config': {
                'input_size': input_size,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'learning_rate': args.lr,
                'batch_size': args.batch_size,
                'sequence_length': args.sequence_length,
            }
        }
        
        all_results[model_type] = results
        
        # Save results to JSON
        results_file = f"trading/results/{model_type}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {results_file}")
        
        # Plot training history
        plot_training_history(
            history,
            f"trading/plots/{model_type}_training_history.png"
        )
        
        # Plot predictions
        plot_predictions(
            predictions,
            test_targets,
            f"trading/plots/{model_type}_predictions.png"
        )
    
    # Compare models
    if len(models_to_train) > 1:
        print("\n" + "=" * 70)
        print("Model Comparison")
        print("=" * 70)
        print(f"{'Model':<10} {'RMSE':<12} {'MAE':<12} {'MAPE':<12} {'R²':<12}")
        print("-" * 70)
        for model_type, results in all_results.items():
            metrics = results['metrics']
            print(f"{model_type.upper():<10} "
                  f"{metrics['rmse']:<12.6f} "
                  f"{metrics['mae']:<12.6f} "
                  f"{metrics['mape']:<12.2f} "
                  f"{metrics['r2']:<12.4f}")
        print("=" * 70)


if __name__ == "__main__":
    main()


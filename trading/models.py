"""
Neural network models for time series price prediction.

Implements RNN, GRU, and LSTM models for SF Compute pricing data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List


class PriceRNN(nn.Module):
    """Simple RNN for price prediction."""
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(PriceRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Take the last time step
        out = out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)
        
        return out


class PriceGRU(nn.Module):
    """GRU network for price prediction."""
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(PriceGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Take the last time step
        out = out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)
        
        return out


class PriceLSTM(nn.Module):
    """LSTM network for price prediction."""
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(PriceLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last time step
        out = out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(out)
        
        return out


class EnhancedPriceLSTM(nn.Module):
    """
    Enhanced LSTM with attention mechanism for price prediction.
    
    This model includes:
    - Bidirectional LSTM
    - Attention mechanism
    - Residual connections
    - Batch normalization
    """
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        num_layers: int = 3,
        output_size: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(EnhancedPriceLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * self.num_directions
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # Project input
        x_proj = self.input_proj(x.reshape(-1, x.size(-1)))
        x_proj = self.input_bn(x_proj)
        x_proj = self.relu(x_proj)
        x_proj = x_proj.reshape(batch_size, seq_len, -1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x_proj)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Take the last time step
        out = attn_out[:, -1, :]
        
        # Fully connected layers with batch norm
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def get_model(
    model_type: str,
    input_size: int = 6,
    hidden_size: int = 64,
    num_layers: int = 2,
    output_size: int = 1,
    dropout: float = 0.2,
    enhanced: bool = False
) -> nn.Module:
    """
    Factory function to create a model.
    
    Args:
        model_type: One of 'rnn', 'gru', 'lstm'
        input_size: Number of input features
        hidden_size: Hidden layer size
        num_layers: Number of recurrent layers
        output_size: Number of output features
        dropout: Dropout rate
        enhanced: Use enhanced LSTM with attention
        
    Returns:
        Neural network model
    """
    model_type = model_type.lower()
    
    if model_type == 'rnn':
        return PriceRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )
    elif model_type == 'gru':
        return PriceGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        )
    elif model_type == 'lstm':
        if enhanced:
            return EnhancedPriceLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout=dropout
            )
        else:
            return PriceLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                dropout=dropout
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'rnn', 'gru', or 'lstm'.")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


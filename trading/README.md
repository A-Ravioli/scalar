# Trading Module - Price Prediction with Deep Learning

This module implements RNN, GRU, and LSTM models for predicting SF Compute marketplace prices.

## Features

- **Data Collection**: Fetches historical pricing data from SF Compute API
- **Synthetic Data Generation**: Creates realistic synthetic data for training when real historical data is unavailable
- **Multiple Model Architectures**: 
  - Simple RNN
  - GRU (Gated Recurrent Unit)
  - LSTM (Long Short-Term Memory)
  - Enhanced LSTM with attention mechanism
- **Comprehensive Training Pipeline**: Includes early stopping, learning rate scheduling, and checkpointing
- **Evaluation Metrics**: MSE, RMSE, MAE, MAPE, R²
- **Visualization**: Training curves and prediction plots

## Installation

Install dependencies:

```bash
cd trading
pip install -r requirements.txt
```

Or from the project root:

```bash
pip install torch numpy pandas scikit-learn matplotlib httpx python-dotenv
```

## Quick Start

### 1. Generate Synthetic Historical Data

```bash
python data_collector.py
```

This will create synthetic historical data at `data/synthetic_historical_data.csv`.

### 2. Train Models

Train all models (RNN, GRU, LSTM):

```bash
python train.py --model all --epochs 100 --batch-size 32
```

Train a specific model:

```bash
python train.py --model lstm --epochs 100 --hidden-size 64 --num-layers 2
```

Training options:
- `--model`: Model type ('rnn', 'gru', 'lstm', or 'all')
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 32)
- `--hidden-size`: Hidden layer size (default: 64)
- `--num-layers`: Number of recurrent layers (default: 2)
- `--lr`: Learning rate (default: 0.001)
- `--sequence-length`: Lookback window size (default: 12)
- `--data-file`: Path to data file

### 3. Make Predictions

Real-time prediction using live data:

```bash
python predict.py --mode realtime
```

Predict from historical data:

```bash
python predict.py --mode file --data-file data/synthetic_historical_data.csv
```

## Data Structure

### Input Features

The models use the following features:

1. **Price Features**:
   - `mid_price`: Mid-point between bid and ask
   - `spread`: Bid-ask spread
   - Price momentum (1, 3, 12 step changes)
   - Moving averages (5, 12, 24 periods)

2. **Volume Features**:
   - `bid_volume`: Total bid volume
   - `ask_volume`: Total ask volume
   - `volume_imbalance`: (bid_volume - ask_volume) / total_volume

3. **Volatility Features**:
   - Rolling standard deviation of prices
   - Spread volatility

4. **Temporal Features**:
   - Hour of day (sin/cos encoding)
   - Day of week (sin/cos encoding)

### Output

The models predict the `mid_price` at the next time step (default: 5 minutes ahead).

## Model Architecture

### Simple RNN

```
Input → RNN Layers → FC → Output
```

- Good for simple patterns
- Fast training
- May struggle with long-term dependencies

### GRU (Gated Recurrent Unit)

```
Input → GRU Layers → FC → Output
```

- Better at capturing long-term dependencies
- More parameters than RNN
- Good balance between performance and complexity

### LSTM (Long Short-Term Memory)

```
Input → LSTM Layers → FC → Output
```

- Best at capturing long-term dependencies
- Most parameters
- Can model complex temporal patterns

### Enhanced LSTM

```
Input → Projection → Bidirectional LSTM → Attention → FC Layers → Output
```

- Includes attention mechanism
- Bidirectional processing
- Batch normalization
- Best performance but highest computational cost

## Training Process

1. **Data Preprocessing**:
   - Feature engineering (momentum, moving averages, volatility)
   - Standardization (zero mean, unit variance)
   - Sequence creation (sliding window)

2. **Train/Val/Test Split**:
   - 70% training
   - 15% validation
   - 15% testing
   - Temporal split (no shuffling to preserve time order)

3. **Training**:
   - Adam optimizer with weight decay
   - Learning rate scheduling (ReduceLROnPlateau or CosineAnnealing)
   - Gradient clipping (prevent exploding gradients)
   - Early stopping (patience: 15 epochs)
   - Model checkpointing (save best model)

4. **Evaluation**:
   - MSE (Mean Squared Error)
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - MAPE (Mean Absolute Percentage Error)
   - R² (Coefficient of Determination)

## File Structure

```
trading/
├── data/                           # Data directory
│   └── synthetic_historical_data.csv
├── checkpoints/                    # Model checkpoints
│   ├── best_PriceRNN.pt
│   ├── best_PriceGRU.pt
│   ├── best_PriceLSTM.pt
│   └── preprocessor.pkl
├── results/                        # Training results (JSON)
│   ├── rnn_results.json
│   ├── gru_results.json
│   └── lstm_results.json
├── plots/                          # Visualizations
│   ├── rnn_training_history.png
│   ├── rnn_predictions.png
│   ├── gru_training_history.png
│   ├── gru_predictions.png
│   ├── lstm_training_history.png
│   └── lstm_predictions.png
├── data_collector.py               # Data collection and generation
├── data_preprocessing.py           # Data preprocessing utilities
├── models.py                       # Neural network models
├── train.py                        # Training script
├── predict.py                      # Inference script
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## Real-Time Data Collection

To collect real-time data from SF Compute (requires valid API key):

```python
from data_collector import SFComputeDataCollector
import asyncio

collector = SFComputeDataCollector()

# Collect data for 24 hours at 5-minute intervals
asyncio.run(collector.collect_historical_data(
    duration_hours=24,
    interval_seconds=300,
    instance_type="8xH100"
))
```

## Tips for Better Performance

1. **More Training Data**: Collect more historical data for better generalization
2. **Hyperparameter Tuning**: Experiment with:
   - Hidden size (32, 64, 128, 256)
   - Number of layers (1, 2, 3, 4)
   - Learning rate (1e-4, 5e-4, 1e-3)
   - Sequence length (6, 12, 24, 48)
3. **Feature Engineering**: Add domain-specific features
4. **Ensemble Methods**: Combine predictions from multiple models
5. **Regularization**: Adjust dropout rate to prevent overfitting

## Example Output

```
Training LSTM model
======================================================================
Model has 45,377 trainable parameters
======================================================================
Epoch   1/100 | Train Loss: 0.012543 | Val Loss: 0.010234 | LR: 0.001000
Epoch   2/100 | Train Loss: 0.008234 | Val Loss: 0.007891 | LR: 0.001000
  → Saved best model (val_loss: 0.007891)
...
Early stopping triggered at epoch 47
======================================================================
Training complete!
Best validation loss: 0.004523

Test Set Evaluation:
==================================================
MSE:   0.004612
RMSE:  0.067912
MAE:   0.052341
MAPE:  3.61%
R²:    0.9234
==================================================
```

## References

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [PyTorch RNN Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- [Time Series Forecasting with Deep Learning](https://arxiv.org/abs/1703.07015)

## License

This module is part of the Scalar project.


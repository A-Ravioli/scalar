# Quick Start Guide

## Installation

```bash
# Navigate to trading directory
cd trading

# Install dependencies
pip install -r requirements.txt
```

## Running the Complete Pipeline

### Step 1: Generate Data (Already Done!)
```bash
python3 data_collector.py
```
✅ Generated: `data/synthetic_historical_data.csv` (8,641 samples)

### Step 2: Train Models (Already Done!)
```bash
# Train all three models
python3 train.py --model all --epochs 30 --batch-size 32
```
✅ Trained: RNN, GRU, LSTM models with excellent performance

### Step 3: Make Predictions

```bash
# Predict with RNN (best performer)
python3 predict.py --mode file \
    --model checkpoints/best_PriceRNN.pt \
    --model-type rnn

# Predict with GRU
python3 predict.py --mode file \
    --model checkpoints/best_PriceGRU.pt \
    --model-type gru

# Predict with LSTM
python3 predict.py --mode file \
    --model checkpoints/best_PriceLSTM.pt \
    --model-type lstm
```

## Quick Examples

### Train a Single Model
```bash
# LSTM with custom settings
python3 train.py \
    --model lstm \
    --epochs 50 \
    --batch-size 64 \
    --hidden-size 128 \
    --num-layers 3
```

### Generate More Data
```python
from data_collector import SFComputeDataCollector

collector = SFComputeDataCollector()
df = collector.generate_synthetic_historical_data(days=90)
```

### Load and Use a Model
```python
from predict import PricePredictor
import pandas as pd

# Load model
predictor = PricePredictor(
    model_path="checkpoints/best_PriceRNN.pt",
    preprocessor_path="checkpoints/preprocessor.pkl",
    model_type='rnn'
)

# Load data
df = pd.read_csv("data/synthetic_historical_data.csv")

# Make predictions
predictions = predictor.predict(df)
print(f"Latest prediction: ${predictions[-1][0]:.4f}")
```

## What's Already Done

✅ **Data Collection**
- 8,641 synthetic data points (30 days @ 5-min intervals)
- Realistic price dynamics with trends, cycles, and noise

✅ **Models Trained**
| Model | MAPE  | R²     | Status |
|-------|-------|--------|--------|
| RNN   | 2.43% | 0.9325 | ✅ Best |
| GRU   | 2.61% | 0.9253 | ✅      |
| LSTM  | 2.77% | 0.9136 | ✅      |

✅ **Generated Artifacts**
- 3 trained model checkpoints
- Preprocessor (for inference)
- Training visualizations (6 plots)
- Results JSON files

## File Structure

```
trading/
├── data/
│   └── synthetic_historical_data.csv (8,641 samples)
├── checkpoints/
│   ├── best_PriceRNN.pt (best model)
│   ├── best_PriceGRU.pt
│   ├── best_PriceLSTM.pt
│   └── preprocessor.pkl
├── results/
│   ├── rnn_results.json
│   ├── gru_results.json
│   └── lstm_results.json
├── plots/
│   ├── rnn_training_history.png
│   ├── rnn_predictions.png
│   ├── gru_training_history.png
│   ├── gru_predictions.png
│   ├── lstm_training_history.png
│   └── lstm_predictions.png
├── data_collector.py
├── data_preprocessing.py
├── models.py
├── train.py
├── predict.py
└── README.md
```

## Common Tasks

### View Training Results
```bash
cat results/rnn_results.json
```

### Check Model Performance
```python
import json

with open("results/rnn_results.json") as f:
    results = json.load(f)
    
print(f"Test MAPE: {results['metrics']['mape']:.2f}%")
print(f"Test R²: {results['metrics']['r2']:.4f}")
```

### Retrain a Model
```bash
# Just run train.py again with desired settings
python3 train.py --model rnn --epochs 50
```

## Next Steps

1. **Collect Real Data**: Modify `data_collector.py` to fetch real SF Compute data
2. **Deploy Model**: Create REST API with FastAPI
3. **Monitor Performance**: Set up logging and metrics tracking
4. **Automate Retraining**: Schedule periodic model updates

## Support

See `README.md` for detailed documentation.
See `SUMMARY.md` for implementation details and results.

---

**System Status**: 🟢 All Systems Operational  
**Models Ready**: 3/3 ✅  
**Training Complete**: ✅  
**Predictions Working**: ✅


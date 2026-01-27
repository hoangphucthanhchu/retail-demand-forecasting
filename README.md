# Demand Forecasting - ML Production Project

A production-ready ML project for demand forecasting using XGBoost/LightGBM with the M5 Forecasting dataset.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Dataset](#dataset)
- [Features](#features)
- [Models](#models)
- [Evaluation](#evaluation)

## Overview

This project implements a complete demand forecasting system with:

- **Data Pipeline**: Load and preprocess M5 Forecasting dataset
- **Feature Engineering**: Create lag features, rolling statistics, calendar features, price features
- **Model Training**: XGBoost and LightGBM with hyperparameter tuning
- **Evaluation**: Multiple metrics (RMSE, MAE, MAPE, WMAPE)
- **Inference Pipeline**: Production-ready prediction pipeline
- **Testing**: Unit tests for main components

## Project Structure

```
deman-forecasting/
├── configs/
│   └── config.yaml              # Configuration file
├── data/
│   ├── raw/                     # Raw M5 dataset (download separately)
│   ├── processed/               # Processed data
│   └── external/                # External data
├── models/                      # Saved models
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 02_training_and_evaluation.ipynb
├── src/
│   ├── data/
│   │   └── loader.py            # Data loading utilities
│   ├── features/
│   │   └── engineering.py       # Feature engineering
│   ├── models/
│   │   └── trainer.py           # Model training
│   ├── evaluation/
│   │   └── metrics.py           # Evaluation metrics
│   ├── pipeline/
│   │   ├── train_pipeline.py    # Training pipeline
│   │   └── inference.py         # Inference pipeline
│   └── utils/
│       ├── config.py            # Config utilities
│       └── logger.py            # Logging utilities
├── tests/
│   ├── test_features.py
│   └── test_metrics.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. **Clone repository and create virtual environment:**

```bash
cd deman-forecasting
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Download M5 Forecasting dataset:**

The dataset can be downloaded from [Kaggle M5 Forecasting Competition](https://www.kaggle.com/c/m5-forecasting-accuracy/data).

Required files:
- `calendar.csv`
- `sales_train_validation.csv`
- `sell_prices.csv`

Place these files in the `data/raw/` directory.

## Usage

### 1. Training Pipeline

Run the complete training pipeline:

```python
from src.pipeline.train_pipeline import TrainingPipeline

pipeline = TrainingPipeline(config_path='configs/config.yaml')
models, results = pipeline.run()
```

Or from command line:

```bash
python -m src.pipeline.train_pipeline
```

### 2. Inference

Use trained model to make predictions:

```python
from src.pipeline.inference import InferencePipeline
import pandas as pd

# Load historical data
df = pd.read_csv('data/processed/train_data.csv')

# Initialize inference pipeline
inference = InferencePipeline(
    config_path='configs/config.yaml',
    model_path='models/xgboost_model.pkl'
)

# Make predictions
predictions = inference.predict(df)
```

### 3. Jupyter Notebooks

Run notebooks to explore data and train models:

```bash
jupyter notebook notebooks/
```

## Dataset

The M5 Forecasting dataset includes:

- **Sales data**: Daily sales for 1,949 products over 1,913 days
- **Calendar**: Calendar events, holidays, snap days
- **Prices**: Weekly prices for each product-store combination

The dataset is reshaped from wide format (columns d_1, d_2, ...) to long format for easier processing.

## Features

### Time-based Features
- **Lag features**: demand_lag_1, demand_lag_7, demand_lag_14, demand_lag_28, ...
- **Rolling statistics**: 
  - Rolling mean/std/min/max with windows 7, 14, 28, 30, 60, 90 days

### Calendar Features
- Day of week, day of month, month, week of year
- Is weekend, is month start/end, is quarter start/end
- Cyclical encoding (sin/cos) for periodic features

### Price Features
- Price lags
- Price changes (absolute and percentage)
- Rolling price statistics

### Event Features
- Event indicators from M5 calendar
- SNAP days indicators

## Models

### XGBoost
- Regression objective with square error loss
- Early stopping to prevent overfitting
- Hyperparameters can be configured in `configs/config.yaml`

### LightGBM
- Regression with RMSE metric
- Early stopping
- Hyperparameters can be configured in `configs/config.yaml`

## Evaluation

Metrics used:

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **WMAPE**: Weighted Mean Absolute Percentage Error

Evaluation is performed on:
- Validation set (during training)
- Test set (final evaluation)

## Testing

Run tests:

```bash
pytest tests/
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Configuration

All configuration is defined in `configs/config.yaml`:

- Data paths and split ratios
- Feature engineering parameters
- Model hyperparameters
- Training parameters
- Evaluation metrics

## Notes

- M5 dataset needs to be downloaded separately from Kaggle
- Models are saved as pickle files
- Logging is configured in the config file
- Code structure follows best practices for ML production

## Future Improvements

- [ ] Add hyperparameter tuning with Optuna/Hyperopt
- [ ] Implement ensemble methods
- [ ] Add MLflow tracking
- [ ] Create API endpoint with FastAPI
- [ ] Add Docker containerization
- [ ] Implement online learning capabilities
- [ ] Add more sophisticated feature engineering (e.g., Fourier features)

## License

MIT License. See LICENSE for details.

## Author

**hoangphucthanhchu**
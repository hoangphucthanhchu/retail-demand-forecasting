"""
Model training utilities for XGBoost and LightGBM
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import joblib
from pathlib import Path

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    logging.warning("XGBoost not installed")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    logging.warning("LightGBM not installed")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trainer for XGBoost and LightGBM models"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Model Trainer
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        self.xgb_params = self.config.get('model', {}).get('xgboost', {})
        self.lgb_params = self.config.get('model', {}).get('lightgbm', {})
        self.models = {}
        
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame, y_val: pd.Series,
                     feature_names: Optional[List[str]] = None) -> xgb.XGBRegressor:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            feature_names: List of feature names
            
        Returns:
            Trained XGBoost model
        """
        if xgb is None:
            raise ImportError("XGBoost is not installed")
        
        logger.info("Training XGBoost model")
        
        # Prepare parameters
        params = self.xgb_params.copy()
        n_estimators = params.pop('n_estimators', 1000)
        early_stopping_rounds = params.pop('early_stopping_rounds', 50)
        
        # Create model
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            **params
        )
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        logger.info(f"XGBoost training completed. Best iteration: {model.best_iteration}")
        self.models['xgboost'] = model
        return model
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series,
                      feature_names: Optional[List[str]] = None) -> lgb.LGBMRegressor:
        """
        Train LightGBM model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            feature_names: List of feature names
            
        Returns:
            Trained LightGBM model
        """
        if lgb is None:
            raise ImportError("LightGBM is not installed")
        
        logger.info("Training LightGBM model")
        
        # Prepare parameters
        params = self.lgb_params.copy()
        n_estimators = params.pop('n_estimators', 1000)
        early_stopping_rounds = params.pop('early_stopping_rounds', 50)
        
        # Create model
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            **params
        )
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
        )
        
        logger.info(f"LightGBM training completed. Best iteration: {model.best_iteration_}")
        self.models['lightgbm'] = model
        return model
    
    def train_both(self, X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """
        Train both XGBoost and LightGBM models
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary with trained models
        """
        models = {}
        
        if xgb is not None:
            models['xgboost'] = self.train_xgboost(X_train, y_train, X_val, y_val)
        
        if lgb is not None:
            models['lightgbm'] = self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        return models
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a trained model
        
        Args:
            model_name: Name of the model ('xgboost' or 'lightgbm')
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        predictions = model.predict(X)
        
        # Ensure non-negative predictions for demand
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def save_model(self, model_name: str, path: str):
        """
        Save trained model
        
        Args:
            model_name: Name of the model to save
            path: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.models[model_name], model_path)
        logger.info(f"Model {model_name} saved to {model_path}")
    
    def load_model(self, model_name: str, path: str):
        """
        Load saved model
        
        Args:
            model_name: Name of the model
            path: Path to the saved model
        """
        model = joblib.load(path)
        self.models[model_name] = model
        logger.info(f"Model {model_name} loaded from {path}")
    
    def get_feature_importance(self, model_name: str, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            return importance_df
        else:
            logger.warning(f"Model {model_name} does not have feature_importances_ attribute")
            return pd.DataFrame()

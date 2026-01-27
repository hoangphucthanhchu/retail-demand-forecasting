"""
Inference pipeline for demand forecasting
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging
import joblib

from src.features.engineering import FeatureEngineer
from src.utils.config import load_config
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Inference pipeline for making predictions"""
    
    def __init__(self, config_path: str = "configs/config.yaml", model_path: Optional[str] = None):
        """
        Initialize inference pipeline
        
        Args:
            config_path: Path to configuration file
            model_path: Path to saved model (if None, will look in models directory)
        """
        self.config = load_config(config_path)
        setup_logging(
            log_level=self.config.get('logging', {}).get('level', 'INFO'),
            log_file=None
        )
        
        self.feature_engineer = FeatureEngineer(
            config=self.config.get('features', {})
        )
        
        # Load model
        if model_path is None:
            models_dir = self.config.get('paths', {}).get('models_dir', 'models')
            # Try to load XGBoost by default, fallback to LightGBM
            xgb_path = Path(models_dir) / "xgboost_model.pkl"
            lgb_path = Path(models_dir) / "lightgbm_model.pkl"
            
            if xgb_path.exists():
                model_path = str(xgb_path)
            elif lgb_path.exists():
                model_path = str(lgb_path)
            else:
                raise FileNotFoundError(f"No model found in {models_dir}")
        
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features for inference
        
        Args:
            df: Input dataframe with historical data
            
        Returns:
            Feature dataframe and list of feature names
        """
        logger.info("Preparing features for inference...")
        
        # Create all features
        df = self.feature_engineer.create_all_features(
            df,
            group_cols=['id'],
            target_col='demand',
            date_col='date'
        )
        
        # Select feature columns (same as training)
        exclude_cols = [
            'demand', 'id', 'date', 'item_id', 'dept_id', 'cat_id', 
            'store_id', 'state_id', 'wm_yr_wk', 'weekday', 'wday', 
            'month', 'year', 'd', 'event_name_1', 'event_name_2', 
            'event_type_1', 'event_type_2'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return df, feature_cols
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            df: Input dataframe with features
            
        Returns:
            Predictions array
        """
        logger.info("Making predictions...")
        
        df, feature_cols = self.prepare_features(df)
        
        # Get features for prediction (use most recent row for each id)
        X = df.groupby('id').tail(1)[feature_cols].fillna(0)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Ensure non-negative
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_future(self, df: pd.DataFrame, future_dates: pd.DatetimeIndex,
                      id_col: str = 'id') -> pd.DataFrame:
        """
        Predict demand for future dates
        
        Args:
            df: Historical data
            future_dates: Dates to predict for
            id_col: ID column name
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Predicting for {len(future_dates)} future dates...")
        
        # Get unique IDs
        unique_ids = df[id_col].unique()
        
        # Create future dataframe
        future_df = pd.DataFrame({
            id_col: np.repeat(unique_ids, len(future_dates)),
            'date': np.tile(future_dates, len(unique_ids))
        })
        
        # Merge with historical data to get metadata
        metadata_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        available_metadata = [col for col in metadata_cols if col in df.columns]
        
        if available_metadata:
            metadata = df.groupby(id_col)[available_metadata].first().reset_index()
            future_df = future_df.merge(metadata, on=id_col, how='left')
        
        # Merge with calendar if available
        if 'calendar' in self.config:
            # This would require calendar data to be loaded
            pass
        
        # For now, use the most recent historical data to predict
        # In production, you'd want to update features for each future date
        predictions = self.predict(df)
        
        # Assign predictions (this is simplified - in reality you'd need to
        # update features for each future date)
        future_df['predicted_demand'] = np.tile(predictions, len(future_dates))
        
        return future_df

"""
Training pipeline for demand forecasting
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from src.data.loader import M5DataLoader
from src.features.engineering import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import Evaluator
from src.utils.config import load_config
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize training pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        setup_logging(
            log_level=self.config.get('logging', {}).get('level', 'INFO'),
            log_file=None
        )
        
        self.data_loader = M5DataLoader(
            data_path=self.config.get('data', {}).get('raw_data_path', 'data/raw')
        )
        self.feature_engineer = FeatureEngineer(
            config=self.config.get('features', {})
        )
        self.model_trainer = ModelTrainer(
            config=self.config
        )
        self.evaluator = Evaluator(
            config=self.config
        )
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data"""
        logger.info("Loading data...")
        calendar, sales, prices = self.data_loader.load_all()
        
        if calendar is None or sales is None:
            raise ValueError("Failed to load required data files. Please download M5 dataset.")
        
        df = self.data_loader.create_base_dataset()
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
        """
        Prepare features for training
        
        Args:
            df: Input dataframe
            
        Returns:
            Feature dataframe and list of feature names
        """
        logger.info("Preparing features...")
        
        # Create all features
        df = self.feature_engineer.create_all_features(
            df,
            group_cols=['id'],
            target_col='demand',
            date_col='date'
        )
        
        # Select feature columns (exclude target and metadata)
        exclude_cols = [
            'demand', 'id', 'date', 'item_id', 'dept_id', 'cat_id', 
            'store_id', 'state_id', 'wm_yr_wk', 'weekday', 'wday', 
            'month', 'year', 'd', 'event_name_1', 'event_name_2', 
            'event_type_1', 'event_type_2'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with NaN in target (needed for lag features)
        df = df.dropna(subset=['demand'])
        
        logger.info(f"Prepared {len(feature_cols)} features")
        return df, feature_cols
    
    def split_data(self, df: pd.DataFrame, date_col: str = 'date',
                  test_size: int = 28, validation_size: int = 28) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Input dataframe
            date_col: Date column name
            test_size: Number of days for test set
            validation_size: Number of days for validation set
            
        Returns:
            Train, validation, and test dataframes
        """
        logger.info("Splitting data...")
        
        df = df.sort_values([date_col]).reset_index(drop=True)
        dates = df[date_col].unique()
        dates = sorted(dates)
        
        # Split: train -> validation -> test
        split_idx_test = len(dates) - test_size
        split_idx_val = split_idx_test - validation_size
        
        train_dates = dates[:split_idx_val]
        val_dates = dates[split_idx_val:split_idx_test]
        test_dates = dates[split_idx_test:]
        
        train_df = df[df[date_col].isin(train_dates)].copy()
        val_df = df[df[date_col].isin(val_dates)].copy()
        test_df = df[df[date_col].isin(test_dates)].copy()
        
        logger.info(f"Train: {len(train_df)} rows ({train_dates[0]} to {train_dates[-1]})")
        logger.info(f"Validation: {len(val_df)} rows ({val_dates[0]} to {val_dates[-1]})")
        logger.info(f"Test: {len(test_df)} rows ({test_dates[0]} to {test_dates[-1]})")
        
        return train_df, val_df, test_df
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
             feature_cols: list) -> Dict:
        """
        Train models
        
        Args:
            train_df: Training data
            val_df: Validation data
            feature_cols: List of feature column names
            
        Returns:
            Dictionary with trained models
        """
        logger.info("Training models...")
        
        # Prepare data
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df['demand']
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df['demand']
        
        # Train models
        models = self.model_trainer.train_both(X_train, y_train, X_val, y_val)
        
        # Evaluate on validation set
        logger.info("\nValidation Set Performance:")
        for model_name, model in models.items():
            y_pred = self.model_trainer.predict(model_name, X_val)
            metrics = self.evaluator.evaluate(y_val.values, y_pred)
            logger.info(f"\n{model_name.upper()}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return models
    
    def evaluate(self, test_df: pd.DataFrame, feature_cols: list) -> Dict:
        """
        Evaluate models on test set
        
        Args:
            test_df: Test data
            feature_cols: List of feature column names
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating on test set...")
        
        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df['demand']
        
        results = {}
        
        for model_name in self.model_trainer.models.keys():
            y_pred = self.model_trainer.predict(model_name, X_test)
            metrics = self.evaluator.evaluate(y_test.values, y_pred)
            results[model_name] = metrics
            
            logger.info(f"\n{model_name.upper()} Test Performance:")
            for metric, value in metrics.items():
                logger.info(f"  {metric.upper()}: {value:.4f}")
        
        return results
    
    def save_models(self, models_dir: str = "models"):
        """
        Save trained models
        
        Args:
            models_dir: Directory to save models
        """
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        
        for model_name in self.model_trainer.models.keys():
            model_path = models_path / f"{model_name}_model.pkl"
            self.model_trainer.save_model(model_name, str(model_path))
    
    def run(self):
        """Run complete training pipeline"""
        logger.info("=" * 60)
        logger.info("Starting Training Pipeline")
        logger.info("=" * 60)
        
        # Load data
        df = self.load_data()
        
        # Prepare features
        df, feature_cols = self.prepare_features(df)
        
        # Split data
        train_df, val_df, test_df = self.split_data(
            df,
            test_size=self.config.get('training', {}).get('test_size', 28),
            validation_size=self.config.get('training', {}).get('validation_size', 28)
        )
        
        # Train models
        models = self.train(train_df, val_df, feature_cols)
        
        # Evaluate on test set
        test_results = self.evaluate(test_df, feature_cols)
        
        # Save models
        self.save_models(
            models_dir=self.config.get('paths', {}).get('models_dir', 'models')
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("Training Pipeline Completed")
        logger.info("=" * 60)
        
        return models, test_results


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    models, results = pipeline.run()

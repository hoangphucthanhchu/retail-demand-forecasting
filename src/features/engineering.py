"""
Feature engineering for demand forecasting
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for time series demand forecasting"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Feature Engineer
        
        Args:
            config: Configuration dictionary with feature engineering parameters
        """
        self.config = config or {}
        self.lag_features = self.config.get('lag_features', {}).get('lags', [1, 7, 14, 28])
        self.rolling_windows = self.config.get('rolling_features', {}).get('windows', [7, 14, 28])
        self.rolling_functions = self.config.get('rolling_features', {}).get('functions', ['mean', 'std'])
        
    def create_lag_features(self, df: pd.DataFrame, group_cols: List[str], 
                           target_col: str = 'demand') -> pd.DataFrame:
        """
        Create lag features
        
        Args:
            df: Input dataframe
            group_cols: Columns to group by (e.g., ['id'])
            target_col: Target column name
            
        Returns:
            Dataframe with lag features
        """
        logger.info(f"Creating lag features: {self.lag_features}")
        df = df.copy()
        
        for lag in self.lag_features:
            df[f'{target_col}_lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, group_cols: List[str],
                               target_col: str = 'demand') -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: Input dataframe
            group_cols: Columns to group by
            target_col: Target column name
            
        Returns:
            Dataframe with rolling features
        """
        logger.info(f"Creating rolling features: windows={self.rolling_windows}, functions={self.rolling_functions}")
        df = df.copy()
        
        for window in self.rolling_windows:
            for func in self.rolling_functions:
                if func == 'mean':
                    df[f'{target_col}_rolling_mean_{window}'] = (
                        df.groupby(group_cols)[target_col]
                        .shift(1)
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )
                elif func == 'std':
                    df[f'{target_col}_rolling_std_{window}'] = (
                        df.groupby(group_cols)[target_col]
                        .shift(1)
                        .rolling(window=window, min_periods=1)
                        .std()
                        .reset_index(level=0, drop=True)
                    )
                elif func == 'min':
                    df[f'{target_col}_rolling_min_{window}'] = (
                        df.groupby(group_cols)[target_col]
                        .shift(1)
                        .rolling(window=window, min_periods=1)
                        .min()
                        .reset_index(level=0, drop=True)
                    )
                elif func == 'max':
                    df[f'{target_col}_rolling_max_{window}'] = (
                        df.groupby(group_cols)[target_col]
                        .shift(1)
                        .rolling(window=window, min_periods=1)
                        .max()
                        .reset_index(level=0, drop=True)
                    )
        
        return df
    
    def create_calendar_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Create calendar-based features
        
        Args:
            df: Input dataframe
            date_col: Date column name
            
        Returns:
            Dataframe with calendar features
        """
        logger.info("Creating calendar features")
        df = df.copy()
        
        if date_col not in df.columns:
            logger.warning(f"Date column {date_col} not found")
            return df
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic time features
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['day_of_month'] = df[date_col].dt.day
        df['month'] = df[date_col].dt.month
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        df['year'] = df[date_col].dt.year
        df['quarter'] = df[date_col].dt.quarter
        
        # Boolean features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
        
        # Cyclical encoding for periodic features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        return df
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-related features
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with price features
        """
        logger.info("Creating price features")
        df = df.copy()
        
        if 'sell_price' not in df.columns:
            logger.warning("sell_price column not found, skipping price features")
            return df
        
        # Group by item and store to create price features
        group_cols = ['item_id', 'store_id']
        
        # Price lag features
        for lag in [1, 7, 14, 28]:
            df[f'price_lag_{lag}'] = df.groupby(group_cols)['sell_price'].shift(lag)
        
        # Price change features
        df['price_change'] = df.groupby(group_cols)['sell_price'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # Rolling price statistics
        for window in [7, 14, 28]:
            df[f'price_rolling_mean_{window}'] = (
                df.groupby(group_cols)['sell_price']
                .shift(1)
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df[f'price_rolling_std_{window}'] = (
                df.groupby(group_cols)['sell_price']
                .shift(1)
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )
        
        return df
    
    def create_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create event/calendar event features from M5 dataset
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with event features
        """
        logger.info("Creating event features")
        df = df.copy()
        
        # Event columns from M5 calendar
        event_cols = [col for col in df.columns if col.startswith('event') or col.startswith('snap')]
        
        # One-hot encode events
        for col in event_cols:
            if df[col].dtype == 'object':
                # Create binary indicator
                df[f'{col}_indicator'] = df[col].notna().astype(int)
        
        return df
    
    def create_all_features(self, df: pd.DataFrame, group_cols: List[str] = ['id'],
                          target_col: str = 'demand', date_col: str = 'date') -> pd.DataFrame:
        """
        Create all features
        
        Args:
            df: Input dataframe
            group_cols: Columns to group by for time series features
            target_col: Target column name
            date_col: Date column name
            
        Returns:
            Dataframe with all engineered features
        """
        logger.info("Creating all features")
        
        # Calendar features (no dependency on target)
        df = self.create_calendar_features(df, date_col)
        
        # Price features (if available)
        if 'sell_price' in df.columns:
            df = self.create_price_features(df)
        
        # Event features
        df = self.create_event_features(df)
        
        # Lag features (depend on target)
        df = self.create_lag_features(df, group_cols, target_col)
        
        # Rolling features (depend on target)
        df = self.create_rolling_features(df, group_cols, target_col)
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df

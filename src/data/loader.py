"""
Data loading utilities for M5 Forecasting dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class M5DataLoader:
    """Loader for M5 Forecasting dataset"""
    
    def __init__(self, data_path: str = "data/raw"):
        """
        Initialize M5 Data Loader
        
        Args:
            data_path: Path to raw data directory
        """
        self.data_path = Path(data_path)
        self.calendar = None
        self.sales = None
        self.prices = None
        
    def load_calendar(self) -> pd.DataFrame:
        """Load calendar data"""
        calendar_path = self.data_path / "calendar.csv"
        if not calendar_path.exists():
            logger.warning(f"Calendar file not found at {calendar_path}")
            logger.info("Please download M5 dataset from: https://www.kaggle.com/c/m5-forecasting-accuracy/data")
            return None
            
        logger.info(f"Loading calendar from {calendar_path}")
        calendar = pd.read_csv(calendar_path)
        calendar['date'] = pd.to_datetime(calendar['date'])
        self.calendar = calendar
        return calendar
    
    def load_sales(self) -> pd.DataFrame:
        """Load sales data"""
        sales_path = self.data_path / "sales_train_validation.csv"
        if not sales_path.exists():
            logger.warning(f"Sales file not found at {sales_path}")
            logger.info("Please download M5 dataset from: https://www.kaggle.com/c/m5-forecasting-accuracy/data")
            return None
            
        logger.info(f"Loading sales from {sales_path}")
        sales = pd.read_csv(sales_path)
        self.sales = sales
        return sales
    
    def load_prices(self) -> pd.DataFrame:
        """Load prices data"""
        prices_path = self.data_path / "sell_prices.csv"
        if not prices_path.exists():
            logger.warning(f"Prices file not found at {prices_path}")
            return None
            
        logger.info(f"Loading prices from {prices_path}")
        prices = pd.read_csv(prices_path)
        prices['wm_yr_wk'] = prices['wm_yr_wk'].astype(int)
        self.prices = prices
        return prices
    
    def load_all(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load all M5 datasets"""
        calendar = self.load_calendar()
        sales = self.load_sales()
        prices = self.load_prices()
        return calendar, sales, prices
    
    def reshape_sales_to_long(self, sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
        """
        Reshape sales data from wide to long format
        
        Args:
            sales: Sales dataframe in wide format
            calendar: Calendar dataframe
            
        Returns:
            Long format dataframe with columns: id, date, demand
        """
        logger.info("Reshaping sales data to long format")
        
        # Get date columns (d_1, d_2, ..., d_1913)
        date_cols = [col for col in sales.columns if col.startswith('d_')]
        
        # Melt sales data
        sales_long = sales.melt(
            id_vars=[col for col in sales.columns if col not in date_cols],
            value_vars=date_cols,
            var_name='d',
            value_name='demand'
        )
        
        # Map d_* to actual dates
        d_to_date = dict(zip(
            [f'd_{i}' for i in range(1, len(calendar) + 1)],
            calendar['date'].values
        ))
        
        sales_long['date'] = sales_long['d'].map(d_to_date)
        sales_long = sales_long.drop('d', axis=1)
        sales_long = sales_long.sort_values(['id', 'date']).reset_index(drop=True)
        
        logger.info(f"Reshaped to {len(sales_long)} rows")
        return sales_long
    
    def create_base_dataset(self) -> pd.DataFrame:
        """
        Create base dataset by combining all data sources
        
        Returns:
            Combined dataframe with all features
        """
        if self.calendar is None or self.sales is None:
            logger.error("Please load calendar and sales data first")
            return None
        
        logger.info("Creating base dataset")
        
        # Reshape sales to long format
        sales_long = self.reshape_sales_to_long(self.sales, self.calendar)
        
        # Merge with calendar
        df = sales_long.merge(
            self.calendar,
            on='date',
            how='left'
        )
        
        # Merge with prices if available
        if self.prices is not None:
            # Create a mapping for prices
            df = df.merge(
                self.prices,
                on=['store_id', 'item_id', 'wm_yr_wk'],
                how='left'
            )
        
        logger.info(f"Base dataset created with {len(df)} rows and {len(df.columns)} columns")
        return df

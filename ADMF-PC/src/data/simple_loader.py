"""
Simple data loader with train/test split support.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path


class SimpleDataLoader:
    """Simple data loader that handles train/test splits."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with data configuration."""
        self.config = config
        self.file_path = config.get('file_path', 'data/SYNTH_1min.csv')
        self.split_ratio = config.get('split_ratio', 0.8)  # 80% train by default
        
        # Full dataset
        self.full_data: Optional[pd.DataFrame] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        
    def load_full_data(self, max_bars: Optional[int] = None) -> pd.DataFrame:
        """Load full dataset from CSV."""
        print(f"Loading data from: {self.file_path}")
        
        # Load CSV
        df = pd.read_csv(self.file_path, parse_dates=['timestamp'], index_col='timestamp')
        
        # Limit bars if specified
        if max_bars:
            df = df.iloc[:max_bars]
            print(f"Limited to {max_bars} bars")
        
        self.full_data = df
        
        # Calculate split point
        split_idx = int(len(df) * self.split_ratio)
        self.train_data = df.iloc[:split_idx]
        self.test_data = df.iloc[split_idx:]
        
        print(f"Loaded {len(df)} total bars")
        print(f"Train set: {len(self.train_data)} bars ({self.split_ratio:.0%})")
        print(f"Test set: {len(self.test_data)} bars ({1-self.split_ratio:.0%})")
        
        return df
    
    def get_dataset(self, dataset: str = 'full', max_bars: Optional[int] = None) -> pd.DataFrame:
        """
        Get specific dataset (train, test, or full).
        
        Args:
            dataset: 'train', 'test', or 'full'
            max_bars: Limit number of bars (applied after split)
            
        Returns:
            Requested dataset
        """
        # Load data if not already loaded
        if self.full_data is None:
            self.load_full_data()
        
        # Select dataset
        if dataset == 'train':
            data = self.train_data
        elif dataset == 'test':
            data = self.test_data
        else:  # 'full'
            data = self.full_data
        
        # Apply max_bars limit if specified
        if max_bars and len(data) > max_bars:
            data = data.iloc[:max_bars]
            print(f"Limited {dataset} set to {max_bars} bars")
        
        # Print info
        print(f"\nUsing {dataset} dataset:")
        print(f"  Bars: {len(data)}")
        print(f"  Date range: {data.index[0]} to {data.index[-1]}")
        print(f"  Price range: ${data['close'].min():.2f} to ${data['close'].max():.2f}")
        
        # Check for trading opportunities
        buy_opps = (data['close'] <= 90).sum()
        sell_opps = (data['close'] >= 100).sum()
        print(f"  Buy opportunities (price <= $90): {buy_opps}")
        print(f"  Sell opportunities (price >= $100): {sell_opps}")
        
        return data
    
    def get_train_test_info(self) -> Dict[str, Any]:
        """Get information about train/test split."""
        if self.full_data is None:
            self.load_full_data()
            
        return {
            'total_bars': len(self.full_data),
            'train_bars': len(self.train_data),
            'test_bars': len(self.test_data),
            'split_ratio': self.split_ratio,
            'train_date_range': (self.train_data.index[0], self.train_data.index[-1]),
            'test_date_range': (self.test_data.index[0], self.test_data.index[-1]),
        }
"""
Signal Context Analysis

Joins sparse signal data with source data to analyze performance in different contexts.
This allows post-hoc analysis without modifying the signal generation system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SignalContextAnalyzer:
    """
    Analyzes signals in context by joining with source data and computing features.
    
    This approach:
    1. Loads sparse signal data
    2. Loads source bar data for the same period
    3. Computes features/indicators on source data
    4. Joins signals with features at signal timestamps
    5. Analyzes performance by context
    """
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.signal_data = None
        self.source_data = None
        self.enhanced_signals = None
        
    def load_signal_data(self, strategy_name: str) -> pd.DataFrame:
        """Load sparse signal data for a strategy."""
        signal_file = self.workspace_path / f"{strategy_name}.parquet"
        if not signal_file.exists():
            raise FileNotFoundError(f"Signal file not found: {signal_file}")
            
        self.signal_data = pd.read_parquet(signal_file)
        logger.info(f"Loaded {len(self.signal_data)} signals from {strategy_name}")
        return self.signal_data
    
    def load_source_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load source bar data for the period."""
        # This would load from your data source
        # For now, assuming you have a method to get this
        # self.source_data = load_bars(symbol, start_date, end_date)
        pass
    
    def compute_context_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all context features on source data."""
        df = data.copy()
        
        # Technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        
        # VWAP (reset daily)
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        df['vwap'] = df.groupby('date').apply(
            lambda x: (x['close'] * x['volume']).cumsum() / x['volume'].cumsum()
        ).reset_index(0, drop=True)
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        df['atr_14'] = df['tr'].rolling(14).mean()
        
        # Relative positions
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20'] * 100
        df['price_vs_sma200'] = (df['close'] - df['sma_200']) / df['sma_200'] * 100
        df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        
        # Market regimes
        df['trend_regime'] = pd.cut(
            df['price_vs_sma200'],
            bins=[-np.inf, -2, 2, np.inf],
            labels=['DOWNTREND', 'SIDEWAYS', 'UPTREND']
        )
        
        # Volatility regime (ATR percentile)
        df['atr_percentile'] = df['atr_14'].rolling(100).rank(pct=True)
        df['vol_regime'] = pd.cut(
            df['atr_percentile'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['LOW_VOL', 'NORMAL_VOL', 'HIGH_VOL']
        )
        
        return df
    
    def join_signals_with_context(self) -> pd.DataFrame:
        """Join sparse signals with context features."""
        if self.signal_data is None or self.source_data is None:
            raise ValueError("Must load signal and source data first")
        
        # Compute features on source data
        source_with_features = self.compute_context_features(self.source_data)
        
        # Join signals with features at signal timestamps
        self.enhanced_signals = pd.merge_asof(
            self.signal_data.sort_values('timestamp'),
            source_with_features.sort_values('timestamp'),
            on='timestamp',
            direction='backward'  # Use most recent feature values
        )
        
        return self.enhanced_signals
    
    def analyze_by_context(self, context_column: str) -> pd.DataFrame:
        """Analyze signal performance by a context column."""
        if self.enhanced_signals is None:
            raise ValueError("Must join signals with context first")
        
        # Add trade outcomes (simplified - you'd calculate actual PnL)
        signals = self.enhanced_signals.copy()
        signals['next_price'] = signals['close'].shift(-1)
        signals['return'] = np.where(
            signals['signal_value'] > 0,
            (signals['next_price'] - signals['close']) / signals['close'],
            np.where(
                signals['signal_value'] < 0,
                (signals['close'] - signals['next_price']) / signals['close'],
                0
            )
        )
        
        # Group by context
        context_stats = signals.groupby(context_column).agg({
            'signal_value': 'count',  # Number of signals
            'return': ['mean', 'std', 'sum']  # Performance metrics
        }).round(4)
        
        context_stats.columns = ['signal_count', 'avg_return', 'return_std', 'total_return']
        context_stats['sharpe'] = context_stats['avg_return'] / context_stats['return_std']
        
        return context_stats
    
    def analyze_entry_conditions(self) -> pd.DataFrame:
        """Analyze what conditions were present at entry."""
        if self.enhanced_signals is None:
            raise ValueError("Must join signals with context first")
        
        # Filter to entry signals only
        entries = self.enhanced_signals[self.enhanced_signals['signal_value'] != 0].copy()
        
        # Summarize conditions at entry
        conditions = pd.DataFrame({
            'total_entries': len(entries),
            'long_entries': len(entries[entries['signal_value'] > 0]),
            'short_entries': len(entries[entries['signal_value'] < 0]),
            
            # Price position
            'pct_above_sma200': (entries['price_vs_sma200'] > 0).sum() / len(entries) * 100,
            'pct_above_vwap': (entries['price_vs_vwap'] > 0).sum() / len(entries) * 100,
            'avg_distance_sma200': entries['price_vs_sma200'].mean(),
            'avg_distance_vwap': entries['price_vs_vwap'].mean(),
            
            # Regime distribution
            'pct_uptrend': (entries['trend_regime'] == 'UPTREND').sum() / len(entries) * 100,
            'pct_downtrend': (entries['trend_regime'] == 'DOWNTREND').sum() / len(entries) * 100,
            'pct_high_vol': (entries['vol_regime'] == 'HIGH_VOL').sum() / len(entries) * 100,
        }, index=[0])
        
        return conditions


# Example usage function
def analyze_bollinger_strategy(workspace_path: Path, symbol: str = 'SPY'):
    """Complete analysis of Bollinger Bands strategy with context."""
    
    analyzer = SignalContextAnalyzer(workspace_path)
    
    # 1. Load signal data
    signals = analyzer.load_signal_data('bollinger_bands')
    
    # 2. Load source data for the same period
    start = signals['timestamp'].min()
    end = signals['timestamp'].max()
    analyzer.load_source_data(symbol, start, end)
    
    # 3. Join with context
    enhanced = analyzer.join_signals_with_context()
    
    # 4. Analyze by different contexts
    print("\n=== Performance by Trend Regime ===")
    print(analyzer.analyze_by_context('trend_regime'))
    
    print("\n=== Performance by Volatility Regime ===")
    print(analyzer.analyze_by_context('vol_regime'))
    
    print("\n=== Entry Conditions Summary ===")
    print(analyzer.analyze_entry_conditions())
    
    # 5. Custom analysis - e.g., only trades when price > VWAP
    above_vwap = enhanced[enhanced['price_vs_vwap'] > 0]
    below_vwap = enhanced[enhanced['price_vs_vwap'] <= 0]
    
    print(f"\n=== VWAP Analysis ===")
    print(f"Signals above VWAP: {len(above_vwap)} ({len(above_vwap)/len(enhanced)*100:.1f}%)")
    print(f"Signals below VWAP: {len(below_vwap)} ({len(below_vwap)/len(enhanced)*100:.1f}%)")
    
    return enhanced
# Custom SQL Functions for Trading Analytics
"""
Trading-specific SQL functions that extend DuckDB capabilities
for ADMF-PC analytics workflows.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import json
from datetime import datetime, timedelta

from .exceptions import AnalyticsError


class TradingFunctions:
    """Container for trading-specific SQL functions"""
    
    def __init__(self, workspace_path: Path):
        """Initialize with workspace context
        
        Args:
            workspace_path: Path to workspace for file access
        """
        self.workspace_path = workspace_path
    
    def load_signals(self, file_path: str) -> pd.DataFrame:
        """Load signal data from file path
        
        Args:
            file_path: Relative path to signal file
            
        Returns:
            DataFrame with signal data
        """
        try:
            full_path = self.workspace_path / file_path
            if not full_path.exists():
                raise AnalyticsError(f"Signal file not found: {file_path}")
            
            if full_path.suffix == '.parquet':
                return pd.read_parquet(full_path)
            elif full_path.suffix == '.json':
                with open(full_path, 'r') as f:
                    data = json.load(f)
                # Convert sparse JSON to DataFrame
                return self._sparse_json_to_dataframe(data)
            else:
                raise AnalyticsError(f"Unsupported signal file format: {full_path.suffix}")
                
        except Exception as e:
            raise AnalyticsError(f"Failed to load signals from {file_path}: {e}")
    
    def load_states(self, file_path: str) -> pd.DataFrame:
        """Load classifier states from file path
        
        Args:
            file_path: Relative path to states file
            
        Returns:
            DataFrame with classifier states
        """
        try:
            full_path = self.workspace_path / file_path
            if not full_path.exists():
                raise AnalyticsError(f"States file not found: {file_path}")
            
            if full_path.suffix == '.parquet':
                return pd.read_parquet(full_path)
            elif full_path.suffix == '.json':
                with open(full_path, 'r') as f:
                    data = json.load(f)
                # Convert sparse JSON to DataFrame
                return self._sparse_states_to_dataframe(data)
            else:
                raise AnalyticsError(f"Unsupported states file format: {full_path.suffix}")
                
        except Exception as e:
            raise AnalyticsError(f"Failed to load states from {file_path}: {e}")
    
    def signal_correlation(self, file_a: str, file_b: str) -> float:
        """Calculate correlation between two signal files
        
        Args:
            file_a: Path to first signal file
            file_b: Path to second signal file
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        try:
            signals_a = self.load_signals(file_a)
            signals_b = self.load_signals(file_b)
            
            return self._correlate_sparse_signals(signals_a, signals_b)
            
        except Exception as e:
            raise AnalyticsError(f"Failed to calculate signal correlation: {e}")
    
    def expand_signals(self, file_path: str, total_bars: int) -> pd.DataFrame:
        """Expand sparse signals to full timeseries
        
        Args:
            file_path: Path to sparse signal file
            total_bars: Total number of bars in the timeseries
            
        Returns:
            DataFrame with full timeseries (bar_idx, signal)
        """
        try:
            sparse_signals = self.load_signals(file_path)
            
            # Create full range
            full_range = pd.DataFrame({'bar_idx': range(total_bars)})
            
            # Merge with sparse signals
            expanded = full_range.merge(sparse_signals, on='bar_idx', how='left')
            
            # Forward fill signals (last signal persists until changed)
            expanded['signal'] = expanded['signal'].fillna(method='ffill').fillna(0)
            
            return expanded
            
        except Exception as e:
            raise AnalyticsError(f"Failed to expand signals: {e}")
    
    def expand_states(self, file_path: str, total_bars: int) -> pd.DataFrame:
        """Expand sparse classifier states to full timeseries
        
        Args:
            file_path: Path to sparse states file
            total_bars: Total number of bars in the timeseries
            
        Returns:
            DataFrame with full timeseries (bar_idx, regime)
        """
        try:
            sparse_states = self.load_states(file_path)
            
            # Create full range
            full_range = pd.DataFrame({'bar_idx': range(total_bars)})
            
            # Merge with sparse states
            expanded = full_range.merge(sparse_states, on='bar_idx', how='left')
            
            # Forward fill states
            expanded['regime'] = expanded['regime'].fillna(method='ffill').fillna('UNKNOWN')
            
            return expanded
            
        except Exception as e:
            raise AnalyticsError(f"Failed to expand states: {e}")
    
    def signal_stats(self, file_path: str) -> Dict[str, Any]:
        """Calculate signal statistics
        
        Args:
            file_path: Path to signal file
            
        Returns:
            Dictionary with signal statistics
        """
        try:
            signals = self.load_signals(file_path)
            
            stats = {
                'total_changes': len(signals),
                'long_signals': len(signals[signals['signal'] > 0]),
                'short_signals': len(signals[signals['signal'] < 0]),
                'neutral_signals': len(signals[signals['signal'] == 0]),
                'max_signal': float(signals['signal'].max()) if not signals.empty else 0,
                'min_signal': float(signals['signal'].min()) if not signals.empty else 0,
                'signal_range': signals['bar_idx'].max() - signals['bar_idx'].min() if not signals.empty else 0
            }
            
            return stats
            
        except Exception as e:
            raise AnalyticsError(f"Failed to calculate signal stats: {e}")
    
    def regime_stats(self, file_path: str) -> Dict[str, Any]:
        """Calculate regime/classifier statistics
        
        Args:
            file_path: Path to states file
            
        Returns:
            Dictionary with regime statistics
        """
        try:
            states = self.load_states(file_path)
            
            if states.empty:
                return {'regime_counts': {}, 'total_changes': 0}
            
            # Count regime occurrences
            regime_counts = states['regime'].value_counts().to_dict()
            
            # Calculate regime changes
            regime_changes = (states['regime'] != states['regime'].shift(1)).sum() - 1
            
            stats = {
                'regime_counts': regime_counts,
                'total_changes': int(regime_changes),
                'unique_regimes': len(regime_counts),
                'dominant_regime': states['regime'].mode().iloc[0] if not states.empty else None,
                'state_range': states['bar_idx'].max() - states['bar_idx'].min() if not states.empty else 0
            }
            
            return stats
            
        except Exception as e:
            raise AnalyticsError(f"Failed to calculate regime stats: {e}")
    
    def load_events(self, file_path: str) -> pd.DataFrame:
        """Load event data from file path
        
        Args:
            file_path: Relative path to events file
            
        Returns:
            DataFrame with event data
        """
        try:
            full_path = self.workspace_path / file_path
            if not full_path.exists():
                raise AnalyticsError(f"Events file not found: {file_path}")
            
            return pd.read_parquet(full_path)
            
        except Exception as e:
            raise AnalyticsError(f"Failed to load events from {file_path}: {e}")
    
    def sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio
        
        Args:
            returns: List of returns
            risk_free_rate: Risk-free rate (annualized)
            
        Returns:
            Sharpe ratio
        """
        try:
            returns_array = np.array(returns)
            excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
            
            if len(excess_returns) == 0 or np.std(excess_returns) == 0:
                return 0.0
            
            return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))
            
        except Exception as e:
            raise AnalyticsError(f"Failed to calculate Sharpe ratio: {e}")
    
    def max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown
        
        Args:
            returns: List of returns
            
        Returns:
            Maximum drawdown (positive value)
        """
        try:
            returns_array = np.array(returns)
            if len(returns_array) == 0:
                return 0.0
            
            # Calculate cumulative returns
            cumulative = np.cumprod(1 + returns_array)
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative)
            
            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max
            
            return float(-np.min(drawdown))
            
        except Exception as e:
            raise AnalyticsError(f"Failed to calculate max drawdown: {e}")
    
    def win_rate(self, returns: List[float]) -> float:
        """Calculate win rate
        
        Args:
            returns: List of returns
            
        Returns:
            Win rate (0 to 1)
        """
        try:
            returns_array = np.array(returns)
            if len(returns_array) == 0:
                return 0.0
            
            winning_trades = np.sum(returns_array > 0)
            total_trades = len(returns_array)
            
            return float(winning_trades / total_trades)
            
        except Exception as e:
            raise AnalyticsError(f"Failed to calculate win rate: {e}")
    
    def _sparse_json_to_dataframe(self, data: Union[dict, list]) -> pd.DataFrame:
        """Convert sparse JSON signals to DataFrame"""
        if isinstance(data, dict):
            # Handle {'123': 1, '456': -1} format
            df_data = [{'bar_idx': int(k), 'signal': v} for k, v in data.items()]
        elif isinstance(data, list):
            # Handle [{'bar_idx': 123, 'signal': 1}, ...] format
            df_data = data
        else:
            raise AnalyticsError(f"Unsupported JSON format: {type(data)}")
        
        return pd.DataFrame(df_data)
    
    def _sparse_states_to_dataframe(self, data: Union[dict, list]) -> pd.DataFrame:
        """Convert sparse JSON states to DataFrame"""
        if isinstance(data, dict):
            # Handle {'123': 'BULL', '456': 'BEAR'} format
            df_data = [{'bar_idx': int(k), 'regime': v} for k, v in data.items()]
        elif isinstance(data, list):
            # Handle [{'bar_idx': 123, 'regime': 'BULL'}, ...] format
            df_data = data
        else:
            raise AnalyticsError(f"Unsupported JSON format: {type(data)}")
        
        return pd.DataFrame(df_data)
    
    def _correlate_sparse_signals(self, signals_a: pd.DataFrame, signals_b: pd.DataFrame) -> float:
        """Calculate correlation between sparse signal DataFrames"""
        try:
            # Merge on bar_idx
            merged = signals_a.merge(signals_b, on='bar_idx', how='outer', suffixes=('_a', '_b'))
            
            # Fill NaN with 0 (no signal change)
            merged['signal_a'] = merged['signal_a'].fillna(0)
            merged['signal_b'] = merged['signal_b'].fillna(0)
            
            # Calculate correlation
            if len(merged) < 2:
                return 0.0
            
            correlation = merged['signal_a'].corr(merged['signal_b'])
            return float(correlation) if not pd.isna(correlation) else 0.0
            
        except Exception as e:
            raise AnalyticsError(f"Failed to correlate sparse signals: {e}")


def register_functions(conn, workspace_path: Path) -> None:
    """Register custom functions with DuckDB connection
    
    Args:
        conn: DuckDB connection
        workspace_path: Path to workspace for function context
    """
    try:
        functions = TradingFunctions(workspace_path)
        
        # Note: DuckDB Python UDF registration is complex
        # For now, we'll implement these as methods in the workspace class
        # Full UDF registration would require more complex setup
        
        # Store functions instance for later use
        # NOTE: Can't store on conn object - DuckDB connections don't allow attribute setting
        # conn._trading_functions = functions
        
    except Exception as e:
        raise AnalyticsError(f"Failed to register custom functions: {e}")


def get_available_functions() -> List[str]:
    """Get list of available custom functions
    
    Returns:
        List of function names
    """
    return [
        'load_signals',
        'load_states', 
        'load_events',
        'signal_correlation',
        'expand_signals',
        'expand_states',
        'signal_stats',
        'regime_stats',
        'sharpe_ratio',
        'max_drawdown',
        'win_rate'
    ]
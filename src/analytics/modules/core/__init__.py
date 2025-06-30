"""Core utilities for analytics modules."""

from .data_loading import (
    load_global_traces,
    load_strategy_index,
    load_run_index,
    load_execution_trades,
    load_signal_references,
    load_market_data,
    load_trace_metadata
)

from .helpers import (
    format_large_number,
    calculate_returns,
    resample_ohlc
)

from .metrics import (
    calculate_sharpe,
    calculate_compound_sharpe,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_profit_factor
)

__all__ = [
    # Data loading
    'load_global_traces',
    'load_strategy_index',
    'load_run_index',
    'load_execution_trades',
    'load_signal_references',
    'load_market_data',
    'load_trace_metadata',
    
    # Helpers
    'format_large_number',
    'calculate_returns',
    'resample_ohlc',
    
    # Metrics
    'calculate_sharpe',
    'calculate_compound_sharpe',
    'calculate_max_drawdown',
    'calculate_win_rate',
    'calculate_profit_factor'
]
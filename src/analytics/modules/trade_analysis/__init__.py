"""Trade extraction and analysis modules."""

from .extraction import (
    extract_trades,
    extract_trades_with_stops,
    merge_trades_with_metadata
)

from .frequency import (
    calculate_trade_frequency,
    filter_by_frequency,
    analyze_frequency_distribution
)

from .duration import (
    analyze_trade_duration,
    filter_by_duration,
    calculate_holding_periods
)

from .expected_trades import (
    generate_expected_trades,
    aggregate_expected_trades
)

__all__ = [
    # Extraction
    'extract_trades',
    'extract_trades_with_stops',
    'merge_trades_with_metadata',
    
    # Frequency
    'calculate_trade_frequency',
    'filter_by_frequency',
    'analyze_frequency_distribution',
    
    # Duration
    'analyze_trade_duration',
    'filter_by_duration',
    'calculate_holding_periods',
    
    # Expected trades
    'generate_expected_trades',
    'aggregate_expected_trades'
]
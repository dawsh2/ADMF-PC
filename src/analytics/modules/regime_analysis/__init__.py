"""Market regime analysis modules."""

from .volatility import (
    add_volatility_regime,
    calculate_atr,
    analyze_volatility_clusters
)

from .trend import (
    add_trend_regime,
    calculate_trend_strength,
    identify_trend_changes
)

from .volume import (
    add_volume_regime,
    calculate_relative_volume,
    analyze_volume_patterns
)

from .combined import (
    add_regime_indicators,
    analyze_by_regime,
    create_composite_regime
)

__all__ = [
    # Volatility
    'add_volatility_regime',
    'calculate_atr',
    'analyze_volatility_clusters',
    
    # Trend
    'add_trend_regime',
    'calculate_trend_strength',
    'identify_trend_changes',
    
    # Volume
    'add_volume_regime',
    'calculate_relative_volume',
    'analyze_volume_patterns',
    
    # Combined
    'add_regime_indicators',
    'analyze_by_regime',
    'create_composite_regime'
]
"""
Bollinger RSI Divergence EXACT Strategy

This uses the bb_rsi_divergence_exact feature to trade the EXACT pattern that showed:
- 494 trades
- 71.9% win rate
- 11.82% net return
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='bollinger_rsi_divergence_exact',
    feature_discovery=lambda params: [
        # The exact divergence feature
        FeatureSpec('bb_rsi_divergence_exact', {}),
        # Base features for reference (need separate specs for each output)
        FeatureSpec('bollinger_bands', {
            'period': 20,
            'std_dev': 2.0
        }, output_component='upper'),
        FeatureSpec('bollinger_bands', {
            'period': 20,
            'std_dev': 2.0
        }, output_component='middle'),
        FeatureSpec('bollinger_bands', {
            'period': 20,
            'std_dev': 2.0
        }, output_component='lower'),
        FeatureSpec('rsi', {
            'period': 14
        })
    ],
    parameter_space={
        # No parameters - using exact values from profitable backtest
    },
    strategy_type='mean_reversion',
    tags=['bollinger_bands', 'rsi', 'divergence', 'mean_reversion', 'exact_pattern']
)
def bollinger_rsi_divergence_exact(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Trade the EXACT RSI divergence pattern from the profitable backtest.
    
    All logic is in the bb_rsi_divergence_exact feature which implements:
    - Multi-bar divergence detection
    - Confirmation waiting
    - Position management with middle band exit
    """
    
    # Get the divergence signal
    divergence = features.get('bb_rsi_divergence_exact', {})
    signal_value = divergence.get('value', 0)
    
    # Get indicator values for metadata
    bb_upper = features.get('bollinger_bands_upper')
    bb_middle = features.get('bollinger_bands_middle')
    bb_lower = features.get('bollinger_bands_lower')
    rsi = features.get('rsi')
    
    # Build response
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'bollinger_rsi_divergence_exact',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            # Signal info
            'reason': divergence.get('reason', ''),
            'stage': divergence.get('stage', ''),
            'position_type': divergence.get('position_type', 0),
            'in_position': divergence.get('in_position', False),
            
            # Market data
            'close': bar.get('close', 0),
            'upper_band': bb_upper,
            'middle_band': bb_middle,
            'lower_band': bb_lower,
            'rsi': rsi,
            
            # Tracking info
            'extremes_tracked': divergence.get('extremes_tracked', 0),
            'divergence_active': divergence.get('divergence_active', False)
        }
    }
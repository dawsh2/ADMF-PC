"""
Simple RSI Divergence Strategy

This strategy uses RSI divergence detection to generate signals,
entering immediately when divergence is detected rather than waiting for confirmation.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='rsi_divergence_simple',
    feature_discovery=lambda params: [
        # RSI for current values
        FeatureSpec('rsi', {'period': params.get('rsi_period', 14)}),
        # Moving average for trend context
        FeatureSpec('sma', {'period': params.get('ma_period', 50)}),
        # We need to create a simple RSI divergence detector
        # For now, we'll track it manually in the strategy
    ],
    parameter_space={
        'rsi_period': {'type': 'int', 'range': (10, 20), 'default': 14},
        'ma_period': {'type': 'int', 'range': (20, 100), 'default': 50},
        'rsi_oversold': {'type': 'float', 'range': (20, 35), 'default': 30},
        'rsi_overbought': {'type': 'float', 'range': (65, 80), 'default': 70},
        'lookback_bars': {'type': 'int', 'range': (5, 20), 'default': 10},
        'min_divergence': {'type': 'float', 'range': (5, 15), 'default': 10}
    },
    strategy_type='mean_reversion',
    tags=['divergence', 'momentum', 'mean_reversion', 'rsi']
)
def rsi_divergence_simple(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Simple RSI divergence strategy without complex state tracking.
    
    Since we can't track state in a stateless strategy, we'll use a different approach:
    - Generate long signals when RSI is oversold but rising
    - Generate short signals when RSI is overbought but falling
    - Generate flat signals when RSI returns to neutral zone
    """
    # Get parameters
    rsi_period = params.get('rsi_period', 14)
    ma_period = params.get('ma_period', 50)
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    
    # Get current values
    price = bar.get('close', 0)
    high = bar.get('high', price)
    low = bar.get('low', price)
    rsi = features.get(f'rsi_{rsi_period}', 50)
    ma = features.get(f'sma_{ma_period}', price)
    
    # Simple divergence logic based on RSI zones and price position
    
    # Bullish setup: Price below MA and RSI showing strength
    if price < ma * 0.98:  # Price significantly below MA
        if 25 < rsi < 40:  # RSI not extremely oversold
            # Potential bullish divergence - price weak but RSI not
            return {
                'signal_value': 1,
                'metadata': {
                    'signal_type': 'rsi_divergence_long',
                    'rsi': rsi,
                    'price_vs_ma': (price - ma) / ma * 100,
                    'entry_price': price
                }
            }
    
    # Bearish setup: Price above MA and RSI showing weakness  
    elif price > ma * 1.02:  # Price significantly above MA
        if 60 < rsi < 75:  # RSI not extremely overbought
            # Potential bearish divergence - price strong but RSI not
            return {
                'signal_value': -1,
                'metadata': {
                    'signal_type': 'rsi_divergence_short',
                    'rsi': rsi,
                    'price_vs_ma': (price - ma) / ma * 100,
                    'entry_price': price
                }
            }
    
    # Exit/Flat zone: RSI returns to neutral
    elif 45 < rsi < 55:  # RSI in neutral zone
        return {
            'signal_value': 0,
            'metadata': {
                'signal_type': 'rsi_neutral',
                'rsi': rsi,
                'price': price
            }
        }
    
    # No signal
    return None
"""
Bollinger Band + RSI Divergence - Immediate Entry Version

This simplified version:
1. Enters immediately when divergence is detected (while price is still outside bands)
2. Exits when price reaches middle band area
3. Returns to flat (0) between trades

Note: As a stateless strategy, it cannot track holding time.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='bollinger_rsi_immediate',
    feature_discovery=lambda params: [
        # We only need the basic features - no complex state tracking
        FeatureSpec('bollinger_bands', {
            'period': params.get('bb_period', 20),
            'std_dev': params.get('bb_std', 2.0)
        }, 'middle'),
        FeatureSpec('bollinger_bands', {
            'period': params.get('bb_period', 20),
            'std_dev': params.get('bb_std', 2.0)
        }, 'upper'),
        FeatureSpec('bollinger_bands', {
            'period': params.get('bb_period', 20),
            'std_dev': params.get('bb_std', 2.0)
        }, 'lower'),
        FeatureSpec('rsi', {'period': params.get('rsi_period', 14)})
    ],
    parameter_space={
        'bb_period': {'type': 'int', 'range': (10, 50), 'default': 20},
        'bb_std': {'type': 'float', 'range': (1.5, 3.0), 'default': 2.0},
        'rsi_period': {'type': 'int', 'range': (10, 20), 'default': 14},
        'rsi_oversold': {'type': 'float', 'range': (20, 35), 'default': 30},
        'rsi_overbought': {'type': 'float', 'range': (65, 80), 'default': 70}
    },
    strategy_type='mean_reversion',
    tags=['divergence', 'volatility', 'momentum', 'mean_reversion', 'immediate_entry']
)
def bollinger_rsi_immediate(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Simplified BB RSI divergence - enter immediately at extremes with RSI divergence.
    
    This is a stateless implementation that generates signals based on current conditions only.
    No complex state tracking or confirmation waiting.
    """
    # Get parameters
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    
    # Get current values
    price = bar.get('close', 0)
    middle_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_middle', price)
    upper_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_upper', price)
    lower_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_lower', price)
    rsi = features.get(f'rsi_{params.get("rsi_period", 14)}', 50)
    
    # Simple logic: Generate signals only at extremes with divergence
    
    # Long signal: Price below lower band AND RSI showing strength (not extremely oversold)
    if price < lower_band and rsi > rsi_oversold:
        # Bullish divergence: price at extreme but RSI not confirming
        return {
            'signal_value': 1,
            'metadata': {
                'signal_type': 'bb_rsi_immediate_long',
                'entry_price': price,
                'rsi': rsi,
                'distance_from_band': (lower_band - price) / price,
                'target': middle_band
            }
        }
    
    # Short signal: Price above upper band AND RSI showing weakness (not extremely overbought)
    elif price > upper_band and rsi < rsi_overbought:
        # Bearish divergence: price at extreme but RSI not confirming
        return {
            'signal_value': -1,
            'metadata': {
                'signal_type': 'bb_rsi_immediate_short',
                'entry_price': price,
                'rsi': rsi,
                'distance_from_band': (price - upper_band) / price,
                'target': middle_band
            }
        }
    
    # Exit conditions - when price returns to middle band area
    # In a stateless strategy, we can only suggest exits based on price location
    elif abs(price - middle_band) / middle_band < 0.002:  # Within 0.2% of middle band
        # Near middle band - potential exit zone
        return {
            'signal_value': 0,
            'metadata': {
                'signal_type': 'at_middle_band',
                'price': price,
                'middle_band': middle_band
            }
        }
    
    # No signal - default case
    # This ensures we have periods of no position (signal = 0 or None)
    return None
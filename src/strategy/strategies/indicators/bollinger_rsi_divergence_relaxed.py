"""
Bollinger Band + RSI Divergence (Relaxed)

A more relaxed version that looks for simple RSI divergence patterns.
When price is near/outside bands with RSI showing non-confirmation.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='bollinger_rsi_divergence_relaxed',
    feature_discovery=lambda params: [
        # Bollinger Bands
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
        # RSI
        FeatureSpec('rsi', {'period': params.get('rsi_period', 14)})
    ],
    parameter_space={
        'bb_period': {'type': 'int', 'range': (10, 50), 'default': 20},
        'bb_std': {'type': 'float', 'range': (1.5, 3.0), 'default': 2.0},
        'rsi_period': {'type': 'int', 'range': (10, 20), 'default': 14},
        'rsi_threshold': {'type': 'float', 'range': (5, 15), 'default': 10},
        'entry_threshold': {'type': 'float', 'range': (0.0, 0.3), 'default': 0.1}
    },
    strategy_type='mean_reversion',
    tags=['bollinger_bands', 'rsi', 'divergence', 'relaxed']
)
def bollinger_rsi_divergence_relaxed(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Simplified divergence detection using RSI non-confirmation.
    
    Entry logic:
    - Long: Price below lower band but RSI not oversold (divergence)
    - Short: Price above upper band but RSI not overbought (divergence)
    
    Exit logic:
    - Exit when price returns to middle band area
    """
    # Get parameters
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    rsi_period = params.get('rsi_period', 14)
    rsi_threshold = params.get('rsi_threshold', 10)
    entry_threshold = params.get('entry_threshold', 0.1)
    
    # Get current values
    price = bar.get('close', 0)
    middle_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_middle', price)
    upper_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_upper', price)
    lower_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_lower', price)
    rsi = features.get(f'rsi_{rsi_period}', 50)
    
    # Calculate band position
    band_width = upper_band - lower_band
    position_in_bands = (price - lower_band) / band_width if band_width > 0 else 0.5
    
    # ENTRY CONDITIONS
    
    # Bullish divergence: Price below/near lower band but RSI not oversold
    if position_in_bands < entry_threshold:  # Below or very close to lower band
        # RSI showing divergence (not as oversold as price suggests)
        if rsi > (30 + rsi_threshold):  # e.g., RSI > 40 when price is at extreme
            return {
                'signal_value': 1,
                'metadata': {
                    'signal_type': 'bullish_divergence',
                    'reason': 'price_below_band_rsi_divergence',
                    'price': price,
                    'rsi': rsi,
                    'position_in_bands': position_in_bands,
                    'distance_below_band': (lower_band - price) / price * 100
                }
            }
    
    # Bearish divergence: Price above/near upper band but RSI not overbought
    elif position_in_bands > (1.0 - entry_threshold):  # Above or very close to upper band
        # RSI showing divergence (not as overbought as price suggests)
        if rsi < (70 - rsi_threshold):  # e.g., RSI < 60 when price is at extreme
            return {
                'signal_value': -1,
                'metadata': {
                    'signal_type': 'bearish_divergence',
                    'reason': 'price_above_band_rsi_divergence',
                    'price': price,
                    'rsi': rsi,
                    'position_in_bands': position_in_bands,
                    'distance_above_band': (price - upper_band) / price * 100
                }
            }
    
    # EXIT CONDITIONS
    
    # Exit when price returns to middle band area (middle 40% of bands)
    elif 0.3 < position_in_bands < 0.7:
        return {
            'signal_value': 0,
            'metadata': {
                'signal_type': 'middle_band_exit',
                'reason': 'price_returned_to_middle',
                'position_in_bands': position_in_bands,
                'price': price
            }
        }
    
    return None
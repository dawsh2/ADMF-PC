"""
Bollinger Band + RSI with Volatility Filter

Same as bollinger_rsi_simple_signals but with ATR-based volatility filtering.
Based on analysis showing medium volatility underperforms.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='bollinger_rsi_volatility_filtered',
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
        FeatureSpec('rsi', {'period': params.get('rsi_period', 14)}),
        # ATR for volatility filtering
        FeatureSpec('atr', {'period': params.get('atr_period', 14)})
    ],
    parameter_space={
        'bb_period': {'type': 'int', 'range': (10, 50), 'default': 20},
        'bb_std': {'type': 'float', 'range': (1.5, 3.0), 'default': 2.0},
        'rsi_period': {'type': 'int', 'range': (10, 20), 'default': 14},
        'rsi_threshold': {'type': 'float', 'range': (5, 15), 'default': 10},
        'atr_period': {'type': 'int', 'range': (10, 20), 'default': 14},
        'volatility_filter': {'type': 'str', 'values': ['none', 'exclude_medium', 'high_only'], 'default': 'exclude_medium'},
        'low_vol_threshold': {'type': 'float', 'range': (0.3, 0.7), 'default': 0.5},
        'high_vol_threshold': {'type': 'float', 'range': (0.8, 1.5), 'default': 1.0}
    },
    strategy_type='mean_reversion',
    tags=['bollinger_bands', 'rsi', 'mean_reversion', 'volatility_filter']
)
def bollinger_rsi_volatility_filtered(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Bollinger RSI strategy with volatility filtering.
    
    Analysis showed:
    - Low volatility: 0.014% avg return (decent)
    - Medium volatility: -0.006% avg return (poor)
    - High volatility: 0.044% avg return (best)
    
    Default filter excludes medium volatility periods.
    """
    # Get parameters
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    rsi_period = params.get('rsi_period', 14)
    rsi_threshold = params.get('rsi_threshold', 10)
    atr_period = params.get('atr_period', 14)
    vol_filter = params.get('volatility_filter', 'exclude_medium')
    low_threshold = params.get('low_vol_threshold', 0.5)
    high_threshold = params.get('high_vol_threshold', 1.0)
    
    # Get current values
    price = bar.get('close', 0)
    middle_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_middle', price)
    upper_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_upper', price)
    lower_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_lower', price)
    rsi = features.get(f'rsi_{rsi_period}', 50)
    atr = features.get(f'atr_{atr_period}', 0)
    
    # Calculate volatility regime using ATR as percentage of price
    volatility_pct = (atr / price * 100) if price > 0 else 0
    
    # Determine volatility regime
    if volatility_pct < low_threshold:
        vol_regime = 'low'
    elif volatility_pct > high_threshold:
        vol_regime = 'high'
    else:
        vol_regime = 'medium'
    
    # Apply volatility filter
    if vol_filter == 'exclude_medium' and vol_regime == 'medium':
        # Skip signals during medium volatility
        return None
    elif vol_filter == 'high_only' and vol_regime != 'high':
        # Only trade during high volatility
        return None
    
    # Rest of the logic is identical to simple signals
    band_width = upper_band - lower_band
    position_in_bands = (price - lower_band) / band_width if band_width > 0 else 0.5
    
    # LONG CONDITIONS: Price at lower extreme with RSI divergence
    if position_in_bands < 0:  # Below lower band
        if rsi > (30 + rsi_threshold):  # RSI shows divergence
            return {
                'signal_value': 1,
                'metadata': {
                    'reason': 'below_band_rsi_divergence',
                    'price': price,
                    'rsi': rsi,
                    'volatility': vol_regime,
                    'volatility_pct': volatility_pct,
                    'distance_below_band': (lower_band - price) / price * 100
                }
            }
    
    # SHORT CONDITIONS: Price at upper extreme with RSI divergence
    elif position_in_bands > 1:  # Above upper band
        if rsi < (70 - rsi_threshold):  # RSI shows divergence
            return {
                'signal_value': -1,
                'metadata': {
                    'reason': 'above_band_rsi_divergence',
                    'price': price,
                    'rsi': rsi,
                    'volatility': vol_regime,
                    'volatility_pct': volatility_pct,
                    'distance_above_band': (price - upper_band) / price * 100
                }
            }
    
    # FLAT CONDITIONS: Price in middle zone
    elif 0.4 < position_in_bands < 0.6:  # Middle 20% of bands
        return {
            'signal_value': 0,
            'metadata': {
                'reason': 'middle_band_zone',
                'price': price,
                'position_in_bands': position_in_bands,
                'volatility': vol_regime
            }
        }
    
    # No signal - let position ride
    return None
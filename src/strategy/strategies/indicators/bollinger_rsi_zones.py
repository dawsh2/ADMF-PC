"""
Bollinger Band + RSI Zones Strategy

This version defines clear zones for long, short, and flat positions:
- Long zone: Below lower band with RSI > 30
- Short zone: Above upper band with RSI < 70  
- Flat zone: Inside bands OR at extremes without divergence
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='bollinger_rsi_zones',
    feature_discovery=lambda params: [
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
        'rsi_divergence_long': {'type': 'float', 'range': (25, 40), 'default': 30},
        'rsi_divergence_short': {'type': 'float', 'range': (60, 75), 'default': 70},
        'exit_at_middle': {'type': 'bool', 'default': True}
    },
    strategy_type='mean_reversion',
    tags=['divergence', 'volatility', 'momentum', 'mean_reversion', 'zones']
)
def bollinger_rsi_zones(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Zone-based BB RSI strategy with clear position rules.
    
    Returns:
    - 1: Long zone (below lower band with bullish RSI divergence)
    - -1: Short zone (above upper band with bearish RSI divergence)
    - 0: Flat zone (inside bands or no divergence)
    - None: No change from previous signal
    """
    # Get parameters
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    rsi_divergence_long = params.get('rsi_divergence_long', 30)
    rsi_divergence_short = params.get('rsi_divergence_short', 70)
    exit_at_middle = params.get('exit_at_middle', True)
    
    # Get current values
    price = bar.get('close', 0)
    middle_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_middle', price)
    upper_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_upper', price)
    lower_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_lower', price)
    rsi = features.get(f'rsi_{params.get("rsi_period", 14)}', 50)
    
    # Define zones
    
    # LONG ZONE: Price below lower band with bullish divergence
    if price < lower_band:
        if rsi > rsi_divergence_long:
            # Bullish divergence confirmed - enter long
            return {
                'signal_value': 1,
                'metadata': {
                    'zone': 'long',
                    'reason': 'below_lower_band_with_divergence',
                    'price': price,
                    'rsi': rsi,
                    'lower_band': lower_band,
                    'distance_pct': (lower_band - price) / price * 100
                }
            }
        else:
            # Below band but RSI too oversold - stay flat
            return {
                'signal_value': 0,
                'metadata': {
                    'zone': 'flat',
                    'reason': 'below_band_no_divergence',
                    'price': price,
                    'rsi': rsi
                }
            }
    
    # SHORT ZONE: Price above upper band with bearish divergence
    elif price > upper_band:
        if rsi < rsi_divergence_short:
            # Bearish divergence confirmed - enter short
            return {
                'signal_value': -1,
                'metadata': {
                    'zone': 'short',
                    'reason': 'above_upper_band_with_divergence',
                    'price': price,
                    'rsi': rsi,
                    'upper_band': upper_band,
                    'distance_pct': (price - upper_band) / price * 100
                }
            }
        else:
            # Above band but RSI too overbought - stay flat
            return {
                'signal_value': 0,
                'metadata': {
                    'zone': 'flat',
                    'reason': 'above_band_no_divergence',
                    'price': price,
                    'rsi': rsi
                }
            }
    
    # FLAT ZONE: Inside bands
    else:
        # Define middle band zone for exits
        if exit_at_middle:
            middle_zone_width = (upper_band - lower_band) * 0.1  # 10% of band width
            if abs(price - middle_band) < middle_zone_width:
                # Near middle band - explicit flat signal
                return {
                    'signal_value': 0,
                    'metadata': {
                        'zone': 'flat',
                        'reason': 'near_middle_band',
                        'price': price,
                        'middle_band': middle_band,
                        'distance': abs(price - middle_band)
                    }
                }
        
        # Inside bands but not in middle zone - stay with current position
        # Return None to indicate no change
        return None
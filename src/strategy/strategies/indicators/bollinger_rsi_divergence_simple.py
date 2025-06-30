"""
Simplified Bollinger + RSI Divergence Strategy.

Since feature hub doesn't support cross-feature dependencies well,
this implements divergence detection directly in the strategy.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='bollinger_rsi_simple',
    feature_discovery=lambda params: [
        FeatureSpec('bollinger_bands', {
            'period': params.get('bb_period', 20),
            'std_dev': params.get('bb_std', 2.0)
        }, 'upper'),
        FeatureSpec('bollinger_bands', {
            'period': params.get('bb_period', 20),
            'std_dev': params.get('bb_std', 2.0)
        }, 'middle'),
        FeatureSpec('bollinger_bands', {
            'period': params.get('bb_period', 20),
            'std_dev': params.get('bb_std', 2.0)
        }, 'lower'),
        FeatureSpec('rsi', {
            'period': params.get('rsi_period', 14)
        })
    ],
    parameter_space={
        'bb_period': {'type': 'int', 'range': (20, 20), 'default': 20},
        'bb_std': {'type': 'float', 'range': (2.0, 2.0), 'default': 2.0},
        'rsi_period': {'type': 'int', 'range': (14, 14), 'default': 14},
        'rsi_os_level': {'type': 'float', 'range': (20, 40), 'default': 30},
        'rsi_ob_level': {'type': 'float', 'range': (60, 80), 'default': 70},
        'exit_threshold': {'type': 'float', 'range': (0.0, 0.005), 'default': 0.001}
    },
    strategy_type='mean_reversion',
    tags=['bollinger_bands', 'rsi', 'mean_reversion', 'simple']
)
def bollinger_rsi_simple(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Simplified Bollinger + RSI strategy.
    
    This approximates the divergence strategy by:
    - Only entering longs when price is at lower band AND RSI is NOT oversold
    - Only entering shorts when price is at upper band AND RSI is NOT overbought
    - Exit at middle band
    
    The logic is that if price is at an extreme but RSI isn't, it suggests divergence.
    """
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    rsi_period = params.get('rsi_period', 14)
    rsi_os = params.get('rsi_os_level', 30)
    rsi_ob = params.get('rsi_ob_level', 70)
    exit_threshold = params.get('exit_threshold', 0.001)
    
    # Get features
    upper_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_upper')
    middle_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_middle')
    lower_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_lower')
    rsi = features.get(f'rsi_{rsi_period}')
    
    price = bar.get('close', 0)
    
    if any(v is None for v in [upper_band, lower_band, middle_band, rsi]):
        return None
    
    signal_value = 0
    entry_reason = None
    
    # First check exit condition - within threshold of middle band
    if middle_band and abs(price - middle_band) / middle_band <= exit_threshold:
        signal_value = 0  # Exit any position
        entry_reason = "Exit at middle band"
    else:
        # Check for divergence-like conditions
        # Long: Price at lower band but RSI NOT oversold (suggests bullish divergence)
        if price <= lower_band and rsi > rsi_os:
            signal_value = 1
            entry_reason = f"Price at lower band ({price:.2f}) but RSI not oversold ({rsi:.1f} > {rsi_os})"
        
        # Short: Price at upper band but RSI NOT overbought (suggests bearish divergence)
        elif price >= upper_band and rsi < rsi_ob:
            signal_value = -1
            entry_reason = f"Price at upper band ({price:.2f}) but RSI not overbought ({rsi:.1f} < {rsi_ob})"
    
    # Always return signal
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'bollinger_rsi_simple',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'price': price,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'middle_band': middle_band,
            'rsi': rsi,
            'band_width': upper_band - lower_band,
            'band_position': (price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5,
            'rsi_oversold': rsi < rsi_os,
            'rsi_overbought': rsi > rsi_ob,
            'reason': entry_reason or 'No signal'
        }
    }
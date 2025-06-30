"""
Bollinger Band + RSI Divergence Strategy using dependent features.

This strategy properly uses the feature dependency system where the
bb_rsi_dependent feature declares dependencies on BB and RSI features.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='bollinger_rsi_dependent',
    feature_discovery=lambda params: [
        # The dependent feature that tracks divergences
        FeatureSpec('bb_rsi_dependent', {
            'lookback': params.get('lookback', 20),
            'rsi_divergence_threshold': params.get('rsi_divergence_threshold', 5.0),
            'confirmation_bars': params.get('confirmation_bars', 10),
            'bb_period': params.get('bb_period', 20),
            'bb_std': params.get('bb_std', 2.0),
            'rsi_period': params.get('rsi_period', 14)
        }),
        # Bollinger bands for exit signals
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
        # RSI is computed as a dependency of bb_rsi_dependent
        FeatureSpec('rsi', {'period': params.get('rsi_period', 14)})
    ],
    parameter_space={
        'bb_period': {'type': 'int', 'range': (10, 50), 'default': 20},
        'bb_std': {'type': 'float', 'range': (1.5, 3.0), 'default': 2.0},
        'rsi_period': {'type': 'int', 'range': (10, 20), 'default': 14},
        'lookback': {'type': 'int', 'range': (10, 30), 'default': 20},
        'rsi_divergence_threshold': {'type': 'float', 'range': (3.0, 10.0), 'default': 5.0},
        'confirmation_bars': {'type': 'int', 'range': (5, 20), 'default': 10},
        'exit_at_middle_band': {'type': 'bool', 'default': True}
    },
    strategy_type='mean_reversion',
    tags=['divergence', 'volatility', 'momentum', 'mean_reversion', 'multi_bar_pattern']
)
def bollinger_rsi_dependent(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Uses the bb_rsi_dependent feature which properly depends on BB and RSI.
    
    This is the correct architectural approach - state in features, 
    strategies remain stateless.
    
    Entry: When bb_rsi_dependent confirms a divergence
    Exit: At middle band or opposite band
    """
    # Get parameters
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    exit_at_middle_band = params.get('exit_at_middle_band', True)
    
    # Get divergence signals
    # Build the full feature key with parameters
    lookback = params.get('lookback', 20)
    rsi_divergence_threshold = params.get('rsi_divergence_threshold', 5.0)
    confirmation_bars = params.get('confirmation_bars', 10)
    rsi_period = params.get('rsi_period', 14)
    
    feature_prefix = f'bb_rsi_dependent_{lookback}_{rsi_divergence_threshold}_{confirmation_bars}_{bb_period}_{bb_std}_{rsi_period}'
    
    confirmed_long = features.get(f'{feature_prefix}_confirmed_long', False)
    confirmed_short = features.get(f'{feature_prefix}_confirmed_short', False)
    
    # Get current price and bands for exit logic
    price = bar.get('close', 0)
    middle_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_middle', price)
    upper_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_upper', price)
    lower_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_lower', price)
    
    # Entry signals
    if confirmed_long:
        print(f"[STRATEGY] Generating LONG signal at bar {bar.get('timestamp', 0)}")
        return {
            'signal_value': 1,  # Changed from 'signal' to 'signal_value'
            'metadata': {
                'signal_type': 'bb_rsi_divergence_long',
            'has_divergence': features.get(f'{feature_prefix}_has_bullish_divergence', False),
            'divergence_strength': features.get(f'{feature_prefix}_divergence_strength', 0.0),
            'bars_since_divergence': features.get(f'{feature_prefix}_bars_since_divergence'),
            'entry_price': price,
            'target_price': middle_band if exit_at_middle_band else upper_band
            }
        }
    elif confirmed_short:
        print(f"[STRATEGY] Generating SHORT signal at bar {bar.get('timestamp', 0)}")
        return {
            'signal_value': -1,  # Changed from 'signal' to 'signal_value'
            'metadata': {
                'signal_type': 'bb_rsi_divergence_short',
            'has_divergence': features.get(f'{feature_prefix}_has_bearish_divergence', False),
            'divergence_strength': features.get(f'{feature_prefix}_divergence_strength', 0.0),
            'bars_since_divergence': features.get(f'{feature_prefix}_bars_since_divergence'),
            'entry_price': price,
            'target_price': middle_band if exit_at_middle_band else lower_band
            }
        }
    
    # Exit signals - close positions at middle band or opposite band
    # Note: In a real system, we'd track open positions. Here we use price location as proxy.
    if exit_at_middle_band:
        # Exit longs if price reaches middle band from below
        if price >= middle_band and price < upper_band:
            # Check if we were likely in a long position (price was recently below lower band)
            if features.get(f'{feature_prefix}_bars_since_divergence') and features.get(f'{feature_prefix}_bars_since_divergence') < 20:
                return {
                    'signal_value': 0,  # Changed from 'signal' to 'signal_value'
                    'metadata': {
                        'signal_type': 'exit_at_middle',
                    'exit_price': price
                    }
                }
        # Exit shorts if price reaches middle band from above
        elif price <= middle_band and price > lower_band:
            # Check if we were likely in a short position
            if features.get(f'{feature_prefix}_bars_since_divergence') and features.get(f'{feature_prefix}_bars_since_divergence') < 20:
                return {
                    'signal_value': 0,  # Changed from 'signal' to 'signal_value'
                    'metadata': {
                        'signal_type': 'exit_at_middle',
                    'exit_price': price
                    }
                }
    
    # No signal
    return None
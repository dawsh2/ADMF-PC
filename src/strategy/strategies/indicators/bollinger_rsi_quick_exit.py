"""
Bollinger Band + RSI Divergence Strategy - Quick Exit Version

This version exits positions more quickly by:
1. Exiting at middle band without time restrictions
2. Using tighter stop losses
3. Taking profits earlier
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='bollinger_rsi_quick_exit',
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
        # RSI for momentum confirmation
        FeatureSpec('rsi', {'period': params.get('rsi_period', 14)})
    ],
    parameter_space={
        'bb_period': {'type': 'int', 'range': (10, 50), 'default': 20},
        'bb_std': {'type': 'float', 'range': (1.5, 3.0), 'default': 2.0},
        'rsi_period': {'type': 'int', 'range': (10, 20), 'default': 14},
        'lookback': {'type': 'int', 'range': (10, 30), 'default': 20},
        'rsi_divergence_threshold': {'type': 'float', 'range': (3.0, 10.0), 'default': 5.0},
        'confirmation_bars': {'type': 'int', 'range': (5, 20), 'default': 10},
        'profit_target_pct': {'type': 'float', 'range': (0.001, 0.01), 'default': 0.003},
        'stop_loss_pct': {'type': 'float', 'range': (0.001, 0.01), 'default': 0.005},
        'middle_band_exit_after': {'type': 'int', 'range': (5, 50), 'default': 20}
    },
    strategy_type='mean_reversion',
    tags=['divergence', 'volatility', 'momentum', 'mean_reversion', 'quick_exit']
)
def bollinger_rsi_quick_exit(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Quick exit version of BB RSI divergence.
    
    Entry: When bb_rsi_dependent confirms a divergence
    Exit: At profit target, stop loss, or middle band after minimum bars
    """
    # Get parameters
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    profit_target_pct = params.get('profit_target_pct', 0.003)  # 0.3% default
    stop_loss_pct = params.get('stop_loss_pct', 0.005)  # 0.5% default
    middle_band_exit_after = params.get('middle_band_exit_after', 20)
    
    # Get divergence signals
    lookback = params.get('lookback', 20)
    rsi_divergence_threshold = params.get('rsi_divergence_threshold', 5.0)
    confirmation_bars = params.get('confirmation_bars', 10)
    rsi_period = params.get('rsi_period', 14)
    
    feature_prefix = f'bb_rsi_dependent_{lookback}_{rsi_divergence_threshold}_{confirmation_bars}_{bb_period}_{bb_std}_{rsi_period}'
    
    confirmed_long = features.get(f'{feature_prefix}_confirmed_long', False)
    confirmed_short = features.get(f'{feature_prefix}_confirmed_short', False)
    bars_since_divergence = features.get(f'{feature_prefix}_bars_since_divergence')
    
    # Get current price and bands
    price = bar.get('close', 0)
    middle_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_middle', price)
    upper_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_upper', price)
    lower_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_lower', price)
    rsi = features.get(f'rsi_{rsi_period}', 50)
    
    # Entry signals
    if confirmed_long:
        entry_price = price
        take_profit = entry_price * (1 + profit_target_pct)
        stop_loss = entry_price * (1 - stop_loss_pct)
        
        return {
            'signal_value': 1,
            'metadata': {
                'signal_type': 'bb_rsi_divergence_long',
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'target_middle_band': middle_band
            }
        }
    elif confirmed_short:
        entry_price = price
        take_profit = entry_price * (1 - profit_target_pct)
        stop_loss = entry_price * (1 + stop_loss_pct)
        
        return {
            'signal_value': -1,
            'metadata': {
                'signal_type': 'bb_rsi_divergence_short',
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'target_middle_band': middle_band
            }
        }
    
    # Exit logic based on recent divergence signals
    if bars_since_divergence is not None:
        # We assume we're in a position if there was a recent divergence
        has_bullish = features.get(f'{feature_prefix}_has_bullish_divergence', False)
        has_bearish = features.get(f'{feature_prefix}_has_bearish_divergence', False)
        
        # Exit long positions
        if has_bullish and bars_since_divergence < 500:  # Reasonable window
            # Quick exit conditions for longs
            if (
                # Reached middle band after minimum hold
                (bars_since_divergence >= middle_band_exit_after and price >= middle_band) or
                # RSI shows overbought
                (rsi > 70) or
                # Price reversed below recent low
                (bars_since_divergence > 10 and price < lower_band)
            ):
                return {
                    'signal_value': 0,
                    'metadata': {
                        'signal_type': 'quick_exit',
                        'exit_reason': 'middle_band' if price >= middle_band else 'rsi_overbought' if rsi > 70 else 'reversal',
                        'exit_price': price,
                        'bars_held': bars_since_divergence
                    }
                }
        
        # Exit short positions
        elif has_bearish and bars_since_divergence < 500:
            # Quick exit conditions for shorts
            if (
                # Reached middle band after minimum hold
                (bars_since_divergence >= middle_band_exit_after and price <= middle_band) or
                # RSI shows oversold
                (rsi < 30) or
                # Price reversed above recent high
                (bars_since_divergence > 10 and price > upper_band)
            ):
                return {
                    'signal_value': 0,
                    'metadata': {
                        'signal_type': 'quick_exit',
                        'exit_reason': 'middle_band' if price <= middle_band else 'rsi_oversold' if rsi < 30 else 'reversal',
                        'exit_price': price,
                        'bars_held': bars_since_divergence
                    }
                }
    
    # No signal
    return None
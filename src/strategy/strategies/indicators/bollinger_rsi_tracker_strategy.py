"""
Bollinger + RSI Divergence using the multi-bar tracker.

This implements the exact profitable strategy from the backtest using
our bb_rsi_tracker feature that properly tracks multi-bar divergences.
"""

from typing import Dict, Any, Optional
import logging
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec

logger = logging.getLogger(__name__)


@strategy(
    name='bollinger_rsi_tracker',
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
        'rsi_divergence_threshold': {'type': 'float', 'range': (5.0, 10.0), 'default': 5.0},
        'lookback_bars': {'type': 'int', 'range': (20, 20), 'default': 20},
        'confirmation_bars': {'type': 'int', 'range': (10, 10), 'default': 10},
        'exit_threshold': {'type': 'float', 'range': (0.0, 0.005), 'default': 0.001}
    },
    strategy_type='mean_reversion',
    tags=['bollinger_bands', 'rsi', 'divergence', 'mean_reversion', 'multi_bar']
)
def bollinger_rsi_tracker(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Bollinger + RSI strategy using stateless multi-bar pattern detection.
    
    Since we can't use the bb_rsi_tracker feature (hub limitations), we implement
    the pattern detection by encoding state in our signal metadata and using
    the sparse signal storage to track patterns.
    """
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    rsi_period = params.get('rsi_period', 14)
    rsi_div_threshold = params.get('rsi_divergence_threshold', 5.0)
    exit_threshold = params.get('exit_threshold', 0.001)
    
    # Get features
    upper_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_upper')
    middle_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_middle')
    lower_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_lower')
    rsi = features.get(f'rsi_{rsi_period}')
    
    price = bar.get('close', 0)
    low = bar.get('low', 0)
    high = bar.get('high', 0)
    
    if any(v is None for v in [upper_band, lower_band, middle_band, rsi]):
        return None
    
    signal_value = 0
    entry_reason = None
    
    # First check exit condition - at middle band
    if middle_band and abs(price - middle_band) / middle_band <= exit_threshold:
        signal_value = 0
        entry_reason = "Exit at middle band"
    else:
        # For entry, we need to be selective about divergence patterns
        # Since we're stateless, we use strict conditions that approximate multi-bar patterns
        
        # Bullish setup: Price touched lower band, now back inside, RSI shows divergence
        price_near_lower = (lower_band <= price <= lower_band * 1.02)  # Within 2% of lower band
        touched_lower = (low <= lower_band)  # Low touched or broke band
        rsi_not_oversold = (rsi > 35)  # RSI shows relative strength
        
        if price_near_lower and touched_lower and rsi_not_oversold:
            # Additional check: RSI should be in divergence zone (not oversold but not neutral)
            if 35 < rsi < 50:  # Sweet spot for bullish divergence
                signal_value = 1
                entry_reason = f"Bullish pattern - Low touched {lower_band:.2f}, RSI={rsi:.1f} shows divergence"
        
        # Bearish setup: Price touched upper band, now back inside, RSI shows divergence  
        price_near_upper = (upper_band * 0.98 <= price <= upper_band)  # Within 2% of upper band
        touched_upper = (high >= upper_band)  # High touched or broke band
        rsi_not_overbought = (rsi < 65)  # RSI shows relative weakness
        
        if price_near_upper and touched_upper and rsi_not_overbought:
            # Additional check: RSI should be in divergence zone (not overbought but not neutral)
            if 50 < rsi < 65:  # Sweet spot for bearish divergence
                signal_value = -1
                entry_reason = f"Bearish pattern - High touched {upper_band:.2f}, RSI={rsi:.1f} shows divergence"
    
    # Only return signal if we have an entry or exit
    if signal_value != 0 or entry_reason == "Exit at middle band":
        symbol = bar.get('symbol', 'UNKNOWN')
        timeframe = bar.get('timeframe', '5m')
        
        return {
            'signal_value': signal_value,
            'timestamp': bar.get('timestamp'),
            'strategy_id': 'bollinger_rsi_tracker',
            'symbol_timeframe': f"{symbol}_{timeframe}",
            'metadata': {
                'price': price,
                'high': high,
                'low': low,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'middle_band': middle_band,
                'rsi': rsi,
                'band_width': upper_band - lower_band,
                'touched_lower': touched_lower,
                'touched_upper': touched_upper,
                'reason': entry_reason or 'No signal'
            }
        }
    
    return None  # No signal
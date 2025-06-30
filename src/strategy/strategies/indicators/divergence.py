"""
Divergence-based indicator strategies.

All divergence strategies that detect when price and indicators move in
opposite directions, signaling potential reversals.

These use simplified divergence detection by comparing recent highs/lows
in price vs indicators over a lookback period.
"""

from typing import Dict, Any, Optional
import logging
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec

logger = logging.getLogger(__name__)


def find_recent_high_low(values: list, lookback: int = 20) -> tuple:
    """Find the highest high and lowest low in recent bars."""
    if len(values) < lookback:
        return None, None
    
    recent = values[-lookback:]
    return max(recent), min(recent)


@strategy(
    name='rsi_divergence',
    feature_discovery=lambda params: [
        FeatureSpec('rsi', {'period': params.get('rsi_period', 14)}),
        FeatureSpec('swing_points', {'lookback': params.get('swing_lookback', 5)})
    ],
    parameter_space={
        'rsi_period': {'type': 'int', 'range': (7, 30), 'default': 14},
        'divergence_lookback': {'type': 'int', 'range': (10, 50), 'default': 20},
        'swing_lookback': {'type': 'int', 'range': (3, 10), 'default': 5},
        'rsi_threshold': {'type': 'float', 'range': (20, 80), 'default': 30}
    },
    strategy_type='reversal',
    tags=['divergence', 'rsi', 'reversal']
)
def rsi_divergence(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    RSI divergence strategy using swing points.
    
    Entry signals:
    - 1: Bullish divergence at swing low (price lower, RSI higher)
    - -1: Bearish divergence at swing high (price higher, RSI lower)
    - 0: No divergence
    
    Works best when RSI is in extreme zones.
    """
    rsi_period = params.get('rsi_period', 14)
    rsi_threshold = params.get('rsi_threshold', 30)
    
    # Get current values
    rsi = features.get(f'rsi_{rsi_period}')
    swing_data = features.get(f'swing_points_{params.get("swing_lookback", 5)}')
    
    if rsi is None or not swing_data:
        return None
    
    price = bar.get('close', 0)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    signal_value = 0
    divergence_type = None
    
    # Check for bullish divergence at swing low
    if swing_data.get('is_swing_low') and rsi < rsi_threshold + 20:
        last_swing_low = swing_data.get('last_swing_low')
        
        if last_swing_low and last_swing_low > price:
            # Price made lower low, check if RSI made higher low
            # Since we don't have RSI history, we use current RSI vs threshold
            # as a proxy for divergence
            if rsi > rsi_threshold:
                signal_value = 1
                divergence_type = 'bullish'
    
    # Check for bearish divergence at swing high  
    elif swing_data.get('is_swing_high') and rsi > (100 - rsi_threshold - 20):
        last_swing_high = swing_data.get('last_swing_high')
        
        if last_swing_high and last_swing_high < price:
            # Price made higher high, check if RSI made lower high
            if rsi < (100 - rsi_threshold):
                signal_value = -1
                divergence_type = 'bearish'
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'rsi_divergence',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'rsi': rsi,
            'price': price,
            'divergence_type': divergence_type,
            'is_swing_high': swing_data.get('is_swing_high', False),
            'is_swing_low': swing_data.get('is_swing_low', False),
            'reason': f'{divergence_type} divergence at swing point' if divergence_type else 'No divergence'
        }
    }


@strategy(
    name='macd_histogram_divergence',
    feature_discovery=lambda params: [
        FeatureSpec('macd', {
            'fast_period': params.get('fast_period', 12),
            'slow_period': params.get('slow_period', 26),
            'signal_period': params.get('signal_period', 9)
        }, 'histogram'),
        FeatureSpec('swing_points', {'lookback': params.get('swing_lookback', 5)})
    ],
    parameter_space={
        'fast_period': {'type': 'int', 'range': (8, 15), 'default': 12},
        'slow_period': {'type': 'int', 'range': (20, 30), 'default': 26},
        'signal_period': {'type': 'int', 'range': (5, 15), 'default': 9},
        'swing_lookback': {'type': 'int', 'range': (3, 10), 'default': 5},
        'histogram_threshold': {'type': 'float', 'range': (0.0, 0.01), 'default': 0.001}
    },
    strategy_type='reversal',
    tags=['divergence', 'macd', 'reversal']
)
def macd_histogram_divergence(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    MACD histogram divergence strategy.
    
    Entry signals:
    - 1: Bullish divergence (price lower low, histogram less negative)
    - -1: Bearish divergence (price higher high, histogram less positive)
    - 0: No divergence
    
    MACD histogram divergences often precede trend reversals.
    """
    fast = params.get('fast_period', 12)
    slow = params.get('slow_period', 26)
    signal = params.get('signal_period', 9)
    hist_threshold = params.get('histogram_threshold', 0.001)
    
    # Get current values
    histogram = features.get(f'macd_{fast}_{slow}_{signal}_histogram')
    swing_data = features.get(f'swing_points_{params.get("swing_lookback", 5)}')
    
    if histogram is None or not swing_data:
        return None
    
    price = bar.get('close', 0)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    signal_value = 0
    divergence_type = None
    
    # Bullish divergence at swing low
    if swing_data.get('is_swing_low') and histogram < 0:
        # Histogram is negative but less negative than expected
        if histogram > -hist_threshold:
            signal_value = 1
            divergence_type = 'bullish_histogram'
    
    # Bearish divergence at swing high
    elif swing_data.get('is_swing_high') and histogram > 0:
        # Histogram is positive but less positive than expected
        if histogram < hist_threshold:
            signal_value = -1
            divergence_type = 'bearish_histogram'
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'macd_histogram_divergence',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'histogram': histogram,
            'price': price,
            'divergence_type': divergence_type,
            'is_swing_high': swing_data.get('is_swing_high', False),
            'is_swing_low': swing_data.get('is_swing_low', False),
            'reason': f'{divergence_type} detected' if divergence_type else 'No divergence'
        }
    }


@strategy(
    name='stochastic_divergence',
    feature_discovery=lambda params: [
        FeatureSpec('stochastic', {
            'k_period': params.get('k_period', 14),
            'd_period': params.get('d_period', 3)
        }, 'k'),
        FeatureSpec('swing_points', {'lookback': params.get('swing_lookback', 5)})
    ],
    parameter_space={
        'k_period': {'type': 'int', 'range': (5, 20), 'default': 14},
        'd_period': {'type': 'int', 'range': (3, 10), 'default': 3},
        'swing_lookback': {'type': 'int', 'range': (3, 10), 'default': 5},
        'oversold_level': {'type': 'float', 'range': (10, 30), 'default': 20},
        'overbought_level': {'type': 'float', 'range': (70, 90), 'default': 80}
    },
    strategy_type='reversal',
    tags=['divergence', 'stochastic', 'reversal']
)
def stochastic_divergence(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Stochastic oscillator divergence strategy.
    
    Entry signals:
    - 1: Bullish divergence in oversold zone at swing low
    - -1: Bearish divergence in overbought zone at swing high
    - 0: No divergence or not in extreme zone
    
    Most reliable when divergence occurs in extreme zones.
    """
    k_period = params.get('k_period', 14)
    d_period = params.get('d_period', 3)
    oversold = params.get('oversold_level', 20)
    overbought = params.get('overbought_level', 80)
    
    # Get current values
    stoch_k = features.get(f'stochastic_{k_period}_{d_period}_k')
    swing_data = features.get(f'swing_points_{params.get("swing_lookback", 5)}')
    
    if stoch_k is None or not swing_data:
        return None
    
    price = bar.get('close', 0)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    signal_value = 0
    divergence_type = None
    zone = None
    
    # Bullish divergence at swing low (in oversold zone)
    if swing_data.get('is_swing_low') and stoch_k < oversold + 10:
        # Stochastic showing strength in oversold area
        if stoch_k > oversold:
            signal_value = 1
            divergence_type = 'bullish'
            zone = 'oversold'
    
    # Bearish divergence at swing high (in overbought zone)
    elif swing_data.get('is_swing_high') and stoch_k > overbought - 10:
        # Stochastic showing weakness in overbought area
        if stoch_k < overbought:
            signal_value = -1
            divergence_type = 'bearish'
            zone = 'overbought'
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'stochastic_divergence',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'stoch_k': stoch_k,
            'price': price,
            'divergence_type': divergence_type,
            'zone': zone,
            'in_oversold': stoch_k < oversold,
            'in_overbought': stoch_k > overbought,
            'reason': f'{divergence_type} divergence in {zone} zone' if divergence_type else 'No divergence'
        }
    }


@strategy(
    name='momentum_divergence',
    feature_discovery=lambda params: [
        FeatureSpec('momentum', {'period': params.get('momentum_period', 10)}),
        FeatureSpec('swing_points', {'lookback': params.get('swing_lookback', 5)})
    ],
    parameter_space={
        'momentum_period': {'type': 'int', 'range': (5, 30), 'default': 10},
        'swing_lookback': {'type': 'int', 'range': (3, 10), 'default': 5},
        'momentum_threshold': {'type': 'float', 'range': (0.0, 0.02), 'default': 0.005}
    },
    strategy_type='reversal',
    tags=['divergence', 'momentum', 'reversal']
)
def momentum_divergence(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Momentum (Rate of Change) divergence strategy.
    
    Entry signals:
    - 1: Bullish divergence (price lower low, momentum less negative)
    - -1: Bearish divergence (price higher high, momentum less positive)
    - 0: No divergence
    
    Momentum divergences indicate waning trend strength.
    """
    period = params.get('momentum_period', 10)
    mom_threshold = params.get('momentum_threshold', 0.005)
    
    # Get current values
    momentum = features.get(f'momentum_{period}')
    swing_data = features.get(f'swing_points_{params.get("swing_lookback", 5)}')
    
    if momentum is None or not swing_data:
        return None
    
    price = bar.get('close', 0)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    # Calculate momentum as percentage
    momentum_pct = momentum / price if price > 0 else 0
    
    signal_value = 0
    divergence_type = None
    
    # Bullish divergence at swing low
    if swing_data.get('is_swing_low'):
        # Momentum showing less negative than expected
        if momentum_pct > -mom_threshold and momentum < 0:
            signal_value = 1
            divergence_type = 'bullish_momentum'
    
    # Bearish divergence at swing high
    elif swing_data.get('is_swing_high'):
        # Momentum showing less positive than expected
        if momentum_pct < mom_threshold and momentum > 0:
            signal_value = -1
            divergence_type = 'bearish_momentum'
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'momentum_divergence',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'momentum': momentum,
            'momentum_pct': momentum_pct,
            'price': price,
            'divergence_type': divergence_type,
            'is_swing_high': swing_data.get('is_swing_high', False),
            'is_swing_low': swing_data.get('is_swing_low', False),
            'reason': f'{divergence_type} detected' if divergence_type else 'No divergence'
        }
    }


@strategy(
    name='obv_price_divergence',
    feature_discovery=lambda params: [
        FeatureSpec('obv', {}),
        FeatureSpec('swing_points', {'lookback': params.get('swing_lookback', 5)})
    ],
    parameter_space={
        'obv_sma_period': {'type': 'int', 'range': (10, 50), 'default': 20},
        'swing_lookback': {'type': 'int', 'range': (3, 10), 'default': 5}
    },
    strategy_type='reversal',
    tags=['divergence', 'volume', 'reversal']
)
def obv_price_divergence(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    On-Balance Volume divergence strategy.
    
    Entry signals:
    - 1: Bullish divergence (price at swing low, OBV > OBV_SMA)
    - -1: Bearish divergence (price at swing high, OBV < OBV_SMA)
    - 0: No divergence
    
    OBV divergences reveal hidden accumulation/distribution.
    """
    obv_sma_period = params.get('obv_sma_period', 20)
    
    # Get current values
    obv = features.get('obv')
    swing_data = features.get(f'swing_points_{params.get("swing_lookback", 5)}')
    
    if obv is None or not swing_data:
        return None
    
    price = bar.get('close', 0)
    volume = bar.get('volume', 0)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    
    signal_value = 0
    divergence_type = None
    
    # Use OBV direction for trend (positive = accumulation, negative = distribution)
    obv_trend = 'up' if obv > 0 else 'down'
    
    # Bullish divergence at swing low
    if swing_data.get('is_swing_low') and obv_trend == 'up':
        # Price at low but OBV showing accumulation
        signal_value = 1
        divergence_type = 'bullish_accumulation'
    
    # Bearish divergence at swing high
    elif swing_data.get('is_swing_high') and obv_trend == 'down':
        # Price at high but OBV showing distribution
        signal_value = -1
        divergence_type = 'bearish_distribution'
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'obv_price_divergence',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'obv': obv,
            'obv_trend': obv_trend,
            'price': price,
            'volume': volume,
            'divergence_type': divergence_type,
            'reason': f'{divergence_type} detected' if divergence_type else 'No divergence'
        }
    }
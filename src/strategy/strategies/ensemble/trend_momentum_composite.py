"""
Example Ensemble Strategy: Trend + Momentum Confirmation.

This demonstrates how to combine multiple indicator strategies
to create more sophisticated trading signals.
"""

from typing import Dict, Any, Optional
import logging
from ....core.components.discovery import strategy

# Import indicator strategies to compose
from ..indicators.crossovers import sma_crossover
from ..indicators.oscillators import rsi_bands

logger = logging.getLogger(__name__)


@strategy(
    name='trend_momentum_composite',
    feature_config=['sma', 'rsi']  # Topology builder infers parameters from strategy logic
)
def trend_momentum_composite(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Ensemble strategy combining MA crossover with RSI confirmation.
    
    Logic:
    1. Primary signal: MA crossover (trend direction)
    2. Confirmation: RSI not in extreme zones (avoid overbought/oversold)
    3. Only trade when both conditions align
    
    Args:
        features: Calculated features from FeatureHub
        bar: Current bar data
        params: Strategy parameters
        
    Returns:
        Combined signal or None
    """
    # Get individual indicator signals
    ma_signal = _get_ma_crossover_signal(features, bar, params)
    rsi_filter = _get_rsi_filter(features, bar, params)
    
    # Combine signals with logic
    if ma_signal == 0:
        # No trend signal
        signal_value = 0
    elif rsi_filter == 0:
        # RSI is neutral, allow trend signal
        signal_value = ma_signal
    elif ma_signal == rsi_filter:
        # Trend and momentum agree (rare but strong)
        signal_value = ma_signal
    else:
        # Trend vs momentum disagreement, be cautious
        signal_value = 0
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
            'symbol_timeframe': f"{symbol}_{timeframe}",
            'signal_value': signal_value,
            'timestamp': bar.get('timestamp'),
            'strategy_id': 'trend_momentum_composite',
            'metadata': {
                'components': {
                    'ma_crossover': ma_signal,
                    'rsi_filter': rsi_filter
                },
                'logic': 'trend_with_momentum_filter',
                'price': bar.get('close', 0)
            }
    }


def _get_ma_crossover_signal(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> int:
    """Get the MA crossover signal component."""
    fast_period = params.get('fast_period', 10)
    slow_period = params.get('slow_period', 20)
    
    fast_ma = features.get(f'sma_{fast_period}')
    slow_ma = features.get(f'sma_{slow_period}')
    
    if fast_ma is None or slow_ma is None:
        return 0
    
    if fast_ma > slow_ma:
        return 1
    elif fast_ma < slow_ma:
        return -1
    else:
        return 0


def _get_rsi_filter(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> int:
    """Get RSI filter signal (neutral zone preferred)."""
    rsi_period = params.get('rsi_period', 14)
    upper_threshold = params.get('rsi_upper', 70)
    lower_threshold = params.get('rsi_lower', 30)
    
    rsi = features.get(f'rsi_{rsi_period}') or features.get('rsi')
    
    if rsi is None:
        return 0
    
    if rsi > upper_threshold:
        return -1  # Overbought, prefer short
    elif rsi < lower_threshold:
        return 1   # Oversold, prefer long
    else:
        return 0   # Neutral zone, no preference


# Alternative: Voting-based ensemble strategy
@strategy(
    name='multi_indicator_voting',
    feature_config=['sma', 'rsi', 'bollinger']  # Topology builder infers parameters from strategy logic
)
def multi_indicator_voting(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Voting-based ensemble strategy.
    
    Combines multiple indicators and uses majority vote.
    """
    # Get individual signals
    ma_signal = _get_ma_crossover_signal(features, bar, params)
    rsi_signal = _get_rsi_bands_signal(features, bar, params)
    bb_signal = _get_bollinger_signal(features, bar, params)
    
    # Collect votes
    signals = [ma_signal, rsi_signal, bb_signal]
    
    # Count votes
    long_votes = sum(1 for s in signals if s == 1)
    short_votes = sum(1 for s in signals if s == -1)
    
    # Majority vote
    if long_votes >= 2:
        signal_value = 1
    elif short_votes >= 2:
        signal_value = -1
    else:
        signal_value = 0
    
    # Always return current signal state (sustained while conditions hold)
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '5m')
    
    return {
            'symbol_timeframe': f"{symbol}_{timeframe}",
            'signal_value': signal_value,
            'timestamp': bar.get('timestamp'),
            'strategy_id': 'multi_indicator_voting',
            'metadata': {
                'votes': {
                    'ma_crossover': ma_signal,
                    'rsi_bands': rsi_signal,
                    'bollinger_bands': bb_signal
                },
                'vote_counts': {
                    'long': long_votes,
                    'short': short_votes
                },
                'price': bar.get('close', 0)
            }
    }


def _get_rsi_bands_signal(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> int:
    """Get RSI bands signal."""
    rsi_period = params.get('rsi_period', 14)
    rsi = features.get(f'rsi_{rsi_period}') or features.get('rsi')
    
    if rsi is None:
        return 0
    
    if rsi > 70:
        return -1  # Overbought
    elif rsi < 30:
        return 1   # Oversold
    else:
        return 0


def _get_bollinger_signal(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> int:
    """Get Bollinger Bands signal (mean reversion)."""
    period = params.get('period', 20)
    std_dev = params.get('std_dev', 2.0)
    
    upper_band = features.get(f'bollinger_{period}_{std_dev}_upper')
    lower_band = features.get(f'bollinger_{period}_{std_dev}_lower')
    price = bar.get('close', 0)
    
    if upper_band is None or lower_band is None:
        return 0
    
    if price > upper_band:
        return -1  # Mean reversion short
    elif price < lower_band:
        return 1   # Mean reversion long
    else:
        return 0
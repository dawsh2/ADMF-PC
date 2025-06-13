"""
Simple momentum strategy that generates both entry and exit signals.

This strategy is designed to test signal performance calculations by
generating paired entry/exit signals based on momentum conditions.
"""

from typing import Dict, Any, Optional
import logging
from ...core.components.discovery import strategy

logger = logging.getLogger(__name__)


@strategy(
    name='simple_momentum',
    feature_config={
        'sma': {'params': ['sma_period'], 'defaults': {'sma_period': 20}},
        'rsi': {'params': [], 'default': 14}  # RSI with default period
    },
    validate_features=False  # Disable validation since features are dynamically named
)
def simple_momentum_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Simple momentum strategy that generates entry and exit signals.
    
    Uses a simple holding period approach to ensure we get exit signals.
    
    Args:
        features: Calculated features from FeatureHub
        bar: Current bar data
        params: Strategy parameters
        
    Returns:
        Signal dict or None
    """
    # Get parameters
    sma_period = params.get('sma_period', 20)
    rsi_threshold_long = params.get('rsi_threshold_long', 30)
    rsi_threshold_short = params.get('rsi_threshold_short', 70)
    exit_bars = params.get('exit_bars', 10)  # Exit after N bars
    
    # Get features
    sma = features.get(f'sma_{sma_period}') or features.get('sma_20')
    rsi = features.get('rsi')
    price = bar.get('close', 0)
    symbol = bar.get('symbol', 'UNKNOWN')
    
    # Track position state in features
    position_state = features.get('position_state', 'flat')
    entry_bar = features.get('entry_bar', -999)
    current_bar = features.get('bar_count', 0)
    
    logger.debug(f"Simple momentum: price={price}, sma={sma}, rsi={rsi}, position={position_state}")
    
    # Check if we have required features
    if sma is None or rsi is None:
        logger.debug(f"Missing features for {symbol}: sma={sma}, rsi={rsi}")
        return None
    
    # Generate signal
    signal = None
    
    if position_state == 'flat':
        # Look for entry signals
        # Long signal: Price above SMA and RSI oversold
        if price > sma and rsi < rsi_threshold_long:
            signal = {
                'symbol': symbol,
                'direction': 'long',
                'signal_type': 'entry',
                'strength': min(1.0, (rsi_threshold_long - rsi) / rsi_threshold_long),
                'price': price,
                'reason': f'Momentum long: price > SMA{sma_period} and RSI < {rsi_threshold_long}',
                'indicators': {
                    'price': price,
                    'sma': sma,
                    'rsi': rsi
                }
            }
            logger.info(f"Generated LONG entry signal for {symbol}: price={price}, sma={sma}, rsi={rsi}")
        
        # Short signal: Price below SMA and RSI overbought
        elif price < sma and rsi > rsi_threshold_short:
            signal = {
                'symbol': symbol,
                'direction': 'short',
                'signal_type': 'entry',
                'strength': min(1.0, (rsi - rsi_threshold_short) / (100 - rsi_threshold_short)),
                'price': price,
                'reason': f'Momentum short: price < SMA{sma_period} and RSI > {rsi_threshold_short}',
                'indicators': {
                    'price': price,
                    'sma': sma,
                    'rsi': rsi
                }
            }
            logger.info(f"Generated SHORT entry signal for {symbol}: price={price}, sma={sma}, rsi={rsi}")
    
    else:  # We have a position
        # Exit after holding period or on reverse signal
        bars_held = current_bar - entry_bar
        
        # Check exit conditions
        should_exit = False
        exit_reason = ""
        
        # Time-based exit
        if bars_held >= exit_bars:
            should_exit = True
            exit_reason = f"Exit after {bars_held} bars"
        
        # Condition-based exit for longs
        elif position_state == 'long' and (price < sma or rsi > 80):
            should_exit = True
            exit_reason = f"Exit long: {'Price < SMA' if price < sma else f'RSI overbought ({rsi:.1f})'}"
        
        # Condition-based exit for shorts
        elif position_state == 'short' and (price > sma or rsi < 20):
            should_exit = True
            exit_reason = f"Exit short: {'Price > SMA' if price > sma else f'RSI oversold ({rsi:.1f})'}"
        
        if should_exit:
            signal = {
                'symbol': symbol,
                'direction': position_state,  # Direction of position we're closing
                'signal_type': 'exit',
                'strength': 1.0,
                'price': price,
                'reason': exit_reason,
                'indicators': {
                    'price': price,
                    'sma': sma,
                    'rsi': rsi,
                    'bars_held': bars_held
                }
            }
            logger.info(f"Generated exit signal for {symbol} {position_state} position: {exit_reason}")
    
    return signal
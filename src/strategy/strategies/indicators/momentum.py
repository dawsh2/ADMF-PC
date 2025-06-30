"""
Momentum-based indicator strategies.

All momentum strategies that generate signals based on momentum indicators
like MACD, ROC, ADX, Aroon, and Vortex. These strategies are stateless and
use the FeatureHub for indicator computation.

Follows strategy-interface.md best practices:
- Simplified feature_config list format
- Protocol + composition architecture (no inheritance)
- Stateless pure functions with @strategy decorator
"""

from typing import Dict, Any, Optional
import logging
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec

logger = logging.getLogger(__name__)


@strategy(
    name='macd_crossover_strategy',
    feature_config=['macd'],  # Topology builder infers parameters from strategy logic
    param_feature_mapping={
        'fast_period': 'macd_{fast_period}_{slow_period}_{signal_period}',
        'slow_period': 'macd_{fast_period}_{slow_period}_{signal_period}',
        'signal_period': 'macd_{fast_period}_{slow_period}_{signal_period}'
    },
    parameter_space={
        'fast_period': {'type': 'int', 'range': (5, 50), 'default': 10},
        'min_threshold': {'type': 'float', 'range': (0.0, 0.1), 'default': 0.001},
        'signal_period': {'type': 'int', 'range': (5, 20), 'default': 9},
        'slow_period': {'type': 'int', 'range': (20, 200), 'default': 20}
    }
)
def macd_crossover_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    MACD crossover strategy with signal line.
    
    Entry signals:
    - Long when MACD line crosses above signal line
    - Short when MACD line crosses below signal line
    
    Uses MACD histogram for signal strength measurement.
    """
    # Parameters with defaults
    fast_period = params.get('fast_period', 12)
    slow_period = params.get('slow_period', 26) 
    signal_period = params.get('signal_period', 9)
    min_threshold = params.get('min_threshold', 0.001)
    
    # Get MACD features using standard naming convention
    macd_data = features.get(f'macd_{fast_period}_{slow_period}_{signal_period}')
    
    if not macd_data or not isinstance(macd_data, dict):
        return None
    
    macd_line = macd_data.get('macd')
    signal_line = macd_data.get('signal') 
    histogram = macd_data.get('histogram')
    
    if macd_line is None or signal_line is None or histogram is None:
        return None
    
    # Get bar info
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    current_price = bar.get('close', 0)
    
    # Generate signal based on histogram (MACD - Signal)
    signal_value = 0
    if histogram > min_threshold:
        signal_value = 1  # Bullish crossover
    elif histogram < -min_threshold:
        signal_value = -1  # Bearish crossover
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'macd_crossover',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram,
            'price': current_price,
            'reason': f'MACD histogram: {histogram:.4f}'
        }
    }


@strategy(
    name='momentum_breakout',
    feature_config=['momentum'],  # Rate of change momentum
    param_feature_mapping={
        'momentum_period': 'momentum_{momentum_period}'
    }
)
def momentum_breakout_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Momentum breakout strategy using rate of change.
    
    Entry signals:
    - Long when momentum exceeds positive threshold
    - Short when momentum falls below negative threshold
    
    Measures price momentum over a specified period.
    """
    # Parameters
    momentum_period = params.get('momentum_period', 10)
    breakout_threshold = params.get('breakout_threshold', 0.02)  # 2%
    
    # Get momentum feature
    momentum_value = features.get(f'momentum_{momentum_period}')
    
    if momentum_value is None:
        return None
    
    # Get bar info
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    current_price = bar.get('close', 0)
    
    # Calculate momentum as percentage
    momentum_pct = momentum_value / current_price if current_price > 0 else 0
    
    # Generate breakout signals
    signal_value = 0
    if momentum_pct > breakout_threshold:
        signal_value = 1  # Bullish breakout
    elif momentum_pct < -breakout_threshold:
        signal_value = -1  # Bearish breakout
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'momentum_breakout',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'momentum_value': momentum_value,
            'momentum_pct': momentum_pct,
            'threshold': breakout_threshold,
            'price': current_price,
            'reason': f'Momentum: {momentum_pct:.3f} vs threshold {breakout_threshold:.3f}'
        }
    }


@strategy(
    name='roc_trend',
    feature_discovery=lambda params: [
        FeatureSpec('roc', {'period': params.get('roc_period', 10)})
    ]
)
def roc_trend_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rate of Change trend following strategy.
    
    Entry signals:
    - Long when ROC > positive threshold (uptrend acceleration)
    - Short when ROC < negative threshold (downtrend acceleration)
    
    ROC measures percentage change over specified period.
    """
    # Parameters
    roc_period = params.get('roc_period', 12)
    trend_threshold = params.get('trend_threshold', 1.0)  # 1%
    
    # Get ROC feature
    roc_value = features.get(f'roc_{roc_period}')
    
    if roc_value is None:
        return None
    
    # Get bar info
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    current_price = bar.get('close', 0)
    
    # Generate trend signals
    signal_value = 0
    if roc_value > trend_threshold:
        signal_value = 1  # Strong uptrend
    elif roc_value < -trend_threshold:
        signal_value = -1  # Strong downtrend
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'roc_trend',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'roc_value': roc_value,
            'threshold': trend_threshold,
            'price': current_price,
            'reason': f'ROC: {roc_value:.2f}% vs threshold {trend_threshold:.2f}%'
        }
    }


@strategy(
    name='adx_trend_strength',
    feature_discovery=lambda params: [
        FeatureSpec('adx', {'period': params.get('adx_period', 14)}, 'adx'),
        FeatureSpec('adx', {'period': params.get('adx_period', 14)}, 'di_plus'),
        FeatureSpec('adx', {'period': params.get('adx_period', 14)}, 'di_minus')
    ]
)
def adx_trend_strength_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    ADX trend strength strategy with directional indicators.
    
    Entry signals:
    - Long when ADX > threshold AND DI+ > DI- (strong uptrend)
    - Short when ADX > threshold AND DI- > DI+ (strong downtrend)
    
    ADX measures trend strength, DI+ and DI- measure direction.
    """
    # Parameters
    adx_period = params.get('adx_period', 14)
    trend_strength_threshold = params.get('trend_strength_threshold', 25)
    di_spread_threshold = params.get('di_spread_threshold', 2)  # Minimum DI spread
    
    # Get ADX features
    adx_data = features.get(f'adx_{adx_period}')
    
    if not adx_data or not isinstance(adx_data, dict):
        return None
    
    adx = adx_data.get('adx')
    di_plus = adx_data.get('di_plus')
    di_minus = adx_data.get('di_minus')
    
    if adx is None or di_plus is None or di_minus is None:
        return None
    
    # Get bar info
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    current_price = bar.get('close', 0)
    
    # Calculate directional spread
    di_spread = abs(di_plus - di_minus)
    
    # Generate signals based on trend strength and direction
    signal_value = 0
    if adx > trend_strength_threshold and di_spread > di_spread_threshold:
        if di_plus > di_minus:
            signal_value = 1  # Strong uptrend
        else:
            signal_value = -1  # Strong downtrend
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'adx_trend_strength',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus,
            'di_spread': di_spread,
            'trend_strength_threshold': trend_strength_threshold,
            'price': current_price,
            'reason': f'ADX: {adx:.1f}, DI+: {di_plus:.1f}, DI-: {di_minus:.1f}'
        }
    }


@strategy(
    name='aroon_oscillator',
    feature_discovery=lambda params: [
        FeatureSpec('aroon', {'period': params.get('aroon_period', 25)}, 'up'),
        FeatureSpec('aroon', {'period': params.get('aroon_period', 25)}, 'down')
    ]
)
def aroon_oscillator_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Aroon oscillator strategy for trend identification.
    
    Entry signals:
    - Long when Aroon oscillator > positive threshold (uptrend)
    - Short when Aroon oscillator < negative threshold (downtrend)
    
    Aroon oscillator = Aroon Up - Aroon Down, measures trend direction.
    """
    # Parameters
    aroon_period = params.get('aroon_period', 25)
    oscillator_threshold = params.get('oscillator_threshold', 50)
    
    # Get Aroon features (they are decomposed into separate keys)
    aroon_up = features.get(f'aroon_{aroon_period}_up')
    aroon_down = features.get(f'aroon_{aroon_period}_down')
    aroon_oscillator = features.get(f'aroon_{aroon_period}_oscillator')
    
    if aroon_oscillator is None:
        return None
    
    # Get bar info
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    current_price = bar.get('close', 0)
    
    # Generate signals based on oscillator
    signal_value = 0
    if aroon_oscillator > oscillator_threshold:
        signal_value = 1  # Strong uptrend
    elif aroon_oscillator < -oscillator_threshold:
        signal_value = -1  # Strong downtrend
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'aroon_oscillator',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'aroon_up': aroon_up,
            'aroon_down': aroon_down,
            'aroon_oscillator': aroon_oscillator,
            'threshold': oscillator_threshold,
            'price': current_price,
            'reason': f'Aroon oscillator: {aroon_oscillator:.1f}'
        }
    }


@strategy(
    name='vortex_trend',
    feature_discovery=lambda params: [
        FeatureSpec('vortex', {'period': params.get('vortex_period', 14)}, 'vi_plus'),
        FeatureSpec('vortex', {'period': params.get('vortex_period', 14)}, 'vi_minus')
    ]
)
def vortex_trend_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Vortex indicator trend strategy.
    
    Entry signals:
    - Long when VI+ crosses above VI- (bullish trend change)
    - Short when VI- crosses above VI+ (bearish trend change)
    
    Vortex indicators measure trend reversals and trend strength.
    """
    # Parameters
    vortex_period = params.get('vortex_period', 14)
    crossover_threshold = params.get('crossover_threshold', 0.02)  # Minimum spread for signal
    
    # Get Vortex features (they are decomposed into separate keys)
    vi_plus = features.get(f'vortex_{vortex_period}_vi_plus')
    vi_minus = features.get(f'vortex_{vortex_period}_vi_minus')
    
    if vi_plus is None or vi_minus is None:
        return None
    
    # Get bar info
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    current_price = bar.get('close', 0)
    
    # Calculate vortex spread
    vi_spread = vi_plus - vi_minus
    
    # Generate crossover signals
    signal_value = 0
    if vi_spread > crossover_threshold:
        signal_value = 1  # VI+ above VI- (bullish)
    elif vi_spread < -crossover_threshold:
        signal_value = -1  # VI- above VI+ (bearish)
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'vortex_trend',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'vi_plus': vi_plus,
            'vi_minus': vi_minus,
            'vi_spread': vi_spread,
            'threshold': crossover_threshold,
            'price': current_price,
            'reason': f'VI+: {vi_plus:.3f}, VI-: {vi_minus:.3f}, spread: {vi_spread:.3f}'
        }
    }


@strategy(
    name='elder_ray',
    feature_discovery=lambda params: [
        FeatureSpec('ema', {'period': params.get('period', 13)})
    ]
)
def elder_ray_strategy(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Elder Ray strategy - Bull Power vs Bear Power analysis.
    
    Entry signals:
    - Long when Bull Power > threshold AND Bear Power is rising
    - Short when Bear Power < threshold AND Bull Power is falling
    
    Elder Ray measures buying vs selling pressure relative to EMA:
    - Bull Power = High - EMA (buying strength above trend)
    - Bear Power = Low - EMA (selling pressure below trend)
    """
    # Parameters
    ema_period = params.get('period', 21)
    bull_threshold = params.get('bull_threshold', 0.001)  # Minimum bull power
    bear_threshold = params.get('bear_threshold', -0.001)  # Maximum bear power (negative)
    
    # Get features
    ema_value = features.get(f'ema_{ema_period}')
    
    if ema_value is None:
        return None
    
    # Get bar info
    symbol = bar.get('symbol', 'UNKNOWN')
    timeframe = bar.get('timeframe', '1m')
    high = bar.get('high', 0)
    low = bar.get('low', 0)
    close = bar.get('close', 0)
    
    if high <= 0 or low <= 0 or ema_value <= 0:
        return None
    
    # Calculate Elder Ray components
    bull_power = high - ema_value  # Buying pressure above EMA
    bear_power = low - ema_value   # Selling pressure below EMA
    
    # Convert to percentage for better comparison
    bull_power_pct = bull_power / ema_value
    bear_power_pct = bear_power / ema_value
    
    # Generate signals based on Bull/Bear power
    signal_value = 0
    reason = ""
    
    # Long signal: Strong buying pressure
    if bull_power_pct > bull_threshold and bear_power_pct > bear_threshold:
        signal_value = 1
        reason = f"Bull power dominant: {bull_power_pct:.4f}, Bear power: {bear_power_pct:.4f}"
    
    # Short signal: Strong selling pressure  
    elif bear_power_pct < bear_threshold and bull_power_pct < bull_threshold:
        signal_value = -1
        reason = f"Bear power dominant: {bear_power_pct:.4f}, Bull power: {bull_power_pct:.4f}"
    
    # Neutral: Mixed or weak signals
    else:
        signal_value = 0
        reason = f"Mixed signals - Bull: {bull_power_pct:.4f}, Bear: {bear_power_pct:.4f}"
    
    return {
        'signal_value': signal_value,
        'timestamp': bar.get('timestamp'),
        'strategy_id': 'elder_ray',
        'symbol_timeframe': f"{symbol}_{timeframe}",
        'metadata': {
            'ema_value': ema_value,
            'bull_power': bull_power,
            'bear_power': bear_power,
            'bull_power_pct': bull_power_pct,
            'bear_power_pct': bear_power_pct,
            'bull_threshold': bull_threshold,
            'bear_threshold': bear_threshold,
            'price': close,
            'reason': reason
        }
    }


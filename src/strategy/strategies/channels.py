"""
Channel and band-based trading strategies.

This module implements event-driven versions of channel trading rules,
including Keltner Channel, Donchian Channel, and Bollinger Bands strategies.
All strategies follow the stateless pattern required by the architecture.
"""

from typing import Dict, Any, Optional
from src.core.features.feature_spec import FeatureSpec
import logging
from ...core.components.discovery import strategy

logger = logging.getLogger(__name__)


@strategy(
    name='rule14_keltner_channel',
    feature_discovery=lambda params: [FeatureSpec('keltner', {})]  # Topology builder infers parameters from strategy logic
)
def rule14_keltner_channel(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rule 14: Keltner Channel Strategy.
    
    Generates short signals when price > upper band,
    long signals when price < lower band.
    
    Args:
        features: Calculated features from FeatureHub
        bar: Current bar data
        params: Strategy parameters (period)
        
    Returns:
        Signal dict or None
    """
    period = params.get('period', 20)
    
    # Get features
    upper_band = features.get(f'keltner_{period}_upper')
    lower_band = features.get(f'keltner_{period}_lower')
    middle_band = features.get(f'keltner_{period}_middle')
    price = bar.get('close', 0)
    symbol = bar.get('symbol', 'UNKNOWN')
    
    if upper_band is None or lower_band is None:
        logger.warning("Keltner Channel features not available - requires implementation in features.py")
        return None
    
    # Track previous price position relative to bands
    prev_price = features.get('prev_close')
    
    if prev_price is not None:
        # Long signal: Price crosses below lower band
        if prev_price >= lower_band and price < lower_band:
            return {
                'symbol': symbol,
                'direction': 'long',
                'signal_type': 'entry',
                'strength': min(1.0, (lower_band - price) / lower_band),
                'price': price,
                'reason': f'Price broke below Keltner lower band',
                'indicators': {
                    'price': price,
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'middle_band': middle_band
                }
            }
        # Short signal: Price crosses above upper band
        elif prev_price <= upper_band and price > upper_band:
            return {
                'symbol': symbol,
                'direction': 'short',
                'signal_type': 'entry',
                'strength': min(1.0, (price - upper_band) / upper_band),
                'price': price,
                'reason': f'Price broke above Keltner upper band',
                'indicators': {
                    'price': price,
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'middle_band': middle_band
                }
            }
        # Exit long: Price crosses back above lower band
        elif prev_price < lower_band and price >= lower_band:
            return {
                'symbol': symbol,
                'direction': 'long',
                'signal_type': 'exit',
                'strength': 0.5,
                'price': price,
                'reason': f'Price crossed back above Keltner lower band',
                'indicators': {
                    'price': price,
                    'lower_band': lower_band
                }
            }
        # Exit short: Price crosses back below upper band
        elif prev_price > upper_band and price <= upper_band:
            return {
                'symbol': symbol,
                'direction': 'short',
                'signal_type': 'exit',
                'strength': 0.5,
                'price': price,
                'reason': f'Price crossed back below Keltner upper band',
                'indicators': {
                    'price': price,
                    'upper_band': upper_band
                }
            }
    
    return None


@strategy(
    name='rule15_donchian_channel',
    feature_discovery=lambda params: [FeatureSpec('donchian', {})]  # Topology builder infers parameters from strategy logic
)
def rule15_donchian_channel(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rule 15: Donchian Channel Strategy.
    
    Generates short signals when price > upper band,
    long signals when price < lower band.
    
    Args:
        features: Calculated features from FeatureHub
        bar: Current bar data
        params: Strategy parameters (period)
        
    Returns:
        Signal dict or None
    """
    period = params.get('period', 20)
    
    # Get features
    upper_band = features.get(f'donchian_{period}_upper')
    lower_band = features.get(f'donchian_{period}_lower')
    middle_band = features.get(f'donchian_{period}_middle')
    price = bar.get('close', 0)
    symbol = bar.get('symbol', 'UNKNOWN')
    
    if upper_band is None or lower_band is None:
        logger.warning("Donchian Channel features not available - requires implementation in features.py")
        return None
    
    # Track previous price position relative to bands
    prev_price = features.get('prev_close')
    
    if prev_price is not None:
        # Long signal: Price crosses below lower band
        if prev_price >= lower_band and price < lower_band:
            return {
                'symbol': symbol,
                'direction': 'long',
                'signal_type': 'entry',
                'strength': min(1.0, (lower_band - price) / lower_band),
                'price': price,
                'reason': f'Price broke below Donchian lower band',
                'indicators': {
                    'price': price,
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'middle_band': middle_band
                }
            }
        # Short signal: Price crosses above upper band
        elif prev_price <= upper_band and price > upper_band:
            return {
                'symbol': symbol,
                'direction': 'short',
                'signal_type': 'entry',
                'strength': min(1.0, (price - upper_band) / upper_band),
                'price': price,
                'reason': f'Price broke above Donchian upper band',
                'indicators': {
                    'price': price,
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'middle_band': middle_band
                }
            }
        # Exit long: Price crosses back above lower band
        elif prev_price < lower_band and price >= lower_band:
            return {
                'symbol': symbol,
                'direction': 'long',
                'signal_type': 'exit',
                'strength': 0.5,
                'price': price,
                'reason': f'Price crossed back above Donchian lower band',
                'indicators': {
                    'price': price,
                    'lower_band': lower_band
                }
            }
        # Exit short: Price crosses back below upper band
        elif prev_price > upper_band and price <= upper_band:
            return {
                'symbol': symbol,
                'direction': 'short',
                'signal_type': 'exit',
                'strength': 0.5,
                'price': price,
                'reason': f'Price crossed back below Donchian upper band',
                'indicators': {
                    'price': price,
                    'upper_band': upper_band
                }
            }
    
    return None


@strategy(
    name='rule16_bollinger_bands',
    feature_discovery=lambda params: [FeatureSpec('bollinger', {})]  # Topology builder infers parameters from strategy logic
)
def rule16_bollinger_bands(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rule 16: Bollinger Bands Strategy.
    
    Generates short signals when price > upper band,
    long signals when price < lower band.
    
    Args:
        features: Calculated features from FeatureHub
        bar: Current bar data
        params: Strategy parameters (period, std_dev)
        
    Returns:
        Signal dict or None
    """
    period = params.get('period', 20)
    std_dev = params.get('std_dev', 2.0)
    
    # Get features - bollinger_bands returns dict with 'upper', 'middle', 'lower'
    upper_band = features.get(f'bollinger_{period}_{std_dev}_upper')
    lower_band = features.get(f'bollinger_{period}_{std_dev}_lower')
    middle_band = features.get(f'bollinger_{period}_{std_dev}_middle')
    price = bar.get('close', 0)
    symbol = bar.get('symbol', 'UNKNOWN')
    
    if upper_band is None or lower_band is None:
        return None
    
    # Track previous price position relative to bands
    prev_price = features.get('prev_close')
    
    if prev_price is not None:
        # Long signal: Price crosses below lower band (mean reversion)
        if prev_price >= lower_band and price < lower_band:
            return {
                'symbol': symbol,
                'direction': 'long',
                'signal_type': 'entry',
                'strength': min(1.0, (lower_band - price) / lower_band),
                'price': price,
                'reason': f'Price broke below Bollinger lower band',
                'indicators': {
                    'price': price,
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'middle_band': middle_band,
                    'band_width': upper_band - lower_band
                }
            }
        # Short signal: Price crosses above upper band (mean reversion)
        elif prev_price <= upper_band and price > upper_band:
            return {
                'symbol': symbol,
                'direction': 'short',
                'signal_type': 'entry',
                'strength': min(1.0, (price - upper_band) / upper_band),
                'price': price,
                'reason': f'Price broke above Bollinger upper band',
                'indicators': {
                    'price': price,
                    'upper_band': upper_band,
                    'lower_band': lower_band,
                    'middle_band': middle_band,
                    'band_width': upper_band - lower_band
                }
            }
        # Exit long: Price crosses back above lower band or reaches middle
        elif prev_price < lower_band and price >= lower_band:
            return {
                'symbol': symbol,
                'direction': 'long',
                'signal_type': 'exit',
                'strength': 0.5,
                'price': price,
                'reason': f'Price crossed back above Bollinger lower band',
                'indicators': {
                    'price': price,
                    'lower_band': lower_band,
                    'middle_band': middle_band
                }
            }
        # Exit short: Price crosses back below upper band or reaches middle
        elif prev_price > upper_band and price <= upper_band:
            return {
                'symbol': symbol,
                'direction': 'short',
                'signal_type': 'exit',
                'strength': 0.5,
                'price': price,
                'reason': f'Price crossed back below Bollinger upper band',
                'indicators': {
                    'price': price,
                    'upper_band': upper_band,
                    'middle_band': middle_band
                }
            }
        # Alternative exit at middle band for mean reversion
        elif features.get('position_state') == 'long' and price >= middle_band:
            return {
                'symbol': symbol,
                'direction': 'long',
                'signal_type': 'exit',
                'strength': 0.7,
                'price': price,
                'reason': f'Price reached Bollinger middle band (target)',
                'indicators': {
                    'price': price,
                    'middle_band': middle_band
                }
            }
        elif features.get('position_state') == 'short' and price <= middle_band:
            return {
                'symbol': symbol,
                'direction': 'short',
                'signal_type': 'exit',
                'strength': 0.7,
                'price': price,
                'reason': f'Price reached Bollinger middle band (target)',
                'indicators': {
                    'price': price,
                    'middle_band': middle_band
                }
            }
    
    return None
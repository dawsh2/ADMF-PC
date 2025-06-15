"""
Oscillator-based trading strategies.

This module implements event-driven versions of oscillator trading rules,
including RSI, CCI, and other momentum oscillator strategies.
All strategies follow the stateless pattern required by the architecture.
"""

from typing import Dict, Any, Optional
import logging
from ...core.components.discovery import strategy

logger = logging.getLogger(__name__)


@strategy(
    name='rule10_rsi_threshold',
    feature_config=['rsi']  # Topology builder infers parameters from strategy logic
)
def rule10_rsi_threshold(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rule 10: RSI Single Threshold Strategy.
    
    Generates long signals when RSI < threshold (oversold),
    short signals when RSI > threshold (overbought).
    
    Args:
        features: Calculated features from FeatureHub
        bar: Current bar data
        params: Strategy parameters (rsi_period, threshold)
        
    Returns:
        Signal dict or None
    """
    rsi_period = params.get('rsi_period', 14)
    threshold = params.get('threshold', 50)
    
    # Get features
    rsi = features.get(f'rsi_{rsi_period}') or features.get('rsi')
    price = bar.get('close', 0)
    symbol = bar.get('symbol', 'UNKNOWN')
    
    if rsi is None:
        return None
    
    # Track previous RSI to detect threshold crossings
    prev_rsi = features.get(f'prev_rsi_{rsi_period}') or features.get('prev_rsi')
    
    if prev_rsi is not None:
        # Long signal: RSI crosses below threshold (entering oversold)
        if prev_rsi >= threshold and rsi < threshold:
            return {
                'symbol': symbol,
                'direction': 'long',
                'signal_type': 'entry',
                'strength': min(1.0, (threshold - rsi) / threshold),
                'price': price,
                'reason': f'RSI crossed below {threshold} (oversold)',
                'indicators': {
                    'rsi': rsi,
                    'threshold': threshold,
                    'price': price
                }
            }
        # Short signal: RSI crosses above threshold (entering overbought)
        elif prev_rsi <= threshold and rsi > threshold:
            return {
                'symbol': symbol,
                'direction': 'short',
                'signal_type': 'entry',
                'strength': min(1.0, (rsi - threshold) / (100 - threshold)),
                'price': price,
                'reason': f'RSI crossed above {threshold} (overbought)',
                'indicators': {
                    'rsi': rsi,
                    'threshold': threshold,
                    'price': price
                }
            }
    
    return None


@strategy(
    name='rule11_cci_threshold',
    feature_config=['cci']  # Topology builder infers parameters from strategy logic
)
def rule11_cci_threshold(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rule 11: CCI Single Threshold Strategy.
    
    Generates long signals when CCI < threshold (oversold),
    short signals when CCI > threshold (overbought).
    
    Args:
        features: Calculated features from FeatureHub
        bar: Current bar data
        params: Strategy parameters (cci_period, threshold)
        
    Returns:
        Signal dict or None
    """
    cci_period = params.get('cci_period', 20)
    threshold = params.get('threshold', 0)
    
    # Get features
    cci = features.get(f'cci_{cci_period}') or features.get('cci')
    price = bar.get('close', 0)
    symbol = bar.get('symbol', 'UNKNOWN')
    
    if cci is None:
        return None
    
    # Track previous CCI to detect threshold crossings
    prev_cci = features.get(f'prev_cci_{cci_period}') or features.get('prev_cci')
    
    if prev_cci is not None:
        # Long signal: CCI crosses below threshold
        if prev_cci >= threshold and cci < threshold:
            return {
                'symbol': symbol,
                'direction': 'long',
                'signal_type': 'entry',
                'strength': min(1.0, abs(cci - threshold) / 100),
                'price': price,
                'reason': f'CCI crossed below {threshold}',
                'indicators': {
                    'cci': cci,
                    'threshold': threshold,
                    'price': price
                }
            }
        # Short signal: CCI crosses above threshold
        elif prev_cci <= threshold and cci > threshold:
            return {
                'symbol': symbol,
                'direction': 'short',
                'signal_type': 'entry',
                'strength': min(1.0, abs(cci - threshold) / 100),
                'price': price,
                'reason': f'CCI crossed above {threshold}',
                'indicators': {
                    'cci': cci,
                    'threshold': threshold,
                    'price': price
                }
            }
    
    return None


@strategy(
    name='rule12_rsi_bands',
    feature_config=['rsi']  # Topology builder infers parameters from strategy logic
)
def rule12_rsi_bands(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rule 12: RSI Band Strategy (Overbought/Oversold).
    
    Generates long signals when RSI < lower_threshold (oversold),
    short signals when RSI > upper_threshold (overbought).
    
    Args:
        features: Calculated features from FeatureHub
        bar: Current bar data
        params: Strategy parameters (rsi_period, upper_threshold, lower_threshold)
        
    Returns:
        Signal dict or None
    """
    rsi_period = params.get('rsi_period', 14)
    upper_threshold = params.get('upper_threshold', 70)
    lower_threshold = params.get('lower_threshold', 30)
    
    # Get features
    rsi = features.get(f'rsi_{rsi_period}') or features.get('rsi')
    price = bar.get('close', 0)
    symbol = bar.get('symbol', 'UNKNOWN')
    
    if rsi is None:
        return None
    
    # Track previous RSI to detect band crossings
    prev_rsi = features.get(f'prev_rsi_{rsi_period}') or features.get('prev_rsi')
    
    if prev_rsi is not None:
        # Long signal: RSI enters oversold region
        if prev_rsi >= lower_threshold and rsi < lower_threshold:
            return {
                'symbol': symbol,
                'direction': 'long',
                'signal_type': 'entry',
                'strength': min(1.0, (lower_threshold - rsi) / lower_threshold),
                'price': price,
                'reason': f'RSI entered oversold region (< {lower_threshold})',
                'indicators': {
                    'rsi': rsi,
                    'upper_threshold': upper_threshold,
                    'lower_threshold': lower_threshold,
                    'price': price
                }
            }
        # Short signal: RSI enters overbought region
        elif prev_rsi <= upper_threshold and rsi > upper_threshold:
            return {
                'symbol': symbol,
                'direction': 'short',
                'signal_type': 'entry',
                'strength': min(1.0, (rsi - upper_threshold) / (100 - upper_threshold)),
                'price': price,
                'reason': f'RSI entered overbought region (> {upper_threshold})',
                'indicators': {
                    'rsi': rsi,
                    'upper_threshold': upper_threshold,
                    'lower_threshold': lower_threshold,
                    'price': price
                }
            }
        # Exit long: RSI exits oversold region
        elif prev_rsi < lower_threshold and rsi >= lower_threshold:
            return {
                'symbol': symbol,
                'direction': 'long',
                'signal_type': 'exit',
                'strength': 0.5,
                'price': price,
                'reason': f'RSI exited oversold region',
                'indicators': {
                    'rsi': rsi,
                    'price': price
                }
            }
        # Exit short: RSI exits overbought region
        elif prev_rsi > upper_threshold and rsi <= upper_threshold:
            return {
                'symbol': symbol,
                'direction': 'short',
                'signal_type': 'exit',
                'strength': 0.5,
                'price': price,
                'reason': f'RSI exited overbought region',
                'indicators': {
                    'rsi': rsi,
                    'price': price
                }
            }
    
    return None


@strategy(
    name='rule13_cci_bands',
    feature_config=['cci']  # Topology builder infers parameters from strategy logic
)
def rule13_cci_bands(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Rule 13: CCI Band Strategy.
    
    Generates long signals when CCI < lower_threshold,
    short signals when CCI > upper_threshold.
    
    Args:
        features: Calculated features from FeatureHub
        bar: Current bar data
        params: Strategy parameters (cci_period, upper_threshold, lower_threshold)
        
    Returns:
        Signal dict or None
    """
    cci_period = params.get('cci_period', 20)
    upper_threshold = params.get('upper_threshold', 100)
    lower_threshold = params.get('lower_threshold', -100)
    
    # Get features
    cci = features.get(f'cci_{cci_period}') or features.get('cci')
    price = bar.get('close', 0)
    symbol = bar.get('symbol', 'UNKNOWN')
    
    if cci is None:
        return None
    
    # Track previous CCI to detect band crossings
    prev_cci = features.get(f'prev_cci_{cci_period}') or features.get('prev_cci')
    
    if prev_cci is not None:
        # Long signal: CCI enters oversold region
        if prev_cci >= lower_threshold and cci < lower_threshold:
            return {
                'symbol': symbol,
                'direction': 'long',
                'signal_type': 'entry',
                'strength': min(1.0, abs(cci - lower_threshold) / 100),
                'price': price,
                'reason': f'CCI entered oversold region (< {lower_threshold})',
                'indicators': {
                    'cci': cci,
                    'upper_threshold': upper_threshold,
                    'lower_threshold': lower_threshold,
                    'price': price
                }
            }
        # Short signal: CCI enters overbought region
        elif prev_cci <= upper_threshold and cci > upper_threshold:
            return {
                'symbol': symbol,
                'direction': 'short',
                'signal_type': 'entry',
                'strength': min(1.0, abs(cci - upper_threshold) / 100),
                'price': price,
                'reason': f'CCI entered overbought region (> {upper_threshold})',
                'indicators': {
                    'cci': cci,
                    'upper_threshold': upper_threshold,
                    'lower_threshold': lower_threshold,
                    'price': price
                }
            }
        # Exit long: CCI exits oversold region
        elif prev_cci < lower_threshold and cci >= lower_threshold:
            return {
                'symbol': symbol,
                'direction': 'long',
                'signal_type': 'exit',
                'strength': 0.5,
                'price': price,
                'reason': f'CCI exited oversold region',
                'indicators': {
                    'cci': cci,
                    'price': price
                }
            }
        # Exit short: CCI exits overbought region
        elif prev_cci > upper_threshold and cci <= upper_threshold:
            return {
                'symbol': symbol,
                'direction': 'short',
                'signal_type': 'exit',
                'strength': 0.5,
                'price': price,
                'reason': f'CCI exited overbought region',
                'indicators': {
                    'cci': cci,
                    'price': price
                }
            }
    
    return None
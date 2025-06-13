"""
Feature calculation and management for ADMF-PC.

This module contains both:
1. Stateless feature calculation functions (Tier 1) - pure functions for parallelization
2. FeatureHub stateful component (Tier 2) - manages incremental feature computation

Following Protocol + Composition architecture:
- Pure functions for batch computation
- Stateful hub for streaming/incremental updates
- No inheritance, just composition
- Maximum parallelization potential for stateless functions

Now includes @feature decorators for automatic discovery.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, Set, List, Deque
from datetime import datetime
from collections import deque, defaultdict
import logging

from ...core.components.discovery import feature

logger = logging.getLogger(__name__)


@feature(
    name='sma',
    params=['period'],
    min_history='period'
)
def sma_feature(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average feature.
    
    Pure function - no state, just computation.
    
    Args:
        prices: Price series (typically close prices)
        period: Number of periods for average
        
    Returns:
        Series with SMA values
    """
    return prices.rolling(window=period, min_periods=period).mean()


@feature(
    name='ema',
    params=['period', 'smoothing'],
    min_history='period'
)
def ema_feature(prices: pd.Series, period: int, smoothing: float = 2.0) -> pd.Series:
    """
    Calculate Exponential Moving Average feature.
    
    Pure function using pandas built-in EMA calculation.
    
    Args:
        prices: Price series
        period: Number of periods for average
        smoothing: Smoothing factor (typically 2)
        
    Returns:
        Series with EMA values
    """
    alpha = smoothing / (period + 1)
    return prices.ewm(alpha=alpha, adjust=False).mean()


@feature(
    name='rsi',
    params=['period'],
    min_history='period + 1'  # Need one extra for diff
)
def rsi_feature(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index feature.
    
    Pure function - RSI calculation using pandas operations.
    
    Args:
        prices: Price series
        period: Number of periods for RSI calculation (default 14)
        
    Returns:
        Series with RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


@feature(
    name='macd',
    params=['fast', 'slow', 'signal'],
    min_history='slow + signal',
    dependencies=['ema']
)
def macd_feature(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    Calculate MACD feature components.
    
    Pure function returning all MACD components.
    
    Args:
        prices: Price series
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
        
    Returns:
        Dict with 'macd', 'signal', and 'histogram' series
    """
    ema_fast = ema_feature(prices, fast)
    ema_slow = ema_feature(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema_feature(macd_line, signal)
    histogram = macd_line - signal_line
    
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }


@feature(
    name='bollinger_bands',
    params=['period', 'std_dev'],
    min_history='period',
    dependencies=['sma']
)
def bollinger_bands_feature(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands feature components.
    
    Pure function returning all Bollinger Band components.
    
    Args:
        prices: Price series
        period: Period for moving average and standard deviation
        std_dev: Number of standard deviations for bands
        
    Returns:
        Dict with 'middle', 'upper', and 'lower' band series
    """
    middle = sma_feature(prices, period)
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return {
        "middle": middle,
        "upper": upper,
        "lower": lower
    }


@feature(
    name='atr',
    params=['period'],
    min_history='period + 1',  # Need one extra for shift
    input_type='ohlc'  # Indicates it needs OHLC data, not just close
)
def atr_feature(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range feature.
    
    Pure function measuring volatility using true range.
    
    Args:
        high: High price series
        low: Low price series  
        close: Close price series
        period: Number of periods for ATR calculation
        
    Returns:
        Series with ATR values
    """
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


@feature(
    name='stochastic',
    params=['k_period', 'd_period'],
    min_history='k_period + d_period',
    input_type='ohlc'
)
def stochastic_feature(high: pd.Series, low: pd.Series, close: pd.Series, 
                      k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """
    Calculate Stochastic Oscillator feature.
    
    Pure function for momentum indicator.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: Period for %K calculation
        d_period: Period for %D smoothing
        
    Returns:
        Dict with 'k' and 'd' series
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return {
        "k": k_percent,
        "d": d_percent
    }


def williams_r_feature(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Williams %R feature.
    
    Pure function for momentum indicator.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Lookback period
        
    Returns:
        Series with Williams %R values (-100 to 0)
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams_r


@feature(
    name='cci',
    params=['period'],
    min_history='period',
    input_type='ohlc'
)
def cci_feature(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Commodity Channel Index feature.
    
    Pure function for momentum indicator.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Period for CCI calculation
        
    Returns:
        Series with CCI values
    """
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
    return cci


def adx_feature(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
    """
    Calculate Average Directional Index feature components.
    
    Pure function for trend strength indicator.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Period for ADX calculation
        
    Returns:
        Dict with 'adx', 'di_plus', and 'di_minus' series
    """
    # Calculate True Range
    atr = atr_feature(high, low, close, period)
    
    # Calculate Directional Movement
    dm_plus = (high.diff().where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0))
    dm_minus = (low.diff().abs().where((low.diff().abs() > high.diff()) & (low.diff() < 0), 0))
    
    # Smooth Directional Movement
    dm_plus_smooth = dm_plus.rolling(window=period).mean()
    dm_minus_smooth = dm_minus.rolling(window=period).mean()
    
    # Calculate Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # Calculate ADX
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()
    
    return {
        "adx": adx,
        "di_plus": di_plus,
        "di_minus": di_minus
    }


def volume_features(volume: pd.Series, close: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
    """
    Calculate volume-based features.
    
    Pure function for volume analysis.
    
    Args:
        volume: Volume series
        close: Close price series
        period: Period for calculations
        
    Returns:
        Dict with volume-based features
    """
    # Volume moving average
    volume_ma = volume.rolling(window=period).mean()
    
    # Volume ratio (current vs average)
    volume_ratio = volume / volume_ma
    
    # On Balance Volume
    price_change = close.diff()
    obv_direction = np.where(price_change > 0, volume, 
                   np.where(price_change < 0, -volume, 0))
    obv = pd.Series(obv_direction, index=volume.index).cumsum()
    
    # Volume Price Trend
    vpt = ((close.diff() / close.shift(1)) * volume).cumsum()
    
    return {
        "volume_ma": volume_ma,
        "volume_ratio": volume_ratio,
        "obv": obv,
        "vpt": vpt
    }


@feature(
    name='momentum',
    params=['period'],
    min_history='period + 1'
)
def momentum_feature(prices: pd.Series, period: int = 10) -> pd.Series:
    """
    Calculate momentum (rate of change) feature.
    
    Pure function for momentum analysis.
    
    Args:
        prices: Price series
        period: Lookback period for momentum
        
    Returns:
        Series with momentum values (percentage change)
    """
    # Price momentum (rate of change)
    momentum = (prices / prices.shift(period) - 1) * 100
    return momentum


@feature(
    name='dema',
    params=['period'],
    min_history='period * 2'  # Need extra for double smoothing
)
def dema_feature(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Double Exponential Moving Average (DEMA) feature.
    
    DEMA = 2 * EMA - EMA(EMA)
    
    Args:
        prices: Price series
        period: Period for DEMA calculation
        
    Returns:
        Series with DEMA values
    """
    ema1 = ema_feature(prices, period)
    ema2 = ema_feature(ema1, period)
    dema = 2 * ema1 - ema2
    return dema


@feature(
    name='tema',
    params=['period'],
    min_history='period * 3'  # Need extra for triple smoothing
)
def tema_feature(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Triple Exponential Moving Average (TEMA) feature.
    
    TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
    
    Args:
        prices: Price series
        period: Period for TEMA calculation
        
    Returns:
        Series with TEMA values
    """
    ema1 = ema_feature(prices, period)
    ema2 = ema_feature(ema1, period)
    ema3 = ema_feature(ema2, period)
    tema = 3 * ema1 - 3 * ema2 + ema3
    return tema


@feature(
    name='vortex',
    params=['period'],
    min_history='period + 1',
    input_type='ohlc'
)
def vortex_feature(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
    """
    Calculate Vortex Indicator feature.
    
    Measures the relationship between closing prices and true price range.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Period for Vortex calculation
        
    Returns:
        Dict with 'vi_plus' and 'vi_minus' series
    """
    # Calculate Vortex Movement
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()
    
    # Calculate True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    # Calculate Vortex Indicators
    vi_plus = vm_plus.rolling(period).sum() / tr.rolling(period).sum()
    vi_minus = vm_minus.rolling(period).sum() / tr.rolling(period).sum()
    
    return {
        "vi_plus": vi_plus,
        "vi_minus": vi_minus
    }


@feature(
    name='ichimoku',
    params=['conversion_period', 'base_period', 'lead_span_b_period'],
    min_history='lead_span_b_period',
    input_type='ohlc'
)
def ichimoku_feature(high: pd.Series, low: pd.Series, close: pd.Series,
                    conversion_period: int = 9, base_period: int = 26, 
                    lead_span_b_period: int = 52) -> Dict[str, pd.Series]:
    """
    Calculate Ichimoku Cloud components.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        conversion_period: Period for Tenkan-sen (default 9)
        base_period: Period for Kijun-sen (default 26)
        lead_span_b_period: Period for Senkou Span B (default 52)
        
    Returns:
        Dict with Ichimoku components
    """
    # Tenkan-sen (Conversion Line)
    high_conversion = high.rolling(conversion_period).max()
    low_conversion = low.rolling(conversion_period).min()
    tenkan_sen = (high_conversion + low_conversion) / 2
    
    # Kijun-sen (Base Line)
    high_base = high.rolling(base_period).max()
    low_base = low.rolling(base_period).min()
    kijun_sen = (high_base + low_base) / 2
    
    # Senkou Span A (Leading Span A) - plotted 26 periods ahead
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B) - plotted 26 periods ahead
    high_lead = high.rolling(lead_span_b_period).max()
    low_lead = low.rolling(lead_span_b_period).min()
    senkou_span_b = (high_lead + low_lead) / 2
    
    # Chikou Span (Lagging Span) - close plotted 26 periods behind
    chikou_span = close.shift(-base_period)
    
    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_span_a": senkou_span_a,
        "senkou_span_b": senkou_span_b,
        "chikou_span": chikou_span
    }


@feature(
    name='keltner',
    params=['period', 'multiplier'],
    min_history='period',
    input_type='ohlc'
)
def keltner_channel_feature(high: pd.Series, low: pd.Series, close: pd.Series, 
                           period: int = 20, multiplier: float = 2.0) -> Dict[str, pd.Series]:
    """
    Calculate Keltner Channel feature.
    
    Uses EMA and ATR to create volatility-based bands.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Period for calculations
        multiplier: ATR multiplier for bands
        
    Returns:
        Dict with 'upper', 'middle', and 'lower' channel series
    """
    # Middle line is EMA of close
    middle = ema_feature(close, period)
    
    # Calculate ATR
    atr = atr_feature(high, low, close, period)
    
    # Calculate bands
    upper = middle + (multiplier * atr)
    lower = middle - (multiplier * atr)
    
    return {
        "upper": upper,
        "middle": middle,
        "lower": lower
    }


@feature(
    name='donchian',
    params=['period'],
    min_history='period',
    input_type='ohlc'
)
def donchian_channel_feature(high: pd.Series, low: pd.Series, close: pd.Series, 
                            period: int = 20) -> Dict[str, pd.Series]:
    """
    Calculate Donchian Channel feature.
    
    Channel based on highest high and lowest low over period.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series (not used but kept for consistency)
        period: Lookback period
        
    Returns:
        Dict with 'upper', 'middle', and 'lower' channel series
    """
    # Upper band: highest high
    upper = high.rolling(period).max()
    
    # Lower band: lowest low
    lower = low.rolling(period).min()
    
    # Middle band: average of upper and lower
    middle = (upper + lower) / 2
    
    return {
        "upper": upper,
        "middle": middle,
        "lower": lower
    }


@feature(
    name='volatility',
    params=['period'],
    min_history='period + 1'  # Need extra for pct_change
)
def volatility_feature(prices: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate price volatility feature (standard deviation of returns).
    
    Pure function for volatility measurement.
    
    Args:
        prices: Price series
        period: Period for volatility calculation
        
    Returns:
        Series with volatility values
    """
    if isinstance(prices, pd.DataFrame):
        prices = prices['close']
    returns = prices.pct_change()
    volatility = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
    return volatility


@feature(
    name='atr_sma',
    params=['atr_period', 'sma_period'],
    min_history='atr_period + sma_period',
    input_type='ohlc'
)
def atr_sma_feature(high: pd.Series, low: pd.Series, close: pd.Series, 
                    atr_period: int = 14, sma_period: int = 20) -> pd.Series:
    """
    Calculate SMA of ATR for volatility smoothing.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        atr_period: Period for ATR calculation
        sma_period: Period for smoothing
        
    Returns:
        Series with smoothed ATR values
    """
    atr = atr_feature(high, low, close, atr_period)
    return atr.rolling(window=sma_period).mean()


@feature(
    name='volatility_sma',
    params=['vol_period', 'sma_period'],
    min_history='vol_period + sma_period + 1'
)
def volatility_sma_feature(prices: pd.Series, vol_period: int = 20, sma_period: int = 20) -> pd.Series:
    """
    Calculate SMA of volatility for smoothing.
    
    Args:
        prices: Price series
        vol_period: Period for volatility calculation
        sma_period: Period for smoothing
        
    Returns:
        Series with smoothed volatility values
    """
    volatility = volatility_feature(prices, vol_period)
    return volatility.rolling(window=sma_period).mean()


def momentum_features(prices: pd.Series, periods: list = [5, 10, 20]) -> Dict[str, pd.Series]:
    """
    Calculate momentum features for multiple periods.
    
    Pure function for momentum analysis.
    
    Args:
        prices: Price series
        periods: List of periods to calculate momentum
        
    Returns:
        Dict with momentum features for each period
    """
    features = {}
    
    for period in periods:
        # Price momentum (rate of change)
        momentum = (prices / prices.shift(period) - 1) * 100
        features[f"momentum_{period}"] = momentum
        
        # Price relative position in range
        highest = prices.rolling(window=period).max()
        lowest = prices.rolling(window=period).min()
        position = (prices - lowest) / (highest - lowest) * 100
        features[f"position_{period}"] = position
    
    return features


def price_action_features(high: pd.Series, low: pd.Series, close: pd.Series, 
                         open_: Optional[pd.Series] = None) -> Dict[str, pd.Series]:
    """
    Calculate price action features.
    
    Pure function for candlestick and price action analysis.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        open_: Open price series (optional)
        
    Returns:
        Dict with price action features
    """
    features = {}
    
    # Basic price statistics
    features["high_low_ratio"] = high / low
    features["close_position"] = (close - low) / (high - low)
    
    # Price ranges
    features["true_range"] = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    if open_ is not None:
        # Candlestick features
        features["body_size"] = (close - open_).abs()
        features["upper_shadow"] = high - pd.concat([close, open_], axis=1).max(axis=1)
        features["lower_shadow"] = pd.concat([close, open_], axis=1).min(axis=1) - low
        features["body_direction"] = np.where(close > open_, 1, -1)
    
    return features


# Feature registry for easy discovery and dynamic creation
FEATURE_REGISTRY = {
    "sma": sma_feature,
    "ema": ema_feature, 
    "rsi": rsi_feature,
    "macd": macd_feature,
    "bollinger": bollinger_bands_feature,
    "atr": atr_feature,
    "stochastic": stochastic_feature,
    "williams_r": williams_r_feature,
    "cci": cci_feature,
    "adx": adx_feature,
    "volume": volume_features,
    "momentum": momentum_feature,  # Changed to single-period version
    "volatility": volatility_feature,  # Added volatility feature
    "atr_sma": atr_sma_feature,  # Added ATR smoothing
    "volatility_sma": volatility_sma_feature,  # Added volatility smoothing
    "price_action": price_action_features
}


def compute_feature(feature_name: str, data: Union[pd.DataFrame, pd.Series], **kwargs) -> Union[pd.Series, Dict[str, pd.Series]]:
    """
    Compute a single feature by name.
    
    Pure function dispatcher for feature calculation.
    
    Args:
        feature_name: Name of feature to compute
        data: Price data (DataFrame with OHLCV or Series)
        **kwargs: Additional parameters for feature calculation
        
    Returns:
        Feature values as Series or Dict of Series
    """
    if feature_name not in FEATURE_REGISTRY:
        raise ValueError(f"Unknown feature: {feature_name}")
    
    feature_func = FEATURE_REGISTRY[feature_name]
    
    # Handle different data input types
    if isinstance(data, pd.DataFrame):
        # Extract common price series from DataFrame
        if feature_name in ["sma", "ema", "rsi"]:
            return feature_func(data["close"], **kwargs)
        elif feature_name == "macd":
            return feature_func(data["close"], **kwargs)
        elif feature_name == "bollinger":
            return feature_func(data["close"], **kwargs)
        elif feature_name in ["atr", "stochastic", "adx", "atr_sma"]:
            return feature_func(data["high"], data["low"], data["close"], **kwargs)
        elif feature_name == "williams_r":
            return feature_func(data["high"], data["low"], data["close"], **kwargs)
        elif feature_name == "cci":
            return feature_func(data["high"], data["low"], data["close"], **kwargs)
        elif feature_name == "volume":
            return feature_func(data["volume"], data["close"], **kwargs)
        elif feature_name in ["momentum", "volatility", "volatility_sma"]:
            return feature_func(data["close"], **kwargs)
        elif feature_name == "price_action":
            open_ = data.get("open")
            return feature_func(data["high"], data["low"], data["close"], open_, **kwargs)
    else:
        # Single series input
        if feature_name in ["sma", "ema", "rsi", "momentum"]:
            return feature_func(data, **kwargs)
        else:
            raise ValueError(f"Feature {feature_name} requires OHLC data, not single series")
    
    return feature_func(data, **kwargs)


def compute_multiple_features(feature_configs: Dict[str, Dict[str, Any]], 
                            data: pd.DataFrame) -> Dict[str, Union[pd.Series, Dict[str, pd.Series]]]:
    """
    Compute multiple features in one call.
    
    Pure function for batch feature computation.
    
    Args:
        feature_configs: Dict mapping feature names to their configurations
        data: OHLCV DataFrame
        
    Returns:
        Dict mapping feature names to their computed values
    
    Example:
        configs = {
            "sma_20": {"feature": "sma", "period": 20},
            "rsi": {"feature": "rsi", "period": 14},
            "macd": {"feature": "macd", "fast": 12, "slow": 26, "signal": 9}
        }
        features = compute_multiple_features(configs, data)
    """
    results = {}
    
    for name, config in feature_configs.items():
        feature_type = config.pop("feature")
        results[name] = compute_feature(feature_type, data, **config)
    
    return results


# ============================================================================
# Stateful Feature Management - FeatureHub
# ============================================================================

class FeatureHub:
    """
    Centralized stateful feature computation engine.
    
    This Tier 2 component manages ALL incremental feature calculation,
    allowing strategies to remain completely stateless.
    
    Key responsibilities:
    - Maintain rolling windows for all features
    - Provide incremental updates for streaming data
    - Cache computed features for efficiency
    - Manage dependencies between features
    """
    
    def __init__(self, symbols: Optional[List[str]] = None):
        """
        Initialize FeatureHub.
        
        Args:
            symbols: List of symbols to track (optional)
        """
        self.symbols = symbols or []
        
        # Stateful data storage - rolling windows per symbol
        self.price_data: Dict[str, Dict[str, Deque[float]]] = defaultdict(lambda: {
            'open': deque(maxlen=1000),
            'high': deque(maxlen=1000), 
            'low': deque(maxlen=1000),
            'close': deque(maxlen=1000),
            'volume': deque(maxlen=1000)
        })
        
        # Feature cache - computed feature values per symbol
        self.feature_cache: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Feature configurations - what features to compute
        self.feature_configs: Dict[str, Dict[str, Any]] = {}
        
        # State tracking
        self.bar_count: Dict[str, int] = defaultdict(int)
        
        logger.info("FeatureHub initialized for symbols: %s", self.symbols)
    
    def configure_features(self, feature_configs: Dict[str, Dict[str, Any]]) -> None:
        """
        Configure which features to compute.
        
        Args:
            feature_configs: Dict mapping feature names to their configurations
            
        Example:
            {
                "sma_20": {"feature": "sma", "period": 20},
                "rsi": {"feature": "rsi", "period": 14},
                "bollinger": {"feature": "bollinger", "period": 20, "std_dev": 2.0}
            }
        """
        self.feature_configs = feature_configs
        logger.info("FeatureHub configured with features: %s", list(feature_configs.keys()))
    
    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to track."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.info("Added symbol to FeatureHub: %s", symbol)
    
    def update_bar(self, symbol: str, bar: Dict[str, float]) -> None:
        """
        Update with new bar data for incremental feature calculation.
        
        Args:
            symbol: Symbol to update
            bar: Bar data with open, high, low, close, volume
        """
        # Store new bar data
        for field in ['open', 'high', 'low', 'close', 'volume']:
            if field in bar:
                self.price_data[symbol][field].append(bar[field])
        
        self.bar_count[symbol] += 1
        
        # Recompute features incrementally
        self._update_features(symbol)
        
        logger.debug("Updated bar for %s, total bars: %d", symbol, self.bar_count[symbol])
    
    def get_features(self, symbol: str) -> Dict[str, Any]:
        """
        Get current feature values for a symbol.
        
        Args:
            symbol: Symbol to get features for
            
        Returns:
            Dict of feature_name -> feature_value
        """
        return self.feature_cache.get(symbol, {}).copy()
    
    def get_all_features(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current feature values for all symbols.
        
        Returns:
            Dict mapping symbol -> feature_dict
        """
        return {
            symbol: self.get_features(symbol)
            for symbol in self.symbols
        }
    
    def has_sufficient_data(self, symbol: str, min_bars: int = 50) -> bool:
        """
        Check if symbol has sufficient data for feature calculation.
        
        Args:
            symbol: Symbol to check
            min_bars: Minimum number of bars required
            
        Returns:
            True if sufficient data available
        """
        return self.bar_count.get(symbol, 0) >= min_bars
    
    def reset(self, symbol: Optional[str] = None) -> None:
        """
        Reset feature computation state.
        
        Args:
            symbol: Symbol to reset (None for all symbols)
        """
        if symbol:
            # Reset specific symbol
            for field in self.price_data[symbol]:
                self.price_data[symbol][field].clear()
            self.feature_cache[symbol].clear()
            self.bar_count[symbol] = 0
            logger.info("Reset FeatureHub state for symbol: %s", symbol)
        else:
            # Reset all symbols
            self.price_data.clear()
            self.feature_cache.clear()
            self.bar_count.clear()
            logger.info("Reset FeatureHub state for all symbols")
    
    def _update_features(self, symbol: str) -> None:
        """
        Update features for a symbol using current data.
        
        This converts the rolling window data to pandas and computes
        features using the stateless feature functions.
        """
        # Convert deques to pandas DataFrame for feature computation
        data_dict = {}
        for field, deque_data in self.price_data[symbol].items():
            if len(deque_data) > 0:
                data_dict[field] = list(deque_data)
        
        if not data_dict or len(data_dict['close']) < 2:
            return
        
        # Create DataFrame for feature computation
        df = pd.DataFrame(data_dict)
        
        # Compute all configured features
        symbol_features = {}
        
        for feature_name, config in self.feature_configs.items():
            try:
                feature_type = config.get('feature')
                if feature_type not in FEATURE_REGISTRY:
                    logger.warning("Unknown feature type: %s", feature_type)
                    continue
                
                # Create config copy without 'feature' key
                feature_params = {k: v for k, v in config.items() if k != 'feature'}
                
                # Compute feature using stateless function
                feature_func = FEATURE_REGISTRY[feature_type]
                
                if feature_type in ['sma', 'ema', 'rsi', 'momentum']:
                    result = feature_func(df['close'], **feature_params)
                elif feature_type in ['macd', 'bollinger']:
                    result = feature_func(df['close'], **feature_params)
                elif feature_type in ['atr', 'stochastic', 'adx', 'williams_r', 'cci']:
                    result = feature_func(df['high'], df['low'], df['close'], **feature_params)
                elif feature_type == 'volume':
                    result = feature_func(df['volume'], df['close'], **feature_params)
                elif feature_type == 'price_action':
                    open_series = df.get('open')
                    result = feature_func(df['high'], df['low'], df['close'], open_series, **feature_params)
                else:
                    logger.warning("Unsupported feature type: %s", feature_type)
                    continue
                
                # Store latest feature value(s)
                if isinstance(result, dict):
                    # Multi-value features (e.g., MACD, Bollinger Bands)
                    for sub_name, series in result.items():
                        if len(series) > 0 and not pd.isna(series.iloc[-1]):
                            symbol_features[f"{feature_name}_{sub_name}"] = float(series.iloc[-1])
                else:
                    # Single-value features (e.g., SMA, RSI)
                    if len(result) > 0 and not pd.isna(result.iloc[-1]):
                        symbol_features[feature_name] = float(result.iloc[-1])
                        
            except Exception as e:
                logger.error("Error computing feature %s for %s: %s", feature_name, symbol, e)
        
        # Update cache
        self.feature_cache[symbol].update(symbol_features)
        
        logger.debug("Updated %d features for %s", len(symbol_features), symbol)
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of FeatureHub state.
        
        Returns:
            Dict with summary information
        """
        return {
            "symbols": len(self.symbols),
            "configured_features": len(self.feature_configs),
            "bar_counts": dict(self.bar_count),
            "feature_counts": {
                symbol: len(features)
                for symbol, features in self.feature_cache.items()
            }
        }


# Factory function for creating FeatureHub
def create_feature_hub(symbols: Optional[List[str]] = None,
                      feature_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> FeatureHub:
    """
    Factory function for creating FeatureHub instances.
    
    Args:
        symbols: List of symbols to track
        feature_configs: Feature configurations
        
    Returns:
        Configured FeatureHub instance
    """
    hub = FeatureHub(symbols)
    
    if feature_configs:
        hub.configure_features(feature_configs)
    
    return hub


# Default feature configurations for common strategies
DEFAULT_MOMENTUM_FEATURES = {
    "sma_fast": {"feature": "sma", "period": 10},
    "sma_slow": {"feature": "sma", "period": 20},
    "rsi": {"feature": "rsi", "period": 14},
    "macd": {"feature": "macd", "fast": 12, "slow": 26, "signal": 9}
}

DEFAULT_MEAN_REVERSION_FEATURES = {
    "bollinger": {"feature": "bollinger", "period": 20, "std_dev": 2.0},
    "rsi": {"feature": "rsi", "period": 14},
    "stochastic": {"feature": "stochastic", "k_period": 14, "d_period": 3}
}

DEFAULT_VOLATILITY_FEATURES = {
    "atr": {"feature": "atr", "period": 14},
    "bollinger": {"feature": "bollinger", "period": 20, "std_dev": 2.0},
    "adx": {"feature": "adx", "period": 14}
}
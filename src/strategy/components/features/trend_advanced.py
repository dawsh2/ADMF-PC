"""
Advanced trend-following indicator features.

Complex trend indicators including SuperTrend, Parabolic SAR, etc.
All functions are pure and stateless for parallelization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ....core.components.discovery import feature


@feature(
    name='supertrend',
    params=['period', 'multiplier'],
    min_history='period',
    input_type='hlc'
)
def supertrend_feature(high: pd.Series, low: pd.Series, close: pd.Series,
                      period: int = 10, multiplier: float = 3.0) -> Dict[str, pd.Series]:
    """
    Calculate SuperTrend feature.
    
    Trend-following indicator based on ATR.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: ATR period (default 10)
        multiplier: ATR multiplier (default 3.0)
        
    Returns:
        Dict with 'trend' and 'direction' series
    """
    # Calculate ATR
    hl = high - low
    hc = np.abs(high - close.shift(1))
    lc = np.abs(low - close.shift(1))
    tr = np.maximum(hl, np.maximum(hc, lc))
    atr = tr.rolling(period).mean()
    
    # Calculate basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize arrays
    trend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=int)
    
    # Calculate SuperTrend
    for i in range(len(close)):
        if i == 0:
            trend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = 1
        else:
            # Update upper and lower bands
            if upper_band.iloc[i] < trend.iloc[i-1] or close.iloc[i-1] > trend.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i]
            else:
                upper_band.iloc[i] = trend.iloc[i-1]
                
            if lower_band.iloc[i] > trend.iloc[i-1] or close.iloc[i-1] < trend.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i]
            else:
                lower_band.iloc[i] = trend.iloc[i-1]
            
            # Determine trend direction
            if close.iloc[i] <= lower_band.iloc[i]:
                direction.iloc[i] = -1
                trend.iloc[i] = lower_band.iloc[i]
            elif close.iloc[i] >= upper_band.iloc[i]:
                direction.iloc[i] = 1
                trend.iloc[i] = upper_band.iloc[i]
            else:
                direction.iloc[i] = direction.iloc[i-1]
                trend.iloc[i] = upper_band.iloc[i] if direction.iloc[i] == 1 else lower_band.iloc[i]
    
    return {
        "trend": trend,
        "direction": direction
    }


@feature(
    name='psar',
    params=['af_start', 'af_max'],
    min_history='2',
    input_type='hlc'
)
def psar_feature(high: pd.Series, low: pd.Series, close: pd.Series,
                af_start: float = 0.02, af_max: float = 0.20) -> Dict[str, pd.Series]:
    """
    Calculate Parabolic SAR feature.
    
    Trend-following indicator that provides stop-loss levels.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        af_start: Initial acceleration factor (default 0.02)
        af_max: Maximum acceleration factor (default 0.20)
        
    Returns:
        Dict with 'sar' and 'trend' series
    """
    n = len(close)
    sar = pd.Series(index=close.index, dtype=float)
    trend = pd.Series(index=close.index, dtype=int)
    
    # Initialize
    sar.iloc[0] = low.iloc[0]
    trend.iloc[0] = 1  # Start with uptrend
    af = af_start
    ep = high.iloc[0]  # Extreme point
    
    for i in range(1, n):
        if trend.iloc[i-1] == 1:  # Uptrend
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            
            # Check for trend reversal
            if low.iloc[i] <= sar.iloc[i]:
                trend.iloc[i] = -1
                sar.iloc[i] = ep
                af = af_start
                ep = low.iloc[i]
            else:
                trend.iloc[i] = 1
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + af_start, af_max)
                
                # SAR should not be above previous two lows
                sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1])
                if i > 1:
                    sar.iloc[i] = min(sar.iloc[i], low.iloc[i-2])
        
        else:  # Downtrend
            sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
            
            # Check for trend reversal
            if high.iloc[i] >= sar.iloc[i]:
                trend.iloc[i] = 1
                sar.iloc[i] = ep
                af = af_start
                ep = high.iloc[i]
            else:
                trend.iloc[i] = -1
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + af_start, af_max)
                
                # SAR should not be below previous two highs
                sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1])
                if i > 1:
                    sar.iloc[i] = max(sar.iloc[i], high.iloc[i-2])
    
    return {
        "sar": sar,
        "trend": trend
    }


@feature(
    name='linear_regression',
    params=['period'],
    min_history='period'
)
def linear_regression_feature(prices: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
    """
    Calculate Linear Regression feature.
    
    Trend indicator using linear regression line.
    
    Args:
        prices: Price series
        period: Period for calculation (default 14)
        
    Returns:
        Dict with 'line' and 'slope' series
    """
    def linear_reg(y_values):
        n = len(y_values)
        x_values = np.arange(n)
        
        # Calculate slope and intercept
        x_mean = x_values.mean()
        y_mean = y_values.mean()
        
        numerator = ((x_values - x_mean) * (y_values - y_mean)).sum()
        denominator = ((x_values - x_mean) ** 2).sum()
        
        if denominator == 0:
            return y_mean, 0
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate the line value at the last point
        line_value = intercept + slope * (n - 1)
        
        return line_value, slope
    
    lr_line = pd.Series(index=prices.index, dtype=float)
    lr_slope = pd.Series(index=prices.index, dtype=float)
    
    for i in range(period - 1, len(prices)):
        window = prices.iloc[i - period + 1:i + 1].values
        line_val, slope_val = linear_reg(window)
        lr_line.iloc[i] = line_val
        lr_slope.iloc[i] = slope_val
    
    return {
        "line": lr_line,
        "slope": lr_slope
    }


@feature(
    name='pivot_points',
    params=['pivot_type'],
    min_history='1',
    input_type='hlc'
)
def pivot_points_feature(high: pd.Series, low: pd.Series, close: pd.Series,
                        pivot_type: str = 'standard') -> Dict[str, pd.Series]:
    """
    Calculate Pivot Points feature.
    
    Support and resistance levels based on previous period's HLOC.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        pivot_type: Type of pivot ('standard', 'fibonacci', 'camarilla')
        
    Returns:
        Dict with pivot levels
    """
    # Use previous day's values
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    
    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3
    
    if pivot_type == 'standard':
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)
        
    elif pivot_type == 'fibonacci':
        diff = prev_high - prev_low
        r1 = pivot + 0.382 * diff
        s1 = pivot - 0.382 * diff
        r2 = pivot + 0.618 * diff
        s2 = pivot - 0.618 * diff
        r3 = pivot + diff
        s3 = pivot - diff
        
    else:  # camarilla
        diff = prev_high - prev_low
        r1 = prev_close + 1.1 * diff / 12
        s1 = prev_close - 1.1 * diff / 12
        r2 = prev_close + 1.1 * diff / 6
        s2 = prev_close - 1.1 * diff / 6
        r3 = prev_close + 1.1 * diff / 4
        s3 = prev_close - 1.1 * diff / 4
    
    return {
        "pivot": pivot,
        "r1": r1,
        "r2": r2,
        "r3": r3,
        "s1": s1,
        "s2": s2,
        "s3": s3
    }


@feature(
    name='fibonacci_retracement',
    params=['period'],
    min_history='period',
    input_type='hl'
)
def fibonacci_retracement_feature(high: pd.Series, low: pd.Series,
                                period: int = 50) -> Dict[str, pd.Series]:
    """
    Calculate Fibonacci Retracement levels.
    
    Key retracement levels based on Fibonacci ratios.
    
    Args:
        high: High price series
        low: Low price series
        period: Period for high/low calculation (default 50)
        
    Returns:
        Dict with Fibonacci levels
    """
    # Calculate rolling high and low
    rolling_high = high.rolling(period).max()
    rolling_low = low.rolling(period).min()
    
    diff = rolling_high - rolling_low
    
    # Calculate Fibonacci levels
    fib_0 = rolling_high
    fib_236 = rolling_high - 0.236 * diff
    fib_382 = rolling_high - 0.382 * diff
    fib_500 = rolling_high - 0.5 * diff
    fib_618 = rolling_high - 0.618 * diff
    fib_786 = rolling_high - 0.786 * diff
    fib_100 = rolling_low
    
    return {
        "fib_0": fib_0,
        "fib_236": fib_236,
        "fib_382": fib_382,
        "fib_500": fib_500,
        "fib_618": fib_618,
        "fib_786": fib_786,
        "fib_100": fib_100
    }


@feature(
    name='support_resistance',
    params=['period', 'threshold'],
    min_history='period',
    input_type='hlc'
)
def support_resistance_feature(high: pd.Series, low: pd.Series, close: pd.Series,
                              period: int = 20, threshold: float = 0.02) -> Dict[str, pd.Series]:
    """
    Calculate Support and Resistance levels.
    
    Dynamic support/resistance based on price action.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Period for calculation (default 20)
        threshold: Threshold for level significance (default 0.02)
        
    Returns:
        Dict with support and resistance levels
    """
    support = pd.Series(index=close.index, dtype=float)
    resistance = pd.Series(index=close.index, dtype=float)
    
    for i in range(period, len(close)):
        window_high = high.iloc[i-period:i]
        window_low = low.iloc[i-period:i]
        
        # Find significant highs and lows
        resistance.iloc[i] = window_high.max()
        support.iloc[i] = window_low.min()
    
    return {
        "support": support,
        "resistance": resistance
    }


@feature(
    name='swing_points',
    params=['period'],
    min_history='period * 2',
    input_type='hlc'
)
def swing_points_feature(high: pd.Series, low: pd.Series, close: pd.Series,
                        period: int = 5) -> Dict[str, pd.Series]:
    """
    Calculate Swing High and Low points.
    
    Identifies significant price reversals.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Period for swing identification (default 5)
        
    Returns:
        Dict with swing highs and lows
    """
    swing_high = pd.Series(index=close.index, dtype=float)
    swing_low = pd.Series(index=close.index, dtype=float)
    
    for i in range(period, len(close) - period):
        # Check for swing high
        window_high = high.iloc[i-period:i+period+1]
        if high.iloc[i] == window_high.max():
            swing_high.iloc[i] = high.iloc[i]
        
        # Check for swing low
        window_low = low.iloc[i-period:i+period+1]
        if low.iloc[i] == window_low.min():
            swing_low.iloc[i] = low.iloc[i]
    
    return {
        "swing_high": swing_high,
        "swing_low": swing_low
    }
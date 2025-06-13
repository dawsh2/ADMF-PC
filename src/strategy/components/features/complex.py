"""
Complex multi-component feature calculations.

Advanced indicators with multiple interdependent components.
All functions are pure and stateless for parallelization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ....core.components.discovery import feature


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
    
    Complex indicator with multiple interdependent components for trend analysis.
    
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
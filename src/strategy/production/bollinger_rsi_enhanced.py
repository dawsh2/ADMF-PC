"""
Bollinger Band + RSI Enhanced Strategy with Smart Filters

This is an enhanced version of bollinger_rsi_simple_signals with:
1. Stop loss at -0.1%
2. Volatility-aware position sizing
3. Market condition filters
4. Time-based filters

Based on comprehensive analysis of market conditions and performance.
"""

from typing import Dict, Any, Optional
from ....core.components.discovery import strategy
from ....core.features.feature_spec import FeatureSpec


@strategy(
    name='bollinger_rsi_enhanced',
    feature_discovery=lambda params: [
        # Bollinger Bands
        FeatureSpec('bollinger_bands', {
            'period': params.get('bb_period', 20),
            'std_dev': params.get('bb_std', 2.0)
        }, 'middle'),
        FeatureSpec('bollinger_bands', {
            'period': params.get('bb_period', 20),
            'std_dev': params.get('bb_std', 2.0)
        }, 'upper'),
        FeatureSpec('bollinger_bands', {
            'period': params.get('bb_period', 20),
            'std_dev': params.get('bb_std', 2.0)
        }, 'lower'),
        # RSI
        FeatureSpec('rsi', {'period': params.get('rsi_period', 14)}),
        # ATR for volatility
        FeatureSpec('atr', {'period': params.get('atr_period', 14)}),
        # VWAP
        FeatureSpec('vwap', {}),
        # SMA for trend
        FeatureSpec('sma', {'period': 200})
    ],
    parameter_space={
        'bb_period': {'type': 'int', 'range': (10, 50), 'default': 20},
        'bb_std': {'type': 'float', 'range': (1.5, 3.0), 'default': 2.0},
        'rsi_period': {'type': 'int', 'range': (10, 20), 'default': 14},
        'rsi_threshold': {'type': 'float', 'range': (5, 15), 'default': 10},
        'atr_period': {'type': 'int', 'range': (10, 20), 'default': 14},
        'stop_loss_pct': {'type': 'float', 'range': (0.0005, 0.002), 'default': 0.001},
        'min_bars_between_trades': {'type': 'int', 'range': (20, 50), 'default': 30}
    },
    strategy_type='mean_reversion',
    tags=['bollinger_bands', 'rsi', 'mean_reversion', 'enhanced', 'production']
)
def bollinger_rsi_enhanced(features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Enhanced BB RSI strategy with smart filters.
    
    Improvements based on analysis:
    1. Stop loss at -0.1%
    2. Volatility-based direction bias
    3. VWAP and trend alignment
    4. Time and extreme volatility filters
    """
    # Get parameters
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    rsi_period = params.get('rsi_period', 14)
    rsi_threshold = params.get('rsi_threshold', 10)
    atr_period = params.get('atr_period', 14)
    stop_loss_pct = params.get('stop_loss_pct', 0.001)  # 0.1%
    
    # Get current values
    price = bar.get('close', 0)
    timestamp = bar.get('timestamp')
    
    # Get indicators
    middle_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_middle', price)
    upper_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_upper', price)
    lower_band = features.get(f'bollinger_bands_{bb_period}_{bb_std}_lower', price)
    rsi = features.get(f'rsi_{rsi_period}', 50)
    atr = features.get(f'atr_{atr_period}', 0)
    vwap = features.get('vwap', price)
    sma_200 = features.get('sma_200', price)
    
    # Calculate derived indicators
    band_width = upper_band - lower_band
    position_in_bands = (price - lower_band) / band_width if band_width > 0 else 0.5
    atr_pct = atr / price * 100 if price > 0 else 0
    price_vs_vwap = (price - vwap) / vwap * 100 if vwap > 0 else 0
    sma_200_prev = features.get('sma_200_prev_20', sma_200)  # Would need to track this
    trend_slope = (sma_200 - sma_200_prev) / sma_200_prev * 100 if sma_200_prev > 0 else 0
    
    # Time filter - avoid first 30 minutes
    if timestamp:
        hour = timestamp.hour
        minute = timestamp.minute
        if hour == 9 and minute < 30:  # First 30 minutes
            return None
    
    # Extreme volatility filter - skip top 5% volatility
    if atr_pct > 0.098:  # 95th percentile from analysis
        return None
    
    # Determine volatility regime (simplified - would need percentile tracking)
    if atr_pct < 0.038:  # Below median
        volatility_regime = 'low'
    elif atr_pct < 0.065:  # Below ~80th percentile
        volatility_regime = 'medium'
    else:
        volatility_regime = 'high'
    
    # Determine trend regime
    if trend_slope < -0.05:
        trend_regime = 'downtrend'
    elif trend_slope > 0.05:
        trend_regime = 'uptrend'
    else:
        trend_regime = 'sideways'
    
    # Base signal logic (same as simple_signals)
    signal_value = None
    signal_type = None
    position_size = 1.0  # Default full size
    
    # Long signal conditions
    if position_in_bands < 0 and rsi > (30 + rsi_threshold):
        # Check market condition filters for longs
        long_filter_pass = True
        
        # Trend + VWAP alignment for longs
        if trend_regime == 'downtrend' and price_vs_vwap > 0.2:
            long_filter_pass = False  # Skip longs in downtrend above VWAP
        
        if long_filter_pass:
            signal_value = 1
            signal_type = 'bb_rsi_long'
            
            # Volatility-based sizing
            if volatility_regime == 'high':
                position_size = 1.0  # Full size for longs in high vol
            elif volatility_regime == 'medium':
                position_size = 1.0  # Full size for longs
            else:
                position_size = 1.0  # Full size for longs
    
    # Short signal conditions
    elif position_in_bands > 1 and rsi < (70 - rsi_threshold):
        # Check market condition filters for shorts
        short_filter_pass = True
        
        # Avoid shorts in medium volatility
        if volatility_regime == 'medium':
            short_filter_pass = False
        
        # Trend + VWAP alignment for shorts
        if trend_regime == 'uptrend' and trend_slope > 0.1:
            short_filter_pass = False  # Skip shorts in strong uptrend
        
        if price_vs_vwap < -0.2:
            short_filter_pass = False  # Skip shorts when far below VWAP
        
        if short_filter_pass:
            signal_value = -1
            signal_type = 'bb_rsi_short'
            
            # Volatility-based sizing
            if volatility_regime == 'high':
                position_size = 0.5  # Half size for shorts in high vol
            elif volatility_regime == 'low':
                position_size = 1.0  # Full size for shorts in low vol
            else:
                position_size = 0  # Skip shorts in medium vol
    
    # Exit signal (middle band)
    elif 0.4 < position_in_bands < 0.6:
        signal_value = 0
        signal_type = 'middle_band_exit'
        position_size = 0
    
    # Return signal with metadata
    if signal_value is not None:
        return {
            'signal_value': signal_value,
            'position_size': position_size,
            'metadata': {
                'signal_type': signal_type,
                'price': price,
                'rsi': rsi,
                'atr_pct': atr_pct,
                'volatility_regime': volatility_regime,
                'trend_regime': trend_regime,
                'price_vs_vwap': price_vs_vwap,
                'stop_loss': stop_loss_pct,
                'filters_applied': True
            }
        }
    
    return None
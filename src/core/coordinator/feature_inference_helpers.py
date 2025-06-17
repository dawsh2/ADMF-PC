"""Helper functions for feature inference with complex indicators."""

def get_feature_parameter_mapping():
    """
    Define how different features map their parameters.
    
    Returns a dict mapping feature names to their parameter patterns.
    """
    return {
        # Simple features with one period parameter
        'sma': '{feature}_{period}',
        'ema': '{feature}_{period}',
        'dema': '{feature}_{period}',
        'tema': '{feature}_{period}',
        'wma': '{feature}_{period}',
        'hma': '{feature}_{period}',
        'rsi': '{feature}_{period}',
        'atr': '{feature}_{period}',
        'adx': '{feature}_{period}',
        'cci': '{feature}_{period}',
        'roc': '{feature}_{period}',
        'aroon': '{feature}_{period}',
        'vortex': '{feature}_{period}',
        
        # Features with multiple parameters
        'bollinger_bands': 'bollinger_bands_{period}_{std_dev}',
        'keltner': 'keltner_{period}_{multiplier}',
        'donchian': 'donchian_{period}',
        'stochastic': 'stochastic_{k_period}_{d_period}',
        'stochastic_rsi': 'stochastic_rsi_{rsi_period}_{stoch_period}',
        'macd': 'macd_{fast_period}_{slow_period}_{signal_period}',
        'supertrend': 'supertrend_{period}_{multiplier}',
        
        # Features with no parameters
        'vwap': 'vwap',
        'obv': 'obv',
        'ad': 'ad',
        'volume': 'volume',
        'close': 'close',
        'open': 'open',
        'high': 'high', 
        'low': 'low'
    }


def infer_feature_name(feature_type: str, params: dict) -> str:
    """
    Infer the actual feature name based on feature type and parameters.
    
    Args:
        feature_type: The base feature type (e.g., 'sma', 'bollinger_bands')
        params: The strategy parameters
        
    Returns:
        The inferred feature name (e.g., 'sma_20', 'bollinger_bands_20_2.0')
    """
    mapping = get_feature_parameter_mapping()
    
    if feature_type not in mapping:
        # Unknown feature, try simple period inference
        period = find_period_param(feature_type, params)
        if period:
            return f"{feature_type}_{period}"
        return feature_type
    
    template = mapping[feature_type]
    
    # Handle features with no parameters
    if template == feature_type:
        return feature_type
    
    # Extract parameters based on feature type
    if feature_type == 'bollinger_bands':
        period = params.get('bollinger_period', params.get('bb_period', params.get('period', 20)))
        std_dev = params.get('bollinger_std_dev', params.get('bb_std', params.get('std_dev', 2.0)))
        return template.format(period=period, std_dev=std_dev)
        
    elif feature_type == 'keltner':
        period = params.get('keltner_period', params.get('kc_period', params.get('period', 20)))
        multiplier = params.get('keltner_multiplier', params.get('kc_mult', params.get('multiplier', 2.0)))
        return template.format(period=period, multiplier=multiplier)
        
    elif feature_type == 'stochastic':
        k_period = params.get('k_period', params.get('stoch_k_period', 14))
        d_period = params.get('d_period', params.get('stoch_d_period', 3))
        return template.format(k_period=k_period, d_period=d_period)
        
    elif feature_type == 'stochastic_rsi':
        rsi_period = params.get('rsi_period', 14)
        stoch_period = params.get('stoch_period', params.get('stochastic_period', 14))
        return template.format(rsi_period=rsi_period, stoch_period=stoch_period)
        
    elif feature_type == 'macd':
        fast = params.get('fast_period', params.get('fast_ema', params.get('macd_fast', 12)))
        slow = params.get('slow_period', params.get('slow_ema', params.get('macd_slow', 26)))
        signal = params.get('signal_period', params.get('signal_ema', params.get('macd_signal', 9)))
        return template.format(fast_period=fast, slow_period=slow, signal_period=signal)
        
    elif feature_type == 'supertrend':
        period = params.get('supertrend_period', params.get('st_period', params.get('period', 10)))
        multiplier = params.get('supertrend_multiplier', params.get('st_mult', params.get('multiplier', 3.0)))
        return template.format(period=period, multiplier=multiplier)
        
    else:
        # Simple period-based features
        period = find_period_param(feature_type, params)
        if period:
            return template.format(feature=feature_type, period=period)
    
    return feature_type


def find_period_param(feature_type: str, params: dict) -> int:
    """Find the period parameter for a given feature type."""
    # Check feature-specific period names first
    period_names = [
        f'{feature_type}_period',
        f'{feature_type[0:3]}_period',  # e.g., sma_period, rsi_period
        'period',
        'lookback',
        'window'
    ]
    
    for name in period_names:
        if name in params:
            return params[name]
    
    # Check for any parameter with 'period' in the name
    for key, value in params.items():
        if 'period' in key.lower() and isinstance(value, (int, float)):
            return int(value)
    
    # Default periods for common indicators
    defaults = {
        'sma': 20,
        'ema': 20,
        'rsi': 14,
        'atr': 14,
        'cci': 20,
        'adx': 14,
        'roc': 10,
        'aroon': 25,
        'vortex': 14
    }
    
    return defaults.get(feature_type, 14)
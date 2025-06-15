#!/usr/bin/env python3
"""
Check data requirements for missing strategies
"""

# Missing strategies and their potential data requirements
missing_strategies = {
    'fibonacci_retracement': {'period': [30, 50, 100]},  # From config
    'pivot_points': {'pivot_type': ['standard', 'fibonacci', 'camarilla']},
    'price_action_swing': {'period': [5, 10, 20]},
    'support_resistance_breakout': {'period': [10, 20, 30]},
    'linear_regression_slope': {'period': [10, 20, 30]},
    'aroon_crossover': {'period': [14, 25, 35]},
    'parabolic_sar': {'af': [0.01, 0.02, 0.03]},
    'supertrend': {'period': [7, 10, 14]},
    'ultimate_oscillator': {'period1': [5, 7, 10], 'period2': [10, 14, 20], 'period3': [20, 28, 35]},
    'stochastic_rsi': {'rsi_period': [10, 14, 20], 'stoch_period': [7, 14, 21]},
    'vortex_crossover': {'vortex_period': [11, 19, 27, 35]},
    'macd_crossover': {'fast_ema': [5, 8, 12], 'slow_ema': [20, 26, 35]},
    'stochastic_crossover': {'k_period': [7, 11, 15], 'd_period': [3, 5, 7]},
    'ichimoku_cloud_position': {'conversion_period': [7, 9, 11], 'base_period': [20, 26, 35]},
    'roc_threshold': {'roc_period': [5, 10, 15]},
    'bollinger_breakout': {'period': [10, 20, 30]},
    'donchian_breakout': {'period': [10, 20, 30]},
    'keltner_breakout': {'period': [10, 20, 30]},
    'obv_trend': {'obv_sma_period': [10, 20, 30]},
    'vwap_deviation': {},
    'accumulation_distribution': {'ad_ema_period': [10, 20, 30]},
    'adx_trend_strength': {'adx_period': [7, 14, 21]}
}

print("Data requirements for missing strategies:")
print("(Running with 300 bars)\n")

for strategy, params in missing_strategies.items():
    max_period = 0
    for param_name, values in params.items():
        if 'period' in param_name and values:
            max_period = max(max_period, max(values))
    
    if max_period > 0:
        # Many indicators need 2-3x their period for warmup
        warmup_needed = max_period * 2
        if warmup_needed > 250:  # Leave 50 bars for actual signals
            print(f"⚠️  {strategy}: Max period {max_period}, needs ~{warmup_needed} bars warmup")
        else:
            print(f"✓ {strategy}: Max period {max_period}, should work with 300 bars")
    else:
        print(f"✓ {strategy}: No period requirements")

print("\nStrategies that might not have enough data with 300 bars:")
print("None! All should work with 300 bars.")
print("\nThe issue must be elsewhere...")
#!/usr/bin/env python3
"""Map strategy IDs to their filter configurations."""

# Based on optimize_keltner_enhanced_5m.yaml structure
def get_strategy_config(strategy_id):
    """Map strategy ID to its configuration."""
    
    configs = []
    idx = 0
    
    # 1. Base winner (1 strategy)
    configs.append({
        'id': idx,
        'period': 50,
        'multiplier': 0.60,
        'filter': 'None (base strategy)'
    })
    idx += 1
    
    # 2. Fine-tune multipliers (6 strategies)
    for mult in [0.58, 0.59, 0.60, 0.61, 0.62, 0.63]:
        configs.append({
            'id': idx,
            'period': 50,
            'multiplier': mult,
            'filter': 'None (multiplier tuning)'
        })
        idx += 1
    
    # 3. VWAP stretch filter (6 strategies)
    for vwap_dist in [0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005]:
        configs.append({
            'id': idx,
            'period': 50,
            'multiplier': 0.60,
            'filter': f'VWAP distance > {vwap_dist}'
        })
        idx += 1
    
    # 4. RSI extremes (9 strategies - 3x3 grid)
    for rsi_low in [25, 30, 35]:
        for rsi_high in [70, 75, 80]:
            configs.append({
                'id': idx,
                'period': 50,
                'multiplier': 0.60,
                'filter': f'RSI < {rsi_low} (long) or RSI > {rsi_high} (short)'
            })
            idx += 1
    
    # 5. Volume spike (4 strategies)
    for vol_spike in [1.5, 2.0, 2.5, 3.0]:
        configs.append({
            'id': idx,
            'period': 50,
            'multiplier': 0.60,
            'filter': f'Volume ratio > {vol_spike}'
        })
        idx += 1
    
    # 6. Volatility regime (3 strategies)
    for vol_pct in [0.6, 0.7, 0.8]:
        configs.append({
            'id': idx,
            'period': 50,
            'multiplier': 0.60,
            'filter': f'Volatility percentile > {vol_pct}'
        })
        idx += 1
    
    # 7. Combined VWAP + RSI (4 strategies - 2x2 grid)
    for vwap_dist in [0.003, 0.004]:
        for rsi_low, rsi_high in [(30, 70), (35, 75)]:
            configs.append({
                'id': idx,
                'period': 50,
                'multiplier': 0.60,
                'filter': f'VWAP > {vwap_dist} AND RSI < {rsi_low}/{rsi_high}'
            })
            idx += 1
    
    # 8. VWAP + Volume (4 strategies - 2x2 grid)
    for vwap_dist in [0.003, 0.0035]:
        for vol_ratio in [1.5, 2.0]:
            configs.append({
                'id': idx,
                'period': 50,
                'multiplier': 0.60,
                'filter': f'VWAP > {vwap_dist} AND Volume > {vol_ratio}x'
            })
            idx += 1
    
    # 9. Trend alignment (3 strategies)
    for trend_ma in [50, 100, 200]:
        configs.append({
            'id': idx,
            'period': 50,
            'multiplier': 0.60,
            'filter': f'Trade with SMA_{trend_ma} trend'
        })
        idx += 1
    
    # 10. Counter-trend RSI (1 strategy)
    configs.append({
        'id': idx,
        'period': 50,
        'multiplier': 0.60,
        'filter': 'Counter-trend with RSI divergence'
    })
    idx += 1
    
    # Remaining strategies...
    
    if strategy_id < len(configs):
        return configs[strategy_id]
    else:
        return {
            'id': strategy_id,
            'period': 50,
            'multiplier': 0.60,
            'filter': f'Other filter config {strategy_id - len(configs) + 1}'
        }

# Map the best performers from the analysis
print("Best Performing Strategies:\n")

# From the analysis output, best performers were:
best_performers = [
    (24, 1.75, 0.97, 713),  # Period 50, Mult 1.75, Edge 0.97 bps
    (25, 1.80, 0.93, 712),  # Period 50, Mult 1.80  
    (41, 0.70, 0.90, 707),  # Period 45, Mult 0.70
]

print("Based on the multipliers, these appear to be from test_keltner_parameter_sweep.yaml")
print("NOT from optimize_keltner_enhanced_5m.yaml!")
print("\nThe enhanced config with filters hasn't been run yet.")
print("\nTo test the filters, you need to run:")
print("python3 main.py --config config/optimize_keltner_enhanced_5m.yaml --optimize --signal-generation")
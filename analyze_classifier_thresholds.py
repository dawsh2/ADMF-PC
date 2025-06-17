#!/usr/bin/env python3
"""
Analyze classifier thresholds and parameters to understand imbalanced distributions.
"""

# Define the classifier configurations with their states
CLASSIFIER_CONFIGS = {
    'multi_timeframe_trend_classifier': {
        'states': ['strong_uptrend', 'weak_uptrend', 'sideways', 'weak_downtrend', 'strong_downtrend'],
        'state_map': {0: 'strong_uptrend', 1: 'weak_uptrend', 2: 'sideways', 3: 'weak_downtrend', 4: 'strong_downtrend'},
        'thresholds': {
            'strong_threshold': 0.02,  # 2%
            'weak_threshold': 0.005    # 0.5%
        },
        'issue': '99.9% in sideways - thresholds too high for typical market movement'
    },
    
    'volatility_momentum_classifier': {
        'states': ['high_vol_bullish', 'high_vol_bearish', 'low_vol_bullish', 'low_vol_bearish', 'neutral'],
        'state_map': {0: 'high_vol_bullish', 1: 'high_vol_bearish', 2: 'low_vol_bullish', 3: 'low_vol_bearish', 4: 'neutral'},
        'thresholds': {
            'vol_threshold': 1.5,      # 1.5% volatility
            'rsi_overbought': 65,
            'rsi_oversold': 35
        },
        'issue': '61.1% neutral - RSI thresholds too extreme (65/35)'
    },
    
    'market_regime_classifier': {
        'states': ['bull_trending', 'bull_ranging', 'bear_trending', 'bear_ranging', 'neutral'],
        'state_map': {0: 'bull_trending', 1: 'bull_ranging', 2: 'bear_trending', 3: 'bear_ranging', 4: 'neutral'},
        'thresholds': {
            'trend_threshold': 0.01,   # 1%
            'vol_threshold': 1.0       # 1%
        },
        'issue': '57.7% neutral - needs better trend/volatility detection'
    },
    
    'microstructure_classifier': {
        'states': ['breakout_up', 'breakout_down', 'consolidation', 'reversal_up', 'reversal_down'],
        'state_map': {0: 'breakout_up', 1: 'breakout_down', 2: 'consolidation', 3: 'reversal_up', 4: 'reversal_down'},
        'thresholds': {
            'breakout_threshold': 0.005,      # 0.5%
            'consolidation_threshold': 0.002   # 0.2%
        },
        'issue': '87.8% consolidation - defaults to consolidation too easily'
    },
    
    'hidden_markov_classifier': {
        'states': ['accumulation', 'markup', 'distribution', 'markdown', 'uncertainty'],
        'state_map': {0: 'accumulation', 1: 'markup', 2: 'distribution', 3: 'markdown', 4: 'uncertainty'},
        'thresholds': {
            'volume_surge_threshold': 1.5,
            'trend_strength_threshold': 0.02,  # 2%
            'volatility_threshold': 1.5        # 1.5%
        },
        'issue': '64.9% uncertainty - falls back to uncertainty too often'
    }
}

print("=== Classifier Threshold Analysis ===\n")

for clf_name, config in CLASSIFIER_CONFIGS.items():
    print(f"\n{clf_name}:")
    print(f"  Issue: {config['issue']}")
    print(f"  Current thresholds:")
    for param, value in config['thresholds'].items():
        print(f"    - {param}: {value}")
    print()

print("\n=== Recommended Threshold Adjustments ===\n")

print("1. multi_timeframe_trend_classifier:")
print("   Problem: Thresholds too high (2% and 0.5%)")
print("   Solution:")
print("     - strong_threshold: 0.02 → 0.01 (1%)")
print("     - weak_threshold: 0.005 → 0.002 (0.2%)")
print("   Rationale: SPY typically moves 0.5-1% per day, current thresholds too strict")

print("\n2. volatility_momentum_classifier:")
print("   Problem: RSI thresholds too extreme (65/35)")
print("   Solution:")
print("     - rsi_overbought: 65 → 60")
print("     - rsi_oversold: 35 → 40")
print("     - vol_threshold: 1.5 → 1.0 (1%)")
print("   Rationale: RSI rarely reaches 65/35 in normal conditions")

print("\n3. market_regime_classifier:")
print("   Problem: Poor trend detection logic")
print("   Solution:")
print("     - trend_threshold: 0.01 → 0.005 (0.5%)")
print("     - vol_threshold: 1.0 → 0.8 (0.8%)")
print("     - Fix logic: Currently requires BOTH trend > threshold AND vol > threshold for trending")
print("     - Should be: trend > threshold OR vol > threshold")

print("\n4. microstructure_classifier:")
print("   Problem: Falls back to consolidation too easily")
print("   Solution:")
print("     - breakout_threshold: 0.005 → 0.003 (0.3%)")
print("     - consolidation_threshold: 0.002 → 0.001 (0.1%)")
print("     - Add RSI bands: reversal_up when RSI < 30, reversal_down when RSI > 70")
print("   Rationale: Current thresholds too strict for intraday moves")

print("\n5. hidden_markov_classifier:")
print("   Problem: Complex conditions rarely met, defaults to uncertainty")
print("   Solution:")
print("     - trend_strength_threshold: 0.02 → 0.01 (1%)")
print("     - volatility_threshold: 1.5 → 1.2 (1.2%)")
print("     - volume_surge_threshold: 1.5 → 1.3")
print("     - Simplify logic: Remove requirement for multiple conditions")

print("\n=== Logic Improvements ===\n")

print("1. multi_timeframe_trend_classifier:")
print("   - Change from average of 3 trends to weighted average")
print("   - Give more weight to price vs short MA (50%)")
print("   - Less weight to medium-term trends (30% + 20%)")

print("\n2. volatility_momentum_classifier:")
print("   - Remove requirement for BOTH RSI extreme AND price momentum")
print("   - Use OR logic: RSI > 60 OR momentum > 0.5%")

print("\n3. market_regime_classifier:")
print("   - Fix is_trending logic: Should use OR not AND")
print("   - Add intermediate RSI ranges (45-55 for neutral)")

print("\n4. microstructure_classifier:")
print("   - Remove default to consolidation")
print("   - Add more nuanced breakout detection")
print("   - Use volume confirmation for breakouts")

print("\n5. hidden_markov_classifier:")
print("   - Simplify phase detection")
print("   - Use single primary indicator per phase")
print("   - Reduce reliance on perfect condition alignment")
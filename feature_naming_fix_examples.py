#!/usr/bin/env python3
"""
Show concrete examples of feature naming mismatches and fixes.
"""

print("=== FEATURE NAMING MISMATCH EXAMPLES ===\n")

examples = [
    {
        'strategy': 'stochastic_crossover',
        'wrong_config': """
features:
  stochastic:  # ❌ Missing parameters in key
    feature: stochastic
    k_period: 14
    d_period: 3""",
        'generates': ['stochastic_k', 'stochastic_d'],
        'expects': ['stochastic_14_3_k', 'stochastic_14_3_d'],
        'correct_config': """
features:
  stochastic_14_3:  # ✅ Parameters in key
    feature: stochastic
    k_period: 14
    d_period: 3"""
    },
    {
        'strategy': 'macd_crossover',
        'wrong_config': """
features:
  macd:  # ❌ Missing parameters
    feature: macd
    fast_ema: 12
    slow_ema: 26
    signal_ema: 9""",
        'generates': ['macd_macd', 'macd_signal', 'macd_histogram'],
        'expects': ['macd_12_26_9_macd', 'macd_12_26_9_signal'],
        'correct_config': """
features:
  macd_12_26_9:  # ✅ All parameters in key
    feature: macd
    fast: 12
    slow: 26
    signal: 9"""
    },
    {
        'strategy': 'stochastic_rsi',
        'wrong_config': """
features:
  stochastic_rsi:  # ❌ Missing periods
    feature: stochastic_rsi
    rsi_period: 14
    stoch_period: 14""",
        'generates': ['stochastic_rsi_k', 'stochastic_rsi_d'],
        'expects': ['stochastic_rsi_14_14_k', 'stochastic_rsi_14_14_d'],
        'correct_config': """
features:
  stochastic_rsi_14_14:  # ✅ Both periods in key
    feature: stochastic_rsi
    rsi_period: 14
    stoch_period: 14"""
    },
    {
        'strategy': 'ultimate_oscillator',
        'wrong_config': """
features:
  ultimate_oscillator:  # ❌ Missing periods
    feature: ultimate_oscillator
    uo_period1: 7
    uo_period2: 14
    uo_period3: 28""",
        'generates': ['ultimate_oscillator'],
        'expects': ['ultimate_oscillator_7_14_28'],
        'correct_config': """
features:
  ultimate_oscillator_7_14_28:  # ✅ All periods in key
    feature: ultimate_oscillator
    uo_period1: 7
    uo_period2: 14
    uo_period3: 28"""
    }
]

for example in examples:
    print(f"STRATEGY: {example['strategy']}")
    print(f"\n❌ WRONG CONFIG:{example['wrong_config']}")
    print(f"\nGenerates: {example['generates']}")
    print(f"Strategy expects: {example['expects']}")
    print(f"Result: MISMATCH - Strategy won't find features!")
    print(f"\n✅ CORRECT CONFIG:{example['correct_config']}")
    print(f"\nGenerates: {example['expects']}")
    print(f"Result: MATCH - Strategy finds its features!")
    print("\n" + "="*60 + "\n")

print("=== PATTERN TO FOLLOW ===\n")
print("""
For multi-parameter features, the configuration key must include ALL parameters:

1. Single parameter features:
   rsi_14:
     feature: rsi
     period: 14
   → Generates: rsi_14

2. Two parameter features:
   stochastic_14_3:
     feature: stochastic
     k_period: 14
     d_period: 3
   → Generates: stochastic_14_3_k, stochastic_14_3_d

3. Three parameter features:
   macd_12_26_9:
     feature: macd
     fast: 12
     slow: 26
     signal: 9
   → Generates: macd_12_26_9_macd, macd_12_26_9_signal, macd_12_26_9_histogram

The key pattern is: {feature_type}_{param1}_{param2}_{...}
""")

print("=== WHY THIS MATTERS ===\n")
print("""
When features are misconfigured:
- Strategies return None (no signal)
- 427 out of 969 components fail to generate signals
- Backtests show no results for these strategies
- The system appears broken but it's just a naming mismatch

The fix is simple: ensure feature config keys match what strategies expect!
""")
#\!/usr/bin/env python3
"""
Analyze the strategies used in the high-performing ensemble vs current configuration.
"""

import json
from pathlib import Path

# Current DEFAULT_REGIME_STRATEGIES (cost-optimized)
CURRENT_REGIME_STRATEGIES = {
    'low_vol_bullish': [
        {'name': 'dema_crossover', 'params': {'fast_dema_period': 19, 'slow_dema_period': 35}},
        {'name': 'macd_crossover', 'params': {'fast_ema': 12, 'slow_ema': 35, 'signal_ema': 9}},
        {'name': 'dema_crossover', 'params': {'fast_dema_period': 7, 'slow_dema_period': 35}},
        {'name': 'macd_crossover', 'params': {'fast_ema': 15, 'slow_ema': 35, 'signal_ema': 7}},
        {'name': 'cci_threshold', 'params': {'cci_period': 11, 'threshold': -40}},
        {'name': 'pivot_channel_bounces', 'params': {'sr_period': 20, 'min_touches': 3, 'bounce_threshold': 0.003}}
    ],
    'low_vol_bearish': [
        {'name': 'stochastic_crossover', 'params': {'k_period': 27, 'd_period': 5}},
        {'name': 'cci_threshold', 'params': {'cci_period': 11, 'threshold': -20}},
        {'name': 'ema_sma_crossover', 'params': {'ema_period': 11, 'sma_period': 15}},
        {'name': 'rsi_bands', 'params': {'rsi_period': 7, 'oversold': 25, 'overbought': 70}},
        {'name': 'pivot_channel_bounces', 'params': {'sr_period': 20, 'min_touches': 2, 'bounce_threshold': 0.001}}
        # REMOVED: keltner_breakout
    ],
    'neutral': [
        {'name': 'stochastic_rsi', 'params': {'rsi_period': 21, 'stoch_period': 21, 'oversold': 15, 'overbought': 80}},
        {'name': 'dema_crossover', 'params': {'fast_dema_period': 19, 'slow_dema_period': 15}},
        {'name': 'vortex_crossover', 'params': {'vortex_period': 27}}
        # REMOVED: bollinger_breakout
    ],
    'high_vol_bullish': [
        {'name': 'atr_channel_breakout', 'params': {'atr_period': 14, 'channel_period': 20, 'atr_multiplier': 2.0}}
        # REMOVED: keltner_breakout, bollinger_breakout
    ],
    'high_vol_bearish': [
        {'name': 'atr_channel_breakout', 'params': {'atr_period': 14, 'channel_period': 20, 'atr_multiplier': 2.0}}
        # REMOVED: keltner_breakout
    ]
}

# What might have been in the original (before cost optimization)
POSSIBLE_ORIGINAL_STRATEGIES = {
    'low_vol_bullish': [
        {'name': 'dema_crossover', 'params': {'fast_dema_period': 19, 'slow_dema_period': 35}},
        {'name': 'macd_crossover', 'params': {'fast_ema': 12, 'slow_ema': 35, 'signal_ema': 9}},
        {'name': 'dema_crossover', 'params': {'fast_dema_period': 7, 'slow_dema_period': 35}},
        {'name': 'macd_crossover', 'params': {'fast_ema': 15, 'slow_ema': 35, 'signal_ema': 7}},
        {'name': 'cci_threshold', 'params': {'cci_period': 11, 'threshold': -40}},
        {'name': 'pivot_channel_bounces', 'params': {'sr_period': 20, 'min_touches': 3, 'bounce_threshold': 0.003}}
    ],
    'low_vol_bearish': [
        {'name': 'stochastic_crossover', 'params': {'k_period': 27, 'd_period': 5}},
        {'name': 'cci_threshold', 'params': {'cci_period': 11, 'threshold': -20}},
        {'name': 'ema_sma_crossover', 'params': {'ema_period': 11, 'sma_period': 15}},
        {'name': 'rsi_bands', 'params': {'rsi_period': 7, 'oversold': 25, 'overbought': 70}},
        {'name': 'pivot_channel_bounces', 'params': {'sr_period': 20, 'min_touches': 2, 'bounce_threshold': 0.001}},
        {'name': 'keltner_breakout', 'params': {'keltner_period': 20, 'atr_multiplier': 2.0}}  # ADDED
    ],
    'neutral': [
        {'name': 'stochastic_rsi', 'params': {'rsi_period': 21, 'stoch_period': 21, 'oversold': 15, 'overbought': 80}},
        {'name': 'dema_crossover', 'params': {'fast_dema_period': 19, 'slow_dema_period': 15}},
        {'name': 'vortex_crossover', 'params': {'vortex_period': 27}},
        {'name': 'bollinger_breakout', 'params': {'bb_period': 20, 'std_mult': 2.0}}  # ADDED
    ],
    'high_vol_bullish': [
        {'name': 'atr_channel_breakout', 'params': {'atr_period': 14, 'channel_period': 20, 'atr_multiplier': 2.0}},
        {'name': 'keltner_breakout', 'params': {'keltner_period': 20, 'atr_multiplier': 2.0}},  # ADDED
        {'name': 'bollinger_breakout', 'params': {'bb_period': 20, 'std_mult': 2.0}}  # ADDED
    ],
    'high_vol_bearish': [
        {'name': 'atr_channel_breakout', 'params': {'atr_period': 14, 'channel_period': 20, 'atr_multiplier': 2.0}},
        {'name': 'keltner_breakout', 'params': {'keltner_period': 20, 'atr_multiplier': 2.0}}  # ADDED
    ]
}

def count_strategies_by_regime(strategies_dict):
    """Count strategies per regime."""
    counts = {}
    for regime, strategies in strategies_dict.items():
        counts[regime] = len(strategies)
    return counts

def get_unique_strategy_names(strategies_dict):
    """Get all unique strategy names."""
    names = set()
    for regime, strategies in strategies_dict.items():
        for strategy in strategies:
            names.add(strategy['name'])
    return sorted(names)

def compare_configurations():
    """Compare current vs possible original configurations."""
    
    print("=== ENSEMBLE STRATEGY COMPARISON ===\n")
    
    # Count strategies per regime
    current_counts = count_strategies_by_regime(CURRENT_REGIME_STRATEGIES)
    original_counts = count_strategies_by_regime(POSSIBLE_ORIGINAL_STRATEGIES)
    
    print("Strategy Count by Regime:")
    print("-" * 50)
    print(f"{'Regime':<20} {'Current':<10} {'Original':<10} {'Difference':<10}")
    print("-" * 50)
    
    for regime in sorted(current_counts.keys()):
        curr = current_counts[regime]
        orig = original_counts[regime]
        diff = orig - curr
        print(f"{regime:<20} {curr:<10} {orig:<10} {diff:<10}")
    
    print(f"\n{'TOTAL':<20} {sum(current_counts.values()):<10} {sum(original_counts.values()):<10} {sum(original_counts.values()) - sum(current_counts.values()):<10}")
    
    # Show unique strategies
    current_strategies = get_unique_strategy_names(CURRENT_REGIME_STRATEGIES)
    original_strategies = get_unique_strategy_names(POSSIBLE_ORIGINAL_STRATEGIES)
    
    print("\n\nUnique Strategy Types:")
    print("-" * 50)
    print(f"Current configuration: {len(current_strategies)} unique strategies")
    print(f"Original configuration: {len(original_strategies)} unique strategies")
    
    removed_strategies = set(original_strategies) - set(current_strategies)
    if removed_strategies:
        print(f"\nRemoved strategies: {', '.join(sorted(removed_strategies))}")
    
    # Detailed breakdown of removed strategies by regime
    print("\n\nDetailed Breakdown of Removed Strategies:")
    print("-" * 50)
    
    for regime in sorted(POSSIBLE_ORIGINAL_STRATEGIES.keys()):
        original = POSSIBLE_ORIGINAL_STRATEGIES[regime]
        current = CURRENT_REGIME_STRATEGIES[regime]
        
        # Find removed strategies
        current_strat_names = {s['name'] for s in current}
        removed_in_regime = []
        
        for strat in original:
            if strat['name'] not in current_strat_names or strat not in current:
                removed_in_regime.append(strat)
        
        if removed_in_regime:
            print(f"\n{regime}:")
            for strat in removed_in_regime:
                print(f"  - {strat['name']} with params: {strat['params']}")
    
    # Performance impact analysis
    print("\n\nPerformance Impact Analysis:")
    print("-" * 50)
    print("The high-performing ensemble (v1_9c2c22c9) likely used the full strategy set")
    print("including bollinger_breakout and keltner_breakout strategies.")
    print("\nThese strategies were removed for cost optimization but may have provided:")
    print("  1. Better volatility regime adaptation (especially in high volatility)")
    print("  2. More diverse signal generation reducing overfitting to specific patterns")
    print("  3. Better breakout detection in trending markets")
    print("\nRecommendation: Consider re-adding these strategies with adjusted parameters")
    print("or create a 'performance' vs 'cost-optimized' configuration option.")

if __name__ == "__main__":
    compare_configurations()
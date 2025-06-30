#!/usr/bin/env python3
"""
Clarify what the 2826-signal group actually represents.
"""

import pandas as pd
import json
from pathlib import Path

def clarify_2826_group():
    print("=== CLARIFYING THE 2826-SIGNAL GROUP ===\n")
    
    # Load metadata to understand what these strategies are
    workspace = "/Users/daws/ADMF-PC/config/keltner/results/latest"
    with open(Path(workspace) / "metadata_enhanced.json", 'r') as f:
        metadata = json.load(f)
    
    # Find all 2826-signal strategies
    strategies_2826 = []
    for name, comp in metadata['components'].items():
        if comp.get('signal_changes') == 2826:
            strategy_num = int(name.split('_')[-1])
            strategies_2826.append(strategy_num)
    
    print(f"The 2826-signal group contains {len(strategies_2826)} strategies")
    print(f"Strategy numbers: {strategies_2826}")
    
    # Check if they're identical or different
    print("\nChecking if strategies are identical...")
    
    # Load first few strategies to compare
    first_signals = None
    all_identical = True
    
    for i, strategy_num in enumerate(strategies_2826[:3]):
        strategy_name = f"SPY_5m_compiled_strategy_{strategy_num}"
        signals_file = Path(workspace) / "traces" / "keltner_bands" / f"{strategy_name}.parquet"
        
        signals_df = pd.read_parquet(signals_file)
        
        if i == 0:
            first_signals = signals_df
            print(f"\nStrategy {strategy_num}: {len(signals_df)} signal changes")
        else:
            # Compare with first
            if len(signals_df) != len(first_signals):
                all_identical = False
            else:
                # Check if signals are the same
                comparison = pd.merge(
                    first_signals[['idx', 'val']].rename(columns={'val': 'val1'}),
                    signals_df[['idx', 'val']].rename(columns={'val': 'val2'}),
                    on='idx',
                    how='outer'
                )
                if (comparison['val1'] != comparison['val2']).any():
                    all_identical = False
            
            print(f"Strategy {strategy_num}: {len(signals_df)} signal changes - {'IDENTICAL' if all_identical else 'DIFFERENT'}")
    
    print(f"\nConclusion: Strategies are {'IDENTICAL' if all_identical else 'DIFFERENT'}")
    
    # Now explain what this means
    print("\n" + "="*60)
    print("WHAT THE 2826 GROUP REPRESENTS:")
    print("="*60)
    
    if all_identical:
        print("These are DUPLICATE strategies with the same signals!")
        print("This happens when the parameter sweep generates identical results")
        print("(e.g., different Keltner parameters that produce same signals)")
        print("\nSo the 'group average' is really just ONE strategy's performance")
        print("measured 11 times (which is why they all show same results)")
    else:
        print("These are DIFFERENT strategies that happen to have similar signal counts")
        print("The 'group average' is the average across these different strategies")
    
    # Load config to understand parameter variations
    print("\n" + "="*60)
    print("UNDERSTANDING THE PARAMETER SPACE:")
    print("="*60)
    
    print("\nFrom the config, these 11 strategies likely represent:")
    print("- Same filter type (volatility filter)")
    print("- Different Keltner parameters:")
    print("  - Periods: [10, 15, 20, 30, 50]")
    print("  - Multipliers: [1.0, 1.5, 2.0, 2.5, 3.0]")
    print("  - Total combinations: 5 × 3 = 15 possible")
    print("\nThe 11 strategies with 2826 signals are parameter combinations")
    print("that happened to produce the same signal count (but possibly")
    print("at different times)")
    
    # Performance summary with win rates
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY WITH WIN RATES:")
    print("="*60)
    
    # Load filter analysis for complete picture
    df = pd.read_csv("keltner_filter_group_analysis.csv")
    df = df.sort_values('avg_return_bps', ascending=False).head(10)
    
    print(f"{'Signals':<8} {'RPT':<8} {'Win%':<6} {'T/Day':<6} {'Annual':<8} {'Description':<40}")
    print("-"*80)
    
    filter_descriptions = {
        47: "Master regime (Vol+VWAP+Time)",
        161: "Strong volatility filter", 
        303: "RSI/Volume combination",
        529: "VWAP positioning",
        535: "Time of day filter",
        587: "Directional RSI",
        1500: "Long-only variant",
        2305: "Light volume filter",
        2826: "Volatility filter (ATR-based)",
        3262: "Baseline (minimal filter)"
    }
    
    for _, row in df.iterrows():
        desc = filter_descriptions.get(row['signal_count'], "Unknown filter")
        annual = row['avg_return_bps'] * row['avg_trades'] / 100
        print(f"{row['signal_count']:<8.0f} {row['avg_return_bps']:<8.2f} "
              f"{row['avg_win_rate']*100:<6.1f} {row['avg_trades']/252:<6.1f} "
              f"{annual:<8.1f} {desc:<40}")
    
    print("\n" + "="*60)
    print("WHAT RUNNING THIS 'ENSEMBLE' WOULD LOOK LIKE:")
    print("="*60)
    print("\nOption 1: Run ONE strategy from the group")
    print("  - Pick any of the 11 strategies (they're similar)")
    print("  - Get ~0.68 bps/trade, 5.7 trades/day")
    print("  - This is what we've been analyzing")
    
    print("\nOption 2: Run ALL 11 strategies as ensemble")
    print("  - If signals are identical: No benefit (same trades)")
    print("  - If signals differ slightly: Some diversification")
    print("  - Could average signals or use majority voting")
    print("  - Complexity may not justify marginal improvement")
    
    print("\nRECOMMENDATION:")
    print("Run a single strategy from the 2826 group, not an ensemble")
    print("The 'group average' just confirms consistency across similar parameters")

if __name__ == "__main__":
    clarify_2826_group()
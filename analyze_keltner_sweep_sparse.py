#!/usr/bin/env python3
"""Analyze Keltner Bands multiplier sweep using sparse trace analysis."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from glob import glob

def analyze_multiplier_sweep(workspace_path: str):
    """Analyze results from multiplier sweep optimization."""
    
    workspace = Path(workspace_path)
    if not workspace.exists():
        print(f"Workspace not found: {workspace_path}")
        return
    
    print(f"Analyzing workspace: {workspace}")
    
    # Find all signal files
    signal_pattern = str(workspace / "traces/SPY_1m/signals/keltner_bands/*.parquet")
    signal_files = sorted(glob(signal_pattern))
    
    print(f"Found {len(signal_files)} signal files")
    
    if not signal_files:
        print("No signal files found!")
        return
    
    # Collect results
    results = []
    
    # Process each strategy
    for signal_file in signal_files:
        file_path = Path(signal_file)
        strategy_name = file_path.stem  # e.g., SPY_compiled_strategy_0
        
        try:
            # Load sparse signals
            signals_df = pd.read_parquet(signal_file)
            
            if signals_df.empty:
                continue
            
            # Extract strategy ID from filename
            strategy_id = int(strategy_name.split('_')[-1])
            
            # Count trades (non-zero signals)
            total_signals = len(signals_df[signals_df['val'] != 0])
            
            if total_signals == 0:
                continue
            
            # Calculate time span
            first_ts = pd.to_datetime(signals_df['ts'].iloc[0])
            last_ts = pd.to_datetime(signals_df['ts'].iloc[-1])
            trading_days = (last_ts - first_ts).days or 1
            
            # Trades per day
            trades_per_day = total_signals / trading_days * 252 / 365
            
            # Calculate returns for each trade
            trade_returns = []
            entry_price = None
            entry_signal = None
            
            for _, row in signals_df.iterrows():
                signal = row['val']
                price = row['px']
                
                if signal != 0 and entry_price is None:
                    # Opening position
                    entry_price = price
                    entry_signal = signal
                elif entry_price is not None and (signal == 0 or signal == -entry_signal):
                    # Closing position
                    # Log return calculation
                    log_return = np.log(price / entry_price) * entry_signal
                    # Apply 2bp round trip cost
                    log_return *= 0.9998
                    trade_returns.append(log_return)
                    
                    # Reset for next trade
                    if signal != 0:
                        # Immediate reversal
                        entry_price = price
                        entry_signal = signal
                    else:
                        entry_price = None
                        entry_signal = None
            
            if not trade_returns:
                continue
            
            # Calculate metrics
            trade_returns_bps = [r * 10000 for r in trade_returns]
            wins = [r for r in trade_returns_bps if r > 0]
            losses = [r for r in trade_returns_bps if r <= 0]
            
            win_rate = len(wins) / len(trade_returns_bps) * 100 if trade_returns_bps else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            edge_bps = np.mean(trade_returns_bps) if trade_returns_bps else 0
            total_return_bps = np.sum(trade_returns_bps)
            
            # Extract parameters from filename or use defaults
            # Format: SPY_compiled_strategy_X
            # We'll need to map strategy ID to parameters later
            
            result = {
                'strategy': strategy_id,
                'strategy_name': strategy_name,
                'edge_bps': edge_bps,
                'trades_per_day': trades_per_day,
                'total_trades': total_signals,
                'win_rate': win_rate,
                'avg_win_bps': avg_win,
                'avg_loss_bps': avg_loss,
                'total_return_bps': total_return_bps,
                'annual_trades': trades_per_day * 252
            }
            
            # Calculate expected annual return
            result['expected_annual_return'] = (
                result['edge_bps'] * result['annual_trades'] / 10000
            )
            
            results.append(result)
            
        except Exception as e:
            print(f"Error analyzing {strategy_name}: {e}")
            continue
    
    if not results:
        print("\nNo results to analyze!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Map strategy IDs to parameters based on the pattern from config
    # Strategies 0-40: period=50, multiplier varies
    # Strategies 41-59: period=45, multiplier varies  
    # Strategies 60-78: period=40, multiplier varies
    
    # Multipliers for period=50 (0-40)
    mult_50 = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
               1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45,
               1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95,
               2.00, 2.10, 2.20, 2.30, 2.40, 2.50, 2.60, 2.70, 2.80, 2.90, 3.00]
    
    # Multipliers for period=45 and 40
    mult_other = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60,
                  1.70, 1.80, 1.90, 2.00, 2.20, 2.40, 2.60, 2.80, 3.00]
    
    # Map parameters
    for idx, row in df.iterrows():
        strat_id = row['strategy']
        if strat_id < 41:
            df.at[idx, 'period'] = 50
            df.at[idx, 'multiplier'] = mult_50[strat_id] if strat_id < len(mult_50) else 'unknown'
        elif strat_id < 60:
            df.at[idx, 'period'] = 45
            df.at[idx, 'multiplier'] = mult_other[strat_id - 41] if (strat_id - 41) < len(mult_other) else 'unknown'
        else:
            df.at[idx, 'period'] = 40
            df.at[idx, 'multiplier'] = mult_other[strat_id - 60] if (strat_id - 60) < len(mult_other) else 'unknown'
    
    # Sort by period and multiplier
    df = df.sort_values(['period', 'multiplier'])
    
    print("\n=== KELTNER MULTIPLIER SWEEP ANALYSIS ===\n")
    print(f"Total strategies analyzed: {len(df)}")
    
    # Analyze by period
    for period in sorted(df['period'].unique()):
        period_df = df[df['period'] == period].copy()
        
        print(f"\n--- Period {period} ---")
        print(f"Multipliers tested: {len(period_df)}")
        
        # Find strategies meeting criteria (>1 bps edge)
        good_strategies = period_df[period_df['edge_bps'] > 1.0]
        
        if not good_strategies.empty:
            print(f"\nStrategies with >1 bps edge:")
            for _, row in good_strategies.iterrows():
                print(f"  Multiplier {row['multiplier']}: "
                      f"{row['edge_bps']:.2f} bps, "
                      f"{row['trades_per_day']:.1f} trades/day, "
                      f"{row['annual_trades']:.0f} trades/year, "
                      f"Win rate: {row['win_rate']:.1f}%")
        else:
            print("  No strategies with >1 bps edge")
        
        # Find optimal trade-off point
        viable = period_df[
            (period_df['annual_trades'] > 100) & 
            (period_df['edge_bps'] > 0)
        ]
        
        if not viable.empty:
            # Sort by expected annual return
            viable = viable.sort_values('expected_annual_return', ascending=False)
            best = viable.iloc[0]
            
            print(f"\nBest trade-off (>100 trades/year):")
            print(f"  Multiplier: {best['multiplier']}")
            print(f"  Edge: {best['edge_bps']:.2f} bps")
            print(f"  Trades/day: {best['trades_per_day']:.1f}")
            print(f"  Annual trades: {best['annual_trades']:.0f}")
            print(f"  Expected annual return: {best['expected_annual_return']:.2%}")
            print(f"  Win rate: {best['win_rate']:.1f}%")
    
    # Overall analysis
    print("\n\n=== TOP PERFORMERS ===\n")
    
    # Show top 10 by expected return
    df_sorted = df.sort_values('expected_annual_return', ascending=False)
    print("Top 10 strategies by expected annual return:")
    print("Period | Mult  | Edge(bps) | Trades/yr | E[Return] | Win%")
    print("-------|-------|-----------|-----------|-----------|-----")
    for _, row in df_sorted.head(10).iterrows():
        print(f"{row['period']:6.0f} | {row['multiplier']:5.2f} | {row['edge_bps']:9.2f} | "
              f"{row['annual_trades']:9.0f} | {row['expected_annual_return']:9.2%} | "
              f"{row['win_rate']:4.1f}%")
    
    # Find the efficient frontier
    print("\n\n=== EFFICIENT FRONTIER ===")
    print("(Best return for given trade frequency)\n")
    
    # Bin by trade frequency
    bins = [0, 50, 100, 200, 500, 1000, 5000, 10000]
    labels = ['<50', '50-100', '100-200', '200-500', '500-1k', '1k-5k', '5k+']
    df['freq_bin'] = pd.cut(df['annual_trades'], bins=bins, labels=labels)
    
    for freq_bin in labels:
        bin_data = df[df['freq_bin'] == freq_bin]
        if not bin_data.empty:
            best = bin_data.loc[bin_data['expected_annual_return'].idxmax()]
            print(f"\n{freq_bin} trades/year:")
            print(f"  Best: Period={best['period']:.0f}, Mult={best['multiplier']:.2f}")
            print(f"  Edge: {best['edge_bps']:.2f} bps")
            print(f"  Expected return: {best['expected_annual_return']:.2%}")
            print(f"  Win rate: {best['win_rate']:.1f}%")
    
    # Key insights
    print("\n\n=== KEY INSIGHTS ===\n")
    
    # Find strategies with good balance of edge and frequency
    balanced = df[(df['edge_bps'] > 0.5) & (df['annual_trades'] > 200)]
    if not balanced.empty:
        print(f"Strategies with >0.5 bps edge AND >200 trades/year: {len(balanced)}")
        best_balanced = balanced.loc[balanced['expected_annual_return'].idxmax()]
        print(f"Best balanced strategy:")
        print(f"  Period={best_balanced['period']:.0f}, Mult={best_balanced['multiplier']:.2f}")
        print(f"  Edge: {best_balanced['edge_bps']:.2f} bps")
        print(f"  Annual trades: {best_balanced['annual_trades']:.0f}")
        print(f"  Expected return: {best_balanced['expected_annual_return']:.2%}")
    
    # Save detailed results
    output_file = "keltner_multiplier_sweep_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n\nDetailed results saved to: {output_file}")
    
    # Create summary report
    summary_file = "KELTNER_MULTIPLIER_ANALYSIS.md"
    with open(summary_file, 'w') as f:
        f.write("# Keltner Bands Multiplier Sweep Analysis\n\n")
        f.write(f"Workspace: {workspace}\n")
        f.write(f"Total strategies tested: {len(df)}\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write(f"- Strategies with positive edge: {len(df[df['edge_bps'] > 0])}\n")
        f.write(f"- Strategies with >1 bps edge: {len(df[df['edge_bps'] > 1])}\n")
        f.write(f"- Average edge across all strategies: {df['edge_bps'].mean():.2f} bps\n")
        f.write(f"- Best edge: {df['edge_bps'].max():.2f} bps\n")
        f.write(f"- Most frequent trader: {df['trades_per_day'].max():.1f} trades/day\n\n")
        
        f.write("## Optimal Strategies\n\n")
        if not df_sorted.empty:
            best = df_sorted.iloc[0]
            f.write(f"### Best Expected Return\n")
            f.write(f"- Period: {best['period']:.0f}\n")
            f.write(f"- Multiplier: {best['multiplier']:.2f}\n")
            f.write(f"- Edge: {best['edge_bps']:.2f} bps\n")
            f.write(f"- Trades/year: {best['annual_trades']:.0f}\n")
            f.write(f"- Expected return: {best['expected_annual_return']:.2%}\n\n")
    
    print(f"Summary report saved to: {summary_file}")

if __name__ == "__main__":
    import sys
    workspace = sys.argv[1] if len(sys.argv) > 1 else "workspaces/signal_generation_3f2b1535"
    analyze_multiplier_sweep(workspace)
#!/usr/bin/env python3
"""
Analyze the 1000-bar parameter expansion results.
"""

import json
from pathlib import Path
import pandas as pd
from src.analytics.signal_reconstruction import SignalReconstructor

def analyze_strategy_signals(signal_file: str, market_data: str):
    """Analyze a single strategy's signal performance."""
    
    # Load signal metadata
    with open(signal_file, 'r') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    
    # Extract strategy name and parameters
    file_path = Path(signal_file)
    strategy_name = file_path.stem.replace('signals_strategy_SPY_', '').split('_2025')[0]
    
    # Use signal reconstructor to get trades
    reconstructor = SignalReconstructor(signal_file, market_data)
    trades = reconstructor.extract_trades()
    
    # Calculate performance metrics
    total_gross = sum(trade.pnl for trade in trades)
    total_trades = len(trades)
    
    # Simple execution costs: $0.01 spread + $0.005 commission per share
    execution_cost_per_trade = 0.01 + (0.005 * 2)  # Round trip
    total_costs = execution_cost_per_trade * total_trades
    net_pnl = total_gross - total_costs
    
    # Calculate signal statistics
    total_bars = metadata['total_bars']
    total_changes = metadata['total_changes']
    compression_ratio = metadata['compression_ratio']
    
    return {
        'strategy': strategy_name,
        'strategy_type': strategy_name.split('_')[0],
        'total_bars': total_bars,
        'signal_changes': total_changes,
        'compression_ratio': compression_ratio,
        'trades': total_trades,
        'gross_pnl': total_gross,
        'total_costs': total_costs,
        'net_pnl': net_pnl,
        'avg_pnl_per_trade': net_pnl / total_trades if total_trades > 0 else 0,
        'signal_frequency': total_changes / total_bars if total_bars > 0 else 0,
        'profitable': net_pnl > 0
    }

def main():
    workspace = "workspaces/tmp/20250611_185728"
    market_data = "data/SPY_1m.csv"
    
    # Get all signal files
    signal_files = sorted(Path(workspace).glob("signals_strategy_*.json"))
    
    print("="*120)
    print("1000-BAR PARAMETER EXPANSION ANALYSIS")
    print("="*120)
    print(f"\nAnalyzing {len(signal_files)} strategies from {workspace}")
    print("Execution cost assumptions: $0.01 spread + $0.01 commission = $0.02 per round-trip trade\n")
    
    # Analyze all strategies
    results = []
    for signal_file in signal_files:
        try:
            result = analyze_strategy_signals(str(signal_file), market_data)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {signal_file}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Group by strategy type for analysis
    print("STRATEGY TYPE SUMMARY:")
    print("-" * 60)
    for strategy_type in df['strategy_type'].unique():
        type_df = df[df['strategy_type'] == strategy_type]
        profitable_count = type_df['profitable'].sum()
        total_count = len(type_df)
        avg_net = type_df['net_pnl'].mean()
        avg_trades = type_df['trades'].mean()
        avg_signal_freq = type_df['signal_frequency'].mean()
        
        print(f"{strategy_type:15} | {total_count:2d} variants | {profitable_count:2d}/{total_count:2d} profitable | "
              f"Avg Net: ${avg_net:7.3f} | Avg Trades: {avg_trades:5.1f} | Signal Freq: {avg_signal_freq:.3f}")
    
    # Top and bottom performers
    df_sorted = df.sort_values('net_pnl', ascending=False)
    
    print(f"\nTOP 10 PERFORMERS:")
    print("-" * 100)
    print(f"{'Strategy':35} | {'Bars':>5} | {'Trades':>6} | {'Gross':>8} | {'Net':>8} | {'Per Trade':>9}")
    print("-" * 100)
    
    for _, row in df_sorted.head(10).iterrows():
        print(f"{row['strategy']:35} | {row['total_bars']:5d} | {row['trades']:6d} | "
              f"${row['gross_pnl']:7.3f} | ${row['net_pnl']:7.3f} | ${row['avg_pnl_per_trade']:8.4f}")
    
    print(f"\nBOTTOM 10 PERFORMERS:")
    print("-" * 100)
    print(f"{'Strategy':35} | {'Bars':>5} | {'Trades':>6} | {'Gross':>8} | {'Net':>8} | {'Per Trade':>9}")
    print("-" * 100)
    
    for _, row in df_sorted.tail(10).iterrows():
        print(f"{row['strategy']:35} | {row['total_bars']:5d} | {row['trades']:6d} | "
              f"${row['gross_pnl']:7.3f} | ${row['net_pnl']:7.3f} | ${row['avg_pnl_per_trade']:8.4f}")
    
    # Overall statistics
    print(f"\nOVERALL STATISTICS:")
    print("=" * 60)
    profitable_count = df['profitable'].sum()
    total_count = len(df)
    
    print(f"Total strategies analyzed: {total_count}")
    print(f"Profitable strategies: {profitable_count} ({profitable_count/total_count*100:.1f}%)")
    print(f"Average bars processed: {df['total_bars'].mean():.0f}")
    print(f"Average trades per strategy: {df['trades'].mean():.1f}")
    print(f"Average signal frequency: {df['signal_frequency'].mean():.3f}")
    print(f"Average compression ratio: {df['compression_ratio'].mean():.3f}")
    print(f"Total gross P&L: ${df['gross_pnl'].sum():.3f}")
    print(f"Total execution costs: ${df['total_costs'].sum():.3f}")
    print(f"Total net P&L: ${df['net_pnl'].sum():.3f}")
    
    # Signal generation efficiency
    print(f"\nSIGNAL GENERATION EFFICIENCY:")
    print("-" * 60)
    print(f"Total bars processed across all strategies: {df['total_bars'].sum():,}")
    print(f"Total signal changes generated: {df['signal_changes'].sum():,}")
    print(f"Average compression (sparse storage): {(1 - df['compression_ratio'].mean())*100:.1f}%")

if __name__ == "__main__":
    main()
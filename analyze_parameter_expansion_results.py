#!/usr/bin/env python3
"""
Analyze all parameter expansion results with realistic execution costs.
"""

from pathlib import Path
from src.analytics.signal_reconstruction import SignalReconstructor
from src.execution.synchronous.models.slippage import PercentageSlippageModel
from src.execution.synchronous.models.commission import PerShareCommissionModel
from src.execution.types import Order, OrderSide, OrderType
from decimal import Decimal
import pandas as pd

def analyze_strategy(signal_file: str, market_data: str):
    """Analyze a single strategy with execution costs."""
    
    # Initialize models
    commission_model = PerShareCommissionModel(rate_per_share=0.005)
    
    # Load and reconstruct
    reconstructor = SignalReconstructor(signal_file, market_data)
    trades = reconstructor.extract_trades()
    
    if not trades:
        return None
    
    # Calculate costs
    total_gross = 0
    total_spread = 0
    total_commission = 0
    
    for trade in trades:
        # Gross P&L
        total_gross += trade.pnl
        
        # Spread cost (1 cent per round trip)
        total_spread += 0.01
        
        # Commission (2x for round trip)
        total_commission += 0.01
    
    total_costs = total_spread + total_commission
    total_net = total_gross - total_costs
    
    # Extract strategy name
    strategy_name = Path(signal_file).stem.replace('signals_strategy_SPY_', '').replace(f'_{Path(signal_file).stem.split("_")[-1]}', '')
    
    return {
        'strategy': strategy_name,
        'trades': len(trades),
        'gross_pnl': total_gross,
        'spread_cost': total_spread,
        'commission': total_commission,
        'total_costs': total_costs,
        'net_pnl': total_net,
        'avg_gross_per_trade': total_gross / len(trades) if trades else 0,
        'avg_net_per_trade': total_net / len(trades) if trades else 0,
        'cost_per_trade': total_costs / len(trades) if trades else 0,
        'profitable': total_net > 0
    }

def main():
    workspace = "workspaces/tmp/20250611_185728"
    market_data = "data/SPY_1m.csv"
    
    # Get all signal files
    signal_files = sorted(Path(workspace).glob("signals_strategy_*.json"))
    
    print("="*100)
    print("PARAMETER EXPANSION ANALYSIS - REALISTIC EXECUTION COSTS")
    print("="*100)
    print("\nExecution assumptions: $0.01 spread + $0.005/share commission = $0.02 per round-trip trade\n")
    
    # Analyze all strategies
    results = []
    for signal_file in signal_files:
        result = analyze_strategy(str(signal_file), market_data)
        if result:
            results.append(result)
    
    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(results)
    
    # Sort by net P&L
    df = df.sort_values('net_pnl', ascending=False)
    
    # Display results
    print("Strategy Performance Ranking (sorted by Net P&L):")
    print("-"*100)
    print(f"{'Strategy':15} | {'Trades':>6} | {'Gross P&L':>10} | {'Costs':>8} | {'Net P&L':>10} | {'Net/Trade':>10} | {'Status':>10}")
    print("-"*100)
    
    for _, row in df.iterrows():
        status = "✓ Profitable" if row['profitable'] else "✗ Loss"
        print(f"{row['strategy']:15} | {row['trades']:6d} | ${row['gross_pnl']:9.4f} | ${row['total_costs']:7.4f} | "
              f"${row['net_pnl']:9.4f} | ${row['avg_net_per_trade']:9.4f} | {status:>10}")
    
    # Summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    profitable_count = df['profitable'].sum()
    total_count = len(df)
    
    print(f"\nProfitable strategies: {profitable_count}/{total_count} ({profitable_count/total_count*100:.1f}%)")
    
    if profitable_count > 0:
        print(f"\nTop 3 Performers:")
        for i, row in df.head(3).iterrows():
            print(f"  {i+1}. {row['strategy']:15} - Net P&L: ${row['net_pnl']:.4f} ({row['trades']} trades)")
    
    # Parameter insights
    print(f"\nParameter Insights:")
    
    # Group by fast period
    for fast in [5, 10, 20]:
        strategies = df[df['strategy'].str.startswith(f'ma_crossover_{fast}_')]
        if not strategies.empty:
            avg_net = strategies['net_pnl'].mean()
            profitable_pct = (strategies['profitable'].sum() / len(strategies)) * 100
            print(f"  Fast MA {fast:2d}: Avg Net P&L ${avg_net:7.4f}, {profitable_pct:5.1f}% profitable")
    
    # Group by slow period  
    print(f"\n  By Slow Period:")
    for slow in [20, 30, 50]:
        strategies = df[df['strategy'].str.endswith(f'_{slow}')]
        if not strategies.empty:
            avg_net = strategies['net_pnl'].mean()
            profitable_pct = (strategies['profitable'].sum() / len(strategies)) * 100
            print(f"  Slow MA {slow:2d}: Avg Net P&L ${avg_net:7.4f}, {profitable_pct:5.1f}% profitable")
    
    # Invalid combinations
    invalid = df[df['strategy'] == 'ma_crossover_20_20']
    if not invalid.empty:
        print(f"\n⚠️  Note: ma_crossover_20_20 has identical fast/slow periods (invalid crossover)")

if __name__ == "__main__":
    main()
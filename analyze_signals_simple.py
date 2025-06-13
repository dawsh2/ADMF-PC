#!/usr/bin/env python3
"""
Analyze signals with simple 1-share-per-trade methodology.
"""

from src.analytics.signal_reconstruction import SignalReconstructor
import json
from pathlib import Path

def analyze_with_simple_metrics(signal_file: str, market_data: str):
    """Analyze signals using 1 share per trade for clarity."""
    
    reconstructor = SignalReconstructor(signal_file, market_data)
    trades = reconstructor.extract_trades()
    
    # Load metadata
    with open(signal_file, 'r') as f:
        data = json.load(f)
    metadata = data['metadata']
    
    print(f"\nStrategy: {Path(signal_file).stem}")
    print("=" * 60)
    
    # Trade details
    print("\nTrade-by-Trade Analysis (1 share per trade):")
    print("-" * 60)
    
    total_pnl = 0
    total_points = 0
    winners = []
    losers = []
    
    for i, trade in enumerate(trades):
        points = trade.pnl  # This is already the dollar P&L for 1 share
        total_pnl += points
        total_points += abs(points)
        
        if trade.is_winner:
            winners.append(points)
        else:
            losers.append(points)
            
        print(f"Trade {i+1}: {trade.direction:5s} | "
              f"Bars: {trade.bars_held:3d} | "
              f"Entry: ${trade.entry_price:6.2f} | "
              f"Exit: ${trade.exit_price:6.2f} | "
              f"P&L: ${points:7.4f}")
    
    # Summary metrics
    print(f"\nSummary Metrics:")
    print("-" * 60)
    print(f"Total trades: {len(trades)}")
    print(f"Winners: {len(winners)} ({len(winners)/len(trades)*100:.1f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/len(trades)*100:.1f}%)")
    print(f"\nDollar Performance (1 share):")
    print(f"  Total P&L: ${total_pnl:.4f}")
    print(f"  Average trade: ${total_pnl/len(trades):.4f}")
    print(f"  Average winner: ${sum(winners)/len(winners):.4f}" if winners else "  Average winner: N/A")
    print(f"  Average loser: ${sum(losers)/len(losers):.4f}" if losers else "  Average loser: N/A")
    print(f"  Largest win: ${max(winners):.4f}" if winners else "  Largest win: N/A")
    print(f"  Largest loss: ${min(losers):.4f}" if losers else "  Largest loss: N/A")
    
    # Risk metrics
    if losers and winners:
        avg_win = sum(winners)/len(winners)
        avg_loss = abs(sum(losers)/len(losers))
        profit_factor = sum(winners) / abs(sum(losers))
        expectancy = (len(winners)/len(trades) * avg_win) - (len(losers)/len(trades) * avg_loss)
        
        print(f"\nRisk Metrics:")
        print(f"  Profit factor: {profit_factor:.2f}")
        print(f"  Win/Loss ratio: {avg_win/avg_loss:.2f}")
        print(f"  Expectancy: ${expectancy:.4f} per trade")
    
    # Signal efficiency
    total_bars = metadata['total_bars']
    bars_in_position = sum(t.bars_held for t in trades)
    
    print(f"\nSignal Efficiency:")
    print(f"  Total bars analyzed: {total_bars}")
    print(f"  Bars in position: {bars_in_position} ({bars_in_position/total_bars*100:.1f}%)")
    print(f"  Signal changes: {metadata['total_changes']}")
    print(f"  Storage compression: {metadata.get('compression_ratio', 0)*100:.1f}%")
    
    # Points per bar
    if bars_in_position > 0:
        print(f"  Points per bar in position: ${total_pnl/bars_in_position:.4f}")
    
    return {
        'strategy': Path(signal_file).stem,
        'trades': len(trades),
        'win_rate': len(winners)/len(trades),
        'total_pnl': total_pnl,
        'avg_trade': total_pnl/len(trades),
        'profit_factor': sum(winners)/abs(sum(losers)) if losers else float('inf'),
        'bars_in_position_pct': bars_in_position/total_bars*100
    }

def main():
    workspace = "workspaces/tmp/20250611_171158"
    market_data = "data/SPY_1m.csv"
    
    # Analyze both strategies
    signal_files = list(Path(workspace).glob("signals_strategy_*.json"))
    
    print("\n" + "="*80)
    print("SIGNAL PERFORMANCE ANALYSIS - 1 Share Per Trade")
    print("="*80)
    
    results = []
    for signal_file in sorted(signal_files):
        result = analyze_with_simple_metrics(str(signal_file), market_data)
        results.append(result)
    
    # Comparison
    if len(results) > 1:
        print("\n" + "="*80)
        print("STRATEGY COMPARISON")
        print("="*80)
        print("\nKey Differences:")
        
        for i, r in enumerate(results):
            print(f"\n{r['strategy']}:")
            print(f"  - {r['trades']} trades over {r['bars_in_position_pct']:.1f}% of time")
            print(f"  - ${r['total_pnl']:.4f} total P&L (${r['avg_trade']:.4f} per trade)")
            print(f"  - {r['win_rate']*100:.1f}% win rate with {r['profit_factor']:.2f} profit factor")

if __name__ == "__main__":
    main()
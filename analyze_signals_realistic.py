#!/usr/bin/env python3
"""
Analyze signals with realistic execution assumptions including slippage and costs.
"""

from src.analytics.signal_reconstruction import SignalReconstructor
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class ExecutionAssumptions:
    """Realistic execution assumptions for SPY."""
    # Slippage assumptions
    market_order_slippage_bps: float = 1.0  # 1 basis point for liquid SPY
    aggressive_fill_slippage_bps: float = 2.0  # When crossing spread aggressively
    
    # Transaction costs
    commission_per_share: float = 0.005  # $0.005 per share (common retail)
    sec_fee_per_dollar: float = 0.0000278  # SEC fee ~$27.80 per $1M
    
    # Market microstructure
    typical_spread_cents: float = 1.0  # SPY typically 1 cent spread
    
    def calculate_realistic_pnl(self, entry_price: float, exit_price: float, 
                              direction: str, shares: int = 1,
                              execution_style: str = 'market') -> Dict[str, float]:
        """Calculate P&L with realistic execution assumptions."""
        
        # Base P&L (no costs)
        if direction == 'long':
            gross_pnl = (exit_price - entry_price) * shares
        else:  # short
            gross_pnl = (entry_price - exit_price) * shares
        
        # Slippage based on execution style
        if execution_style == 'market':
            slippage_bps = self.market_order_slippage_bps
        else:  # aggressive
            slippage_bps = self.aggressive_fill_slippage_bps
            
        # Calculate slippage cost (paid on both entry and exit)
        avg_price = (entry_price + exit_price) / 2
        slippage_cost = avg_price * (slippage_bps / 10000) * 2 * shares  # 2x for round trip
        
        # Commission costs
        commission_cost = self.commission_per_share * shares * 2  # 2x for round trip
        
        # SEC fees (on sell side only)
        sec_fee = exit_price * shares * self.sec_fee_per_dollar
        
        # Half-spread cost (assume we cross half the spread on each side)
        spread_cost = (self.typical_spread_cents / 100) * shares  # Convert cents to dollars
        
        # Total costs
        total_costs = slippage_cost + commission_cost + sec_fee + spread_cost
        
        # Net P&L
        net_pnl = gross_pnl - total_costs
        
        return {
            'gross_pnl': gross_pnl,
            'slippage_cost': slippage_cost,
            'commission_cost': commission_cost,
            'sec_fee': sec_fee,
            'spread_cost': spread_cost,
            'total_costs': total_costs,
            'net_pnl': net_pnl,
            'cost_percentage': (total_costs / (avg_price * shares)) * 100
        }

def analyze_with_execution_costs(signal_file: str, market_data: str, 
                               assumptions: ExecutionAssumptions):
    """Analyze signals with realistic execution costs."""
    
    reconstructor = SignalReconstructor(signal_file, market_data)
    trades = reconstructor.extract_trades()
    
    # Load metadata
    with open(signal_file, 'r') as f:
        data = json.load(f)
    metadata = data['metadata']
    
    print(f"\nStrategy: {Path(signal_file).stem}")
    print("=" * 80)
    
    # Analyze each trade with costs
    print("\nTrade Analysis with Execution Costs (1 share per trade):")
    print("-" * 80)
    print("Trade | Dir   | Bars | Entry    | Exit     | Gross P&L | Costs  | Net P&L | Impact")
    print("-" * 80)
    
    total_gross = 0
    total_net = 0
    total_costs = 0
    net_winners = []
    net_losers = []
    
    for i, trade in enumerate(trades):
        # Calculate with execution costs
        result = assumptions.calculate_realistic_pnl(
            trade.entry_price, trade.exit_price, trade.direction
        )
        
        total_gross += result['gross_pnl']
        total_net += result['net_pnl']
        total_costs += result['total_costs']
        
        if result['net_pnl'] > 0:
            net_winners.append(result['net_pnl'])
        else:
            net_losers.append(result['net_pnl'])
        
        # Display
        impact_pct = ((result['gross_pnl'] - result['net_pnl']) / abs(result['gross_pnl']) * 100 
                     if result['gross_pnl'] != 0 else 100)
        
        print(f"{i+1:5d} | {trade.direction:5s} | {trade.bars_held:4d} | "
              f"${trade.entry_price:7.2f} | ${trade.exit_price:7.2f} | "
              f"${result['gross_pnl']:9.4f} | ${result['total_costs']:6.4f} | "
              f"${result['net_pnl']:8.4f} | {impact_pct:5.1f}%")
    
    # Summary comparison
    print(f"\nExecution Cost Impact Summary:")
    print("-" * 80)
    print(f"Total Gross P&L: ${total_gross:.4f}")
    print(f"Total Costs:     ${total_costs:.4f}")
    print(f"Total Net P&L:   ${total_net:.4f}")
    print(f"Cost Impact:     {(total_costs/abs(total_gross)*100):.1f}% of gross P&L")
    
    # Win rate comparison
    gross_win_rate = len([t for t in trades if t.is_winner]) / len(trades) * 100
    net_win_rate = len(net_winners) / len(trades) * 100
    
    print(f"\nWin Rate Impact:")
    print(f"  Gross win rate: {gross_win_rate:.1f}%")
    print(f"  Net win rate:   {net_win_rate:.1f}%")
    print(f"  Degradation:    {gross_win_rate - net_win_rate:.1f} percentage points")
    
    # Profitability metrics after costs
    if net_winners and net_losers:
        net_profit_factor = sum(net_winners) / abs(sum(net_losers))
        net_expectancy = total_net / len(trades)
        
        print(f"\nNet Performance Metrics:")
        print(f"  Net profit factor: {net_profit_factor:.2f}")
        print(f"  Net expectancy:    ${net_expectancy:.4f} per trade")
        print(f"  Avg net winner:    ${sum(net_winners)/len(net_winners):.4f}")
        print(f"  Avg net loser:     ${sum(net_losers)/len(net_losers):.4f}")
    
    # Break-even analysis
    avg_cost_per_trade = total_costs / len(trades)
    avg_price = sum(t.entry_price for t in trades) / len(trades)
    breakeven_move = (avg_cost_per_trade / avg_price) * 100
    
    print(f"\nBreak-even Analysis:")
    print(f"  Average cost per trade: ${avg_cost_per_trade:.4f}")
    print(f"  Break-even move:        {breakeven_move:.3f}%")
    print(f"  Break-even points:      ${avg_cost_per_trade:.4f}")
    
    # Time efficiency
    total_bars = metadata['total_bars']
    bars_in_position = sum(t.bars_held for t in trades)
    if bars_in_position > 0:
        net_points_per_bar = total_net / bars_in_position
        print(f"\nTime Efficiency:")
        print(f"  Net P&L per bar in position: ${net_points_per_bar:.4f}")
    
    return {
        'strategy': Path(signal_file).stem,
        'trades': len(trades),
        'gross_pnl': total_gross,
        'total_costs': total_costs,
        'net_pnl': total_net,
        'cost_impact_pct': (total_costs/abs(total_gross)*100),
        'gross_win_rate': gross_win_rate,
        'net_win_rate': net_win_rate,
        'net_expectancy': total_net / len(trades),
        'breakeven_move_pct': breakeven_move
    }

def main():
    workspace = "workspaces/tmp/20250611_171158"
    market_data = "data/SPY_1m.csv"
    
    # Standard execution assumptions
    assumptions = ExecutionAssumptions()
    
    print("\n" + "="*80)
    print("SIGNAL PERFORMANCE WITH REALISTIC EXECUTION COSTS")
    print("="*80)
    print(f"\nExecution Assumptions:")
    print(f"  Market order slippage: {assumptions.market_order_slippage_bps} bps")
    print(f"  Commission: ${assumptions.commission_per_share}/share")
    print(f"  Typical spread: {assumptions.typical_spread_cents} cents")
    
    # Analyze both strategies
    signal_files = list(Path(workspace).glob("signals_strategy_*.json"))
    results = []
    
    for signal_file in sorted(signal_files):
        result = analyze_with_execution_costs(str(signal_file), market_data, assumptions)
        results.append(result)
    
    # Final comparison
    if len(results) > 1:
        print("\n" + "="*80)
        print("STRATEGY COMPARISON - NET PERFORMANCE")
        print("="*80)
        
        # Create comparison table
        print("\n{:<50} | {:>12} | {:>12} | {:>10}".format(
            "Strategy", "Gross P&L", "Net P&L", "Cost Impact"
        ))
        print("-" * 90)
        
        for r in results:
            strategy_name = r['strategy'].replace('signals_strategy_', '')
            print("{:<50} | ${:>11.4f} | ${:>11.4f} | {:>9.1f}%".format(
                strategy_name, r['gross_pnl'], r['net_pnl'], r['cost_impact_pct']
            ))
        
        # Identify best strategy after costs
        best_net = max(results, key=lambda x: x['net_pnl'])
        best_efficiency = max(results, key=lambda x: x['net_expectancy'])
        
        print(f"\nBest Net P&L: {best_net['strategy'].replace('signals_strategy_', '')}")
        print(f"Best Expectancy: {best_efficiency['strategy'].replace('signals_strategy_', '')} "
              f"(${best_efficiency['net_expectancy']:.4f}/trade)")
        
        # Warning about strategies that don't survive costs
        for r in results:
            if r['net_pnl'] < 0 and r['gross_pnl'] > 0:
                print(f"\n⚠️  WARNING: {r['strategy']} is profitable gross but loses money after costs!")

if __name__ == "__main__":
    main()
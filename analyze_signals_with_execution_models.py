#!/usr/bin/env python3
"""
Analyze signals using the existing execution models from src/execution.
"""

from src.analytics.signal_reconstruction import SignalReconstructor
from src.execution.synchronous.models.slippage import PercentageSlippageModel, MarketConditions
from src.execution.synchronous.models.commission import PerShareCommissionModel
from src.execution.types import Order, OrderSide, OrderType
import json
from pathlib import Path
from decimal import Decimal

def analyze_with_execution_models(signal_file: str, market_data: str):
    """Analyze signals using proper execution models."""
    
    # Initialize execution models with realistic parameters for SPY
    # For 1 share of SPY, market impact is essentially zero
    # We only pay half the spread when crossing
    slippage_model = PercentageSlippageModel(
        base_slippage_pct=0.0,  # No market impact for 1 share
        volatility_multiplier=0.0,  # No volatility impact for 1 share
        volume_impact_factor=0.0  # No volume impact for 1 share
    )
    
    commission_model = PerShareCommissionModel(
        rate_per_share=0.005,  # $0.005 per share
        minimum_commission=0.0,  # No minimum for our 1-share trades
        maximum_commission=10.0
    )
    
    # Load signals
    reconstructor = SignalReconstructor(signal_file, market_data)
    trades = reconstructor.extract_trades()
    
    with open(signal_file, 'r') as f:
        data = json.load(f)
    metadata = data['metadata']
    
    print(f"\nStrategy: {Path(signal_file).stem}")
    print("=" * 80)
    print("\nExecution Model Parameters (1 share SPY):")
    print(f"  Market impact: None (1 share has no impact)")
    print(f"  Spread cost: $0.01/trade (1 cent spread)")
    print(f"  Commission: ${commission_model.rate_per_share}/share")
    
    # Analyze trades
    print("\nTrade Analysis with Execution Models:")
    print("-" * 80)
    print("Trade | Dir   | Entry    | Exit     | Gross | Spread | Comm | Net P&L")
    print("-" * 80)
    
    total_gross = 0
    total_slippage = 0
    total_commission = 0
    total_net = 0
    
    for i, trade in enumerate(trades):
        # Create mock orders for entry and exit
        entry_order = Order(
            order_id=f"entry_{i}",
            symbol=trade.symbol,
            side=OrderSide.BUY if trade.direction == 'long' else OrderSide.SELL,
            quantity=Decimal("1"),
            order_type=OrderType.MARKET,
            price=None  # Market order, no limit price
        )
        
        exit_order = Order(
            order_id=f"exit_{i}",
            symbol=trade.symbol,
            side=OrderSide.SELL if trade.direction == 'long' else OrderSide.BUY,
            quantity=Decimal("1"),
            order_type=OrderType.MARKET,
            price=None  # Market order, no limit price
        )
        
        # Calculate slippage-adjusted execution prices
        entry_slippage = slippage_model.calculate_slippage(entry_order, trade.entry_price, volume=100000)
        exit_slippage = slippage_model.calculate_slippage(exit_order, trade.exit_price, volume=100000)
        
        # Apply slippage to get actual execution prices
        # For SPY with 1 share, we only pay half the spread (no market impact)
        half_spread = 0.005  # Half of typical 1 cent spread
        
        # For market orders: buy at ask (higher), sell at bid (lower)
        if trade.direction == 'long':
            actual_entry_price = trade.entry_price + half_spread  # Buy at ask
            actual_exit_price = trade.exit_price - half_spread   # Sell at bid
        else:  # short
            actual_entry_price = trade.entry_price - half_spread  # Sell at bid
            actual_exit_price = trade.exit_price + half_spread   # Buy at ask
        
        # Calculate P&L with slippage-adjusted prices
        if trade.direction == 'long':
            pnl_after_slippage = actual_exit_price - actual_entry_price
        else:  # short
            pnl_after_slippage = actual_entry_price - actual_exit_price
        
        # Calculate commission on actual execution prices
        entry_commission = commission_model.calculate_commission(entry_order, actual_entry_price)
        exit_commission = commission_model.calculate_commission(exit_order, actual_exit_price)
        total_trade_commission = entry_commission + exit_commission
        
        # Calculate net P&L
        gross_pnl = trade.pnl
        spread_cost = half_spread * 2  # Pay half spread on each side
        net_pnl = pnl_after_slippage - total_trade_commission
        
        # Track totals
        total_gross += gross_pnl
        total_slippage += spread_cost
        total_commission += total_trade_commission
        total_net += net_pnl
        
        print(f"{i+1:5d} | {trade.direction:5s} | ${trade.entry_price:7.2f} | ${trade.exit_price:7.2f} | "
              f"${gross_pnl:6.3f} | ${spread_cost:4.3f} | ${total_trade_commission:4.3f} | "
              f"${net_pnl:7.3f}")
    
    # Summary
    print(f"\nExecution Cost Summary:")
    print("-" * 80)
    print(f"Total Gross P&L:    ${total_gross:.4f}")
    print(f"Total Spread Cost:  ${total_slippage:.4f}")
    print(f"Total Commission:   ${total_commission:.4f}")
    print(f"Total Net P&L:      ${total_net:.4f}")
    print(f"Cost Impact:        {((total_slippage + total_commission)/abs(total_gross)*100):.1f}% of gross")
    
    # Average costs per trade
    avg_spread = total_slippage / len(trades)
    avg_commission = total_commission / len(trades)
    avg_total_cost = avg_spread + avg_commission
    
    print(f"\nPer-Trade Costs:")
    print(f"  Avg spread cost: ${avg_spread:.4f}")
    print(f"  Avg commission:  ${avg_commission:.4f}")
    print(f"  Total avg cost:  ${avg_total_cost:.4f}")
    print(f"  Break-even move: ${avg_total_cost:.4f} ({avg_total_cost/521*100:.3f}%)")
    
    return {
        'gross_pnl': total_gross,
        'net_pnl': total_net,
        'total_costs': total_slippage + total_commission,
        'trades': len(trades)
    }

def main():
    workspace = "workspaces/tmp/20250611_171158"
    market_data = "data/SPY_1m.csv"
    
    print("\n" + "="*80)
    print("REALISTIC EXECUTION ANALYSIS USING PROPER MODELS")
    print("="*80)
    
    signal_files = list(Path(workspace).glob("signals_strategy_*.json"))
    
    results = []
    for signal_file in sorted(signal_files):
        result = analyze_with_execution_models(str(signal_file), market_data)
        results.append(result)
    
    # Compare strategies
    if len(results) > 1:
        print("\n" + "="*80)
        print("NET PERFORMANCE COMPARISON")
        print("="*80)
        
        for i, (f, r) in enumerate(zip(signal_files, results)):
            strategy = Path(f).stem.replace('signals_strategy_', '')
            profitable = "✓" if r['net_pnl'] > 0 else "✗"
            print(f"{profitable} {strategy}: ${r['gross_pnl']:.4f} gross → ${r['net_pnl']:.4f} net")

if __name__ == "__main__":
    main()
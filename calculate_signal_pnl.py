#!/usr/bin/env python3
"""
Calculate P&L directly from signal changes and prices stored in the parquet file.
"""

import pandas as pd
import numpy as np
from pathlib import Path

WORKSPACE_PATH = "/Users/daws/ADMF-PC/workspaces/duckdb_ensemble_v1_56028885"
TRANSACTION_COST = 0.0  # Zero commission as requested

def calculate_pnl_from_signals():
    """Calculate P&L directly from signal changes with proper transaction cost modeling."""
    
    print("="*80)
    print("📊 P&L CALCULATION FROM SIGNAL CHANGES")
    print("="*80)
    
    # Load signal file
    signal_file = Path(WORKSPACE_PATH) / "traces/SPY_1m/signals/unknown/SPY_adaptive_ensemble_default.parquet"
    signals_df = pd.read_parquet(signal_file)
    
    print(f"\n📈 Signal Data:")
    print(f"  Total signal changes: {len(signals_df):,}")
    print(f"  Columns: {signals_df.columns.tolist()}")
    
    # Check price data
    if 'px' in signals_df.columns:
        print(f"  Price range: ${signals_df['px'].min():.2f} - ${signals_df['px'].max():.2f}")
    else:
        print("  ❌ No price column found!")
        return
    
    # Sort by bar index to ensure chronological order
    signals_df = signals_df.sort_values('idx').reset_index(drop=True)
    
    # Track capital and trades with proper transaction cost modeling
    initial_capital = 1.0  # Start with $1
    capital = initial_capital
    current_position = None
    trades = []
    total_transaction_costs = 0.0
    
    print(f"\n🔄 Processing trades with iterative capital tracking...")
    print(f"Starting capital: ${capital:.6f}")
    
    for i in range(len(signals_df)):
        bar_idx = signals_df.iloc[i]['idx']
        signal = signals_df.iloc[i]['val']
        price = signals_df.iloc[i]['px']
        
        # Skip if price is 0 or missing
        if price == 0 or pd.isna(price):
            continue
        
        # First non-zero signal opens position
        if current_position is None and signal != 0:
            # Entry transaction cost
            entry_cost = capital * TRANSACTION_COST
            capital -= entry_cost
            total_transaction_costs += entry_cost
            
            current_position = {
                'entry_idx': bar_idx,
                'entry_signal': signal,
                'entry_price': price,
                'entry_capital': capital
            }
        
        # Position change
        elif current_position is not None and signal != current_position['entry_signal']:
            # Close current position
            exit_price = price
            entry_price = current_position['entry_price']
            entry_capital = current_position['entry_capital']
            
            # Calculate gross P&L
            if current_position['entry_signal'] == 1:  # Long position
                gross_return = (exit_price - entry_price) / entry_price
            else:  # Short position (-1)
                gross_return = (entry_price - exit_price) / entry_price
            
            # Apply gross return to capital
            capital = entry_capital * (1 + gross_return)
            
            # Exit transaction cost
            exit_cost = capital * TRANSACTION_COST
            capital -= exit_cost
            total_transaction_costs += exit_cost
            
            # Record trade
            trade = {
                'entry_idx': current_position['entry_idx'],
                'exit_idx': bar_idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'signal': current_position['entry_signal'],
                'gross_return': gross_return,
                'entry_capital': entry_capital,
                'pre_exit_capital': entry_capital * (1 + gross_return),
                'final_capital': capital
            }
            trades.append(trade)
            
            # If signal is non-zero, immediately open new position
            if signal != 0:
                # Entry transaction cost for new position
                entry_cost = capital * TRANSACTION_COST
                capital -= entry_cost
                total_transaction_costs += entry_cost
                
                current_position = {
                    'entry_idx': bar_idx,
                    'entry_signal': signal,
                    'entry_price': price,
                    'entry_capital': capital
                }
            else:
                current_position = None
    
    # Calculate final performance metrics
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) > 0:
        # Final returns
        total_return = (capital / initial_capital) - 1
        gross_return = (trades_df['pre_exit_capital'].iloc[-1] / initial_capital) - 1 if len(trades_df) > 0 else 0
        transaction_cost_drag = total_transaction_costs / initial_capital
        
        print("\n" + "="*60)
        print("📊 TRADING SUMMARY - 75% AGREEMENT THRESHOLD (ZERO COMMISSION)")
        print("="*60)
        
        num_trades = len(trades_df)
        
        print(f"\n💰 PERFORMANCE:")
        print(f"  Starting Capital: ${initial_capital:.6f}")
        print(f"  Final Capital: ${capital:.6f}")
        print(f"  Total Return: {total_return:.2%}")
        print(f"  Total Transaction Costs: ${total_transaction_costs:.6f} ({transaction_cost_drag:.2%})")
        print(f"  Completed Trades: {num_trades}")
        
        # Calculate trade statistics
        winning_trades = trades_df[trades_df['gross_return'] > 0]
        losing_trades = trades_df[trades_df['gross_return'] < 0]
        
        print(f"\n📈 TRADE STATISTICS:")
        print(f"  Win Rate: {len(winning_trades)/len(trades_df)*100:.1f}%")
        print(f"  Avg Win: {winning_trades['gross_return'].mean()*100:.2f}%" if len(winning_trades) > 0 else "  Avg Win: N/A")
        print(f"  Avg Loss: {losing_trades['gross_return'].mean()*100:.2f}%" if len(losing_trades) > 0 else "  Avg Loss: N/A")
        print(f"  Best Trade: {trades_df['gross_return'].max()*100:.2f}%")
        print(f"  Worst Trade: {trades_df['gross_return'].min()*100:.2f}%")
        
        # Show impact of transaction costs per trade
        avg_entry_cost = (total_transaction_costs / (num_trades * 2)) / initial_capital * 100
        print(f"\n💸 TRANSACTION COST IMPACT:")
        print(f"  Avg Cost per Side: {avg_entry_cost:.4f}%")
        print(f"  Round-trip Cost: {avg_entry_cost * 2:.4f}%")
        print(f"  Total Trades (round-trips): {num_trades}")
        print(f"  Total Transaction Sides: {num_trades * 2}")
        
        # Capital progression
        print(f"\n📊 CAPITAL PROGRESSION:")
        print("First 3 trades:")
        for i in range(min(3, len(trades_df))):
            t = trades_df.iloc[i]
            direction = "LONG" if t['signal'] == 1 else "SHORT"
            capital_change = t['final_capital'] - t['entry_capital']
            print(f"  {direction}: ${t['entry_capital']:.6f} → ${t['final_capital']:.6f} ({capital_change:+.6f})")
        
        if len(trades_df) > 6:
            print("\nLast 3 trades:")
            for i in range(max(0, len(trades_df)-3), len(trades_df)):
                t = trades_df.iloc[i]
                direction = "LONG" if t['signal'] == 1 else "SHORT"
                capital_change = t['final_capital'] - t['entry_capital']
                print(f"  {direction}: ${t['entry_capital']:.6f} → ${t['final_capital']:.6f} ({capital_change:+.6f})")
        
        # Compare with previous analysis
        print("\n" + "="*60)
        print("📊 COMPARISON WITH PREVIOUS VERSIONS:")
        print("="*60)
        print(f"33% Agreement (with TC): -21.85% net return, 3,511 trades")
        print(f"75% Agreement (zero TC): {total_return:.2%} net return, {num_trades} trades")
        print(f"Trade reduction: {((3511 - num_trades) / 3511 * 100):.1f}% fewer trades")
        
        if transaction_cost_drag > 0:
            print(f"\n🔍 TRANSACTION COST IMPACT:")
            print(f"TC drag with zero commission: {transaction_cost_drag:.2%}")
        else:
            print(f"\n🔍 ZERO COMMISSION RESULT:")
            print(f"Pure strategy performance without any trading costs!")
    
    else:
        print("\n❌ No completed trades found!")

if __name__ == "__main__":
    calculate_pnl_from_signals()
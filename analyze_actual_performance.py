"""Analyze actual performance from the latest run."""

import pandas as pd
from pathlib import Path
import numpy as np

# Load fills data
results_path = Path("config/bollinger/results/latest")
fills_path = results_path / "traces/execution/fills/execution_fills.parquet"

fills_df = pd.read_parquet(fills_path)

print("=== ACTUAL PERFORMANCE ANALYSIS ===\n")

# Calculate trade returns
trades = []
entries = {}

for idx, row in fills_df.iterrows():
    metadata = row['metadata']
    if isinstance(metadata, dict):
        symbol = row.get('symbol', 'SPY_5m')
        side = metadata.get('side', '').lower()
        price = float(metadata.get('price', 0))
        
        nested = metadata.get('metadata', {})
        if isinstance(nested, dict):
            exit_type = nested.get('exit_type')
            
            if not exit_type:  # Entry
                entries[symbol] = {
                    'price': price,
                    'side': side,
                    'idx': idx
                }
            else:  # Exit
                if symbol in entries:
                    entry = entries[symbol]
                    
                    # Calculate return based on position type
                    if entry['side'] == 'sell':  # Short position
                        trade_return = (entry['price'] - price) / entry['price'] * 100
                    else:  # Long position
                        trade_return = (price - entry['price']) / entry['price'] * 100
                    
                    trades.append({
                        'entry_price': entry['price'],
                        'exit_price': price,
                        'return_pct': trade_return,
                        'exit_type': exit_type,
                        'is_short': entry['side'] == 'sell',
                        'exit_reason': nested.get('exit_reason', '')
                    })
                    
                    del entries[symbol]

print(f"Total trades: {len(trades)}")

if trades:
    returns = [t['return_pct'] for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    
    print(f"\nPerformance Metrics:")
    print(f"Total return: {sum(returns):.2f}%")
    print(f"Average return per trade: {np.mean(returns):.4f}%")
    print(f"Win rate: {len(wins)/len(trades)*100:.1f}%")
    print(f"Average win: {np.mean(wins):.4f}%" if wins else "Average win: N/A")
    print(f"Average loss: {np.mean(losses):.4f}%" if losses else "Average loss: N/A")
    
    # Exit type analysis
    stop_losses = [t for t in trades if t['exit_type'] == 'stop_loss']
    take_profits = [t for t in trades if t['exit_type'] == 'take_profit']
    other_exits = [t for t in trades if t['exit_type'] not in ['stop_loss', 'take_profit']]
    
    print(f"\nExit Analysis:")
    print(f"Stop losses: {len(stop_losses)} ({len(stop_losses)/len(trades)*100:.1f}%)")
    print(f"Take profits: {len(take_profits)} ({len(take_profits)/len(trades)*100:.1f}%)")
    print(f"Other exits: {len(other_exits)} ({len(other_exits)/len(trades)*100:.1f}%)")
    
    # Check for trades exiting at exact stop/take levels
    stop_at_exact = sum(1 for t in stop_losses if abs(t['return_pct'] - (-0.075)) < 0.001)
    take_at_exact = sum(1 for t in take_profits if abs(t['return_pct'] - 0.15) < 0.001)
    
    print(f"\nExact Exit Prices:")
    print(f"Stops at exactly -0.075%: {stop_at_exact}/{len(stop_losses)} ({stop_at_exact/len(stop_losses)*100:.1f}%)" if stop_losses else "No stop losses")
    print(f"Takes at exactly 0.15%: {take_at_exact}/{len(take_profits)} ({take_at_exact/len(take_profits)*100:.1f}%)" if take_profits else "No take profits")
    
    # Show some example trades
    print(f"\nFirst 10 trades:")
    for i, trade in enumerate(trades[:10]):
        print(f"\n{i+1}. {'SHORT' if trade['is_short'] else 'LONG'}")
        print(f"   Entry: ${trade['entry_price']:.4f}")
        print(f"   Exit:  ${trade['exit_price']:.4f}")
        print(f"   Return: {trade['return_pct']:.4f}%")
        print(f"   Exit type: {trade['exit_type']}")
        
    # Check for any weird prices
    print(f"\n=== CHECKING FOR WEIRD PRICES ===")
    weird_exits = [t for t in trades if t['exit_price'] < 1]
    if weird_exits:
        print(f"\n🚨 Found {len(weird_exits)} trades with exit price < $1:")
        for w in weird_exits[:5]:
            print(f"   Exit price: ${w['exit_price']:.5f}, Return: {w['return_pct']:.4f}%")
    else:
        print("No trades with abnormally low exit prices")
    
    # Distribution of returns
    print(f"\n=== RETURN DISTRIBUTION ===")
    print(f"Returns < -1%: {sum(1 for r in returns if r < -1)}")
    print(f"Returns -1% to -0.5%: {sum(1 for r in returns if -1 <= r < -0.5)}")
    print(f"Returns -0.5% to 0%: {sum(1 for r in returns if -0.5 <= r < 0)}")
    print(f"Returns 0% to 0.5%: {sum(1 for r in returns if 0 <= r < 0.5)}")
    print(f"Returns 0.5% to 1%: {sum(1 for r in returns if 0.5 <= r < 1)}")
    print(f"Returns > 1%: {sum(1 for r in returns if r >= 1)}")
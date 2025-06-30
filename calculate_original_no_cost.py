"""Calculate original strategy returns without execution costs - properly annualized"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load the original signal data
workspace = Path("workspaces/signal_generation_7ecda4b8")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

print("=== Original Strategy Returns - No Execution Costs ===\n")

# Convert to trades
trades = []
current_position = 0

for i in range(len(signals_df)):
    row = signals_df.iloc[i]
    new_signal = row['val']
    
    if current_position != 0 and new_signal != current_position:
        entry_idx = i - 1
        entry_row = signals_df.iloc[entry_idx]
        
        entry_price = entry_row['px']
        exit_price = row['px']
        pnl_pct = (exit_price / entry_price - 1) * current_position * 100
        
        trades.append({
            'entry_time': entry_row['ts'],
            'exit_time': row['ts'],
            'pnl_pct': pnl_pct,
            'pnl_decimal': pnl_pct / 100
        })
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)

# Calculate metrics
total_trades = len(trades_df)
avg_return_pct = trades_df['pnl_pct'].mean()
avg_return_decimal = avg_return_pct / 100

# Calculate time span
date_range = trades_df['exit_time'].max() - trades_df['entry_time'].min()
days = date_range.days
years = days / 365.25

trades_per_day = total_trades / days
trades_per_year = total_trades / years

print(f"Data Summary:")
print(f"  Total trades: {total_trades}")
print(f"  Date range: {trades_df['entry_time'].min()} to {trades_df['exit_time'].max()}")
print(f"  Total days: {days}")
print(f"  Total years: {years:.2f}")
print(f"  Trades per day: {trades_per_day:.2f}")
print(f"  Trades per year: {trades_per_year:.1f}")

print(f"\nPer-Trade Statistics:")
print(f"  Average return per trade: {avg_return_pct:.4f}% ({avg_return_decimal:.6f})")
print(f"  Win rate: {(trades_df['pnl_pct'] > 0).mean()*100:.1f}%")
print(f"  Std dev: {trades_df['pnl_pct'].std():.4f}%")

# Method 1: Compound each trade
print(f"\n=== Annualized Returns (Different Methods) ===")

# Method 1: Simple compounding
annual_compound = (1 + avg_return_decimal) ** trades_per_year - 1
print(f"\n1. Simple Compounding: (1 + {avg_return_decimal:.6f})^{trades_per_year:.1f} - 1")
print(f"   = {annual_compound*100:.2f}% per year")

# Method 2: Using actual cumulative returns
trades_df['cumulative_return'] = (1 + trades_df['pnl_decimal']).cumprod()
total_return = trades_df['cumulative_return'].iloc[-1] - 1
annual_cagr = (1 + total_return) ** (1/years) - 1
print(f"\n2. Actual CAGR from cumulative returns:")
print(f"   Total return: {total_return*100:.2f}% over {years:.2f} years")
print(f"   CAGR: {annual_cagr*100:.2f}% per year")

# Method 3: Daily returns approach
trades_df['date'] = trades_df['exit_time'].dt.date
daily_returns = trades_df.groupby('date')['pnl_decimal'].sum()

# Fill in missing days with 0 returns
all_dates = pd.date_range(start=daily_returns.index.min(), 
                         end=daily_returns.index.max(), freq='D')
daily_returns = daily_returns.reindex(all_dates.date, fill_value=0)

# Calculate annualized from daily
daily_mean = daily_returns.mean()
annual_from_daily = (1 + daily_mean) ** 252 - 1
print(f"\n3. From Daily Returns:")
print(f"   Average daily return: {daily_mean*100:.4f}%")
print(f"   Annualized (252 days): {annual_from_daily*100:.2f}% per year")

# Calculate Sharpe ratio (using daily returns)
daily_std = daily_returns.std()
sharpe = (daily_mean * 252) / (daily_std * np.sqrt(252)) if daily_std > 0 else 0
print(f"   Daily Sharpe: {daily_mean/daily_std if daily_std > 0 else 0:.3f}")
print(f"   Annual Sharpe: {sharpe:.2f}")

# Summary
print(f"\n=== SUMMARY: Original Strategy with NO Costs ===")
print(f"Average annual return: {annual_compound*100:.1f}% (using per-trade compounding)")
print(f"This would turn $10,000 into ${10000 * (1 + annual_compound):.0f} in one year")
print(f"Sharpe ratio: {sharpe:.2f}")

# Show impact of costs
print(f"\n=== Impact of Execution Costs ===")
for cost_bp in [0, 0.5, 1.0, 2.0]:
    cost_decimal = cost_bp / 100 / 100  # Convert basis points to decimal
    net_return = avg_return_decimal - (2 * cost_decimal)  # Round trip
    
    if net_return > -1:
        annual_net = (1 + net_return) ** trades_per_year - 1
        print(f"{cost_bp}bp cost: {annual_net*100:>6.1f}% annual")
    else:
        print(f"{cost_bp}bp cost: LOSS")
"""Extended stop loss analysis with larger thresholds"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load market data
market_df = pd.read_parquet('./data/SPY_1m.parquet')
market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
market_df = market_df.set_index('timestamp').sort_index()

# Load signals
workspace = Path("workspaces/signal_generation_7ecda4b8")
signal_file = workspace / "traces/SPY_1m/signals/bollinger_rsi_simple_signals/SPY_compiled_strategy_0.parquet"
signals_df = pd.read_parquet(signal_file)
signals_df['ts'] = pd.to_datetime(signals_df['ts'])

print("=== Extended Stop Loss Analysis ===\n")

# Convert sparse signals to trades
trades = []
current_position = 0
entry_time = None
entry_price = None

for i in range(len(signals_df)):
    row = signals_df.iloc[i]
    new_signal = row['val']
    
    if current_position != 0 and new_signal != current_position:
        if entry_time is not None:
            exit_time = row['ts']
            exit_price = row['px']
            pnl_pct = (exit_price / entry_price - 1) * current_position * 100
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'direction': 'long' if current_position > 0 else 'short',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'position': current_position
            })
    
    if new_signal != 0 and new_signal != current_position:
        entry_time = row['ts']
        entry_price = row['px']
    
    current_position = new_signal

trades_df = pd.DataFrame(trades)

# Function to check if trade would be stopped out
def check_stop_hit(trade, stop_pct, market_data):
    """Check if trade would hit stop loss during its lifetime"""
    mask = (market_data.index >= trade['entry_time']) & (market_data.index <= trade['exit_time'])
    trade_bars = market_data[mask]
    
    if len(trade_bars) == 0:
        return False, trade['pnl_pct']
    
    if trade['direction'] == 'long':
        min_low = trade_bars['low'].min()
        worst_drawdown = (min_low / trade['entry_price'] - 1) * 100
        if worst_drawdown < -stop_pct:
            return True, -stop_pct
    else:  # short
        max_high = trade_bars['high'].max()
        worst_drawdown = (trade['entry_price'] / max_high - 1) * 100
        if worst_drawdown < -stop_pct:
            return True, -stop_pct
    
    return False, trade['pnl_pct']

# Extended stop levels - going much larger
stop_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]

print(f"Original: {len(trades_df)} trades, {trades_df['pnl_pct'].mean():.3f}% avg, {(trades_df['pnl_pct'] > 0).mean()*100:.1f}% win rate\n")

# Calculate annualization factors
date_range_days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
trades_per_year = len(trades_df) / date_range_days * 252
exec_cost = 0.0001

print(f"{'Stop Loss':<10} {'Trades Hit':<12} {'Winners Hit':<12} {'Win Rate':<10} {'Avg Return':<12} {'Annual Net':<12} {'Sharpe':<8}")
print("-" * 95)

best_config = {'sharpe': -999}

for stop_pct in stop_levels:
    trades_stopped = 0
    winners_stopped = 0
    modified_returns = []
    
    for idx, trade in trades_df.iterrows():
        was_stopped, final_pnl = check_stop_hit(trade, stop_pct, market_df)
        
        if was_stopped:
            trades_stopped += 1
            if trade['pnl_pct'] > 0:
                winners_stopped += 1
        
        modified_returns.append(final_pnl)
    
    modified_returns = pd.Series(modified_returns)
    avg_return = modified_returns.mean()
    win_rate = (modified_returns > 0).mean()
    
    # Annual return
    net_return_per_trade = (avg_return / 100) - (2 * exec_cost)
    if net_return_per_trade > -1:
        annual_net = (1 + net_return_per_trade) ** trades_per_year - 1
        annual_net_val = annual_net * 100
        annual_net_str = f"{annual_net_val:.1f}%"
    else:
        annual_net_val = -100
        annual_net_str = "LOSS"
    
    # Calculate Sharpe
    # Group by date for daily returns
    trades_copy = trades_df.copy()
    trades_copy['pnl_final'] = modified_returns.values
    trades_copy['date'] = trades_copy['exit_time'].dt.date
    trades_copy['pnl_decimal_net'] = (trades_copy['pnl_final'] / 100) - (2 * exec_cost)
    
    daily_returns = trades_copy.groupby('date')['pnl_decimal_net'].sum()
    date_range = pd.date_range(start=daily_returns.index.min(), 
                              end=daily_returns.index.max(), 
                              freq='D')
    daily_returns = daily_returns.reindex(date_range.date, fill_value=0)
    
    daily_mean = daily_returns.mean()
    daily_std = daily_returns.std()
    
    if daily_std > 0:
        sharpe = (daily_mean * 252) / (daily_std * np.sqrt(252))
    else:
        sharpe = 0
    
    if sharpe > best_config['sharpe'] and annual_net_val > 0:
        best_config = {
            'stop_pct': stop_pct,
            'trades_stopped': trades_stopped,
            'winners_stopped': winners_stopped,
            'avg_return': avg_return,
            'win_rate': win_rate,
            'annual_net': annual_net_val,
            'sharpe': sharpe
        }
    
    print(f"{stop_pct:.2f}%      {trades_stopped:<12} {winners_stopped:<12} {win_rate*100:>8.1f}%   {avg_return:>10.3f}%   {annual_net_str:>11}  {sharpe:>6.2f}")

if 'stop_pct' in best_config:
    print(f"\n*** Best Configuration: {best_config['stop_pct']:.2f}% stop loss ***")
    print(f"Win Rate: {best_config['win_rate']*100:.1f}%")
    print(f"Average Return: {best_config['avg_return']:.3f}%")
    print(f"Annual Return (net): {best_config['annual_net']:.1f}%")
    print(f"Sharpe Ratio: {best_config['sharpe']:.2f}")
else:
    print("\n*** No profitable stop loss configuration found ***")

# Analyze the distribution of adverse excursions
print(f"\n=== Adverse Excursion Analysis ===")
print("What's the worst drawdown for each trade before recovery/exit?\n")

adverse_excursions = []
for idx, trade in trades_df.iterrows():
    mask = (market_df.index >= trade['entry_time']) & (market_df.index <= trade['exit_time'])
    trade_bars = market_df[mask]
    
    if len(trade_bars) > 0:
        if trade['direction'] == 'long':
            min_low = trade_bars['low'].min()
            worst_drawdown = (min_low / trade['entry_price'] - 1) * 100
        else:
            max_high = trade_bars['high'].max()
            worst_drawdown = (trade['entry_price'] / max_high - 1) * 100
        
        adverse_excursions.append({
            'pnl_pct': trade['pnl_pct'],
            'worst_drawdown': worst_drawdown,
            'win': 1 if trade['pnl_pct'] > 0 else 0
        })

ae_df = pd.DataFrame(adverse_excursions)

# Show percentiles
percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
print(f"{'Percentile':<12} {'All Trades':<15} {'Winners Only':<15} {'Losers Only':<15}")
print("-" * 60)

for p in percentiles:
    all_p = ae_df['worst_drawdown'].quantile(p/100)
    win_p = ae_df[ae_df['win']==1]['worst_drawdown'].quantile(p/100) if len(ae_df[ae_df['win']==1]) > 0 else 0
    lose_p = ae_df[ae_df['win']==0]['worst_drawdown'].quantile(p/100) if len(ae_df[ae_df['win']==0]) > 0 else 0
    print(f"{p}th         {all_p:>12.3f}%  {win_p:>12.3f}%  {lose_p:>12.3f}%")

# What percentage of winners have drawdowns beyond certain thresholds?
print(f"\n=== Winners with Large Drawdowns ===")
winners_ae = ae_df[ae_df['win']==1]
thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]

for t in thresholds:
    count = (winners_ae['worst_drawdown'] < -t).sum()
    pct = count / len(winners_ae) * 100 if len(winners_ae) > 0 else 0
    print(f"Winners with >{t:.2f}% drawdown: {count} ({pct:.1f}%)")
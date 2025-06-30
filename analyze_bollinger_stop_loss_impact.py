#!/usr/bin/env python3
"""
Comprehensive analysis of how stop losses could improve Bollinger strategy performance.
Focuses on:
1. Trade duration analysis - identifying long-duration losing trades
2. Return distribution - finding fat tail losses
3. Win/loss characteristics 
4. Stop loss simulation at various levels
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_bollinger_signals(workspace_path, strategy_name="bollinger_bands", max_files=10):
    """Load Bollinger Band signals from workspace."""
    signal_path = Path(workspace_path) / "traces" / "SPY_1m" / "signals" / strategy_name
    if not signal_path.exists():
        print(f"Signal path not found: {signal_path}")
        return []
    
    signal_files = list(signal_path.glob("*.parquet"))[:max_files]
    print(f"Found {len(signal_files)} signal files")
    return signal_files

def extract_trades_from_signals(signals_df):
    """Extract individual trades from signal data."""
    trades = []
    current_position = 0
    entry_price = None
    entry_time = None
    entry_idx = None
    
    for idx, row in signals_df.iterrows():
        signal = row['val'] if 'val' in row else row.get('signal', 0)
        price = row['px'] if 'px' in row else row.get('price', 0)
        timestamp = pd.to_datetime(row['ts']) if 'ts' in row else pd.to_datetime(row.get('timestamp'))
        bar_idx = row.get('idx', idx)
        
        # Entry
        if signal != 0 and current_position == 0:
            current_position = signal
            entry_price = price
            entry_time = timestamp
            entry_idx = bar_idx
        
        # Exit or reversal
        elif current_position != 0 and (signal == 0 or signal == -current_position):
            exit_price = price
            exit_time = timestamp
            
            # Calculate return
            if current_position == 1:  # Long
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:  # Short
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_idx': entry_idx,
                'exit_idx': bar_idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'direction': 'long' if current_position == 1 else 'short',
                'pnl_pct': pnl_pct,
                'duration_bars': bar_idx - entry_idx,
                'duration_minutes': (exit_time - entry_time).total_seconds() / 60 if entry_time else 0
            })
            
            # Handle reversal
            current_position = signal if signal != 0 else 0
            if current_position != 0:
                entry_price = price
                entry_time = timestamp
                entry_idx = bar_idx
    
    return pd.DataFrame(trades)

def analyze_return_distribution(trades_df):
    """Analyze the distribution of returns to identify fat tails."""
    print("\n" + "="*60)
    print("RETURN DISTRIBUTION ANALYSIS")
    print("="*60)
    
    returns = trades_df['pnl_pct']
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"  Mean return: {returns.mean():.3f}%")
    print(f"  Median return: {returns.median():.3f}%")
    print(f"  Std deviation: {returns.std():.3f}%")
    print(f"  Skewness: {returns.skew():.3f}")
    print(f"  Kurtosis: {returns.kurtosis():.3f}")
    
    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nReturn Percentiles:")
    for p in percentiles:
        val = np.percentile(returns, p)
        print(f"  {p:3d}th percentile: {val:7.3f}%")
    
    # Fat tail analysis
    print(f"\nFat Tail Analysis:")
    worst_1pct = returns[returns <= np.percentile(returns, 1)]
    worst_5pct = returns[returns <= np.percentile(returns, 5)]
    best_95pct = returns[returns >= np.percentile(returns, 95)]
    best_99pct = returns[returns >= np.percentile(returns, 99)]
    
    print(f"  Worst 1% ({len(worst_1pct)} trades): avg = {worst_1pct.mean():.3f}%, sum = {worst_1pct.sum():.2f}%")
    print(f"  Worst 5% ({len(worst_5pct)} trades): avg = {worst_5pct.mean():.3f}%, sum = {worst_5pct.sum():.2f}%")
    print(f"  Best 5% ({len(best_95pct)} trades): avg = {best_95pct.mean():.3f}%, sum = {best_95pct.sum():.2f}%")
    print(f"  Best 1% ({len(best_99pct)} trades): avg = {best_99pct.mean():.3f}%, sum = {best_99pct.sum():.2f}%")
    
    # Worst losses
    print(f"\nWorst 10 Losses:")
    worst_trades = trades_df.nsmallest(10, 'pnl_pct')
    for idx, trade in worst_trades.iterrows():
        print(f"  {trade['pnl_pct']:7.3f}% - Duration: {trade['duration_minutes']:6.1f} min, {trade['direction']}")
    
    return returns

def analyze_duration_impact(trades_df):
    """Analyze how trade duration affects performance."""
    print("\n" + "="*60)
    print("DURATION IMPACT ANALYSIS")
    print("="*60)
    
    # Duration buckets
    duration_buckets = [
        (0, 60, "< 1 hour"),
        (60, 240, "1-4 hours"),
        (240, 480, "4-8 hours"),
        (480, 1440, "8-24 hours"),
        (1440, float('inf'), "> 24 hours")
    ]
    
    print(f"\nPerformance by Duration:")
    print(f"{'Duration':<15} {'Count':>8} {'Avg PnL':>10} {'Win Rate':>10} {'Total PnL':>12}")
    print("-" * 60)
    
    for min_dur, max_dur, label in duration_buckets:
        mask = (trades_df['duration_minutes'] >= min_dur) & (trades_df['duration_minutes'] < max_dur)
        bucket_trades = trades_df[mask]
        
        if len(bucket_trades) > 0:
            avg_pnl = bucket_trades['pnl_pct'].mean()
            win_rate = (bucket_trades['pnl_pct'] > 0).mean()
            total_pnl = bucket_trades['pnl_pct'].sum()
            
            print(f"{label:<15} {len(bucket_trades):>8} {avg_pnl:>10.3f}% {win_rate:>10.1%} {total_pnl:>12.2f}%")
    
    # Analyze losing trades by duration
    losing_trades = trades_df[trades_df['pnl_pct'] < 0]
    if len(losing_trades) > 0:
        print(f"\nLosing Trades by Duration:")
        for min_dur, max_dur, label in duration_buckets:
            mask = (losing_trades['duration_minutes'] >= min_dur) & (losing_trades['duration_minutes'] < max_dur)
            bucket_trades = losing_trades[mask]
            
            if len(bucket_trades) > 0:
                avg_loss = bucket_trades['pnl_pct'].mean()
                total_loss = bucket_trades['pnl_pct'].sum()
                print(f"  {label:<15}: {len(bucket_trades):>4} trades, avg loss: {avg_loss:>7.3f}%, total: {total_loss:>8.2f}%")

def simulate_stop_losses(trades_df, stop_levels=[0.1, 0.2, 0.3, 0.5, 1.0, 2.0]):
    """Simulate the impact of different stop loss levels."""
    print("\n" + "="*60)
    print("STOP LOSS SIMULATION")
    print("="*60)
    
    original_stats = {
        'total_trades': len(trades_df),
        'avg_pnl': trades_df['pnl_pct'].mean(),
        'total_pnl': trades_df['pnl_pct'].sum(),
        'win_rate': (trades_df['pnl_pct'] > 0).mean(),
        'avg_win': trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if len(trades_df[trades_df['pnl_pct'] > 0]) > 0 else 0,
        'avg_loss': trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if len(trades_df[trades_df['pnl_pct'] < 0]) > 0 else 0
    }
    
    print(f"\nOriginal Performance:")
    print(f"  Total trades: {original_stats['total_trades']}")
    print(f"  Average PnL: {original_stats['avg_pnl']:.3f}%")
    print(f"  Total PnL: {original_stats['total_pnl']:.2f}%")
    print(f"  Win rate: {original_stats['win_rate']:.1%}")
    print(f"  Avg win: {original_stats['avg_win']:.3f}%")
    print(f"  Avg loss: {original_stats['avg_loss']:.3f}%")
    
    print(f"\nStop Loss Impact:")
    print(f"{'Stop %':>8} {'Stopped':>8} {'New Avg':>10} {'New Total':>12} {'Win Rate':>10} {'Improvement':>12}")
    print("-" * 70)
    
    for stop_pct in stop_levels:
        # Apply stop loss
        modified_trades = trades_df.copy()
        stopped_mask = modified_trades['pnl_pct'] < -stop_pct
        modified_trades.loc[stopped_mask, 'pnl_pct'] = -stop_pct
        
        # Calculate new metrics
        new_avg = modified_trades['pnl_pct'].mean()
        new_total = modified_trades['pnl_pct'].sum()
        new_win_rate = (modified_trades['pnl_pct'] > 0).mean()
        trades_stopped = stopped_mask.sum()
        
        # Calculate improvement
        avg_improvement = new_avg - original_stats['avg_pnl']
        total_improvement = new_total - original_stats['total_pnl']
        
        print(f"{stop_pct:>7.1f}% {trades_stopped:>8} {new_avg:>10.3f}% {new_total:>12.2f}% "
              f"{new_win_rate:>10.1%} {avg_improvement:>+11.3f}%")
    
    # Analyze which trades would benefit most from stops
    print(f"\nTrades That Would Benefit Most from 1% Stop Loss:")
    would_stop = trades_df[trades_df['pnl_pct'] < -1.0].copy()
    would_stop['saved'] = would_stop['pnl_pct'].abs() - 1.0
    top_saves = would_stop.nlargest(10, 'saved')
    
    for idx, trade in top_saves.iterrows():
        print(f"  Loss: {trade['pnl_pct']:7.3f}% → -1.00% (saves {trade['saved']:6.3f}%), "
              f"Duration: {trade['duration_minutes']:6.1f} min")

def analyze_win_loss_characteristics(trades_df):
    """Analyze characteristics of winning vs losing trades."""
    print("\n" + "="*60)
    print("WIN/LOSS CHARACTERISTICS")
    print("="*60)
    
    winners = trades_df[trades_df['pnl_pct'] > 0]
    losers = trades_df[trades_df['pnl_pct'] < 0]
    
    print(f"\nWinning Trades ({len(winners)}):")
    print(f"  Average gain: {winners['pnl_pct'].mean():.3f}%")
    print(f"  Median gain: {winners['pnl_pct'].median():.3f}%")
    print(f"  Avg duration: {winners['duration_minutes'].mean():.1f} minutes")
    print(f"  Max gain: {winners['pnl_pct'].max():.3f}%")
    
    print(f"\nLosing Trades ({len(losers)}):")
    print(f"  Average loss: {losers['pnl_pct'].mean():.3f}%")
    print(f"  Median loss: {losers['pnl_pct'].median():.3f}%")
    print(f"  Avg duration: {losers['duration_minutes'].mean():.1f} minutes")
    print(f"  Max loss: {losers['pnl_pct'].min():.3f}%")
    
    print(f"\nKey Ratios:")
    if len(losers) > 0 and len(winners) > 0:
        win_loss_ratio = abs(winners['pnl_pct'].mean() / losers['pnl_pct'].mean())
        print(f"  Win/Loss ratio: {win_loss_ratio:.2f}")
        print(f"  Avg winner duration / Avg loser duration: {winners['duration_minutes'].mean() / losers['duration_minutes'].mean():.2f}")
    
    # Long vs short analysis
    long_trades = trades_df[trades_df['direction'] == 'long']
    short_trades = trades_df[trades_df['direction'] == 'short']
    
    print(f"\nBy Direction:")
    print(f"  Long trades: {len(long_trades)}, avg: {long_trades['pnl_pct'].mean():.3f}%, win rate: {(long_trades['pnl_pct'] > 0).mean():.1%}")
    print(f"  Short trades: {len(short_trades)}, avg: {short_trades['pnl_pct'].mean():.3f}%, win rate: {(short_trades['pnl_pct'] > 0).mean():.1%}")

def main():
    """Main analysis function."""
    print("BOLLINGER BANDS STOP LOSS IMPACT ANALYSIS")
    print("=" * 80)
    print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Try multiple workspaces
    workspaces = [
        "workspaces/signal_generation_cc984d99",
        "workspaces/signal_generation_f88793ad",
        "workspaces/signal_generation_a15a871e"
    ]
    
    all_trades = []
    
    for workspace in workspaces:
        if Path(workspace).exists():
            print(f"\nAnalyzing workspace: {workspace}")
            signal_files = load_bollinger_signals(workspace, max_files=5)
            
            for signal_file in signal_files:
                try:
                    signals_df = pd.read_parquet(signal_file)
                    trades_df = extract_trades_from_signals(signals_df)
                    
                    if len(trades_df) > 0:
                        trades_df['workspace'] = workspace
                        trades_df['strategy_file'] = signal_file.stem
                        all_trades.append(trades_df)
                        print(f"  Loaded {len(trades_df)} trades from {signal_file.stem}")
                except Exception as e:
                    print(f"  Error processing {signal_file}: {e}")
    
    if not all_trades:
        print("\nNo trades found!")
        return
    
    # Combine all trades
    combined_trades = pd.concat(all_trades, ignore_index=True)
    print(f"\nTotal trades analyzed: {len(combined_trades)}")
    
    # Run analyses
    analyze_return_distribution(combined_trades)
    analyze_duration_impact(combined_trades)
    analyze_win_loss_characteristics(combined_trades)
    simulate_stop_losses(combined_trades)
    
    # Summary recommendations
    print("\n" + "="*60)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*60)
    
    # Calculate key metrics for recommendations
    avg_return = combined_trades['pnl_pct'].mean()
    losing_trades = combined_trades[combined_trades['pnl_pct'] < 0]
    long_losers = losing_trades[losing_trades['duration_minutes'] > 240]  # > 4 hours
    fat_tail_losses = combined_trades[combined_trades['pnl_pct'] < -2.0]
    
    print(f"\nKey Findings:")
    print(f"1. Current average return per trade: {avg_return:.3f}% (0.88 bps mentioned)")
    print(f"2. {len(long_losers)} losing trades last > 4 hours ({len(long_losers)/len(losing_trades)*100:.1f}% of losers)")
    print(f"3. {len(fat_tail_losses)} trades have losses > 2% (fat tails)")
    print(f"4. Worst loss: {combined_trades['pnl_pct'].min():.3f}%")
    
    # Estimate improvement potential
    with_1pct_stop = combined_trades.copy()
    with_1pct_stop.loc[with_1pct_stop['pnl_pct'] < -1.0, 'pnl_pct'] = -1.0
    new_avg = with_1pct_stop['pnl_pct'].mean()
    
    print(f"\nStop Loss Potential:")
    print(f"- 1% stop loss could improve avg return from {avg_return:.3f}% to {new_avg:.3f}%")
    print(f"- This is an improvement of {new_avg - avg_return:.3f}% per trade")
    print(f"- Could potentially reach the 1.5-2 bps target mentioned")
    
    # Save detailed results
    output_file = "bollinger_stop_loss_analysis.csv"
    combined_trades.to_csv(output_file, index=False)
    print(f"\nDetailed trade data saved to: {output_file}")

if __name__ == "__main__":
    main()
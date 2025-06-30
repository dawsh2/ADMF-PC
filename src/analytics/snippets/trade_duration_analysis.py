# Trade Duration Analysis
# Analyzes how long trades are held, including with stop/target exits

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def analyze_trade_durations(strategy_hash, trace_path, market_data, 
                           stop_pct=0.075, target_pct=0.1, 
                           execution_cost_bps=1.0):
    """
    Analyze trade durations with and without stop/target modifications.
    
    Returns detailed duration statistics by exit type.
    """
    # Extract trades
    trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)
    
    if len(trades) == 0:
        return None
    
    # Add duration in bars and time
    trades['duration_bars'] = trades['exit_idx'] - trades['entry_idx']
    trades['duration_time'] = trades['exit_time'] - trades['entry_time']
    trades['duration_minutes'] = trades['duration_time'].dt.total_seconds() / 60
    
    print("📊 Trade Duration Analysis")
    print("=" * 80)
    
    # Original durations (no stops/targets)
    print("\n1. Original Trade Durations (No Stops/Targets):")
    print(f"   Total trades: {len(trades)}")
    print(f"   Average duration: {trades['duration_bars'].mean():.1f} bars ({trades['duration_minutes'].mean():.0f} minutes)")
    print(f"   Median duration: {trades['duration_bars'].median():.0f} bars ({trades['duration_minutes'].median():.0f} minutes)")
    print(f"   Min duration: {trades['duration_bars'].min()} bars")
    print(f"   Max duration: {trades['duration_bars'].max()} bars")
    
    # Duration distribution
    print(f"\n   Duration distribution:")
    duration_bins = [0, 5, 10, 20, 50, 100, 200, 500, 1000]
    duration_counts = pd.cut(trades['duration_bars'], bins=duration_bins, include_lowest=True).value_counts().sort_index()
    for interval, count in duration_counts.items():
        pct = count / len(trades) * 100
        print(f"     {interval}: {count} trades ({pct:.1f}%)")
    
    # Now analyze with stops/targets
    print(f"\n2. Modified Durations with {stop_pct}% Stop / {target_pct}% Target:")
    
    modified_durations = []
    exit_type_durations = {'stop': [], 'target': [], 'signal': []}
    
    for _, trade in trades.iterrows():
        trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
        
        if len(trade_prices) == 0:
            modified_durations.append(trade['duration_bars'])
            exit_type_durations['signal'].append(trade['duration_bars'])
            continue
        
        entry_price = trade['entry_price']
        direction = trade['direction']
        entry_idx = trade['entry_idx']
        
        # Set stop and target prices
        if direction == 1:  # Long
            stop_price = entry_price * (1 - stop_pct/100)
            target_price = entry_price * (1 + target_pct/100)
        else:  # Short
            stop_price = entry_price * (1 + stop_pct/100)
            target_price = entry_price * (1 - target_pct/100)
        
        # Find actual exit bar
        exit_bar = None
        exit_type = 'signal'
        
        for i, (_, bar) in enumerate(trade_prices.iterrows()):
            if direction == 1:  # Long
                if bar['low'] <= stop_price:
                    exit_bar = i
                    exit_type = 'stop'
                    break
                elif bar['high'] >= target_price:
                    exit_bar = i
                    exit_type = 'target'
                    break
            else:  # Short
                if bar['high'] >= stop_price:
                    exit_bar = i
                    exit_type = 'stop'
                    break
                elif bar['low'] <= target_price:
                    exit_bar = i
                    exit_type = 'target'
                    break
        
        if exit_bar is None:
            # Original exit
            duration = trade['duration_bars']
        else:
            # Modified exit
            duration = exit_bar + 1  # +1 because we exit at the bar, not before
        
        modified_durations.append(duration)
        exit_type_durations[exit_type].append(duration)
    
    modified_durations = np.array(modified_durations)
    
    # Overall statistics with stops/targets
    print(f"   Average duration: {modified_durations.mean():.1f} bars ({modified_durations.mean() * 5:.0f} minutes)")
    print(f"   Median duration: {np.median(modified_durations):.0f} bars")
    
    # Duration by exit type
    print(f"\n   Duration by exit type:")
    for exit_type, durations in exit_type_durations.items():
        if durations:
            avg_dur = np.mean(durations)
            med_dur = np.median(durations)
            count = len(durations)
            pct = count / len(trades) * 100
            print(f"     {exit_type.capitalize()} exits: {count} ({pct:.1f}%)")
            print(f"       Average: {avg_dur:.1f} bars ({avg_dur * 5:.0f} minutes)")
            print(f"       Median: {med_dur:.0f} bars ({med_dur * 5:.0f} minutes)")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Original vs Modified duration distribution
    ax = axes[0, 0]
    bins = np.logspace(0, 3, 30)  # Log scale for better visualization
    ax.hist(trades['duration_bars'], bins=bins, alpha=0.5, label='Original', color='blue')
    ax.hist(modified_durations, bins=bins, alpha=0.5, label=f'With {stop_pct}/{target_pct}%', color='red')
    ax.set_xscale('log')
    ax.set_xlabel('Duration (bars)')
    ax.set_ylabel('Count')
    ax.set_title('Trade Duration Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Duration by exit type (box plot)
    ax = axes[0, 1]
    exit_data = []
    for exit_type, durations in exit_type_durations.items():
        if durations:
            for dur in durations:
                exit_data.append({'Exit Type': exit_type.capitalize(), 'Duration': dur})
    
    if exit_data:
        exit_df = pd.DataFrame(exit_data)
        exit_df.boxplot(column='Duration', by='Exit Type', ax=ax)
        ax.set_ylabel('Duration (bars)')
        ax.set_title('Duration Distribution by Exit Type')
        ax.set_yscale('log')
    
    # 3. Cumulative duration reduction
    ax = axes[1, 0]
    original_cum = np.sort(trades['duration_bars'])
    modified_cum = np.sort(modified_durations)
    
    ax.plot(np.arange(len(original_cum)), original_cum, label='Original', linewidth=2)
    ax.plot(np.arange(len(modified_cum)), modified_cum, label=f'With {stop_pct}/{target_pct}%', linewidth=2)
    ax.set_xlabel('Trade Number (sorted by duration)')
    ax.set_ylabel('Duration (bars)')
    ax.set_title('Cumulative Duration Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate time saved
    total_original_bars = trades['duration_bars'].sum()
    total_modified_bars = modified_durations.sum()
    bars_saved = total_original_bars - total_modified_bars
    time_saved_hours = (bars_saved * 5) / 60  # 5 minutes per bar
    
    summary_text = f"Duration Impact Summary\n\n"
    summary_text += f"Original average: {trades['duration_bars'].mean():.1f} bars\n"
    summary_text += f"Modified average: {modified_durations.mean():.1f} bars\n"
    summary_text += f"Reduction: {(1 - modified_durations.mean()/trades['duration_bars'].mean())*100:.1f}%\n\n"
    
    summary_text += f"Total bars traded:\n"
    summary_text += f"  Original: {total_original_bars:,} bars\n"
    summary_text += f"  Modified: {total_modified_bars:,} bars\n"
    summary_text += f"  Saved: {bars_saved:,} bars ({time_saved_hours:.1f} hours)\n\n"
    
    summary_text += f"Quick exits (≤5 bars):\n"
    summary_text += f"  Original: {(trades['duration_bars'] <= 5).sum()} ({(trades['duration_bars'] <= 5).mean()*100:.1f}%)\n"
    summary_text += f"  Modified: {(modified_durations <= 5).sum()} ({(modified_durations <= 5).mean()*100:.1f}%)\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=12, family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    return {
        'original_avg_duration': trades['duration_bars'].mean(),
        'modified_avg_duration': modified_durations.mean(),
        'stop_avg_duration': np.mean(exit_type_durations['stop']) if exit_type_durations['stop'] else 0,
        'target_avg_duration': np.mean(exit_type_durations['target']) if exit_type_durations['target'] else 0,
        'signal_avg_duration': np.mean(exit_type_durations['signal']) if exit_type_durations['signal'] else 0,
        'time_saved_hours': time_saved_hours
    }

# Run analysis
if len(performance_df) > 0:
    print("🕐 Analyzing Trade Durations")
    print("=" * 80)
    
    # Analyze the strategy
    strategy = performance_df.iloc[0]
    
    duration_stats = analyze_trade_durations(
        strategy['strategy_hash'],
        strategy['trace_path'],
        market_data,
        stop_pct=0.075,
        target_pct=0.10,
        execution_cost_bps=execution_cost_bps
    )
    
    if duration_stats:
        print(f"\n💡 Key Insights:")
        print(f"1. Average trade duration reduces from {duration_stats['original_avg_duration']:.1f} to {duration_stats['modified_avg_duration']:.1f} bars")
        print(f"2. Stop exits occur quickly (avg {duration_stats['stop_avg_duration']:.1f} bars)")
        print(f"3. Target exits take {duration_stats['target_avg_duration']:.1f} bars on average")
        print(f"4. Total time saved: {duration_stats['time_saved_hours']:.1f} hours over {len(performance_df.iloc[0]['num_trades'])} trades")
else:
    print("❌ No performance data available")
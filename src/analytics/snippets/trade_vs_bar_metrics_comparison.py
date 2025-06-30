# Comprehensive comparison of trade-based vs bar-based metrics
# This reveals the fundamental difference in how performance is calculated

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_calculation_methods(strategy_hash, trace_path, market_data, execution_cost_bps=1.0):
    """
    Compare bar-based vs trade-based calculation methods for the same strategy.
    Shows exactly why the metrics differ so dramatically.
    """
    # First calculate bar-based metrics (original method)
    signals_path = run_dir / trace_path
    signals = pd.read_parquet(signals_path)
    signals['ts'] = pd.to_datetime(signals['ts'])
    
    # Merge with market data
    df = market_data.merge(
        signals[['ts', 'val', 'px']], 
        left_on='timestamp', 
        right_on='ts', 
        how='left'
    )
    
    # Forward fill signals
    df['signal'] = df['val'].ffill().fillna(0)
    df['position'] = df['signal'].replace({0: 0, 1: 1, -1: -1})
    
    # Calculate bar-by-bar returns
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
    
    # Bar-based metrics
    bars_with_position = df[df['signal'].shift(1) != 0]['strategy_returns']
    bar_positive_returns = bars_with_position[bars_with_position > 0]
    bar_negative_returns = bars_with_position[bars_with_position < 0]
    
    bar_metrics = {
        'total_bars': len(bars_with_position),
        'winning_bars': len(bar_positive_returns),
        'losing_bars': len(bar_negative_returns),
        'bar_win_rate': len(bar_positive_returns) / len(bars_with_position) if len(bars_with_position) > 0 else 0,
        'avg_bar_return': bars_with_position.mean(),
        'total_return_bars': (1 + bars_with_position).cumprod().iloc[-1] - 1 if len(bars_with_position) > 0 else 0
    }
    
    # Now calculate trade-based metrics
    trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)
    
    if len(trades) > 0:
        winning_trades = trades[trades['net_return'] > 0]
        losing_trades = trades[trades['net_return'] <= 0]
        
        trade_metrics = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'trade_win_rate': len(winning_trades) / len(trades),
            'avg_trade_return': trades['net_return'].mean(),
            'total_return_trades': (1 + trades['net_return']).cumprod().iloc[-1] - 1,
            'avg_bars_per_trade': bar_metrics['total_bars'] / len(trades) if len(trades) > 0 else 0
        }
    else:
        trade_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'trade_win_rate': 0,
            'avg_trade_return': 0,
            'total_return_trades': 0,
            'avg_bars_per_trade': 0
        }
    
    return bar_metrics, trade_metrics, df, trades

# Analyze top strategies with both methods
if len(top_overall) > 0:
    print("🔬 Deep Dive: Bar-Based vs Trade-Based Calculation Methods")
    print("=" * 80)
    
    comparison_results = []
    
    # Analyze top 5 strategies
    for idx, (_, strategy) in enumerate(top_overall.head(5).iterrows()):
        print(f"\n📊 Strategy {idx+1}: {strategy['strategy_type']} - {strategy['strategy_hash'][:8]}")
        
        bar_metrics, trade_metrics, df_with_signals, trades = analyze_calculation_methods(
            strategy['strategy_hash'], 
            strategy['trace_path'], 
            market_data, 
            execution_cost_bps
        )
        
        # Show the comparison
        print("\nBar-Based Calculation:")
        print(f"  Total bars with position: {bar_metrics['total_bars']}")
        print(f"  Winning bars: {bar_metrics['winning_bars']} ({bar_metrics['bar_win_rate']:.1%})")
        print(f"  Losing bars: {bar_metrics['losing_bars']}")
        print(f"  Average return per bar: {bar_metrics['avg_bar_return']*100:.3f}%")
        
        print("\nTrade-Based Calculation:")
        print(f"  Total trades: {trade_metrics['total_trades']}")
        print(f"  Winning trades: {trade_metrics['winning_trades']} ({trade_metrics['trade_win_rate']:.1%})")
        print(f"  Losing trades: {trade_metrics['losing_trades']}")
        print(f"  Average return per trade: {trade_metrics['avg_trade_return']*100:.3f}%")
        print(f"  Average bars per trade: {trade_metrics['avg_bars_per_trade']:.1f}")
        
        # Key insight
        print("\n💡 Key Insight:")
        print(f"  Each trade spans ~{trade_metrics['avg_bars_per_trade']:.0f} bars on average")
        print(f"  This explains why bar win rate ({bar_metrics['bar_win_rate']:.1%}) << trade win rate ({trade_metrics['trade_win_rate']:.1%})")
        
        comparison_results.append({
            'strategy': f"{strategy['strategy_type']}_{strategy['strategy_hash'][:8]}",
            'bar_win_rate': bar_metrics['bar_win_rate'],
            'trade_win_rate': trade_metrics['trade_win_rate'],
            'bars_per_trade': trade_metrics['avg_bars_per_trade'],
            'total_bars': bar_metrics['total_bars'],
            'total_trades': trade_metrics['total_trades']
        })
    
    # Visualize the comparison
    comparison_df = pd.DataFrame(comparison_results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Win rate comparison
    ax = axes[0, 0]
    x = np.arange(len(comparison_df))
    width = 0.35
    ax.bar(x - width/2, comparison_df['bar_win_rate'] * 100, width, label='Bar-Based', alpha=0.7)
    ax.bar(x + width/2, comparison_df['trade_win_rate'] * 100, width, label='Trade-Based', alpha=0.7)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rate: Bar-Based vs Trade-Based Calculation')
    ax.set_xticks(x)
    ax.set_xticklabels([s.split('_')[0] for s in comparison_df['strategy']], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Bars per trade
    ax = axes[0, 1]
    ax.bar(range(len(comparison_df)), comparison_df['bars_per_trade'])
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Average Bars per Trade')
    ax.set_title('Trade Duration in Bars')
    ax.set_xticks(range(len(comparison_df)))
    ax.set_xticklabels([s.split('_')[0] for s in comparison_df['strategy']], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 3. Sample trade visualization
    if len(trades) > 0 and len(df_with_signals) > 0:
        ax = axes[1, 0]
        # Show first 500 bars and trades
        sample_df = df_with_signals.head(500).copy()
        sample_df['bar_return'] = sample_df['strategy_returns'] * 100
        
        # Plot bar returns
        bars_with_pos = sample_df[sample_df['signal'].shift(1) != 0]
        positive_bars = bars_with_pos[bars_with_pos['bar_return'] > 0]
        negative_bars = bars_with_pos[bars_with_pos['bar_return'] < 0]
        
        ax.scatter(positive_bars.index, positive_bars['bar_return'], c='green', alpha=0.5, s=10, label='Positive bars')
        ax.scatter(negative_bars.index, negative_bars['bar_return'], c='red', alpha=0.5, s=10, label='Negative bars')
        
        # Overlay trade boundaries
        sample_trades = trades[trades['entry_idx'] < 500]
        for _, trade in sample_trades.iterrows():
            if trade['exit_idx'] < 500:
                ax.axvspan(trade['entry_idx'], trade['exit_idx'], 
                          alpha=0.2, color='blue' if trade['net_return'] > 0 else 'orange')
        
        ax.set_xlabel('Bar Index')
        ax.set_ylabel('Bar Return (%)')
        ax.set_title('Sample: Individual Bar Returns vs Complete Trades')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Win rate relationship
    ax = axes[1, 1]
    ax.scatter(comparison_df['bars_per_trade'], comparison_df['trade_win_rate'] * 100, s=100)
    for i, txt in enumerate(comparison_df.index):
        ax.annotate(f"S{i+1}", (comparison_df['bars_per_trade'].iloc[i], 
                               comparison_df['trade_win_rate'].iloc[i] * 100))
    ax.set_xlabel('Average Bars per Trade')
    ax.set_ylabel('Trade Win Rate (%)')
    ax.set_title('Trade Win Rate vs Trade Duration')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary insights
    print("\n📋 Summary of Findings:")
    print("=" * 60)
    print(f"Average bars per trade across strategies: {comparison_df['bars_per_trade'].mean():.1f}")
    print(f"Average bar-based win rate: {comparison_df['bar_win_rate'].mean():.1%}")
    print(f"Average trade-based win rate: {comparison_df['trade_win_rate'].mean():.1%}")
    print(f"\nRatio of trade win rate to bar win rate: {comparison_df['trade_win_rate'].mean() / comparison_df['bar_win_rate'].mean():.1f}x")
    
    print("\n🎯 Recommendations:")
    print("1. Always use trade-based metrics for strategy comparison")
    print("2. Bar-based metrics are useful for understanding intra-trade dynamics")
    print("3. Stop loss analysis must use trade-based metrics for consistency")
    print("4. Profit factor calculations should use completed trades, not individual bars")
    
    # Export comparison data
    comparison_df.to_csv(run_dir / 'bar_vs_trade_metrics.csv', index=False)
    print(f"\n✅ Detailed comparison saved to: bar_vs_trade_metrics.csv")
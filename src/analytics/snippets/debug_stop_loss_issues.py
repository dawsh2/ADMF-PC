# Debug Stop Loss Issues
# Analyzes why trailing stops aren't triggering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def debug_stop_behavior(strategy_hash, trace_path, market_data, execution_cost_bps=1.0):
    """
    Debug why stops aren't triggering by examining individual trades
    """
    # Extract trades
    trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)
    
    if len(trades) == 0:
        print("No trades found!")
        return
    
    print(f"\n📊 Trade Statistics:")
    print(f"Total trades: {len(trades)}")
    print(f"Average duration: {trades['duration_minutes'].mean():.1f} minutes")
    print(f"Max duration: {trades['duration_minutes'].max():.1f} minutes")
    print(f"Min duration: {trades['duration_minutes'].min():.1f} minutes")
    
    # Analyze trade drawdowns
    print("\n📉 Intra-trade Drawdown Analysis:")
    
    trade_drawdowns = []
    
    for idx, trade in trades.head(10).iterrows():  # Analyze first 10 trades
        # Get intraday prices
        trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
        
        if len(trade_prices) < 2:
            continue
            
        entry_price = trade['entry_price']
        direction = trade['direction']
        
        # Calculate maximum adverse excursion (MAE)
        if direction == 1:  # Long
            worst_price = trade_prices['low'].min()
            mae = (entry_price - worst_price) / entry_price
            best_price = trade_prices['high'].max()
            mfe = (best_price - entry_price) / entry_price  # Maximum favorable excursion
        else:  # Short
            worst_price = trade_prices['high'].max()
            mae = (worst_price - entry_price) / entry_price
            best_price = trade_prices['low'].min()
            mfe = (entry_price - best_price) / entry_price
        
        trade_drawdowns.append({
            'trade_idx': idx,
            'duration_minutes': trade['duration_minutes'],
            'direction': 'Long' if direction == 1 else 'Short',
            'mae_pct': mae * 100,
            'mfe_pct': mfe * 100,
            'final_return_pct': trade['net_return'] * 100,
            'bars_in_trade': len(trade_prices)
        })
        
        # Show details for first 3 trades
        if idx < 3:
            print(f"\nTrade {idx+1} ({trade_drawdowns[-1]['direction']}):")
            print(f"  Duration: {trade_drawdowns[-1]['duration_minutes']:.1f} min ({trade_drawdowns[-1]['bars_in_trade']} bars)")
            print(f"  Max Adverse Excursion: -{trade_drawdowns[-1]['mae_pct']:.2f}%")
            print(f"  Max Favorable Excursion: +{trade_drawdowns[-1]['mfe_pct']:.2f}%")
            print(f"  Final Return: {trade_drawdowns[-1]['final_return_pct']:.2f}%")
            
            # Check why stops wouldn't trigger
            for stop_level in [0.25, 0.5, 1.0, 2.0]:
                if mae * 100 > stop_level:
                    print(f"  ⚠️ Would hit {stop_level}% stop (MAE: {mae*100:.2f}%)")
                else:
                    print(f"  ✓ Would NOT hit {stop_level}% stop")
    
    drawdown_df = pd.DataFrame(trade_drawdowns)
    
    # Visualize trade behavior
    if len(drawdown_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. MAE distribution
        ax = axes[0, 0]
        ax.hist(drawdown_df['mae_pct'], bins=20, alpha=0.7, color='red')
        ax.axvline(0.5, color='black', linestyle='--', label='0.5% stop')
        ax.axvline(1.0, color='black', linestyle='--', label='1.0% stop')
        ax.axvline(2.0, color='black', linestyle='--', label='2.0% stop')
        ax.set_xlabel('Maximum Adverse Excursion (%)')
        ax.set_ylabel('Count')
        ax.set_title('Trade Drawdown Distribution')
        ax.legend()
        
        # 2. MAE vs Final Return
        ax = axes[0, 1]
        ax.scatter(drawdown_df['mae_pct'], drawdown_df['final_return_pct'], alpha=0.6)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, label='1% stop level')
        ax.set_xlabel('Maximum Adverse Excursion (%)')
        ax.set_ylabel('Final Return (%)')
        ax.set_title('Drawdown vs Final Outcome')
        ax.legend()
        
        # 3. Trade duration distribution
        ax = axes[1, 0]
        ax.hist(drawdown_df['duration_minutes'], bins=20, alpha=0.7)
        ax.set_xlabel('Trade Duration (minutes)')
        ax.set_ylabel('Count')
        ax.set_title('Trade Duration Distribution')
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = "Stop Loss Trigger Analysis:\n\n"
        for stop_level in [0.25, 0.5, 1.0, 2.0]:
            would_trigger = (drawdown_df['mae_pct'] > stop_level).sum()
            pct_triggered = would_trigger / len(drawdown_df) * 100
            summary_text += f"{stop_level:>4.2f}% stop: {would_trigger:>3} trades ({pct_triggered:>5.1f}%)\n"
        
        summary_text += f"\nAverage MAE: {drawdown_df['mae_pct'].mean():.2f}%"
        summary_text += f"\nMedian MAE: {drawdown_df['mae_pct'].median():.2f}%"
        summary_text += f"\nMax MAE: {drawdown_df['mae_pct'].max():.2f}%"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=12, family='monospace')
        
        plt.tight_layout()
        plt.show()
    
    return drawdown_df

# Debug top strategies
if len(top_overall) > 0:
    print("🔍 Debugging Stop Loss Behavior")
    print("=" * 80)
    print("Analyzing why trailing stops aren't triggering...\n")
    
    # Focus on the strategy with most trades
    most_active = performance_df.nlargest(1, 'num_trades').iloc[0]
    print(f"Analyzing most active strategy: {most_active['strategy_type']} - {most_active['strategy_hash'][:8]}")
    print(f"Total trades: {most_active['num_trades']}")
    
    drawdown_analysis = debug_stop_behavior(
        most_active['strategy_hash'],
        most_active['trace_path'],
        market_data,
        execution_cost_bps
    )
    
    # Key insights
    print("\n💡 Key Insights:")
    print("1. If MAE is very small, stops won't trigger")
    print("2. Short duration trades may not have enough price movement")
    print("3. Bollinger Bands with std_dev=3.0 create very wide bands")
    print("4. Consider tighter parameters or different strategy types for more active trading")
    
    # Recommendations
    print("\n🎯 Recommendations:")
    print("1. Test with tighter Bollinger Band parameters (std_dev=1.0-2.0)")
    print("2. Try strategies designed for higher frequency (momentum, mean reversion)")
    print("3. Use 1-minute data for more granular entry/exit")
    print("4. Consider volatility-adjusted position sizing instead of fixed stops")
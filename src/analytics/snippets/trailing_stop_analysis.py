# Trailing Stop Loss Analysis
# Analyzes how trailing stops could improve strategy performance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_trailing_stop(strategy_hash, trace_path, market_data, trailing_stop_levels, execution_cost_bps=1.0):
    """
    Analyze impact of trailing stop losses on strategy performance.
    
    Trailing stop: Stop loss that moves up with favorable price movement but never down.
    
    Args:
        strategy_hash: Strategy identifier
        trace_path: Path to trace file
        market_data: Market price data with high/low
        trailing_stop_levels: List of trailing stop percentages (e.g., [1.0, 2.0, 3.0])
        execution_cost_bps: Execution cost in basis points
    
    Returns:
        DataFrame with performance metrics for each trailing stop level
    """
    # Extract original trades
    trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)
    
    if len(trades) == 0:
        return pd.DataFrame()
    
    results = []
    
    for ts_pct in trailing_stop_levels:
        ts_decimal = ts_pct / 100
        
        trades_with_ts = []
        stopped_out_count = 0
        
        # Process each trade with trailing stop
        for _, trade in trades.iterrows():
            # Get intraday prices for this trade
            trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
            
            if len(trade_prices) == 0:
                continue
            
            entry_price = trade['entry_price']
            direction = trade['direction']
            
            # Initialize trailing stop
            if direction == 1:  # Long position
                # Initial stop is entry price minus stop distance
                trailing_stop_price = entry_price * (1 - ts_decimal)
                best_price = entry_price
            else:  # Short position
                # Initial stop is entry price plus stop distance
                trailing_stop_price = entry_price * (1 + ts_decimal)
                best_price = entry_price
            
            # Track through the trade
            stopped = False
            exit_price = trade['exit_price']
            exit_time = trade['exit_time']
            exit_idx = trade['exit_idx']
            
            for idx, bar in trade_prices.iterrows():
                if direction == 1:  # Long
                    # Update best price and trailing stop
                    if bar['high'] > best_price:
                        best_price = bar['high']
                        # Move stop up (but never down)
                        new_stop = best_price * (1 - ts_decimal)
                        trailing_stop_price = max(trailing_stop_price, new_stop)
                    
                    # Check if stopped out
                    if bar['low'] <= trailing_stop_price:
                        stopped = True
                        stopped_out_count += 1
                        exit_price = trailing_stop_price
                        exit_time = bar['timestamp']
                        exit_idx = idx
                        break
                        
                else:  # Short
                    # Update best price and trailing stop
                    if bar['low'] < best_price:
                        best_price = bar['low']
                        # Move stop down (but never up)
                        new_stop = best_price * (1 + ts_decimal)
                        trailing_stop_price = min(trailing_stop_price, new_stop)
                    
                    # Check if stopped out
                    if bar['high'] >= trailing_stop_price:
                        stopped = True
                        stopped_out_count += 1
                        exit_price = trailing_stop_price
                        exit_time = bar['timestamp']
                        exit_idx = idx
                        break
            
            # Calculate return with trailing stop
            if direction == 1:  # Long
                raw_return = (exit_price - entry_price) / entry_price
            else:  # Short
                raw_return = (entry_price - exit_price) / entry_price
            
            # Apply execution costs
            net_return = raw_return - trade['execution_cost']
            
            trade_result = trade.copy()
            trade_result['raw_return'] = raw_return
            trade_result['net_return'] = net_return
            trade_result['trailing_stopped'] = stopped
            trade_result['best_price'] = best_price
            trade_result['final_stop_price'] = trailing_stop_price
            if stopped:
                trade_result['exit_price'] = exit_price
                trade_result['exit_time'] = exit_time
                trade_result['exit_idx'] = exit_idx
            
            trades_with_ts.append(trade_result)
        
        trades_with_ts_df = pd.DataFrame(trades_with_ts)
        
        if len(trades_with_ts_df) > 0:
            # Calculate metrics
            winning_trades = trades_with_ts_df[trades_with_ts_df['net_return'] > 0]
            losing_trades = trades_with_ts_df[trades_with_ts_df['net_return'] <= 0]
            
            # Profit factor
            if len(losing_trades) > 0 and losing_trades['net_return'].sum() != 0:
                profit_factor = winning_trades['net_return'].sum() / abs(losing_trades['net_return'].sum())
            else:
                profit_factor = 999.99 if len(winning_trades) > 0 else 0
            
            # Calculate cumulative return
            trades_with_ts_df['cum_return'] = (1 + trades_with_ts_df['net_return']).cumprod()
            total_return = trades_with_ts_df['cum_return'].iloc[-1] - 1
            
            # Sharpe ratio
            if trades_with_ts_df['net_return'].std() > 0:
                days_in_data = (trades_with_ts_df['exit_time'].max() - trades_with_ts_df['entry_time'].min()).days
                if days_in_data > 0:
                    trades_per_day = len(trades_with_ts_df) / days_in_data
                    annualization_factor = np.sqrt(252 * trades_per_day)
                else:
                    annualization_factor = np.sqrt(252)
                sharpe = trades_with_ts_df['net_return'].mean() / trades_with_ts_df['net_return'].std() * annualization_factor
            else:
                sharpe = 0
            
            results.append({
                'trailing_stop_pct': ts_pct,
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'avg_return_per_trade': trades_with_ts_df['net_return'].mean(),
                'win_rate': len(winning_trades) / len(trades_with_ts_df),
                'profit_factor': profit_factor,
                'num_trades': len(trades_with_ts_df),
                'stopped_out_count': stopped_out_count,
                'stopped_out_rate': stopped_out_count / len(trades_with_ts_df),
                'avg_winner': winning_trades['net_return'].mean() if len(winning_trades) > 0 else 0,
                'avg_loser': losing_trades['net_return'].mean() if len(losing_trades) > 0 else 0,
                'max_winner': winning_trades['net_return'].max() if len(winning_trades) > 0 else 0,
                'max_loser': losing_trades['net_return'].min() if len(losing_trades) > 0 else 0
            })
    
    return pd.DataFrame(results)

# Main analysis
if len(performance_df) > 0 and len(top_overall) > 0:
    print("🎯 Trailing Stop Loss Analysis")
    print("=" * 80)
    print("Trailing stops move with favorable price movement to lock in profits\n")
    
    # Define trailing stop levels to test
    trailing_stop_levels = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    # Analyze top strategies with positive profit factor
    strategies_to_analyze = []
    
    # First, add any strategies with profit factor > 1.0
    profitable_strategies = performance_df[performance_df.get('profit_factor', 0) > 1.0]
    if len(profitable_strategies) > 0:
        strategies_to_analyze.extend(profitable_strategies.to_dict('records'))
    
    # Then add top strategies by Sharpe (even if profit factor < 1)
    top_by_sharpe = performance_df.nlargest(5, 'sharpe_ratio')
    for _, strategy in top_by_sharpe.iterrows():
        if strategy['strategy_hash'] not in [s['strategy_hash'] for s in strategies_to_analyze]:
            strategies_to_analyze.append(strategy.to_dict())
    
    # Limit to 10 strategies
    strategies_to_analyze = strategies_to_analyze[:10]
    
    all_results = []
    strategy_details = []
    
    for idx, strategy in enumerate(strategies_to_analyze):
        print(f"\nAnalyzing strategy {idx+1}/{len(strategies_to_analyze)}: {strategy['strategy_type']} - {strategy['strategy_hash'][:8]}")
        
        # Show current performance
        print(f"  Current Performance:")
        print(f"    Sharpe: {strategy.get('sharpe_ratio', 0):.2f}")
        print(f"    Total Return: {strategy.get('total_return', 0)*100:.2f}%")
        print(f"    Win Rate: {strategy.get('win_rate', 0)*100:.1f}%")
        print(f"    Profit Factor: {strategy.get('profit_factor', 0):.2f}")
        
        # Analyze with trailing stops
        results = analyze_trailing_stop(
            strategy['strategy_hash'],
            strategy['trace_path'],
            market_data,
            trailing_stop_levels,
            execution_cost_bps
        )
        
        if results is not None and len(results) > 0:
            # Find optimal trailing stop
            optimal_idx = results['sharpe_ratio'].idxmax()
            optimal = results.iloc[optimal_idx]
            
            print(f"  Optimal Trailing Stop: {optimal['trailing_stop_pct']:.2f}%")
            print(f"    New Sharpe: {optimal['sharpe_ratio']:.2f} (vs {strategy.get('sharpe_ratio', 0):.2f})")
            print(f"    New Return: {optimal['total_return']*100:.2f}% (vs {strategy.get('total_return', 0)*100:.2f}%)")
            print(f"    New Win Rate: {optimal['win_rate']*100:.1f}% (vs {strategy.get('win_rate', 0)*100:.1f}%)")
            print(f"    New Profit Factor: {optimal['profit_factor']:.2f} (vs {strategy.get('profit_factor', 0):.2f})")
            print(f"    Stopped Out: {optimal['stopped_out_rate']*100:.1f}% of trades")
            
            results['strategy_hash'] = strategy['strategy_hash']
            results['strategy_type'] = strategy['strategy_type']
            results['base_sharpe'] = strategy.get('sharpe_ratio', 0)
            results['base_return'] = strategy.get('total_return', 0)
            results['base_profit_factor'] = strategy.get('profit_factor', 0)
            
            all_results.append(results)
            
            strategy_details.append({
                'hash': strategy['strategy_hash'][:8],
                'type': strategy['strategy_type'],
                'params': f"period={strategy.get('period', 'N/A')}, std_dev={strategy.get('std_dev', 'N/A')}",
                'base_sharpe': strategy.get('sharpe_ratio', 0),
                'base_return': strategy.get('total_return', 0),
                'optimal_ts': optimal['trailing_stop_pct'],
                'optimal_sharpe': optimal['sharpe_ratio'],
                'optimal_return': optimal['total_return']
            })
    
    # Combine and visualize results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Sharpe ratio improvement
        ax = axes[0, 0]
        for strategy_hash in combined_results['strategy_hash'].unique()[:5]:  # Top 5
            strategy_data = combined_results[combined_results['strategy_hash'] == strategy_hash]
            improvement = strategy_data['sharpe_ratio'] - strategy_data['base_sharpe'].iloc[0]
            ax.plot(strategy_data['trailing_stop_pct'], improvement, 
                   marker='o', label=f"S{list(combined_results['strategy_hash'].unique()).index(strategy_hash)+1}")
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Trailing Stop %')
        ax.set_ylabel('Sharpe Ratio Improvement')
        ax.set_title('Sharpe Ratio Improvement vs Base Strategy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Total return by trailing stop
        ax = axes[0, 1]
        pivot_return = combined_results.groupby('trailing_stop_pct')['total_return'].mean()
        ax.plot(pivot_return.index, pivot_return.values * 100, marker='o', linewidth=2)
        ax.set_xlabel('Trailing Stop %')
        ax.set_ylabel('Average Total Return %')
        ax.set_title('Average Return Across All Strategies')
        ax.grid(True, alpha=0.3)
        
        # 3. Win rate by trailing stop
        ax = axes[0, 2]
        pivot_winrate = combined_results.groupby('trailing_stop_pct')['win_rate'].mean()
        ax.plot(pivot_winrate.index, pivot_winrate.values * 100, marker='o', linewidth=2)
        ax.set_xlabel('Trailing Stop %')
        ax.set_ylabel('Average Win Rate %')
        ax.set_title('Win Rate vs Trailing Stop Level')
        ax.grid(True, alpha=0.3)
        
        # 4. Profit factor improvement
        ax = axes[1, 0]
        pivot_pf = combined_results.groupby('trailing_stop_pct')['profit_factor'].mean()
        ax.plot(pivot_pf.index, pivot_pf.values, marker='o', linewidth=2)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Breakeven')
        ax.set_xlabel('Trailing Stop %')
        ax.set_ylabel('Average Profit Factor')
        ax.set_title('Profit Factor vs Trailing Stop Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Stop out rate
        ax = axes[1, 1]
        pivot_stoprate = combined_results.groupby('trailing_stop_pct')['stopped_out_rate'].mean()
        ax.plot(pivot_stoprate.index, pivot_stoprate.values * 100, marker='o', linewidth=2)
        ax.set_xlabel('Trailing Stop %')
        ax.set_ylabel('% of Trades Stopped Out')
        ax.set_title('Percentage of Trades Hit Trailing Stop')
        ax.grid(True, alpha=0.3)
        
        # 6. Summary table
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = "Optimal Trailing Stops by Strategy:\n\n"
        for detail in strategy_details[:5]:  # Top 5
            summary_text += f"{detail['type']} ({detail['hash']}):\n"
            summary_text += f"  Optimal TS: {detail['optimal_ts']:.2f}%\n"
            summary_text += f"  Sharpe: {detail['base_sharpe']:.2f} → {detail['optimal_sharpe']:.2f}\n"
            summary_text += f"  Return: {detail['base_return']*100:.1f}% → {detail['optimal_return']*100:.1f}%\n\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10, family='monospace')
        
        plt.tight_layout()
        plt.show()
        
        # Key insights
        print("\n📊 Key Insights from Trailing Stop Analysis:")
        print("=" * 60)
        
        # Find overall optimal trailing stop
        avg_sharpe_by_ts = combined_results.groupby('trailing_stop_pct')['sharpe_ratio'].mean()
        optimal_ts_overall = avg_sharpe_by_ts.idxmax()
        
        print(f"1. Optimal trailing stop across all strategies: {optimal_ts_overall:.2f}%")
        print(f"2. Average Sharpe improvement at optimal: {(avg_sharpe_by_ts.max() - combined_results['base_sharpe'].mean()):.2f}")
        
        # Count strategies that improve with trailing stops
        improvement_count = 0
        for strategy_hash in combined_results['strategy_hash'].unique():
            strategy_data = combined_results[combined_results['strategy_hash'] == strategy_hash]
            if strategy_data['sharpe_ratio'].max() > strategy_data['base_sharpe'].iloc[0]:
                improvement_count += 1
        
        print(f"3. {improvement_count}/{len(combined_results['strategy_hash'].unique())} strategies improve with trailing stops")
        
        # Average win rate change
        base_avg_winrate = performance_df['win_rate'].mean()
        ts_avg_winrate = combined_results[combined_results['trailing_stop_pct'] == optimal_ts_overall]['win_rate'].mean()
        print(f"4. Win rate change at optimal TS: {base_avg_winrate*100:.1f}% → {ts_avg_winrate*100:.1f}%")
        
        # Save detailed results
        combined_results.to_csv(run_dir / 'trailing_stop_analysis.csv', index=False)
        print(f"\n✅ Detailed results saved to: trailing_stop_analysis.csv")
        
        # Recommendations
        print("\n🎯 Recommendations:")
        print(f"1. Consider implementing a {optimal_ts_overall:.2f}% trailing stop for most strategies")
        print("2. Tighter trailing stops (0.5-1.0%) work well for high-frequency strategies")
        print("3. Wider trailing stops (2-3%) better for trend-following with larger moves")
        print("4. Monitor stop-out rates - if >50%, the stop may be too tight")
        
else:
    print("⚠️ No performance data available for trailing stop analysis")
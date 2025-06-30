# Micro-Movement Trading Analysis
# Optimized for strategies with <0.2% average movements

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration for micro-movements
MICRO_STOP_LEVELS = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3]
PROFIT_TARGET_LEVELS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

def analyze_micro_movements(strategy_hash, trace_path, market_data, execution_cost_bps=1.0):
    """
    Analyze strategies with very small price movements
    Tests both tight stops and profit targets
    """
    # Extract trades
    trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)
    
    if len(trades) == 0:
        return None
    
    results = []
    
    # Test stop loss + profit target combinations
    for stop_pct in MICRO_STOP_LEVELS:
        for target_pct in PROFIT_TARGET_LEVELS:
            if target_pct <= stop_pct:
                continue  # Skip invalid combinations
            
            trades_modified = []
            stops_hit = 0
            targets_hit = 0
            
            for _, trade in trades.iterrows():
                # Get intraday prices
                trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
                
                if len(trade_prices) == 0:
                    continue
                
                entry_price = trade['entry_price']
                direction = trade['direction']
                
                # Calculate stop and target prices
                if direction == 1:  # Long
                    stop_price = entry_price * (1 - stop_pct/100)
                    target_price = entry_price * (1 + target_pct/100)
                else:  # Short
                    stop_price = entry_price * (1 + stop_pct/100)
                    target_price = entry_price * (1 - target_pct/100)
                
                # Check each bar for stop or target hit
                exit_price = trade['exit_price']
                exit_type = 'signal'
                
                for idx, bar in trade_prices.iterrows():
                    if direction == 1:  # Long
                        if bar['low'] <= stop_price:
                            exit_price = stop_price
                            exit_type = 'stop'
                            stops_hit += 1
                            break
                        elif bar['high'] >= target_price:
                            exit_price = target_price
                            exit_type = 'target'
                            targets_hit += 1
                            break
                    else:  # Short
                        if bar['high'] >= stop_price:
                            exit_price = stop_price
                            exit_type = 'stop'
                            stops_hit += 1
                            break
                        elif bar['low'] <= target_price:
                            exit_price = target_price
                            exit_type = 'target'
                            targets_hit += 1
                            break
                
                # Calculate return
                if direction == 1:
                    raw_return = (exit_price - entry_price) / entry_price
                else:
                    raw_return = (entry_price - exit_price) / entry_price
                
                net_return = raw_return - trade['execution_cost']
                
                trades_modified.append({
                    'net_return': net_return,
                    'exit_type': exit_type
                })
            
            trades_df = pd.DataFrame(trades_modified)
            
            if len(trades_df) > 0:
                # Calculate metrics
                total_return = (1 + trades_df['net_return']).cumprod().iloc[-1] - 1
                win_rate = (trades_df['net_return'] > 0).mean()
                
                # Calculate Sharpe
                if trades_df['net_return'].std() > 0:
                    sharpe = trades_df['net_return'].mean() / trades_df['net_return'].std() * np.sqrt(252 * 78)
                else:
                    sharpe = 0
                
                results.append({
                    'stop_pct': stop_pct,
                    'target_pct': target_pct,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe,
                    'win_rate': win_rate,
                    'stops_hit': stops_hit,
                    'targets_hit': targets_hit,
                    'stops_pct': stops_hit / len(trades_df) * 100,
                    'targets_pct': targets_hit / len(trades_df) * 100,
                    'signal_exits_pct': (len(trades_df) - stops_hit - targets_hit) / len(trades_df) * 100
                })
    
    return pd.DataFrame(results)

# Main analysis
if len(performance_df) > 0:
    print("🔬 Micro-Movement Trading Analysis")
    print("=" * 80)
    print(f"Testing stop levels: {MICRO_STOP_LEVELS}")
    print(f"Testing profit targets: {PROFIT_TARGET_LEVELS}")
    
    # Get high-frequency strategies
    trading_days = len(market_data['timestamp'].dt.date.unique())
    high_freq_df = performance_df[performance_df['num_trades'] >= MIN_TRADES_PER_DAY * trading_days]
    
    if len(high_freq_df) > 0:
        # Analyze top 5 high-frequency strategies
        print("\n📊 Analyzing Stop + Target Combinations:")
        
        all_results = []
        
        for idx, row in high_freq_df.head(5).iterrows():
            print(f"\nStrategy {idx+1}: {row['strategy_type']} - {row['strategy_hash'][:8]}")
            print(f"  Base performance: Sharpe={row['sharpe_ratio']:.2f}, Return={row['total_return']*100:.2f}%")
            
            # Analyze with micro stops and targets
            micro_results = analyze_micro_movements(
                row['strategy_hash'],
                row['trace_path'],
                market_data,
                execution_cost_bps
            )
            
            if micro_results is not None and len(micro_results) > 0:
                # Find optimal combination
                optimal_idx = micro_results['sharpe_ratio'].idxmax()
                optimal = micro_results.iloc[optimal_idx]
                
                print(f"  Optimal: Stop={optimal['stop_pct']:.3f}%, Target={optimal['target_pct']:.2f}%")
                print(f"  New Sharpe: {optimal['sharpe_ratio']:.2f} (was {row['sharpe_ratio']:.2f})")
                print(f"  New Return: {optimal['total_return']*100:.2f}% (was {row['total_return']*100:.2f}%)")
                print(f"  Exit breakdown: Stops={optimal['stops_pct']:.1f}%, Targets={optimal['targets_pct']:.1f}%, Signal={optimal['signal_exits_pct']:.1f}%")
                
                micro_results['strategy_hash'] = row['strategy_hash'][:8]
                all_results.append(micro_results)
        
        # Visualize results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            
            # Create visualizations
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Sharpe ratio heatmap (for first strategy)
            ax = axes[0, 0]
            first_strategy = all_results[0]
            pivot_sharpe = first_strategy.pivot_table(
                values='sharpe_ratio',
                index='stop_pct',
                columns='target_pct'
            )
            sns.heatmap(pivot_sharpe, cmap='RdYlGn', center=0, ax=ax, 
                       cbar_kws={'label': 'Sharpe Ratio'})
            ax.set_title('Sharpe Ratio by Stop/Target Combination')
            ax.set_xlabel('Profit Target %')
            ax.set_ylabel('Stop Loss %')
            
            # 2. Win rate heatmap
            ax = axes[0, 1]
            pivot_winrate = first_strategy.pivot_table(
                values='win_rate',
                index='stop_pct',
                columns='target_pct'
            )
            sns.heatmap(pivot_winrate * 100, cmap='RdYlGn', center=50, ax=ax,
                       cbar_kws={'label': 'Win Rate %'})
            ax.set_title('Win Rate by Stop/Target Combination')
            ax.set_xlabel('Profit Target %')
            ax.set_ylabel('Stop Loss %')
            
            # 3. Exit type distribution
            ax = axes[1, 0]
            # Average across all strategies
            avg_exits = combined_results.groupby(['stop_pct', 'target_pct'])[['stops_pct', 'targets_pct', 'signal_exits_pct']].mean()
            optimal_combos = combined_results.groupby('strategy_hash')['sharpe_ratio'].idxmax()
            
            # Show exit distribution for optimal combinations
            exit_data = []
            for strategy, idx in optimal_combos.items():
                row = combined_results.iloc[idx]
                exit_data.append({
                    'Strategy': strategy,
                    'Stops': row['stops_pct'],
                    'Targets': row['targets_pct'],
                    'Signal': row['signal_exits_pct']
                })
            
            exit_df = pd.DataFrame(exit_data)
            exit_df.set_index('Strategy').plot(kind='bar', stacked=True, ax=ax)
            ax.set_ylabel('Exit Type %')
            ax.set_title('Exit Type Distribution (Optimal Settings)')
            ax.legend(title='Exit Type')
            
            # 4. Improvement summary
            ax = axes[1, 1]
            ax.axis('off')
            
            summary_text = "Optimal Stop/Target Combinations:\n\n"
            for strategy in combined_results['strategy_hash'].unique()[:5]:
                strategy_data = combined_results[combined_results['strategy_hash'] == strategy]
                if len(strategy_data) > 0:
                    optimal_idx = strategy_data['sharpe_ratio'].idxmax()
                    optimal = strategy_data.iloc[optimal_idx]
                    
                    # Get original performance
                    orig = high_freq_df[high_freq_df['strategy_hash'].str.contains(strategy)]
                if len(orig) > 0:
                    orig_sharpe = orig.iloc[0]['sharpe_ratio']
                    improvement = optimal['sharpe_ratio'] - orig_sharpe
                    
                    summary_text += f"{strategy}:\n"
                    summary_text += f"  Stop: {optimal['stop_pct']:.3f}%, Target: {optimal['target_pct']:.2f}%\n"
                    summary_text += f"  Sharpe: {orig_sharpe:.2f} → {optimal['sharpe_ratio']:.2f} "
                    summary_text += f"({improvement:+.2f})\n\n"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=11, family='monospace')
            
            plt.tight_layout()
            plt.show()
            
            # Key insights
            print("\n💡 Key Insights:")
            print("=" * 60)
            
            # Find most common optimal settings
            optimal_stops = []
            optimal_targets = []
            for strategy in combined_results['strategy_hash'].unique():
                strategy_data = combined_results[combined_results['strategy_hash'] == strategy]
                optimal_idx = strategy_data['sharpe_ratio'].idxmax()
                optimal = strategy_data.iloc[optimal_idx]
                optimal_stops.append(optimal['stop_pct'])
                optimal_targets.append(optimal['target_pct'])
            
            print(f"1. Most common optimal stop: {np.median(optimal_stops):.3f}%")
            print(f"2. Most common optimal target: {np.median(optimal_targets):.2f}%")
            print(f"3. Stop/Target ratio: ~1:{np.median(optimal_targets)/np.median(optimal_stops):.1f}")
            
            # Save results
            combined_results.to_csv(run_dir / 'micro_movement_analysis.csv', index=False)
            print(f"\n✅ Saved analysis to: micro_movement_analysis.csv")
            
            print("\n🎯 Recommendations:")
            print("1. Use very tight stops (0.1-0.2%) to limit losses")
            print("2. Set profit targets at 2-3x stop distance")
            print("3. Focus on high win rate rather than big moves")
            print("4. Consider commission impact on such small moves")
            
    else:
        print("❌ No high-frequency strategies found")
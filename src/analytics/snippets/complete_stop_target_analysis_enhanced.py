# Complete Stop/Target Analysis - Enhanced Version
# Tests stop loss + profit target combinations on all strategies
# Includes test set validation and detailed metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from helpers import get_actual_trading_days

# Configuration
STOP_TARGET_PAIRS = [
    (0.05, 0.10),    # 2:1 reward/risk
    (0.075, 0.10),   # 1.33:1 (optimal from training)
    (0.10, 0.15),    # 1.5:1
    (0.10, 0.20),    # 2:1
    (0.15, 0.30),    # 2:1
    (0.20, 0.40),    # 2:1
    (0, 0),          # No stop/target (baseline)
]

def apply_stop_target(trades_df, stop_pct, target_pct, market_data):
    """Apply stop loss and profit target to trades"""
    if stop_pct == 0 and target_pct == 0:
        # No modification - return original
        return trades_df['net_return'].values, {'stop': 0, 'target': 0, 'signal': len(trades_df)}
    
    modified_returns = []
    exit_types = {'stop': 0, 'target': 0, 'signal': 0}
    
    for _, trade in trades_df.iterrows():
        trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
        
        if len(trade_prices) == 0:
            modified_returns.append(trade['net_return'])
            exit_types['signal'] += 1
            continue
        
        entry_price = trade['entry_price']
        direction = trade['direction']
        
        # Set stop and target prices
        if target_pct > 0:  # Use profit target
            if direction == 1:  # Long
                stop_price = entry_price * (1 - stop_pct/100) if stop_pct > 0 else 0
                target_price = entry_price * (1 + target_pct/100)
            else:  # Short
                stop_price = entry_price * (1 + stop_pct/100) if stop_pct > 0 else float('inf')
                target_price = entry_price * (1 - target_pct/100)
        else:  # Stop only
            if direction == 1:
                stop_price = entry_price * (1 - stop_pct/100)
                target_price = float('inf')
            else:
                stop_price = entry_price * (1 + stop_pct/100)
                target_price = 0
        
        # Check each bar for exit
        exit_price = trade['exit_price']
        exit_type = 'signal'
        
        for _, bar in trade_prices.iterrows():
            if direction == 1:  # Long
                if stop_pct > 0 and bar['low'] <= stop_price:
                    exit_price = stop_price
                    exit_type = 'stop'
                    break
                elif target_pct > 0 and bar['high'] >= target_price:
                    exit_price = target_price
                    exit_type = 'target'
                    break
            else:  # Short
                if stop_pct > 0 and bar['high'] >= stop_price:
                    exit_price = stop_price
                    exit_type = 'stop'
                    break
                elif target_pct > 0 and bar['low'] <= target_price:
                    exit_price = target_price
                    exit_type = 'target'
                    break
        
        exit_types[exit_type] += 1
        
        # Calculate return
        if direction == 1:
            raw_return = (exit_price - entry_price) / entry_price
        else:
            raw_return = (entry_price - exit_price) / entry_price
        
        net_return = raw_return - trade['execution_cost']
        modified_returns.append(net_return)
    
    return np.array(modified_returns), exit_types

def calculate_detailed_metrics(returns_array, exit_types, trades_per_day):
    """Calculate comprehensive performance metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['total_return'] = (1 + returns_array).prod() - 1
    metrics['win_rate'] = (returns_array > 0).mean()
    metrics['num_trades'] = len(returns_array)
    
    # Risk metrics
    metrics['return_mean'] = returns_array.mean()
    metrics['return_std'] = returns_array.std()
    metrics['max_drawdown'] = calculate_max_drawdown(returns_array)
    
    # Sharpe ratio
    if metrics['return_std'] > 0:
        metrics['sharpe_ratio'] = metrics['return_mean'] / metrics['return_std'] * np.sqrt(252 * trades_per_day)
    else:
        metrics['sharpe_ratio'] = 0
    
    # Exit type percentages
    total_exits = sum(exit_types.values())
    if total_exits > 0:
        metrics['stop_pct'] = exit_types['stop'] / total_exits * 100
        metrics['target_pct'] = exit_types['target'] / total_exits * 100
        metrics['signal_pct'] = exit_types['signal'] / total_exits * 100
    
    # Profit factor
    winners = returns_array[returns_array > 0]
    losers = returns_array[returns_array < 0]
    if len(losers) > 0 and losers.sum() != 0:
        metrics['profit_factor'] = winners.sum() / abs(losers.sum())
    else:
        metrics['profit_factor'] = np.inf if len(winners) > 0 else 0
    
    # Average win/loss
    metrics['avg_win'] = winners.mean() if len(winners) > 0 else 0
    metrics['avg_loss'] = losers.mean() if len(losers) > 0 else 0
    
    return metrics

def calculate_max_drawdown(returns_array):
    """Calculate maximum drawdown from returns array"""
    cumulative = (1 + returns_array).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

# Main analysis
print("🎯 Complete Stop/Target Analysis - Enhanced")
print("=" * 80)

# Check if this is training or test data
current_time_range = f"{market_data['timestamp'].min()} to {market_data['timestamp'].max()}"
print(f"\nAnalyzing data from: {current_time_range}")

if len(performance_df) > 0:
    # Basic statistics - use correct trading days calculation
    trading_days = get_actual_trading_days(performance_df, market_data)
    print(f"Dataset: {trading_days} trading days (actual trading period)")
    print(f"Total strategies: {len(performance_df)}")
    
    # Trade frequency stats
    performance_df['trades_per_day'] = performance_df['num_trades'] / trading_days
    print(f"\nTrade frequency distribution:")
    print(f"  Mean: {performance_df['trades_per_day'].mean():.2f} trades/day")
    print(f"  Median: {performance_df['trades_per_day'].median():.2f} trades/day")
    print(f"  Max: {performance_df['trades_per_day'].max():.2f} trades/day")
    print(f"  Strategies with >1 trade/day: {(performance_df['trades_per_day'] > 1).sum()}")
    
    # Get strategies to analyze (all with reasonable trade count)
    MIN_TOTAL_TRADES = 20  # Minimum trades for meaningful statistics
    valid_strategies = performance_df[performance_df['num_trades'] >= MIN_TOTAL_TRADES].copy()
    
    print(f"\nAnalyzing {len(valid_strategies)} strategies with at least {MIN_TOTAL_TRADES} trades")
    
    if len(valid_strategies) > 0:
        # Sort by trade count for better statistics
        valid_strategies = valid_strategies.sort_values('num_trades', ascending=False)
        
        # Analyze each strategy with different stop/target combinations
        all_results = []
        
        print("\n📊 Processing strategies...")
        
        for idx, row in valid_strategies.iterrows():
            # Show progress
            if idx % 10 == 0:
                print(f"  Processing strategy {idx+1}/{len(valid_strategies)}...")
            
            # Extract trades once
            trades = extract_trades(row['strategy_hash'], row['trace_path'], market_data, execution_cost_bps)
            
            if len(trades) < MIN_TOTAL_TRADES:
                continue
            
            strategy_results = {
                'strategy_hash': row['strategy_hash'],
                'strategy_type': row['strategy_type'],
                'num_trades': len(trades),
                'trades_per_day': len(trades) / trading_days,
                'base_sharpe': row['sharpe_ratio'],
                'base_return': row['total_return'],
                'base_win_rate': row.get('win_rate', 0),
                'period': row.get('period', 'N/A'),
                'std_dev': row.get('std_dev', 'N/A')
            }
            
            # Test each stop/target combination
            for stop_pct, target_pct in STOP_TARGET_PAIRS:
                # Apply stop/target
                returns_array, exit_types = apply_stop_target(trades, stop_pct, target_pct, market_data)
                
                # Calculate detailed metrics
                metrics = calculate_detailed_metrics(returns_array, exit_types, strategy_results['trades_per_day'])
                
                # Store results
                key = f"stop_{stop_pct}_target_{target_pct}"
                for metric_name, value in metrics.items():
                    strategy_results[f"{key}_{metric_name}"] = value
            
            all_results.append(strategy_results)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        if len(results_df) > 0:
            # Summary statistics for each configuration
            print("\n📈 Configuration Performance Summary:")
            print("=" * 80)
            
            config_summary = []
            
            for stop, target in STOP_TARGET_PAIRS:
                key = f"stop_{stop}_target_{target}"
                
                config_stats = {
                    'Stop %': stop,
                    'Target %': target,
                    'Avg Return %': results_df[f"{key}_total_return"].mean() * 100,
                    'Avg Sharpe': results_df[f"{key}_sharpe_ratio"].mean(),
                    'Avg Win Rate %': results_df[f"{key}_win_rate"].mean() * 100,
                    'Avg Profit Factor': results_df[f"{key}_profit_factor"][results_df[f"{key}_profit_factor"] != np.inf].mean(),
                    'Improves %': (results_df[f"{key}_sharpe_ratio"] > results_df['base_sharpe']).mean() * 100
                }
                
                config_summary.append(config_stats)
            
            config_df = pd.DataFrame(config_summary)
            print(config_df.to_string(index=False, float_format='%.2f'))
            
            # Focus on 0.075/0.1 combination
            print("\n🎯 Detailed Analysis: 0.075% Stop / 0.1% Target:")
            print("=" * 80)
            
            key_075_10 = "stop_0.075_target_0.1"
            
            # Performance distribution
            returns_075_10 = results_df[f"{key_075_10}_total_return"] * 100
            sharpes_075_10 = results_df[f"{key_075_10}_sharpe_ratio"]
            
            print(f"\nReturn Distribution:")
            print(f"  Mean: {returns_075_10.mean():.2f}%")
            print(f"  Median: {returns_075_10.median():.2f}%")
            print(f"  25th percentile: {returns_075_10.quantile(0.25):.2f}%")
            print(f"  75th percentile: {returns_075_10.quantile(0.75):.2f}%")
            print(f"  Positive returns: {(returns_075_10 > 0).sum()}/{len(returns_075_10)} ({(returns_075_10 > 0).mean()*100:.1f}%)")
            
            print(f"\nSharpe Ratio Distribution:")
            print(f"  Mean: {sharpes_075_10.mean():.2f}")
            print(f"  Median: {sharpes_075_10.median():.2f}")
            print(f"  >1.0: {(sharpes_075_10 > 1).sum()} strategies")
            print(f"  >2.0: {(sharpes_075_10 > 2).sum()} strategies")
            print(f"  >3.0: {(sharpes_075_10 > 3).sum()} strategies")
            
            # Exit type analysis
            avg_stop_rate = results_df[f"{key_075_10}_stop_pct"].mean()
            avg_target_rate = results_df[f"{key_075_10}_target_pct"].mean()
            avg_signal_rate = results_df[f"{key_075_10}_signal_pct"].mean()
            
            print(f"\nAverage Exit Types:")
            print(f"  Stops hit: {avg_stop_rate:.1f}%")
            print(f"  Targets hit: {avg_target_rate:.1f}%")
            print(f"  Signal exits: {avg_signal_rate:.1f}%")
            
            # Top performers
            print(f"\n🏆 Top 10 Performers with 0.075/0.1 Configuration:")
            top_10 = results_df.nlargest(10, f"{key_075_10}_sharpe_ratio")
            
            for i, (idx, row) in enumerate(top_10.iterrows()):
                print(f"\n{i+1}. {row['strategy_type']} - {row['strategy_hash'][:8]}")
                print(f"   Period: {row['period']}, Std Dev: {row['std_dev']}")
                print(f"   Trades: {row['num_trades']} ({row['trades_per_day']:.1f}/day)")
                print(f"   Return: {row[f'{key_075_10}_total_return']*100:.2f}%")
                print(f"   Sharpe: {row[f'{key_075_10}_sharpe_ratio']:.2f}")
                print(f"   Win Rate: {row[f'{key_075_10}_win_rate']*100:.1f}%")
                print(f"   Profit Factor: {row[f'{key_075_10}_profit_factor']:.2f}")
                print(f"   Max DD: {row[f'{key_075_10}_max_drawdown']*100:.2f}%")
            
            # Visualizations
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Sharpe improvement distribution
            ax = axes[0, 0]
            sharpe_improvements = []
            for stop, target in STOP_TARGET_PAIRS[:-1]:  # Exclude baseline
                key = f"stop_{stop}_target_{target}"
                improvements = results_df[f"{key}_sharpe_ratio"] - results_df['base_sharpe']
                sharpe_improvements.append({
                    'Config': f"{stop}/{target}",
                    'Mean Improvement': improvements.mean(),
                    'Positive %': (improvements > 0).mean() * 100
                })
            
            improvement_df = pd.DataFrame(sharpe_improvements)
            x = range(len(improvement_df))
            ax.bar(x, improvement_df['Mean Improvement'])
            ax.set_xticks(x)
            ax.set_xticklabels(improvement_df['Config'], rotation=45)
            ax.set_ylabel('Mean Sharpe Improvement')
            ax.set_title('Average Sharpe Improvement by Configuration')
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
            
            # 2. Return scatter: Base vs 0.075/0.1
            ax = axes[0, 1]
            base_returns = results_df['base_return'] * 100
            modified_returns = results_df['stop_0.075_target_0.1_total_return'] * 100
            
            ax.scatter(base_returns, modified_returns, alpha=0.6)
            ax.plot([-50, 100], [-50, 100], 'r--', alpha=0.5)  # y=x line
            ax.set_xlabel('Base Return %')
            ax.set_ylabel('Return with 0.075/0.1 %')
            ax.set_title('Return Comparison: Base vs 0.075/0.1 Stop/Target')
            ax.grid(True, alpha=0.3)
            
            # Add quadrant counts
            improved = ((modified_returns > 0) & (modified_returns > base_returns)).sum()
            ax.text(0.05, 0.95, f"Improved: {improved}/{len(results_df)}", 
                    transform=ax.transAxes, verticalalignment='top')
            
            # 3. Sharpe distribution histogram
            ax = axes[0, 2]
            ax.hist(results_df['base_sharpe'], bins=30, alpha=0.5, label='Base', color='blue')
            ax.hist(results_df['stop_0.075_target_0.1_sharpe_ratio'], bins=30, alpha=0.5, 
                   label='With 0.075/0.1', color='green')
            ax.set_xlabel('Sharpe Ratio')
            ax.set_ylabel('Count')
            ax.set_title('Sharpe Ratio Distribution')
            ax.legend()
            ax.axvline(0, color='red', linestyle='--', alpha=0.5)
            
            # 4. Exit type pie chart (average)
            ax = axes[1, 0]
            exit_data = [avg_stop_rate, avg_target_rate, avg_signal_rate]
            labels = ['Stops', 'Targets', 'Signals']
            colors = ['red', 'green', 'blue']
            ax.pie(exit_data, labels=labels, colors=colors, autopct='%1.1f%%')
            ax.set_title('Average Exit Type Distribution (0.075/0.1)')
            
            # 5. Win rate vs Sharpe scatter
            ax = axes[1, 1]
            win_rates = results_df['stop_0.075_target_0.1_win_rate'] * 100
            sharpes = results_df['stop_0.075_target_0.1_sharpe_ratio']
            
            scatter = ax.scatter(win_rates, sharpes, 
                               c=results_df['trades_per_day'], 
                               cmap='viridis', alpha=0.6)
            ax.set_xlabel('Win Rate %')
            ax.set_ylabel('Sharpe Ratio')
            ax.set_title('Win Rate vs Sharpe Ratio (0.075/0.1)')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Trades/Day')
            
            # 6. Summary table
            ax = axes[1, 2]
            ax.axis('off')
            
            summary_text = "📊 Summary Statistics\n\n"
            summary_text += f"Dataset: {current_time_range.split(' to ')[0][:10]} to\n"
            summary_text += f"         {current_time_range.split(' to ')[1][:10]}\n\n"
            summary_text += f"Strategies analyzed: {len(results_df)}\n"
            summary_text += f"Trading days: {trading_days}\n"
            summary_text += f"Total trades: {results_df['num_trades'].sum():,}\n\n"
            
            summary_text += "0.075/0.1 Stop/Target Results:\n"
            summary_text += f"  Avg Return: {returns_075_10.mean():.2f}%\n"
            summary_text += f"  Avg Sharpe: {sharpes_075_10.mean():.2f}\n"
            summary_text += f"  Success Rate: {(returns_075_10 > 0).mean()*100:.1f}%\n"
            summary_text += f"  Best Sharpe: {sharpes_075_10.max():.2f}\n"
            summary_text += f"  Best Return: {returns_075_10.max():.2f}%\n\n"
            
            summary_text += "Execution Assumptions:\n"
            summary_text += f"  Cost: {execution_cost_bps} bps/trade\n"
            summary_text += "  Fill: At stop/target price\n"
            summary_text += "  No slippage or market impact"
            
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=10, family='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5))
            
            plt.tight_layout()
            plt.show()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_df.to_csv(run_dir / f'stop_target_analysis_{timestamp}.csv', index=False)
            
            # Save summary
            summary_df = pd.DataFrame({
                'Metric': ['Total Strategies', 'Avg Base Sharpe', 'Avg 0.075/0.1 Sharpe', 
                          'Avg Base Return %', 'Avg 0.075/0.1 Return %', 'Improvement Rate %'],
                'Value': [
                    len(results_df),
                    results_df['base_sharpe'].mean(),
                    sharpes_075_10.mean(),
                    results_df['base_return'].mean() * 100,
                    returns_075_10.mean(),
                    (sharpes_075_10 > results_df['base_sharpe']).mean() * 100
                ]
            })
            summary_df.to_csv(run_dir / f'stop_target_summary_{timestamp}.csv', index=False)
            
            print(f"\n✅ Analysis complete! Results saved to:")
            print(f"  - {run_dir}/stop_target_analysis_{timestamp}.csv")
            print(f"  - {run_dir}/stop_target_summary_{timestamp}.csv")
            
            # Final recommendations
            print("\n🎯 Key Findings:")
            print("=" * 60)
            
            if sharpes_075_10.mean() > results_df['base_sharpe'].mean():
                improvement_pct = ((sharpes_075_10.mean() - results_df['base_sharpe'].mean()) / 
                                 abs(results_df['base_sharpe'].mean()) * 100)
                print(f"✅ 0.075/0.1 stop/target improves average Sharpe by {improvement_pct:.1f}%")
            
            print(f"\n📋 Recommendations:")
            print("1. Focus on strategies with >1 trade/day for reliable statistics")
            print("2. Monitor actual fill quality vs theoretical stop/target prices")
            print("3. Consider market impact of frequent small trades")
            print("4. Test with real execution to validate assumptions")
            
    else:
        print("\n❌ No strategies found with sufficient trades for analysis")
else:
    print("❌ No performance data available")
    print("\nMake sure to run:")
    print("1. Load market data")
    print("2. Run strategy calculation")
    print("3. Run this analysis")
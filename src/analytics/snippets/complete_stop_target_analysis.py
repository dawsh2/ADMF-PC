# Complete Stop/Target Analysis - All Inclusive
# Tests stop loss + profit target combinations on all strategies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
from pathlib import Path

# Add the snippets directory to Python path for imports
sys.path.append(str(Path.cwd() / 'src' / 'analytics' / 'snippets'))

# Now we can import helpers
try:
    from helpers import get_actual_trading_days
except ImportError:
    # If import fails, define the function locally
    def get_actual_trading_days(performance_df: pd.DataFrame, market_data: pd.DataFrame) -> int:
        """
        Calculate actual trading days based on when strategies were active.
        This avoids the bug of using all days in the market data.
        """
        # Method 1: If performance_df has trade timing info
        if 'first_signal_date' in performance_df.columns and 'last_signal_date' in performance_df.columns:
            first_date = pd.to_datetime(performance_df['first_signal_date'].min())
            last_date = pd.to_datetime(performance_df['last_signal_date'].max())
            trading_days = len(pd.bdate_range(first_date, last_date))
            print(f"Trading period from signals: {first_date.date()} to {last_date.date()}")
            return trading_days
        
        # Method 2: Infer from trace files if available
        if 'trace_path' in performance_df.columns and len(performance_df) > 0:
            try:
                sample_trace = pd.read_parquet(performance_df.iloc[0]['trace_path'])
                if 'timestamp' in sample_trace.columns and len(sample_trace) > 0:
                    first_date = sample_trace['timestamp'].min()
                    last_date = sample_trace['timestamp'].max()
                    trading_days = len(pd.bdate_range(first_date, last_date))
                    print(f"Trading period from traces: {first_date.date()} to {last_date.date()}")
                    return trading_days
            except Exception as e:
                print(f"Could not read trace file: {e}")
        
        # Method 3: Last resort - warn and use market data
        print("⚠️ WARNING: Could not determine actual trading period from signals/traces.")
        print("⚠️ Using full market data range - this may overstate trading days!")
        trading_days = len(market_data['timestamp'].dt.date.unique())
        date_range = f"{market_data['timestamp'].min().date()} to {market_data['timestamp'].max().date()}"
        print(f"Market data period: {date_range}")
        return trading_days

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

# Main analysis
print("🎯 Complete Stop/Target Analysis")
print("=" * 80)

if len(performance_df) > 0:
    # Basic statistics - use correct trading days calculation
    trading_days = get_actual_trading_days(performance_df, market_data)
    print(f"Dataset: {trading_days} trading days (actual trading period)")
    print(f"Total strategies: {len(performance_df)}")
    
    # Trade frequency stats
    performance_df['trades_per_day'] = performance_df['num_trades'] / trading_days
    print(f"\nTrade frequency:")
    print(f"  Mean: {performance_df['trades_per_day'].mean():.2f} trades/day")
    print(f"  Max: {performance_df['trades_per_day'].max():.2f} trades/day")
    
    # Get top strategies (adjust number as needed)
    TOP_N = min(20, len(performance_df))
    top_strategies = performance_df.nlargest(TOP_N, 'num_trades')  # Sort by trade count for better statistics
    
    print(f"\nAnalyzing top {len(top_strategies)} strategies by trade count")
    
    # Analyze each strategy with different stop/target combinations
    all_results = []
    
    for idx, row in top_strategies.iterrows():
        # Extract trades once
        trades = extract_trades(row['strategy_hash'], row['trace_path'], market_data, execution_cost_bps)
        
        if len(trades) < 10:  # Skip if too few trades
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
            
            # Calculate metrics
            total_return = (1 + returns_array).prod() - 1
            win_rate = (returns_array > 0).mean()
            
            if returns_array.std() > 0:
                sharpe = returns_array.mean() / returns_array.std() * np.sqrt(252 * len(trades) / trading_days)
            else:
                sharpe = 0
            
            # Store results
            key = f"stop_{stop_pct}_target_{target_pct}"
            strategy_results[f"{key}_return"] = total_return
            strategy_results[f"{key}_sharpe"] = sharpe
            strategy_results[f"{key}_win_rate"] = win_rate
            strategy_results[f"{key}_stop_pct"] = exit_types['stop'] / len(returns_array) * 100
            strategy_results[f"{key}_target_pct"] = exit_types['target'] / len(returns_array) * 100
        
        all_results.append(strategy_results)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) > 0:
        # Find best stop/target for each strategy
        print("\n📊 Optimal Stop/Target Combinations:")
        print("=" * 80)
        
        optimal_configs = []
        
        for idx, row in results_df.iterrows():
            best_sharpe = row['base_sharpe']
            best_config = 'No stop/target'
            best_stop = 0
            best_target = 0
            
            for stop, target in STOP_TARGET_PAIRS:
                key = f"stop_{stop}_target_{target}"
                if f"{key}_sharpe" in row and row[f"{key}_sharpe"] > best_sharpe:
                    best_sharpe = row[f"{key}_sharpe"]
                    best_config = f"Stop={stop}%, Target={target}%"
                    best_stop = stop
                    best_target = target
            
            optimal_configs.append({
                'strategy': f"{row['strategy_type']}_{row['strategy_hash'][:8]}",
                'period': row['period'],
                'std_dev': row['std_dev'],
                'trades': row['num_trades'],
                'trades_per_day': row['trades_per_day'],
                'base_sharpe': row['base_sharpe'],
                'base_return': row['base_return'] * 100,
                'best_config': best_config,
                'best_sharpe': best_sharpe,
                'best_return': row[f"stop_{best_stop}_target_{best_target}_return"] * 100 if best_stop > 0 or best_target > 0 else row['base_return'] * 100,
                'improvement': best_sharpe - row['base_sharpe']
            })
        
        optimal_df = pd.DataFrame(optimal_configs)
        
        # Show top 10 improvements
        print("\nTop 10 strategies by Sharpe improvement:")
        top_improvements = optimal_df.nlargest(10, 'improvement')
        
        for idx, row in top_improvements.iterrows():
            print(f"\n{row['strategy']} (period={row['period']}, std_dev={row['std_dev']})")
            print(f"  Trades: {row['trades']} ({row['trades_per_day']:.1f}/day)")
            print(f"  Base: Sharpe={row['base_sharpe']:.2f}, Return={row['base_return']:.2f}%")
            print(f"  Best: {row['best_config']} → Sharpe={row['best_sharpe']:.2f}, Return={row['best_return']:.2f}%")
            print(f"  Improvement: Sharpe +{row['improvement']:.2f}")
        
        # Aggregate analysis
        print("\n📈 Aggregate Analysis:")
        print("=" * 60)
        
        # Which stop/target works best overall?
        config_performance = {}
        
        for stop, target in STOP_TARGET_PAIRS:
            key = f"stop_{stop}_target_{target}"
            sharpe_col = f"{key}_sharpe"
            
            if sharpe_col in results_df.columns:
                avg_sharpe = results_df[sharpe_col].mean()
                avg_return = results_df[f"{key}_return"].mean()
                win_count = (results_df[sharpe_col] > results_df['base_sharpe']).sum()
                
                config_performance[f"{stop}/{target}"] = {
                    'avg_sharpe': avg_sharpe,
                    'avg_return': avg_return * 100,
                    'win_count': win_count,
                    'win_rate': win_count / len(results_df) * 100
                }
        
        print("\nAverage performance by stop/target configuration:")
        config_df = pd.DataFrame(config_performance).T
        config_df = config_df.sort_values('avg_sharpe', ascending=False)
        
        for config, metrics in config_df.iterrows():
            print(f"\nStop/Target = {config}%:")
            print(f"  Avg Sharpe: {metrics['avg_sharpe']:.2f}")
            print(f"  Avg Return: {metrics['avg_return']:.2f}%")
            print(f"  Improves {metrics['win_count']:.0f}/{len(results_df)} strategies ({metrics['win_rate']:.1f}%)")
        
        # Test the specific 0.075/0.1 combination
        print("\n🎯 Focus: 0.075% Stop / 0.1% Target Performance:")
        print("=" * 60)
        
        key_075_10 = "stop_0.075_target_0.1"
        if f"{key_075_10}_sharpe" in results_df.columns:
            # Performance stats
            avg_return_075_10 = results_df[f"{key_075_10}_return"].mean() * 100
            avg_sharpe_075_10 = results_df[f"{key_075_10}_sharpe"].mean()
            avg_stop_rate = results_df[f"{key_075_10}_stop_pct"].mean()
            avg_target_rate = results_df[f"{key_075_10}_target_pct"].mean()
            
            print(f"Average return: {avg_return_075_10:.2f}%")
            print(f"Average Sharpe: {avg_sharpe_075_10:.2f}")
            print(f"Average stop hit rate: {avg_stop_rate:.1f}%")
            print(f"Average target hit rate: {avg_target_rate:.1f}%")
            
            # Compare to base
            avg_base_return = results_df['base_return'].mean() * 100
            avg_base_sharpe = results_df['base_sharpe'].mean()
            
            print(f"\nImprovement over base:")
            print(f"  Return: {avg_base_return:.2f}% → {avg_return_075_10:.2f}% ({avg_return_075_10 - avg_base_return:+.2f}%)")
            print(f"  Sharpe: {avg_base_sharpe:.2f} → {avg_sharpe_075_10:.2f} ({avg_sharpe_075_10 - avg_base_sharpe:+.2f})")
            
            # Best performers with this config
            print(f"\nTop 5 performers with 0.075/0.1 stop/target:")
            top_with_config = results_df.nlargest(5, f"{key_075_10}_sharpe")
            
            for idx, row in top_with_config.iterrows():
                print(f"\n{row['strategy_type']} (period={row['period']}, std_dev={row['std_dev']})")
                print(f"  Return: {row[f'{key_075_10}_return']*100:.2f}%")
                print(f"  Sharpe: {row[f'{key_075_10}_sharpe']:.2f}")
                print(f"  Win Rate: {row[f'{key_075_10}_win_rate']*100:.1f}%")
                print(f"  Stops: {row[f'{key_075_10}_stop_pct']:.1f}%, Targets: {row[f'{key_075_10}_target_pct']:.1f}%")
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Sharpe improvement heatmap
        ax = axes[0, 0]
        sharpe_improvements = pd.DataFrame()
        for stop, target in STOP_TARGET_PAIRS[:-1]:  # Exclude no stop/target
            key = f"stop_{stop}_target_{target}"
            if f"{key}_sharpe" in results_df.columns:
                sharpe_improvements[f"{stop}/{target}"] = results_df[f"{key}_sharpe"] - results_df['base_sharpe']
        
        if not sharpe_improvements.empty:
            avg_improvements = sharpe_improvements.mean()
            ax.bar(range(len(avg_improvements)), avg_improvements.values)
            ax.set_xticks(range(len(avg_improvements)))
            ax.set_xticklabels(avg_improvements.index, rotation=45)
            ax.set_ylabel('Average Sharpe Improvement')
            ax.set_title('Sharpe Improvement by Stop/Target Config')
            ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # 2. Return distribution
        ax = axes[0, 1]
        if f"stop_0.075_target_0.1_return" in results_df.columns:
            base_returns = results_df['base_return'] * 100
            modified_returns = results_df['stop_0.075_target_0.1_return'] * 100
            
            ax.scatter(base_returns, modified_returns, alpha=0.6)
            ax.plot([-50, 50], [-50, 50], 'r--', alpha=0.5)  # y=x line
            ax.set_xlabel('Base Return %')
            ax.set_ylabel('Return with 0.075/0.1 Stop/Target %')
            ax.set_title('Return Comparison')
            ax.grid(True, alpha=0.3)
        
        # 3. Exit type distribution
        ax = axes[1, 0]
        if f"stop_0.075_target_0.1_stop_pct" in results_df.columns:
            exit_data = pd.DataFrame({
                'Stops': results_df['stop_0.075_target_0.1_stop_pct'].values,
                'Targets': results_df['stop_0.075_target_0.1_target_pct'].values,
                'Signals': 100 - results_df['stop_0.075_target_0.1_stop_pct'].values - results_df['stop_0.075_target_0.1_target_pct'].values
            })
            
            exit_data.mean().plot(kind='bar', ax=ax)
            ax.set_ylabel('Percentage of Trades')
            ax.set_title('Average Exit Type Distribution (0.075/0.1 Config)')
            ax.set_xticklabels(['Stops', 'Targets', 'Signals'], rotation=0)
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = "Summary Statistics:\n\n"
        summary_text += f"Strategies analyzed: {len(results_df)}\n"
        summary_text += f"Average trades/strategy: {results_df['num_trades'].mean():.0f}\n"
        summary_text += f"Average trades/day: {results_df['trades_per_day'].mean():.1f}\n\n"
        
        if '0.075/0.1' in config_df.index:
            summary_text += "0.075/0.1 Stop/Target Performance:\n"
            summary_text += f"  Avg Return: {config_df.loc['0.075/0.1', 'avg_return']:.2f}%\n"
            summary_text += f"  Avg Sharpe: {config_df.loc['0.075/0.1', 'avg_sharpe']:.2f}\n"
            summary_text += f"  Success Rate: {config_df.loc['0.075/0.1', 'win_rate']:.1f}%\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=12, family='monospace')
        
        plt.tight_layout()
        plt.show()
        
        # Save results
        results_df.to_csv(run_dir / 'stop_target_analysis.csv', index=False)
        optimal_df.to_csv(run_dir / 'optimal_configurations.csv', index=False)
        
        print(f"\n✅ Analysis complete! Results saved to:")
        print(f"  - {run_dir}/stop_target_analysis.csv")
        print(f"  - {run_dir}/optimal_configurations.csv")
        
    else:
        print("\n❌ No valid strategies found for analysis")
else:
    print("❌ No performance data available")
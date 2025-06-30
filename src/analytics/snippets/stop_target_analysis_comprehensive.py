# Comprehensive Stop Loss and Profit Target Analysis
# This provides the CORRECT implementation for analyzing both stops and targets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

def apply_stop_target(trades_df, stop_pct, target_pct, market_data):
    """
    Apply both stop loss and profit target to trades.
    
    CORRECT IMPLEMENTATION:
    - Checks intraday prices bar by bar
    - Exits at first target hit (stop or profit)
    - Properly tracks which exit was triggered
    - Works for both long and short positions
    
    Args:
        trades_df: DataFrame of trades
        stop_pct: Stop loss percentage (e.g., 0.1 for 0.1%)
        target_pct: Profit target percentage (e.g., 0.2 for 0.2%)
        market_data: Market data with OHLC prices
    
    Returns:
        Tuple of (modified_returns_array, exit_types_dict)
    """
    if stop_pct == 0 and target_pct == 0:
        # No modification - return original
        return trades_df['net_return'].values, {'stop': 0, 'target': 0, 'signal': len(trades_df)}
    
    modified_returns = []
    exit_types = {'stop': 0, 'target': 0, 'signal': 0}
    stopped_winners = 0
    
    for _, trade in trades_df.iterrows():
        trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
        
        if len(trade_prices) == 0:
            modified_returns.append(trade['net_return'])
            exit_types['signal'] += 1
            continue
        
        entry_price = trade['entry_price']
        direction = trade['direction']
        original_return = trade['net_return']
        
        # Set stop and target prices
        if direction == 1:  # Long
            stop_price = entry_price * (1 - stop_pct/100) if stop_pct > 0 else 0
            target_price = entry_price * (1 + target_pct/100) if target_pct > 0 else float('inf')
        else:  # Short
            stop_price = entry_price * (1 + stop_pct/100) if stop_pct > 0 else float('inf')
            target_price = entry_price * (1 - target_pct/100) if target_pct > 0 else 0
        
        # Check each bar for exit
        exit_price = trade['exit_price']
        exit_type = 'signal'
        
        for _, bar in trade_prices.iterrows():
            if direction == 1:  # Long
                # Check stop first (more conservative)
                if stop_pct > 0 and bar['low'] <= stop_price:
                    exit_price = stop_price
                    exit_type = 'stop'
                    if original_return > 0:
                        stopped_winners += 1
                    break
                # Then check target
                elif target_pct > 0 and bar['high'] >= target_price:
                    exit_price = target_price
                    exit_type = 'target'
                    break
            else:  # Short
                # Check stop first
                if stop_pct > 0 and bar['high'] >= stop_price:
                    exit_price = stop_price
                    exit_type = 'stop'
                    if original_return > 0:
                        stopped_winners += 1
                    break
                # Then check target
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
    
    # Add stopped winners info
    exit_types['stopped_winners'] = stopped_winners
    
    return np.array(modified_returns), exit_types


def analyze_stop_target_combinations(trades_df, market_data, 
                                   stop_levels=None, target_levels=None,
                                   execution_cost_bps=1.0):
    """
    Analyze all combinations of stop loss and profit target levels.
    
    Args:
        trades_df: DataFrame of trades
        market_data: Market data with OHLC
        stop_levels: List of stop loss percentages
        target_levels: List of profit target percentages
        execution_cost_bps: Execution cost in basis points
    
    Returns:
        DataFrame with results for each combination
    """
    if stop_levels is None:
        stop_levels = [0, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0]
    
    if target_levels is None:
        target_levels = [0, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0]
    
    results = []
    
    for stop, target in product(stop_levels, target_levels):
        # Skip if both are 0 (baseline)
        if stop == 0 and target == 0:
            baseline_return = trades_df['net_return'].sum()
            baseline_sharpe = calculate_sharpe(trades_df['net_return'])
            results.append({
                'stop_pct': 0,
                'target_pct': 0,
                'total_return': baseline_return,
                'sharpe_ratio': baseline_sharpe,
                'win_rate': (trades_df['net_return'] > 0).mean(),
                'num_trades': len(trades_df),
                'stop_exits': 0,
                'target_exits': 0,
                'signal_exits': len(trades_df),
                'stopped_winners': 0,
                'risk_reward_ratio': 0
            })
            continue
        
        # Apply stop/target
        returns, exit_types = apply_stop_target(trades_df, stop, target, market_data)
        
        # Calculate metrics
        total_return = returns.sum()
        sharpe = calculate_sharpe(returns)
        win_rate = (returns > 0).mean()
        
        # Risk/reward ratio
        risk_reward = target / stop if stop > 0 else float('inf')
        
        results.append({
            'stop_pct': stop,
            'target_pct': target,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'num_trades': len(returns),
            'stop_exits': exit_types['stop'],
            'target_exits': exit_types['target'],
            'signal_exits': exit_types['signal'],
            'stopped_winners': exit_types.get('stopped_winners', 0),
            'risk_reward_ratio': risk_reward,
            'stop_exit_pct': exit_types['stop'] / len(returns) * 100,
            'target_exit_pct': exit_types['target'] / len(returns) * 100,
            'signal_exit_pct': exit_types['signal'] / len(returns) * 100
        })
    
    return pd.DataFrame(results)


def calculate_sharpe(returns, periods_per_year=252):
    """Calculate Sharpe ratio from returns array"""
    if len(returns) == 0 or returns.std() == 0:
        return 0
    return returns.mean() / returns.std() * np.sqrt(periods_per_year)


def visualize_stop_target_heatmap(results_df, metric='sharpe_ratio', 
                                 title_suffix="", figsize=(12, 10)):
    """
    Create heatmap visualization of stop/target combinations.
    
    Args:
        results_df: DataFrame from analyze_stop_target_combinations
        metric: Metric to plot ('sharpe_ratio', 'total_return', 'win_rate')
        title_suffix: Additional text for title
        figsize: Figure size
    """
    # Pivot for heatmap
    pivot = results_df.pivot(index='stop_pct', columns='target_pct', values=metric)
    
    plt.figure(figsize=figsize)
    
    # Create heatmap
    if metric == 'sharpe_ratio':
        cmap = 'RdYlGn'
        center = 0
        fmt = '.2f'
    elif metric == 'total_return':
        cmap = 'RdYlGn'
        center = 0
        fmt = '.1%'
    else:  # win_rate
        cmap = 'YlOrRd'
        center = None
        fmt = '.1%'
    
    # Plot
    sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, center=center,
                cbar_kws={'label': metric.replace('_', ' ').title()})
    
    plt.title(f'{metric.replace("_", " ").title()} by Stop/Target Combination{title_suffix}')
    plt.xlabel('Profit Target %')
    plt.ylabel('Stop Loss %')
    
    # Highlight optimal
    optimal_idx = results_df[metric].idxmax()
    optimal = results_df.iloc[optimal_idx]
    plt.text(0.02, 0.98, f'Optimal: Stop={optimal["stop_pct"]:.3f}%, Target={optimal["target_pct"]:.3f}%',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    return optimal


def create_comprehensive_report(trades_df, market_data, strategy_name="Strategy"):
    """
    Create a comprehensive stop/target analysis report.
    """
    print(f"📊 Comprehensive Stop/Target Analysis for {strategy_name}")
    print("=" * 80)
    
    # Baseline metrics
    baseline_return = trades_df['net_return'].sum()
    baseline_sharpe = calculate_sharpe(trades_df['net_return'])
    
    print(f"\nBaseline Performance (No Stops/Targets):")
    print(f"  Total Return: {baseline_return*100:.2f}%")
    print(f"  Sharpe Ratio: {baseline_sharpe:.2f}")
    print(f"  Win Rate: {(trades_df['net_return'] > 0).mean()*100:.1f}%")
    print(f"  Number of Trades: {len(trades_df)}")
    
    # Analyze combinations
    print("\nAnalyzing stop/target combinations...")
    results = analyze_stop_target_combinations(trades_df, market_data)
    
    # Find optimal by different metrics
    optimal_sharpe = results.iloc[results['sharpe_ratio'].idxmax()]
    optimal_return = results.iloc[results['total_return'].idxmax()]
    optimal_winrate = results.iloc[results['win_rate'].idxmax()]
    
    print("\n🎯 Optimal Configurations:")
    print(f"\nBest Sharpe Ratio: {optimal_sharpe['sharpe_ratio']:.2f}")
    print(f"  Stop: {optimal_sharpe['stop_pct']:.3f}%, Target: {optimal_sharpe['target_pct']:.3f}%")
    print(f"  Return: {optimal_sharpe['total_return']*100:.2f}%, Win Rate: {optimal_sharpe['win_rate']*100:.1f}%")
    print(f"  Exit Distribution: Stops={optimal_sharpe['stop_exit_pct']:.1f}%, Targets={optimal_sharpe['target_exit_pct']:.1f}%, Signals={optimal_sharpe['signal_exit_pct']:.1f}%")
    
    print(f"\nBest Total Return: {optimal_return['total_return']*100:.2f}%")
    print(f"  Stop: {optimal_return['stop_pct']:.3f}%, Target: {optimal_return['target_pct']:.3f}%")
    print(f"  Sharpe: {optimal_return['sharpe_ratio']:.2f}, Win Rate: {optimal_return['win_rate']*100:.1f}%")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Sharpe ratio heatmap
    plt.sca(axes[0, 0])
    pivot_sharpe = results.pivot(index='stop_pct', columns='target_pct', values='sharpe_ratio')
    sns.heatmap(pivot_sharpe, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
    plt.title('Sharpe Ratio Heatmap')
    
    # 2. Total return heatmap
    plt.sca(axes[0, 1])
    pivot_return = results.pivot(index='stop_pct', columns='target_pct', values='total_return')
    sns.heatmap(pivot_return * 100, annot=True, fmt='.1f', cmap='RdYlGn', center=0)
    plt.title('Total Return % Heatmap')
    
    # 3. Stop exit percentage
    plt.sca(axes[1, 0])
    pivot_stops = results.pivot(index='stop_pct', columns='target_pct', values='stop_exit_pct')
    sns.heatmap(pivot_stops, annot=True, fmt='.0f', cmap='Reds')
    plt.title('Stop Exit % Heatmap')
    
    # 4. Target exit percentage
    plt.sca(axes[1, 1])
    pivot_targets = results.pivot(index='stop_pct', columns='target_pct', values='target_exit_pct')
    sns.heatmap(pivot_targets, annot=True, fmt='.0f', cmap='Greens')
    plt.title('Target Exit % Heatmap')
    
    plt.tight_layout()
    plt.show()
    
    # Risk/Reward analysis
    print("\n📈 Risk/Reward Analysis:")
    rr_groups = results[results['stop_pct'] > 0].groupby('risk_reward_ratio')
    
    print("\nAverage Sharpe by Risk/Reward Ratio:")
    for rr, group in rr_groups:
        if rr <= 5:  # Reasonable ratios only
            avg_sharpe = group['sharpe_ratio'].mean()
            avg_return = group['total_return'].mean()
            print(f"  R/R = {rr:.1f}: Sharpe={avg_sharpe:.2f}, Return={avg_return*100:.1f}%")
    
    return results, optimal_sharpe


# Example usage:
if __name__ == "__main__":
    print("Example usage:")
    print("trades = extract_trades(strategy_hash, trace_path, market_data)")
    print("results, optimal = create_comprehensive_report(trades, market_data, 'My Strategy')")
    print("\n# For specific stop/target analysis:")
    print("returns, exits = apply_stop_target(trades, stop_pct=0.1, target_pct=0.2, market_data=market_data)")
    print("print(f'Stop exits: {exits[\"stop\"]}, Target exits: {exits[\"target\"]}')")
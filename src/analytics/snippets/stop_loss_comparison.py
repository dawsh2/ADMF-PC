# Comprehensive Stop Loss Comparison
# Compares fixed stops, trailing stops, and ATR-based stops

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_atr(market_data, period=14):
    """Calculate Average True Range for dynamic stop losses"""
    high = market_data['high']
    low = market_data['low']
    close = market_data['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr

def compare_stop_strategies(strategy_hash, trace_path, market_data, execution_cost_bps=1.0):
    """
    Compare different stop loss strategies:
    1. No stop loss (baseline)
    2. Fixed percentage stop
    3. Trailing stop
    4. ATR-based stop (volatility adjusted)
    """
    # Extract original trades
    trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)
    
    if len(trades) == 0:
        return pd.DataFrame()
    
    # Calculate ATR for the entire dataset
    market_data['atr'] = calculate_atr(market_data)
    
    # Test different stop strategies
    stop_strategies = {
        'no_stop': {'type': 'none'},
        'fixed_0.5%': {'type': 'fixed', 'level': 0.5},
        'fixed_1.0%': {'type': 'fixed', 'level': 1.0},
        'fixed_2.0%': {'type': 'fixed', 'level': 2.0},
        'trailing_0.5%': {'type': 'trailing', 'level': 0.5},
        'trailing_1.0%': {'type': 'trailing', 'level': 1.0},
        'trailing_2.0%': {'type': 'trailing', 'level': 2.0},
        'atr_1x': {'type': 'atr', 'multiplier': 1.0},
        'atr_2x': {'type': 'atr', 'multiplier': 2.0},
        'atr_3x': {'type': 'atr', 'multiplier': 3.0}
    }
    
    results = []
    
    for strategy_name, stop_config in stop_strategies.items():
        trades_with_stop = []
        stopped_count = 0
        
        for _, trade in trades.iterrows():
            # Get intraday prices for this trade
            trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
            
            if len(trade_prices) == 0:
                continue
            
            entry_price = trade['entry_price']
            entry_idx = int(trade['entry_idx'])
            direction = trade['direction']
            
            # Initialize stop based on strategy
            if stop_config['type'] == 'none':
                stop_price = 0 if direction == 1 else float('inf')
            elif stop_config['type'] == 'fixed':
                sl_decimal = stop_config['level'] / 100
                stop_price = entry_price * (1 - sl_decimal) if direction == 1 else entry_price * (1 + sl_decimal)
            elif stop_config['type'] == 'trailing':
                sl_decimal = stop_config['level'] / 100
                stop_price = entry_price * (1 - sl_decimal) if direction == 1 else entry_price * (1 + sl_decimal)
                best_price = entry_price
            elif stop_config['type'] == 'atr':
                entry_atr = market_data.iloc[entry_idx]['atr']
                if pd.notna(entry_atr):
                    stop_distance = entry_atr * stop_config['multiplier']
                    stop_price = entry_price - stop_distance if direction == 1 else entry_price + stop_distance
                else:
                    # Fallback to 2% if ATR not available
                    stop_price = entry_price * 0.98 if direction == 1 else entry_price * 1.02
            
            # Track through the trade
            stopped = False
            exit_price = trade['exit_price']
            exit_time = trade['exit_time']
            
            for idx, bar in trade_prices.iterrows():
                # Update trailing stop if applicable
                if stop_config['type'] == 'trailing':
                    if direction == 1 and bar['high'] > best_price:
                        best_price = bar['high']
                        new_stop = best_price * (1 - sl_decimal)
                        stop_price = max(stop_price, new_stop)
                    elif direction == -1 and bar['low'] < best_price:
                        best_price = bar['low']
                        new_stop = best_price * (1 + sl_decimal)
                        stop_price = min(stop_price, new_stop)
                
                # Check if stopped out
                if direction == 1 and bar['low'] <= stop_price and stop_config['type'] != 'none':
                    stopped = True
                    exit_price = stop_price
                    exit_time = bar['timestamp']
                    break
                elif direction == -1 and bar['high'] >= stop_price and stop_config['type'] != 'none':
                    stopped = True
                    exit_price = stop_price
                    exit_time = bar['timestamp']
                    break
            
            if stopped:
                stopped_count += 1
            
            # Calculate return
            if direction == 1:
                raw_return = (exit_price - entry_price) / entry_price
            else:
                raw_return = (entry_price - exit_price) / entry_price
            
            net_return = raw_return - trade['execution_cost']
            
            trade_result = {
                'net_return': net_return,
                'stopped': stopped
            }
            trades_with_stop.append(trade_result)
        
        trades_df = pd.DataFrame(trades_with_stop)
        
        if len(trades_df) > 0:
            # Calculate metrics
            winning_trades = trades_df[trades_df['net_return'] > 0]
            losing_trades = trades_df[trades_df['net_return'] <= 0]
            
            # Cumulative return
            cum_return = (1 + trades_df['net_return']).cumprod().iloc[-1] - 1
            
            # Sharpe ratio
            if trades_df['net_return'].std() > 0:
                sharpe = trades_df['net_return'].mean() / trades_df['net_return'].std() * np.sqrt(252 * 78)
            else:
                sharpe = 0
            
            # Max drawdown
            cumulative = (1 + trades_df['net_return']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative / running_max - 1).min()
            
            # Profit factor
            if len(losing_trades) > 0 and losing_trades['net_return'].sum() != 0:
                profit_factor = winning_trades['net_return'].sum() / abs(losing_trades['net_return'].sum())
            else:
                profit_factor = 999.99 if len(winning_trades) > 0 else 0
            
            results.append({
                'strategy': strategy_name,
                'stop_type': stop_config['type'],
                'total_return': cum_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': drawdown,
                'win_rate': len(winning_trades) / len(trades_df),
                'profit_factor': profit_factor,
                'trades_stopped': stopped_count,
                'stop_rate': stopped_count / len(trades_df),
                'avg_return': trades_df['net_return'].mean(),
                'return_std': trades_df['net_return'].std()
            })
    
    return pd.DataFrame(results)

# Main comparison analysis
if len(top_overall) > 0:
    print("🔍 Stop Loss Strategy Comparison")
    print("=" * 80)
    print("Comparing: No stops, Fixed %, Trailing %, and ATR-based stops\n")
    
    # Analyze top strategies
    comparison_results = []
    
    for idx, (_, strategy) in enumerate(top_overall.head(5).iterrows()):
        print(f"\nStrategy {idx+1}: {strategy['strategy_type']} - {strategy['strategy_hash'][:8]}")
        print(f"Base Performance: Sharpe={strategy['sharpe_ratio']:.2f}, Return={strategy['total_return']*100:.2f}%")
        
        # Compare stop strategies
        comparison = compare_stop_strategies(
            strategy['strategy_hash'],
            strategy['trace_path'],
            market_data,
            execution_cost_bps
        )
        
        if len(comparison) > 0:
            comparison['strategy_hash'] = strategy['strategy_hash']
            comparison['strategy_type'] = strategy['strategy_type']
            comparison_results.append(comparison)
            
            # Show best stop strategy
            best_idx = comparison['sharpe_ratio'].idxmax()
            best = comparison.iloc[best_idx]
            print(f"\nBest Stop Strategy: {best['strategy']}")
            print(f"  Sharpe: {best['sharpe_ratio']:.2f}")
            print(f"  Return: {best['total_return']*100:.2f}%")
            print(f"  Win Rate: {best['win_rate']*100:.1f}%")
            print(f"  Stop Rate: {best['stop_rate']*100:.1f}%")
    
    # Combine and visualize
    if comparison_results:
        all_comparisons = pd.concat(comparison_results, ignore_index=True)
        
        # Create comparison visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Average Sharpe by stop type
        ax = axes[0, 0]
        stop_types = ['none', 'fixed', 'trailing', 'atr']
        avg_sharpe = []
        for stop_type in stop_types:
            type_data = all_comparisons[all_comparisons['stop_type'] == stop_type]
            avg_sharpe.append(type_data['sharpe_ratio'].mean())
        
        ax.bar(stop_types, avg_sharpe)
        ax.set_ylabel('Average Sharpe Ratio')
        ax.set_title('Performance by Stop Type')
        ax.grid(True, alpha=0.3)
        
        # 2. Win rate vs stop rate scatter
        ax = axes[0, 1]
        colors = {'none': 'gray', 'fixed': 'red', 'trailing': 'blue', 'atr': 'green'}
        for stop_type in stop_types:
            type_data = all_comparisons[all_comparisons['stop_type'] == stop_type]
            ax.scatter(type_data['stop_rate'] * 100, type_data['win_rate'] * 100,
                      label=stop_type, color=colors[stop_type], alpha=0.6, s=100)
        
        ax.set_xlabel('Stop Rate (%)')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate vs Stop Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Return distribution by stop type
        ax = axes[1, 0]
        stop_categories = all_comparisons['strategy'].unique()
        x = np.arange(len(stop_categories))
        width = 0.8 / len(all_comparisons['strategy_hash'].unique())
        
        for i, strategy_hash in enumerate(all_comparisons['strategy_hash'].unique()):
            strategy_data = all_comparisons[all_comparisons['strategy_hash'] == strategy_hash]
            returns = strategy_data['total_return'].values * 100
            ax.bar(x + i * width, returns, width, label=f'S{i+1}', alpha=0.7)
        
        ax.set_xlabel('Stop Strategy')
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Returns by Stop Strategy')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(stop_categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Summary table
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calculate summary statistics
        summary_stats = all_comparisons.groupby('stop_type').agg({
            'sharpe_ratio': 'mean',
            'total_return': 'mean',
            'win_rate': 'mean',
            'stop_rate': 'mean'
        })
        
        summary_text = "Average Performance by Stop Type:\n\n"
        summary_text += f"{'Type':<10} {'Sharpe':>8} {'Return':>10} {'Win Rate':>10}\n"
        summary_text += "-" * 40 + "\n"
        
        for stop_type in stop_types:
            if stop_type in summary_stats.index:
                row = summary_stats.loc[stop_type]
                summary_text += f"{stop_type:<10} {row['sharpe_ratio']:>8.2f} "
                summary_text += f"{row['total_return']*100:>9.1f}% {row['win_rate']*100:>9.1f}%\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=12, family='monospace')
        
        plt.tight_layout()
        plt.show()
        
        # Key findings
        print("\n📊 Key Findings:")
        print("=" * 60)
        
        best_overall = all_comparisons.loc[all_comparisons['sharpe_ratio'].idxmax()]
        print(f"1. Best overall: {best_overall['strategy']} (Sharpe: {best_overall['sharpe_ratio']:.2f})")
        
        print(f"\n2. Average Sharpe by stop type:")
        for stop_type, sharpe in zip(stop_types, avg_sharpe):
            print(f"   {stop_type}: {sharpe:.2f}")
        
        print(f"\n3. ATR-based stops adapt to market volatility")
        print(f"4. Trailing stops lock in profits during trends")
        print(f"5. Fixed stops are simple but may be too rigid")
        
        # Save comparison
        all_comparisons.to_csv(run_dir / 'stop_loss_comparison.csv', index=False)
        print(f"\n✅ Detailed comparison saved to: stop_loss_comparison.csv")
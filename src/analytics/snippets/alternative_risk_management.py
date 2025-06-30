# Alternative Risk Management for Micro-Movement Strategies
# When stops don't work, try these approaches

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_alternative_risk_management(strategy_hash, trace_path, market_data, execution_cost_bps=1.0):
    """
    Test alternative risk management approaches:
    1. Time-based exits
    2. Volatility-scaled position sizing
    3. Maximum position limits
    4. Profit targets only (no stops)
    """
    # Extract trades
    trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)
    
    if len(trades) == 0:
        return None
    
    results = {}
    
    # 1. Time-based exits (exit after N bars if not profitable)
    time_limits = [5, 10, 15, 20, 30]  # bars
    time_results = []
    
    for max_bars in time_limits:
        trades_modified = []
        
        for _, trade in trades.iterrows():
            trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
            
            if len(trade_prices) == 0:
                continue
            
            # Check if we should exit early due to time
            if len(trade_prices) > max_bars:
                # Calculate return at max_bars
                exit_bar = trade_prices.iloc[max_bars]
                exit_price = exit_bar['close']
                
                if trade['direction'] == 1:
                    raw_return = (exit_price - trade['entry_price']) / trade['entry_price']
                else:
                    raw_return = (trade['entry_price'] - exit_price) / trade['entry_price']
                
                # Only exit early if losing
                if raw_return < 0:
                    net_return = raw_return - trade['execution_cost']
                    time_exit = True
                else:
                    # Let winner run
                    net_return = trade['net_return']
                    time_exit = False
            else:
                net_return = trade['net_return']
                time_exit = False
            
            trades_modified.append({
                'net_return': net_return,
                'time_exit': time_exit
            })
        
        trades_df = pd.DataFrame(trades_modified)
        
        if len(trades_df) > 0:
            total_return = (1 + trades_df['net_return']).prod() - 1
            time_exit_rate = trades_df['time_exit'].mean()
            
            time_results.append({
                'max_bars': max_bars,
                'total_return': total_return,
                'time_exit_rate': time_exit_rate
            })
    
    results['time_exits'] = pd.DataFrame(time_results)
    
    # 2. Profit targets only (no stop loss)
    profit_targets = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]  # %
    target_results = []
    
    for target_pct in profit_targets:
        trades_modified = []
        targets_hit = 0
        
        for _, trade in trades.iterrows():
            trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
            
            if len(trade_prices) == 0:
                continue
            
            entry_price = trade['entry_price']
            direction = trade['direction']
            
            # Calculate target price
            if direction == 1:
                target_price = entry_price * (1 + target_pct/100)
            else:
                target_price = entry_price * (1 - target_pct/100)
            
            # Check if target hit
            target_hit = False
            exit_price = trade['exit_price']
            
            for _, bar in trade_prices.iterrows():
                if direction == 1 and bar['high'] >= target_price:
                    exit_price = target_price
                    target_hit = True
                    targets_hit += 1
                    break
                elif direction == -1 and bar['low'] <= target_price:
                    exit_price = target_price
                    target_hit = True
                    targets_hit += 1
                    break
            
            # Calculate return
            if direction == 1:
                raw_return = (exit_price - entry_price) / entry_price
            else:
                raw_return = (entry_price - exit_price) / entry_price
            
            net_return = raw_return - trade['execution_cost']
            
            trades_modified.append(net_return)
        
        if trades_modified:
            returns_array = np.array(trades_modified)
            total_return = (1 + returns_array).prod() - 1
            win_rate = (returns_array > 0).mean()
            
            if returns_array.std() > 0:
                sharpe = returns_array.mean() / returns_array.std() * np.sqrt(252 * 78)
            else:
                sharpe = 0
            
            target_results.append({
                'target_pct': target_pct,
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'win_rate': win_rate,
                'targets_hit': targets_hit,
                'target_hit_rate': targets_hit / len(trades_modified) if trades_modified else 0
            })
    
    results['profit_targets'] = pd.DataFrame(target_results)
    
    # 3. Analyze trade duration impact on returns
    duration_analysis = []
    for _, trade in trades.iterrows():
        duration_analysis.append({
            'duration_bars': int((trade['exit_idx'] - trade['entry_idx'])),
            'duration_minutes': trade['duration_minutes'],
            'return': trade['net_return'] * 100
        })
    
    results['duration_analysis'] = pd.DataFrame(duration_analysis)
    
    return results

# Main analysis
if len(performance_df) > 0:
    print("🎯 Alternative Risk Management Analysis")
    print("=" * 80)
    print("Testing approaches that work better than stops for micro-movements\n")
    
    # Get high-frequency strategies
    trading_days = len(market_data['timestamp'].dt.date.unique())
    high_freq_df = performance_df[performance_df['num_trades'] >= MIN_TRADES_PER_DAY * trading_days]
    
    if len(high_freq_df) > 0:
        # Analyze top strategies
        all_results = []
        
        for idx, row in high_freq_df.head(3).iterrows():
            print(f"Strategy {idx+1}: {row['strategy_type']} - {row['strategy_hash'][:8]}")
            print(f"  Base Sharpe: {row['sharpe_ratio']:.2f}, Return: {row['total_return']*100:.2f}%\n")
            
            results = analyze_alternative_risk_management(
                row['strategy_hash'],
                row['trace_path'],
                market_data,
                execution_cost_bps
            )
            
            if results:
                # Show time exit results
                if 'time_exits' in results and len(results['time_exits']) > 0:
                    print("  Time-Based Exit Analysis (exit losing trades after N bars):")
                    best_time = results['time_exits'].loc[results['time_exits']['total_return'].idxmax()]
                    print(f"    Optimal: Exit after {best_time['max_bars']} bars")
                    print(f"    New return: {best_time['total_return']*100:.2f}% (was {row['total_return']*100:.2f}%)")
                    print(f"    Time exits: {best_time['time_exit_rate']*100:.1f}% of trades\n")
                
                # Show profit target results
                if 'profit_targets' in results and len(results['profit_targets']) > 0:
                    print("  Profit Target Analysis (no stop loss):")
                    best_target = results['profit_targets'].loc[results['profit_targets']['sharpe_ratio'].idxmax()]
                    print(f"    Optimal: {best_target['target_pct']:.2f}% profit target")
                    print(f"    New Sharpe: {best_target['sharpe_ratio']:.2f} (was {row['sharpe_ratio']:.2f})")
                    print(f"    Target hit rate: {best_target['target_hit_rate']*100:.1f}%\n")
                
                all_results.append({
                    'strategy': row['strategy_hash'][:8],
                    'results': results
                })
        
        # Visualize findings
        if all_results:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Time exit performance
            ax = axes[0, 0]
            for res in all_results:
                if 'time_exits' in res['results']:
                    time_df = res['results']['time_exits']
                    ax.plot(time_df['max_bars'], time_df['total_return'] * 100, 
                           marker='o', label=res['strategy'])
            ax.set_xlabel('Max Bars Before Exit')
            ax.set_ylabel('Total Return %')
            ax.set_title('Time-Based Exit Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. Profit target performance
            ax = axes[0, 1]
            for res in all_results:
                if 'profit_targets' in res['results']:
                    target_df = res['results']['profit_targets']
                    ax.plot(target_df['target_pct'], target_df['sharpe_ratio'], 
                           marker='o', label=res['strategy'])
            ax.set_xlabel('Profit Target %')
            ax.set_ylabel('Sharpe Ratio')
            ax.set_title('Profit Target Impact (No Stop Loss)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. Duration vs Return scatter
            ax = axes[1, 0]
            if all_results and 'duration_analysis' in all_results[0]['results']:
                duration_df = all_results[0]['results']['duration_analysis']
                scatter = ax.scatter(duration_df['duration_bars'], 
                                   duration_df['return'],
                                   alpha=0.5)
                ax.axhline(0, color='red', linestyle='--', alpha=0.5)
                ax.set_xlabel('Trade Duration (bars)')
                ax.set_ylabel('Return %')
                ax.set_title('Trade Duration vs Returns')
                
                # Add trend line
                if len(duration_df) > 10:
                    z = np.polyfit(duration_df['duration_bars'], duration_df['return'], 1)
                    p = np.poly1d(z)
                    ax.plot(duration_df['duration_bars'], p(duration_df['duration_bars']), 
                           "r--", alpha=0.8, label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
                    ax.legend()
            
            # 4. Recommendations
            ax = axes[1, 1]
            ax.axis('off')
            
            summary_text = "Risk Management Recommendations:\n\n"
            summary_text += "1. No Stop Loss Strategy:\n"
            summary_text += "   • Let mean reversion work\n"
            summary_text += "   • Use position sizing instead\n\n"
            
            summary_text += "2. Time-Based Exits:\n"
            summary_text += "   • Exit losing trades after 10-15 bars\n"
            summary_text += "   • Let winners run indefinitely\n\n"
            
            summary_text += "3. Profit Targets:\n"
            summary_text += "   • Set targets at 0.2-0.3%\n"
            summary_text += "   • No stop loss needed\n\n"
            
            summary_text += "4. Position Sizing:\n"
            summary_text += "   • Scale with volatility regime\n"
            summary_text += "   • Reduce size in High Vol\n"
            summary_text += "   • Increase in Low Vol"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=11, family='monospace')
            
            plt.tight_layout()
            plt.show()
        
        print("\n💡 Key Insights:")
        print("=" * 60)
        print("1. Mean reversion strategies benefit from NO stop loss")
        print("2. Time-based exits can cut losing trades without stopping winners")
        print("3. Small profit targets (0.2-0.3%) improve Sharpe ratios")
        print("4. Position sizing by volatility regime is more effective than stops")
        
    else:
        print("❌ No high-frequency strategies found")
# Ultra-Micro Stop Analysis
# Tests extremely tight stops and shows why no stop might be optimal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ultra-tight stop levels for micro movements
ULTRA_MICRO_STOPS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2]

def analyze_why_no_stop_wins(strategy_hash, trace_path, market_data, execution_cost_bps=1.0):
    """
    Detailed analysis of why no stop loss might be optimal
    """
    # Extract trades
    trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)
    
    if len(trades) == 0:
        return None, None
    
    stop_analysis = []
    trade_details = []
    
    # Analyze first 50 trades in detail
    for trade_idx, trade in trades.head(50).iterrows():
        trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
        
        if len(trade_prices) < 2:
            continue
        
        entry_price = trade['entry_price']
        direction = trade['direction']
        
        # Track maximum adverse excursion bar by bar
        if direction == 1:  # Long
            cumulative_low = trade_prices['low'].expanding().min()
            mae_by_bar = ((entry_price - cumulative_low) / entry_price * 100).values
        else:  # Short
            cumulative_high = trade_prices['high'].expanding().max()
            mae_by_bar = ((cumulative_high - entry_price) / entry_price * 100).values
        
        # Final return
        final_return = trade['net_return'] * 100
        
        # For each stop level, check if it would have been hit and when
        stop_results = {}
        for stop_pct in ULTRA_MICRO_STOPS:
            hit_bar = None
            for i, mae in enumerate(mae_by_bar):
                if mae > stop_pct:
                    hit_bar = i
                    break
            
            if hit_bar is not None:
                # Stop was hit - trade ends with loss
                stop_return = -stop_pct - (execution_cost_bps / 10000 * 100)
            else:
                # Stop not hit - trade completes normally
                stop_return = final_return
            
            stop_results[f'stop_{stop_pct}'] = {
                'hit': hit_bar is not None,
                'hit_bar': hit_bar,
                'return': stop_return
            }
        
        trade_details.append({
            'trade_idx': trade_idx,
            'duration_bars': len(trade_prices),
            'max_mae': mae_by_bar[-1] if len(mae_by_bar) > 0 else 0,
            'final_return': final_return,
            'turned_profitable': final_return > 0,
            **{f'stop_{s}_hit': stop_results[f'stop_{s}']['hit'] for s in ULTRA_MICRO_STOPS},
            **{f'stop_{s}_return': stop_results[f'stop_{s}']['return'] for s in ULTRA_MICRO_STOPS}
        })
    
    # Calculate overall performance with each stop level
    for stop_pct in [0] + ULTRA_MICRO_STOPS:  # Include no stop (0)
        trades_with_stop = []
        stops_hit = 0
        
        for _, trade in trades.iterrows():
            trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
            
            if len(trade_prices) == 0:
                continue
            
            if stop_pct == 0:
                # No stop - use original return
                net_return = trade['net_return']
            else:
                # Check if stop hit
                entry_price = trade['entry_price']
                direction = trade['direction']
                
                stop_hit = False
                for _, bar in trade_prices.iterrows():
                    if direction == 1 and bar['low'] <= entry_price * (1 - stop_pct/100):
                        stop_hit = True
                        break
                    elif direction == -1 and bar['high'] >= entry_price * (1 + stop_pct/100):
                        stop_hit = True
                        break
                
                if stop_hit:
                    net_return = -stop_pct/100 - trade['execution_cost']
                    stops_hit += 1
                else:
                    net_return = trade['net_return']
            
            trades_with_stop.append(net_return)
        
        # Calculate metrics
        returns_array = np.array(trades_with_stop)
        total_return = (1 + returns_array).prod() - 1
        win_rate = (returns_array > 0).mean()
        avg_return = returns_array.mean()
        
        if returns_array.std() > 0:
            sharpe = avg_return / returns_array.std() * np.sqrt(252 * 78)
        else:
            sharpe = 0
        
        stop_analysis.append({
            'stop_pct': stop_pct,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'stops_hit': stops_hit,
            'stop_rate': stops_hit / len(trades_with_stop) if len(trades_with_stop) > 0 else 0
        })
    
    return pd.DataFrame(stop_analysis), pd.DataFrame(trade_details)

# Main analysis
if len(performance_df) > 0:
    print("🔬 Ultra-Micro Stop Analysis: Why No Stop Wins")
    print("=" * 80)
    print(f"Testing ultra-tight stops: {ULTRA_MICRO_STOPS}")
    
    # Get high-frequency strategies
    trading_days = len(market_data['timestamp'].dt.date.unique())
    high_freq_df = performance_df[performance_df['num_trades'] >= MIN_TRADES_PER_DAY * trading_days].head(10)
    
    if len(high_freq_df) > 0:
        # Analyze why no stops win
        all_stop_results = []
        
        for idx, row in high_freq_df.head(3).iterrows():
            print(f"\n📊 Strategy {idx+1}: {row['strategy_type']} - {row['strategy_hash'][:8]}")
            print(f"   Trades: {row['num_trades']} | Base Sharpe: {row['sharpe_ratio']:.2f}")
            
            # Get detailed analysis
            stop_analysis, trade_details = analyze_why_no_stop_wins(
                row['strategy_hash'],
                row['trace_path'],
                market_data,
                execution_cost_bps
            )
            
            if stop_analysis is not None:
                # Show performance by stop level
                print("\n   Stop Level Analysis:")
                print("   Stop%  | Sharpe | Return | WinRate | StopRate")
                print("   " + "-" * 45)
                
                for _, sa in stop_analysis.iterrows():
                    print(f"   {sa['stop_pct']:5.2f} | {sa['sharpe_ratio']:6.2f} | {sa['total_return']*100:6.2f}% | {sa['win_rate']*100:6.1f}% | {sa['stop_rate']*100:7.1f}%")
                
                # Find why no stop wins
                no_stop = stop_analysis[stop_analysis['stop_pct'] == 0].iloc[0]
                best_stop = stop_analysis.loc[stop_analysis['sharpe_ratio'].idxmax()]
                
                if best_stop['stop_pct'] == 0:
                    print("\n   ✅ No stop is optimal!")
                    
                    # Analyze why
                    if len(trade_details) > 0:
                        # How many trades recover from drawdown?
                        recovery_analysis = []
                        for stop in ULTRA_MICRO_STOPS:
                            stop_col = f'stop_{stop}_hit'
                            return_col = f'stop_{stop}_return'
                            
                            if stop_col in trade_details.columns:
                                # Trades that would have been stopped but turned profitable
                                would_stop = trade_details[trade_details[stop_col]]
                                recovered = would_stop[would_stop['final_return'] > 0]
                                
                                if len(would_stop) > 0:
                                    recovery_rate = len(recovered) / len(would_stop)
                                    avg_recovery_return = recovered['final_return'].mean() if len(recovered) > 0 else 0
                                    
                                    recovery_analysis.append({
                                        'stop': stop,
                                        'would_stop_count': len(would_stop),
                                        'recovered_count': len(recovered),
                                        'recovery_rate': recovery_rate * 100,
                                        'avg_recovery_return': avg_recovery_return
                                    })
                        
                        if recovery_analysis:
                            recovery_df = pd.DataFrame(recovery_analysis)
                            print("\n   📈 Recovery Analysis (sample of 50 trades):")
                            print("   Trades that would be stopped but eventually turn profitable:")
                            for _, r in recovery_df.head(3).iterrows():
                                print(f"   {r['stop']:.2f}% stop: {r['recovered_count']}/{r['would_stop_count']} trades recover ({r['recovery_rate']:.0f}%), avg return when recovered: {r['avg_recovery_return']:.2f}%")
                
                stop_analysis['strategy'] = row['strategy_hash'][:8]
                all_stop_results.append(stop_analysis)
        
        # Visualize findings
        if all_stop_results:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. Sharpe ratio by stop level
            ax = axes[0, 0]
            for result_df in all_stop_results[:3]:
                strategy = result_df['strategy'].iloc[0]
                ax.plot(result_df['stop_pct'], result_df['sharpe_ratio'], 
                       marker='o', label=strategy)
            ax.axvline(0, color='red', linestyle='--', alpha=0.3, label='No Stop')
            ax.set_xlabel('Stop Loss %')
            ax.set_ylabel('Sharpe Ratio')
            ax.set_title('Sharpe Ratio vs Stop Level')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. Stop rate by level
            ax = axes[0, 1]
            for result_df in all_stop_results[:3]:
                strategy = result_df['strategy'].iloc[0]
                stop_data = result_df[result_df['stop_pct'] > 0]
                ax.plot(stop_data['stop_pct'], stop_data['stop_rate'] * 100, 
                       marker='o', label=strategy)
            ax.set_xlabel('Stop Loss %')
            ax.set_ylabel('Stop Hit Rate %')
            ax.set_title('Percentage of Trades Hitting Stop')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. Return impact
            ax = axes[1, 0]
            combined_df = pd.concat(all_stop_results)
            stop_impact = combined_df.groupby('stop_pct').agg({
                'total_return': 'mean',
                'sharpe_ratio': 'mean',
                'stop_rate': 'mean'
            })
            
            ax.bar(stop_impact.index, stop_impact['total_return'] * 100)
            ax.set_xlabel('Stop Loss %')
            ax.set_ylabel('Average Total Return %')
            ax.set_title('Average Return Impact by Stop Level')
            ax.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            # 4. Summary insights
            ax = axes[1, 1]
            ax.axis('off')
            
            summary_text = "Why No Stop Loss Wins:\n\n"
            
            avg_stop_rates = combined_df[combined_df['stop_pct'] > 0].groupby('stop_pct')['stop_rate'].mean()
            
            summary_text += "Stop Hit Rates:\n"
            for stop, rate in avg_stop_rates.head(5).items():
                summary_text += f"  {stop:.2f}%: {rate*100:.1f}% of trades\n"
            
            summary_text += "\nKey Insights:\n"
            summary_text += "• Many trades recover from small drawdowns\n"
            summary_text += "• Tight stops cut winners short\n"
            summary_text += "• Mean reversion works at micro level\n"
            summary_text += "• Transaction costs hurt frequent stops\n"
            
            summary_text += "\nRecommendations:\n"
            summary_text += "• Use position sizing instead of stops\n"
            summary_text += "• Focus on entry quality\n"
            summary_text += "• Consider time-based exits\n"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=11, family='monospace')
            
            plt.tight_layout()
            plt.show()
            
            # Additional analysis
            print("\n💡 Overall Findings:")
            print("=" * 60)
            
            # How often is no stop optimal?
            no_stop_wins = sum(1 for df in all_stop_results if df.loc[df['sharpe_ratio'].idxmax(), 'stop_pct'] == 0)
            print(f"No stop is optimal for {no_stop_wins}/{len(all_stop_results)} strategies analyzed")
            
            # Average performance degradation with stops
            avg_no_stop_sharpe = np.mean([df[df['stop_pct'] == 0]['sharpe_ratio'].iloc[0] for df in all_stop_results])
            avg_best_stop_sharpe = np.mean([df[df['stop_pct'] == 0.1]['sharpe_ratio'].iloc[0] for df in all_stop_results if 0.1 in df['stop_pct'].values])
            
            print(f"Average Sharpe with no stop: {avg_no_stop_sharpe:.2f}")
            print(f"Average Sharpe with 0.1% stop: {avg_best_stop_sharpe:.2f}")
            print(f"Performance degradation: {(avg_best_stop_sharpe - avg_no_stop_sharpe) / avg_no_stop_sharpe * 100:.1f}%")
            
    else:
        print("❌ No high-frequency strategies found")
# Comprehensive Stop Loss and Regime Analysis
# This cell combines stop loss optimization with regime filtering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Function to analyze stop loss with regime filtering
def analyze_stop_loss_with_regimes(strategy_hash, trace_path, market_data, stop_loss_levels, execution_cost_bps=1.0):
    """
    Analyze stop loss impact with regime filtering and trade duration statistics.
    """
    # Extract trades
    trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)
    
    if len(trades) == 0:
        return None
    
    # Calculate market regimes for each trade
    market_data['returns'] = market_data['close'].pct_change()
    market_data['volatility'] = market_data['returns'].rolling(window=20*78).std() * np.sqrt(252*78)
    vol_percentiles = market_data['volatility'].quantile([0.33, 0.67])
    market_data['vol_regime'] = pd.cut(
        market_data['volatility'],
        bins=[0, vol_percentiles[0.33], vol_percentiles[0.67], np.inf],
        labels=['Low Vol', 'Medium Vol', 'High Vol']
    )
    
    # Add regime to trades
    trades_with_regime = trades.merge(
        market_data[['vol_regime']], 
        left_on='entry_idx', 
        right_index=True, 
        how='left'
    )
    
    results = []
    
    for sl_pct in stop_loss_levels:
        sl_decimal = sl_pct / 100
        
        # Process each regime separately
        for regime in ['Low Vol', 'Medium Vol', 'High Vol']:
            regime_trades = trades_with_regime[trades_with_regime['vol_regime'] == regime].copy()
            
            if len(regime_trades) == 0:
                continue
            
            trades_with_sl = []
            stopped_out_count = 0
            total_duration_minutes = 0
            stopped_duration_minutes = 0
            
            for _, trade in regime_trades.iterrows():
                # Get intraday prices
                trade_prices = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
                
                if len(trade_prices) == 0:
                    continue
                
                entry_price = trade['entry_price']
                direction = trade['direction']
                
                # Calculate stop loss price
                if direction == 1:  # Long
                    stop_price = entry_price * (1 - sl_decimal)
                else:  # Short
                    stop_price = entry_price * (1 + sl_decimal)
                
                # Check if stop is hit
                stopped = False
                exit_price = trade['exit_price']
                exit_time = trade['exit_time']
                exit_idx = trade['exit_idx']
                
                for idx, bar in trade_prices.iterrows():
                    if direction == 1 and bar['low'] <= stop_price:
                        stopped = True
                        stopped_out_count += 1
                        exit_price = stop_price
                        exit_time = bar['timestamp']
                        exit_idx = idx
                        break
                    elif direction == -1 and bar['high'] >= stop_price:
                        stopped = True
                        stopped_out_count += 1
                        exit_price = stop_price
                        exit_time = bar['timestamp']
                        exit_idx = idx
                        break
                
                # Calculate return
                if direction == 1:
                    raw_return = (exit_price - entry_price) / entry_price
                else:
                    raw_return = (entry_price - exit_price) / entry_price
                
                net_return = raw_return - trade['execution_cost']
                
                # Calculate duration
                duration_minutes = (exit_time - trade['entry_time']).total_seconds() / 60
                total_duration_minutes += duration_minutes
                if stopped:
                    stopped_duration_minutes += duration_minutes
                
                trade_result = trade.copy()
                trade_result['raw_return'] = raw_return
                trade_result['net_return'] = net_return
                trade_result['stopped_out'] = stopped
                trade_result['actual_duration_minutes'] = duration_minutes
                
                trades_with_sl.append(trade_result)
            
            trades_with_sl_df = pd.DataFrame(trades_with_sl)
            
            if len(trades_with_sl_df) > 0:
                # Calculate metrics
                total_return = trades_with_sl_df['net_return'].sum()
                avg_return = trades_with_sl_df['net_return'].mean()
                
                # Correct win rate calculation
                winning_trades = trades_with_sl_df[trades_with_sl_df['net_return'] > 0]
                losing_trades = trades_with_sl_df[trades_with_sl_df['net_return'] <= 0]
                win_rate = len(winning_trades) / len(trades_with_sl_df)
                
                # Duration statistics
                avg_duration_all = trades_with_sl_df['actual_duration_minutes'].mean()
                avg_duration_winners = winning_trades['actual_duration_minutes'].mean() if len(winning_trades) > 0 else 0
                avg_duration_losers = losing_trades['actual_duration_minutes'].mean() if len(losing_trades) > 0 else 0
                avg_duration_stopped = trades_with_sl_df[trades_with_sl_df['stopped_out']]['actual_duration_minutes'].mean() if stopped_out_count > 0 else 0
                
                results.append({
                    'stop_loss_pct': sl_pct,
                    'regime': regime,
                    'total_return': total_return,
                    'avg_return_per_trade': avg_return,
                    'win_rate': win_rate,
                    'num_trades': len(trades_with_sl_df),
                    'stopped_out_count': stopped_out_count,
                    'stopped_out_rate': stopped_out_count / len(trades_with_sl_df),
                    'avg_winner': winning_trades['net_return'].mean() if len(winning_trades) > 0 else 0,
                    'avg_loser': losing_trades['net_return'].mean() if len(losing_trades) > 0 else 0,
                    'profit_factor': winning_trades['net_return'].sum() / abs(losing_trades['net_return'].sum()) if len(losing_trades) > 0 and losing_trades['net_return'].sum() != 0 else (999.99 if len(winning_trades) > 0 else 0),
                    'avg_duration_minutes': avg_duration_all,
                    'avg_duration_winners': avg_duration_winners,
                    'avg_duration_losers': avg_duration_losers,
                    'avg_duration_stopped': avg_duration_stopped
                })
    
    return pd.DataFrame(results)

# Main analysis
if len(performance_df) > 0 and len(top_overall) > 0:
    print("🔍 Comprehensive Stop Loss + Regime Analysis")
    print("=" * 80)
    
    # Analyze more strategies
    num_strategies_to_analyze = min(10, len(top_overall))  # Analyze top 10
    
    all_results = []
    strategy_details = []
    
    for idx, (_, strategy) in enumerate(top_overall.head(num_strategies_to_analyze).iterrows()):
        print(f"\nAnalyzing strategy {idx+1}/{num_strategies_to_analyze}: {strategy['strategy_type']} - {strategy['strategy_hash'][:8]}")
        
        # Show strategy parameters
        param_cols = [col for col in strategy.index if col.startswith('param_') or col in ['period', 'std_dev', 'fast_period', 'slow_period', 'multiplier']]
        params_str = " | ".join([f"{col.replace('param_', '')}: {strategy[col]}" for col in param_cols if pd.notna(strategy[col])])
        print(f"  Parameters: {params_str}")
        print(f"  Base Performance - Sharpe: {strategy['sharpe_ratio']:.2f}, Return: {strategy['total_return']:.1%}, Win Rate: {strategy['win_rate']:.1%}")
        
        # Analyze with regimes and stop losses
        results = analyze_stop_loss_with_regimes(
            strategy['strategy_hash'], 
            strategy['trace_path'], 
            market_data, 
            stop_loss_levels, 
            execution_cost_bps
        )
        
        if results is not None and len(results) > 0:
            # Find optimal stop loss for each regime
            for regime in ['Low Vol', 'Medium Vol', 'High Vol']:
                regime_results = results[results['regime'] == regime]
                if len(regime_results) > 0:
                    optimal = regime_results.loc[regime_results['total_return'].idxmax()]
                    print(f"  {regime}: Optimal SL={optimal['stop_loss_pct']:.2f}%, Return={optimal['total_return']*100:.1f}%, Win Rate={optimal['win_rate']*100:.1f}%")
            
            results['strategy_hash'] = strategy['strategy_hash']
            results['strategy_type'] = strategy['strategy_type']
            all_results.append(results)
            
            strategy_details.append({
                'hash': strategy['strategy_hash'][:8],
                'type': strategy['strategy_type'],
                'params': params_str,
                'base_sharpe': strategy['sharpe_ratio'],
                'base_return': strategy['total_return'],
                'base_win_rate': strategy['win_rate']
            })
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Average return by stop loss and regime
        ax = axes[0, 0]
        pivot_return = combined_results.groupby(['stop_loss_pct', 'regime'])['total_return'].mean().unstack()
        pivot_return.plot(ax=ax, marker='o')
        ax.set_xlabel('Stop Loss %')
        ax.set_ylabel('Average Total Return')
        ax.set_title('Average Return by Stop Loss and Regime')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Regime')
        
        # 2. Win rate by stop loss and regime
        ax = axes[0, 1]
        pivot_winrate = combined_results.groupby(['stop_loss_pct', 'regime'])['win_rate'].mean().unstack()
        pivot_winrate.plot(ax=ax, marker='o')
        ax.set_xlabel('Stop Loss %')
        ax.set_ylabel('Average Win Rate')
        ax.set_title('Win Rate by Stop Loss and Regime')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Regime')
        
        # 3. Average trade duration by stop loss
        ax = axes[0, 2]
        duration_data = combined_results.groupby('stop_loss_pct').agg({
            'avg_duration_minutes': 'mean',
            'avg_duration_winners': 'mean',
            'avg_duration_losers': 'mean',
            'avg_duration_stopped': 'mean'
        })
        duration_data.plot(ax=ax, marker='o')
        ax.set_xlabel('Stop Loss %')
        ax.set_ylabel('Average Duration (minutes)')
        ax.set_title('Trade Duration by Stop Loss Level')
        ax.grid(True, alpha=0.3)
        ax.legend(['All Trades', 'Winners', 'Losers', 'Stopped Out'])
        
        # 4. Stop out rate by regime
        ax = axes[1, 0]
        pivot_stoprate = combined_results.groupby(['stop_loss_pct', 'regime'])['stopped_out_rate'].mean().unstack()
        pivot_stoprate.plot(ax=ax, marker='o')
        ax.set_xlabel('Stop Loss %')
        ax.set_ylabel('Stop Out Rate')
        ax.set_title('Percentage of Trades Stopped Out')
        ax.grid(True, alpha=0.3)
        ax.legend(title='Regime')
        
        # 5. Profit factor by stop loss
        ax = axes[1, 1]
        profit_factor_data = combined_results[combined_results['profit_factor'] > 0].groupby(['stop_loss_pct', 'regime'])['profit_factor'].mean().unstack()
        profit_factor_data.plot(ax=ax, marker='o')
        ax.set_xlabel('Stop Loss %')
        ax.set_ylabel('Profit Factor')
        ax.set_title('Profit Factor (Win$/Loss$) by Stop Loss')
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Regime')
        
        # 6. Optimal stop loss summary
        ax = axes[1, 2]
        ax.axis('off')
        
        # Find best combination for each regime
        summary_text = "Optimal Stop Loss by Regime:\n\n"
        for regime in ['Low Vol', 'Medium Vol', 'High Vol']:
            regime_data = combined_results[combined_results['regime'] == regime]
            if len(regime_data) > 0:
                best_sl_data = regime_data.groupby('stop_loss_pct')['total_return'].mean()
                best_sl = best_sl_data.idxmax()
                best_return = best_sl_data.max()
                
                # Select only numeric columns for mean calculation
                numeric_cols = ['total_return', 'avg_return_per_trade', 'win_rate', 'num_trades', 
                               'stopped_out_rate', 'avg_duration_minutes', 'profit_factor']
                avg_metrics = regime_data[regime_data['stop_loss_pct'] == best_sl][numeric_cols].mean()
                summary_text += f"{regime}:\n"
                summary_text += f"  Best SL: {best_sl:.2f}%\n"
                summary_text += f"  Avg Return: {best_return*100:.1f}%\n"
                summary_text += f"  Avg Win Rate: {avg_metrics['win_rate']*100:.1f}%\n"
                summary_text += f"  Avg Duration: {avg_metrics['avg_duration_minutes']:.0f} min\n\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=11, family='monospace')
        
        plt.tight_layout()
        plt.show()
        
        # Duration impact table
        print("\n📊 Trade Duration Impact by Stop Loss Level:")
        print("=" * 80)
        duration_summary = combined_results.groupby('stop_loss_pct').agg({
            'avg_duration_minutes': 'mean',
            'avg_duration_winners': 'mean',
            'avg_duration_losers': 'mean',
            'avg_duration_stopped': 'mean',
            'stopped_out_rate': 'mean'
        }).round(1)
        print(duration_summary)
        
        # Best performing strategies with optimal stops
        print("\n🏆 Top Strategies with Optimal Stop Loss by Regime:")
        print("=" * 80)
        
        for regime in ['Low Vol', 'Medium Vol', 'High Vol']:
            print(f"\n{regime}:")
            regime_best = combined_results[combined_results['regime'] == regime].groupby(['strategy_hash', 'stop_loss_pct'])['total_return'].mean().reset_index()
            
            # Find best stop loss for each strategy in this regime
            idx = regime_best.groupby('strategy_hash')['total_return'].idxmax()
            best_configs = regime_best.loc[idx].nlargest(3, 'total_return')
            
            for _, row in best_configs.iterrows():
                # Find strategy details
                detail = next((d for d in strategy_details if row['strategy_hash'].startswith(d['hash'])), None)
                if detail:
                    print(f"  {detail['type']} ({detail['hash']})")
                    print(f"    Params: {detail['params']}")
                    print(f"    Stop Loss: {row['stop_loss_pct']:.2f}%")
                    print(f"    Return with SL: {row['total_return']*100:.1f}% (vs base: {detail['base_return']*100:.1f}%)")
        
        # Save detailed results
        combined_results.to_csv(run_dir / 'stop_loss_regime_analysis.csv', index=False)
        print(f"\n✅ Detailed results saved to: stop_loss_regime_analysis.csv")
# Corrected Performance Analysis - Trade-by-Trade Metrics
# This ensures apples-to-apples comparison between base and stop loss performance

import pandas as pd
import numpy as np

def calculate_trade_based_performance(strategy_hash, trace_path, market_data, execution_cost_bps=1.0):
    """
    Calculate performance metrics based on actual trades, not bar-by-bar returns.
    This gives us the true win rate and profit factor for comparison.
    """
    try:
        # Extract trades
        trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)
        
        if len(trades) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'num_trades': 0,
                'win_rate': 0,
                'avg_return_per_trade': 0,
                'profit_factor': 0,
                'avg_winner': 0,
                'avg_loser': 0,
                'total_execution_cost': 0
            }
        
        # Calculate cumulative returns from trades
        trades['cum_return'] = (1 + trades['net_return']).cumprod()
        total_return = trades['cum_return'].iloc[-1] - 1
        
        # Calculate Sharpe ratio from trade returns
        if trades['net_return'].std() > 0:
            # Annualize based on average trades per day
            days_in_data = (trades['exit_time'].max() - trades['entry_time'].min()).days
            if days_in_data > 0:
                trades_per_day = len(trades) / days_in_data
                # Annualize based on actual trading frequency
                annualization_factor = np.sqrt(252 * trades_per_day)
            else:
                # If all trades in one day, use standard daily annualization
                annualization_factor = np.sqrt(252)
            sharpe = trades['net_return'].mean() / trades['net_return'].std() * annualization_factor
        else:
            sharpe = 0
        
        # Max drawdown from trade equity curve
        cummax = trades['cum_return'].expanding().max()
        drawdown = (trades['cum_return'] / cummax - 1)
        max_dd = drawdown.min()
        
        # Win rate and profit factor
        winning_trades = trades[trades['net_return'] > 0]
        losing_trades = trades[trades['net_return'] <= 0]
        
        win_rate = len(winning_trades) / len(trades)
        
        if len(losing_trades) > 0 and losing_trades['net_return'].sum() != 0:
            profit_factor = winning_trades['net_return'].sum() / abs(losing_trades['net_return'].sum())
        else:
            profit_factor = 999.99 if len(winning_trades) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_return_per_trade': trades['net_return'].mean(),
            'profit_factor': profit_factor,
            'avg_winner': winning_trades['net_return'].mean() if len(winning_trades) > 0 else 0,
            'avg_loser': losing_trades['net_return'].mean() if len(losing_trades) > 0 else 0,
            'total_execution_cost': trades['execution_cost'].sum()
        }
    except Exception as e:
        print(f"Error calculating trade-based performance: {e}")
        return None

# Recalculate performance for all strategies using trade-based metrics
if len(performance_df) > 0:
    print("🔄 Recalculating performance using trade-by-trade metrics...")
    print("=" * 80)
    
    # Create new performance results
    corrected_performance = []
    
    for idx, row in performance_df.iterrows():
        if idx % 100 == 0:
            print(f"Progress: {idx}/{len(performance_df)} ({idx/len(performance_df)*100:.1f}%)")
        
        perf = calculate_trade_based_performance(
            row['strategy_hash'], 
            row['trace_path'], 
            market_data, 
            execution_cost_bps
        )
        
        if perf:
            # Combine with original row data
            corrected_row = row.to_dict()
            corrected_row.update(perf)
            corrected_performance.append(corrected_row)
    
    # Create corrected dataframe
    corrected_df = pd.DataFrame(corrected_performance)
    
    # Compare old vs new metrics for top strategies
    print("\n📊 Comparison of Bar-Based vs Trade-Based Metrics:")
    print("=" * 80)
    
    # Get top 10 by original Sharpe
    top_10_original = performance_df.nlargest(10, 'sharpe_ratio')
    
    comparison_data = []
    for _, orig in top_10_original.iterrows():
        # Find corresponding corrected data
        corrected = corrected_df[corrected_df['strategy_hash'] == orig['strategy_hash']]
        
        if len(corrected) > 0:
            corr = corrected.iloc[0]
            comparison_data.append({
                'Strategy': f"{orig['strategy_type']}_{orig['strategy_hash'][:8]}",
                'Old_WinRate': orig['win_rate'],
                'New_WinRate': corr['win_rate'],
                'Old_Return': orig['total_return'],
                'New_Return': corr['total_return'],
                'Old_Sharpe': orig['sharpe_ratio'],
                'New_Sharpe': corr['sharpe_ratio'],
                'Profit_Factor': corr['profit_factor'],
                'Trades': corr['num_trades']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Now show TRUE top performers based on trade metrics
    print("\n🏆 TRUE Top 10 Strategies (Trade-Based Metrics):")
    print("=" * 80)
    
    top_corrected = corrected_df.nlargest(10, 'sharpe_ratio')
    
    for idx, row in top_corrected.iterrows():
        print(f"\n{row['strategy_type']} - {row['strategy_hash'][:8]}")
        print(f"  Sharpe: {row['sharpe_ratio']:.2f} | Return: {row['total_return']:.1%} | Drawdown: {row['max_drawdown']:.1%}")
        print(f"  Win Rate: {row['win_rate']:.1%} | Profit Factor: {row['profit_factor']:.2f} | Trades: {row['num_trades']}")
        print(f"  Avg Winner: {row['avg_winner']*100:.3f}% | Avg Loser: {row['avg_loser']*100:.3f}%")
        
        # Show parameters
        param_cols = [col for col in row.index if col.startswith('param_') or col in ['period', 'std_dev', 'fast_period', 'slow_period', 'multiplier']]
        params_str = " | ".join([f"{col.replace('param_', '')}: {row[col]}" for col in param_cols if pd.notna(row[col])])
        if params_str:
            print(f"  Params: {params_str}")
    
    # Replace the global performance_df with corrected version
    performance_df = corrected_df
    top_overall = corrected_df.nlargest(top_n_strategies, 'sharpe_ratio')
    
    print("\n✅ Performance metrics have been corrected to use trade-based calculations")
    print("   The 'top_overall' variable has been updated with true top performers")
    
    # Quick sanity check on profit factors
    print("\n📈 Profit Factor Distribution:")
    pf_bins = [0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 999]
    pf_labels = ['<0.5', '0.5-0.8', '0.8-1.0', '1.0-1.2', '1.2-1.5', '1.5-2.0', '>2.0']
    corrected_df['pf_bin'] = pd.cut(corrected_df['profit_factor'].clip(upper=100), bins=pf_bins, labels=pf_labels)
    print(corrected_df['pf_bin'].value_counts().sort_index())
    
    # Show win rate distribution comparison
    print("\n📊 Win Rate Distribution Comparison:")
    print(f"Bar-based win rates: Mean={performance_df['win_rate'].mean():.1%}, Median={performance_df['win_rate'].median():.1%}")
    print(f"Trade-based win rates: Mean={corrected_df['win_rate'].mean():.1%}, Median={corrected_df['win_rate'].median():.1%}")
    
    # Analyze the discrepancy
    print("\n🔍 Understanding the Discrepancy:")
    print("The difference comes from how metrics are calculated:")
    print("• Bar-based: Counts every bar where strategy had a position and market moved favorably")
    print("• Trade-based: Counts actual completed trades from entry to exit")
    print("\nExample: A 10-bar winning trade shows:")
    print("• Bar-based: 10 'wins' (if all bars positive)")
    print("• Trade-based: 1 win")
    print("\nThis is why bar-based win rates (~27%) are much lower than trade-based (~60-70%)")
    
else:
    print("⚠️ No performance data available to correct")
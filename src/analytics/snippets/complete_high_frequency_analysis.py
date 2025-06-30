# Complete High-Frequency Strategy Analysis Pipeline
# This replaces the standard analysis with frequency-filtered approach

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
MIN_TRADES_PER_DAY = 2  # Minimum trade frequency requirement
MIN_TOTAL_TRADES = 100  # Absolute minimum to ensure statistical significance

# Step 1: Filter by trade frequency
print("📊 Step 1: Filtering by Trade Frequency")
print("=" * 80)

if len(performance_df) > 0 and market_data is not None:
    # Calculate trading days
    trading_days = len(market_data['timestamp'].dt.date.unique())
    print(f"Dataset spans {trading_days} trading days")
    print(f"Minimum requirement: {MIN_TRADES_PER_DAY} trades/day = {MIN_TRADES_PER_DAY * trading_days} total trades")
    
    # Apply filters
    frequency_mask = (performance_df['num_trades'] >= MIN_TRADES_PER_DAY * trading_days) & \
                    (performance_df['num_trades'] >= MIN_TOTAL_TRADES)
    
    high_freq_df = performance_df[frequency_mask].copy()
    
    print(f"\nResults:")
    print(f"  Total strategies tested: {len(performance_df)}")
    print(f"  High-frequency strategies: {len(high_freq_df)} ({len(high_freq_df)/len(performance_df)*100:.1f}%)")
    
    if len(high_freq_df) == 0:
        print("\n❌ No strategies meet the frequency requirements!")
        print("\nTrade frequency distribution (all strategies):")
        trades_per_day = performance_df['num_trades'] / trading_days
        print(f"  Min: {trades_per_day.min():.2f} trades/day")
        print(f"  Median: {trades_per_day.median():.2f} trades/day") 
        print(f"  Max: {trades_per_day.max():.2f} trades/day")
        print("\n💡 Suggestions:")
        print("  1. Lower MIN_TRADES_PER_DAY requirement")
        print("  2. Use shorter timeframe data (1m)")
        print("  3. Test more aggressive parameters")
    else:
        # Sort by Sharpe ratio
        high_freq_df = high_freq_df.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)
        
        # Update global variables
        performance_df_original = performance_df.copy()  # Keep original for comparison
        performance_df = high_freq_df
        top_overall = high_freq_df.head(20)
        
        print(f"\n✅ Updated analysis to focus on {len(high_freq_df)} high-frequency strategies")
        
        # Step 2: Performance Overview
        print("\n📈 Step 2: High-Frequency Strategy Performance")
        print("=" * 80)
        
        # Show top 10
        print("\nTop 10 High-Frequency Strategies:")
        for idx, row in high_freq_df.head(10).iterrows():
            trades_per_day = row['num_trades'] / trading_days
            print(f"\n{idx+1}. {row['strategy_type']} - {row['strategy_hash'][:8]}")
            print(f"   Trades: {row['num_trades']} ({trades_per_day:.1f}/day)")
            print(f"   Sharpe: {row['sharpe_ratio']:.2f} | Return: {row['total_return']*100:.2f}%")
            print(f"   Win Rate: {row['win_rate']*100:.1f}% | Profit Factor: {row.get('profit_factor', 0):.2f}")
            
            # Parameters
            params = []
            for p in ['period', 'std_dev', 'fast_period', 'slow_period', 'multiplier']:
                if p in row and pd.notna(row[p]):
                    params.append(f"{p}={row[p]}")
            if params:
                print(f"   Params: {' | '.join(params)}")
        
        # Step 3: Parameter Analysis
        print("\n🔍 Step 3: Parameter Analysis")
        print("=" * 80)
        
        # Create parameter heatmaps
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Sharpe heatmap
        ax = axes[0, 0]
        if 'period' in high_freq_df.columns and 'std_dev' in high_freq_df.columns:
            pivot_sharpe = high_freq_df.pivot_table(
                values='sharpe_ratio',
                index='period',
                columns='std_dev',
                aggfunc='mean'
            )
            if not pivot_sharpe.empty:
                sns.heatmap(pivot_sharpe, cmap='RdYlGn', center=0, ax=ax,
                           cbar_kws={'label': 'Sharpe Ratio'})
                ax.set_title('Sharpe Ratio by Parameters')
        
        # 2. Return heatmap
        ax = axes[0, 1]
        if 'period' in high_freq_df.columns and 'std_dev' in high_freq_df.columns:
            pivot_return = high_freq_df.pivot_table(
                values='total_return',
                index='period',
                columns='std_dev',
                aggfunc='mean'
            )
            if not pivot_return.empty:
                sns.heatmap(pivot_return * 100, cmap='RdYlGn', center=0, ax=ax,
                           cbar_kws={'label': 'Return %'})
                ax.set_title('Total Return by Parameters')
        
        # 3. Trade frequency heatmap
        ax = axes[0, 2]
        if 'period' in high_freq_df.columns and 'std_dev' in high_freq_df.columns:
            high_freq_df['trades_per_day'] = high_freq_df['num_trades'] / trading_days
            pivot_freq = high_freq_df.pivot_table(
                values='trades_per_day',
                index='period',
                columns='std_dev',
                aggfunc='mean'
            )
            if not pivot_freq.empty:
                sns.heatmap(pivot_freq, cmap='YlOrRd', ax=ax,
                           cbar_kws={'label': 'Trades/Day'})
                ax.set_title('Trade Frequency by Parameters')
        
        # 4. Profit factor heatmap
        ax = axes[1, 0]
        if 'period' in high_freq_df.columns and 'std_dev' in high_freq_df.columns and 'profit_factor' in high_freq_df.columns:
            pivot_pf = high_freq_df.pivot_table(
                values='profit_factor',
                index='period',
                columns='std_dev',
                aggfunc='mean'
            )
            if not pivot_pf.empty:
                sns.heatmap(pivot_pf, cmap='RdYlGn', center=1.0, ax=ax,
                           cbar_kws={'label': 'Profit Factor'})
                ax.set_title('Profit Factor by Parameters')
        
        # 5. Performance scatter
        ax = axes[1, 1]
        scatter = ax.scatter(high_freq_df['num_trades'], 
                           high_freq_df['sharpe_ratio'],
                           c=high_freq_df['total_return'] * 100,
                           s=50, alpha=0.6, cmap='RdYlGn')
        ax.set_xlabel('Number of Trades')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe vs Trade Count (color = return %)')
        plt.colorbar(scatter, ax=ax, label='Return %')
        
        # 6. Summary stats
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = "High-Frequency Strategy Summary:\n\n"
        summary_text += f"Total Strategies: {len(high_freq_df)}\n"
        summary_text += f"Profitable (return > 0): {(high_freq_df['total_return'] > 0).sum()}\n"
        summary_text += f"Positive Sharpe: {(high_freq_df['sharpe_ratio'] > 0).sum()}\n"
        summary_text += f"Profit Factor > 1: {(high_freq_df['profit_factor'] > 1).sum()}\n\n"
        
        summary_text += "Average Metrics:\n"
        summary_text += f"Sharpe: {high_freq_df['sharpe_ratio'].mean():.2f}\n"
        summary_text += f"Return: {high_freq_df['total_return'].mean()*100:.2f}%\n"
        summary_text += f"Win Rate: {high_freq_df['win_rate'].mean()*100:.1f}%\n"
        summary_text += f"Trades/Day: {(high_freq_df['num_trades']/trading_days).mean():.1f}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=12, family='monospace')
        
        plt.tight_layout()
        plt.show()
        
        # Step 4: Stop Loss Analysis on High-Frequency Strategies
        print("\n🛑 Step 4: Stop Loss Analysis")
        print("=" * 80)
        
        # Test stops on top 3 strategies
        stop_levels = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
        
        for idx, row in high_freq_df.head(3).iterrows():
            print(f"\nStrategy {idx+1}: {row['strategy_type']} - {row['strategy_hash'][:8]}")
            print(f"  Base: Sharpe={row['sharpe_ratio']:.2f}, Return={row['total_return']*100:.2f}%")
            
            # Extract trades
            trades = extract_trades(row['strategy_hash'], row['trace_path'], market_data, execution_cost_bps)
            
            if len(trades) > 0:
                # Test stops
                sl_results = calculate_stop_loss_impact(trades, stop_levels, market_data)
                
                if len(sl_results) > 0:
                    # Find optimal
                    optimal_idx = sl_results['total_return'].idxmax()
                    optimal = sl_results.iloc[optimal_idx]
                    
                    print(f"  Optimal Stop: {optimal['stop_loss_pct']}%")
                    print(f"  New Return: {optimal['total_return']*100:.2f}% (was {row['total_return']*100:.2f}%)")
                    print(f"  Stop Rate: {optimal['stopped_out_rate']*100:.1f}%")
        
        # Step 5: Save results
        print("\n💾 Step 5: Saving Results")
        print("=" * 80)
        
        # Save high-frequency strategies
        high_freq_df.to_csv(run_dir / 'high_frequency_strategies.csv', index=False)
        print(f"Saved {len(high_freq_df)} strategies to: high_frequency_strategies.csv")
        
        # Save parameter summary
        if 'period' in high_freq_df.columns and 'std_dev' in high_freq_df.columns:
            param_summary = high_freq_df.groupby(['period', 'std_dev']).agg({
                'sharpe_ratio': ['mean', 'std', 'count'],
                'total_return': 'mean',
                'num_trades': 'mean'
            }).round(3)
            param_summary.to_csv(run_dir / 'parameter_summary.csv')
            print("Saved parameter summary to: parameter_summary.csv")
        
        print("\n✅ High-frequency analysis complete!")
        print(f"   Working with {len(high_freq_df)} strategies that trade ≥{MIN_TRADES_PER_DAY} times per day")
        
else:
    print("⚠️ No performance data available")
# High Frequency Strategy Analysis
# Filters strategies by minimum trade frequency, then analyzes performance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Filter strategies by minimum trade frequency
MIN_TRADES_PER_DAY = 2  # Configurable threshold

if len(performance_df) > 0 and market_data is not None:
    print("🔍 High-Frequency Strategy Analysis")
    print("=" * 80)
    
    # Calculate trading days in the dataset
    trading_days = len(market_data['timestamp'].dt.date.unique())
    print(f"Dataset spans {trading_days} trading days")
    print(f"Minimum trade frequency requirement: {MIN_TRADES_PER_DAY} trades/day")
    print(f"Minimum total trades required: {MIN_TRADES_PER_DAY * trading_days}")
    
    # Filter strategies by trade frequency
    high_freq_strategies = performance_df[performance_df['num_trades'] >= MIN_TRADES_PER_DAY * trading_days].copy()
    
    print(f"\n📊 Trade Frequency Distribution:")
    print(f"Total strategies analyzed: {len(performance_df)}")
    print(f"Strategies meeting frequency requirement: {len(high_freq_strategies)}")
    print(f"Percentage qualifying: {len(high_freq_strategies)/len(performance_df)*100:.1f}%")
    
    if len(high_freq_strategies) > 0:
        # Sort by Sharpe ratio after filtering
        high_freq_strategies = high_freq_strategies.sort_values('sharpe_ratio', ascending=False)
        
        # Show trade frequency statistics
        print("\n📈 Trade Frequency Statistics (qualifying strategies):")
        print(f"Average trades per day: {high_freq_strategies['num_trades'].mean() / trading_days:.1f}")
        print(f"Median trades per day: {high_freq_strategies['num_trades'].median() / trading_days:.1f}")
        print(f"Max trades per day: {high_freq_strategies['num_trades'].max() / trading_days:.1f}")
        
        # Show top performers among high-frequency strategies
        print(f"\n🏆 Top 10 High-Frequency Strategies (>={MIN_TRADES_PER_DAY} trades/day):")
        print("=" * 100)
        
        top_hf = high_freq_strategies.head(10)
        
        for idx, row in top_hf.iterrows():
            trades_per_day = row['num_trades'] / trading_days
            print(f"\n{row['strategy_type']} - {row['strategy_hash'][:8]}")
            print(f"  Trades/Day: {trades_per_day:.1f} | Total Trades: {row['num_trades']}")
            print(f"  Sharpe: {row['sharpe_ratio']:.2f} | Return: {row['total_return']*100:.2f}% | Win Rate: {row['win_rate']*100:.1f}%")
            print(f"  Profit Factor: {row.get('profit_factor', 0):.2f} | Avg Return/Trade: {row.get('avg_return_per_trade', 0)*100:.3f}%")
            
            # Show parameters
            param_cols = [col for col in row.index if col in ['period', 'std_dev', 'fast_period', 'slow_period', 'multiplier']]
            if param_cols:
                params_str = " | ".join([f"{col}: {row[col]}" for col in param_cols if pd.notna(row[col])])
                print(f"  Params: {params_str}")
        
        # Visualize high-frequency strategies
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Trade frequency distribution
        ax = axes[0, 0]
        trades_per_day_all = performance_df['num_trades'] / trading_days
        trades_per_day_hf = high_freq_strategies['num_trades'] / trading_days
        
        ax.hist(trades_per_day_all, bins=50, alpha=0.5, label='All Strategies', color='gray')
        ax.hist(trades_per_day_hf, bins=30, alpha=0.7, label='High Frequency', color='blue')
        ax.axvline(MIN_TRADES_PER_DAY, color='red', linestyle='--', label=f'Min Requirement ({MIN_TRADES_PER_DAY})')
        ax.set_xlabel('Trades per Day')
        ax.set_ylabel('Count')
        ax.set_title('Trade Frequency Distribution')
        ax.legend()
        ax.set_xlim(0, 20)  # Focus on reasonable range
        
        # 2. Sharpe vs Trade Frequency
        ax = axes[0, 1]
        ax.scatter(trades_per_day_hf, high_freq_strategies['sharpe_ratio'], alpha=0.6)
        ax.set_xlabel('Trades per Day')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe Ratio vs Trade Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add correlation
        corr = trades_per_day_hf.corr(high_freq_strategies['sharpe_ratio'])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes)
        
        # 3. Returns vs Trade Frequency
        ax = axes[0, 2]
        ax.scatter(trades_per_day_hf, high_freq_strategies['total_return'] * 100, alpha=0.6)
        ax.set_xlabel('Trades per Day')
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Returns vs Trade Frequency')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        # 4. Win Rate vs Trade Frequency
        ax = axes[1, 0]
        ax.scatter(trades_per_day_hf, high_freq_strategies['win_rate'] * 100, alpha=0.6)
        ax.set_xlabel('Trades per Day')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate vs Trade Frequency')
        ax.grid(True, alpha=0.3)
        
        # 5. Parameter heatmap for high-frequency strategies
        ax = axes[1, 1]
        if 'period' in high_freq_strategies.columns and 'std_dev' in high_freq_strategies.columns:
            # Create pivot table for Sharpe ratio
            pivot_sharpe = high_freq_strategies.pivot_table(
                values='sharpe_ratio',
                index='period',
                columns='std_dev',
                aggfunc='mean'
            )
            
            if not pivot_sharpe.empty:
                sns.heatmap(pivot_sharpe, cmap='RdYlGn', center=0, 
                           cbar_kws={'label': 'Sharpe Ratio'}, ax=ax)
                ax.set_title('Sharpe Ratio Heatmap (High-Freq Strategies)')
            else:
                ax.text(0.5, 0.5, 'Insufficient data for heatmap', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # 6. Strategy type distribution
        ax = axes[1, 2]
        strategy_counts = high_freq_strategies['strategy_type'].value_counts()
        ax.pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%')
        ax.set_title('High-Frequency Strategies by Type')
        
        plt.tight_layout()
        plt.show()
        
        # Parameter analysis for high-frequency strategies
        print("\n📊 Parameter Analysis (High-Frequency Strategies):")
        print("=" * 60)
        
        if 'period' in high_freq_strategies.columns:
            print(f"Period range: {high_freq_strategies['period'].min()} - {high_freq_strategies['period'].max()}")
            print(f"Most common period: {high_freq_strategies['period'].mode().values[0] if len(high_freq_strategies['period'].mode()) > 0 else 'N/A'}")
        
        if 'std_dev' in high_freq_strategies.columns:
            print(f"Std dev range: {high_freq_strategies['std_dev'].min():.1f} - {high_freq_strategies['std_dev'].max():.1f}")
            print(f"Most common std dev: {high_freq_strategies['std_dev'].mode().values[0] if len(high_freq_strategies['std_dev'].mode()) > 0 else 'N/A':.1f}")
        
        # Find parameter sweet spots
        if len(high_freq_strategies) > 10:
            print("\n🎯 Parameter Sweet Spots (top 25% by Sharpe):")
            top_quartile = high_freq_strategies.head(int(len(high_freq_strategies) * 0.25))
            
            for param in ['period', 'std_dev', 'fast_period', 'slow_period', 'multiplier']:
                if param in top_quartile.columns and top_quartile[param].notna().any():
                    print(f"{param}: {top_quartile[param].min():.1f} - {top_quartile[param].max():.1f} (avg: {top_quartile[param].mean():.1f})")
        
        # Analyze execution cost impact
        print("\n💰 Execution Cost Impact (High-Frequency):")
        print(f"Average execution cost per strategy: {high_freq_strategies['total_execution_cost'].mean()*100:.3f}%")
        print(f"Execution cost vs trades correlation: {high_freq_strategies['num_trades'].corr(high_freq_strategies['total_execution_cost']):.3f}")
        
        # Create filtered dataset for further analysis
        top_hf_overall = high_freq_strategies.head(20)  # Top 20 for further analysis
        
        # Now analyze stops specifically for high-frequency strategies
        print("\n🛑 Stop Loss Analysis for High-Frequency Strategies:")
        print("=" * 60)
        
        # Quick stop loss test on top 3 high-frequency strategies
        stop_levels_test = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]  # Tighter stops for high frequency
        
        for idx, row in top_hf.head(3).iterrows():
            print(f"\nTesting stops for {row['strategy_type']} - {row['strategy_hash'][:8]} ({row['num_trades']/trading_days:.1f} trades/day)")
            
            # Extract trades and test stops
            trades = extract_trades(row['strategy_hash'], row['trace_path'], market_data, execution_cost_bps)
            
            if len(trades) > 0:
                # Test a few stop levels
                for stop_pct in [0.2, 0.5, 1.0]:
                    sl_impact = calculate_stop_loss_impact(trades, [stop_pct], market_data)
                    if len(sl_impact) > 0:
                        result = sl_impact.iloc[0]
                        print(f"  Stop {stop_pct}%: Return={result['total_return']*100:.2f}%, WinRate={result['win_rate']*100:.1f}%, Stopped={result['stopped_out_rate']*100:.1f}%")
        
        # Save high-frequency analysis
        high_freq_strategies.to_csv(run_dir / 'high_frequency_strategies.csv', index=False)
        print(f"\n✅ High-frequency strategy analysis saved to: high_frequency_strategies.csv")
        
        # Update global variables for subsequent analysis
        print("\n🔄 Updating analysis focus to high-frequency strategies only")
        performance_df = high_freq_strategies
        top_overall = top_hf_overall
        
        print(f"\n✨ Ready for further analysis with {len(high_freq_strategies)} high-frequency strategies")
        print("   Next steps:")
        print("   1. Run trailing stop analysis on these strategies")
        print("   2. Analyze regime performance")
        print("   3. Build ensembles from uncorrelated high-frequency strategies")
        
    else:
        print(f"\n❌ No strategies meet the minimum requirement of {MIN_TRADES_PER_DAY} trades/day!")
        print("   Consider:")
        print("   1. Lowering the frequency requirement")
        print("   2. Testing with shorter timeframe data (1m instead of 5m)")
        print("   3. Adjusting strategy parameters for more signals")
else:
    print("⚠️ No performance data available")
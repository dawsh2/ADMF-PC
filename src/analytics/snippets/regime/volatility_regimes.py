"""
Volatility Regime Analysis

Analyzes strategy performance across different volatility regimes.
This snippet can be loaded and customized in Jupyter notebooks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure variables from notebook are available
if 'performance_df' not in locals() or 'market_data' not in locals():
    print("❌ Error: This snippet requires performance_df and market_data from the main analysis")
    print("   Please run the main analysis cells first.")
else:
    print("🔍 Analyzing performance across volatility regimes...")
    
    # Calculate rolling volatility (20-day)
    market_data['returns'] = market_data['close'].pct_change()
    market_data['volatility'] = market_data['returns'].rolling(window=20*78).std() * np.sqrt(252*78)  # Annualized
    
    # Define volatility regimes
    vol_percentiles = market_data['volatility'].quantile([0.33, 0.67])
    market_data['vol_regime'] = pd.cut(
        market_data['volatility'],
        bins=[0, vol_percentiles[0.33], vol_percentiles[0.67], np.inf],
        labels=['Low Vol', 'Medium Vol', 'High Vol']
    )
    
    # Analyze top strategies in each regime
    top_strategies = performance_df.nlargest(10, 'sharpe_ratio')
    
    regime_results = []
    
    for _, strategy in top_strategies.iterrows():
        try:
            # Load strategy signals
            signals_path = run_dir / strategy['trace_path']
            signals = pd.read_parquet(signals_path)
            signals['ts'] = pd.to_datetime(signals['ts'])
            
            # Merge with market data and regimes
            df = market_data.merge(signals[['ts', 'val']], left_on='timestamp', right_on='ts', how='left')
            df['signal'] = df['val'].ffill().fillna(0)
            df['strategy_returns'] = df['returns'] * df['signal'].shift(1)
            
            # Calculate returns by regime
            for regime in ['Low Vol', 'Medium Vol', 'High Vol']:
                regime_mask = df['vol_regime'] == regime
                regime_returns = df.loc[regime_mask, 'strategy_returns']
                
                if len(regime_returns) > 0:
                    total_return = (1 + regime_returns).prod() - 1
                    sharpe = regime_returns.mean() / regime_returns.std() * np.sqrt(252*78) if regime_returns.std() > 0 else 0
                    
                    regime_results.append({
                        'strategy': f"{strategy['strategy_type']}_{strategy['strategy_hash'][:8]}",
                        'regime': regime,
                        'total_return': total_return,
                        'sharpe_ratio': sharpe,
                        'days_in_regime': regime_mask.sum()
                    })
        except Exception as e:
            print(f"  Warning: Could not analyze {strategy['strategy_hash'][:8]}: {e}")
    
    # Create results DataFrame
    regime_df = pd.DataFrame(regime_results)
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Average Sharpe by regime
    ax = axes[0, 0]
    regime_avg = regime_df.groupby('regime')['sharpe_ratio'].mean()
    regime_avg.plot(kind='bar', ax=ax, color=['green', 'yellow', 'red'])
    ax.set_title('Average Sharpe Ratio by Volatility Regime')
    ax.set_ylabel('Sharpe Ratio')
    ax.grid(True, alpha=0.3)
    
    # 2. Strategy performance heatmap
    ax = axes[0, 1]
    pivot = regime_df.pivot_table(values='sharpe_ratio', index='strategy', columns='regime')
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([s.split('_')[0] for s in pivot.index])
    ax.set_title('Strategy Sharpe Ratios Across Regimes')
    plt.colorbar(im, ax=ax)
    
    # 3. Regime distribution over time
    ax = axes[1, 0]
    regime_counts = market_data['vol_regime'].value_counts()
    regime_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
    ax.set_title('Time Spent in Each Regime')
    
    # 4. Best strategies by regime
    ax = axes[1, 1]
    ax.axis('off')
    
    regime_text = "Best Strategies by Regime:\n\n"
    for regime in ['Low Vol', 'Medium Vol', 'High Vol']:
        best = regime_df[regime_df['regime'] == regime].nlargest(3, 'sharpe_ratio')
        regime_text += f"{regime}:\n"
        for _, row in best.iterrows():
            regime_text += f"  • {row['strategy'].split('_')[0]}: Sharpe {row['sharpe_ratio']:.2f}\n"
        regime_text += "\n"
    
    ax.text(0.1, 0.9, regime_text, transform=ax.transAxes, verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print("\n📊 Regime Performance Summary:")
    print("=" * 60)
    
    for regime in ['Low Vol', 'Medium Vol', 'High Vol']:
        regime_data = regime_df[regime_df['regime'] == regime]
        print(f"\n{regime}:")
        print(f"  Average Sharpe: {regime_data['sharpe_ratio'].mean():.2f}")
        print(f"  Best Sharpe: {regime_data['sharpe_ratio'].max():.2f}")
        print(f"  Worst Sharpe: {regime_data['sharpe_ratio'].min():.2f}")
        print(f"  Days in regime: {regime_data['days_in_regime'].iloc[0] if len(regime_data) > 0 else 0}")
    
    # Recommendations
    print("\n🎯 Regime-Based Recommendations:")
    
    # Find strategies that perform well across all regimes
    consistent_strategies = []
    for strategy in regime_df['strategy'].unique():
        strategy_regimes = regime_df[regime_df['strategy'] == strategy]
        if len(strategy_regimes) == 3:  # Has data for all regimes
            min_sharpe = strategy_regimes['sharpe_ratio'].min()
            if min_sharpe > 0:  # Positive in all regimes
                consistent_strategies.append({
                    'strategy': strategy,
                    'min_sharpe': min_sharpe,
                    'avg_sharpe': strategy_regimes['sharpe_ratio'].mean()
                })
    
    if consistent_strategies:
        consistent_df = pd.DataFrame(consistent_strategies).sort_values('avg_sharpe', ascending=False)
        print("\nStrategies performing well across ALL volatility regimes:")
        for _, row in consistent_df.head(5).iterrows():
            print(f"  • {row['strategy'].split('_')[0]}: Min Sharpe {row['min_sharpe']:.2f}, Avg {row['avg_sharpe']:.2f}")
    
    # Export regime analysis
    regime_df.to_csv(run_dir / 'regime_analysis.csv', index=False)
    print(f"\n✅ Regime analysis saved to: regime_analysis.csv")
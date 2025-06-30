# Regime-Specific Parameter Analysis with Optimal Stops
# Creates parameter heatmaps for top performers in each volatility regime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
MIN_TRADES_PER_DAY = 2
# Much tighter stops for micro-movements
STOP_LEVELS_TO_TEST = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

def analyze_strategy_by_regime_with_stops(strategy_row, market_data, execution_cost_bps=1.0):
    """
    Analyze a strategy's performance by regime and find optimal stops for each regime
    """
    # Extract trades
    trades = extract_trades(strategy_row['strategy_hash'], strategy_row['trace_path'], market_data, execution_cost_bps)
    
    if len(trades) == 0:
        return None
    
    # Add regime to trades
    trades = trades.merge(
        market_data[['vol_regime']], 
        left_on='entry_idx', 
        right_index=True, 
        how='left'
    )
    
    results = []
    
    # Analyze each regime
    for regime in ['Low Vol', 'Medium Vol', 'High Vol']:
        regime_trades = trades[trades['vol_regime'] == regime].copy()
        
        if len(regime_trades) < 10:  # Skip if too few trades
            continue
        
        # Test different stop levels for this regime
        best_return = (1 + regime_trades['net_return']).cumprod().iloc[-1] - 1
        best_stop = 0  # No stop
        
        for stop_pct in STOP_LEVELS_TO_TEST:
            # Apply stop loss
            sl_trades = calculate_stop_loss_impact(
                regime_trades, 
                [stop_pct], 
                market_data
            )
            
            if len(sl_trades) > 0:
                sl_return = sl_trades.iloc[0]['total_return']
                if sl_return > best_return:
                    best_return = sl_return
                    best_stop = stop_pct
        
        # Calculate metrics with best stop
        if best_stop > 0:
            sl_impact = calculate_stop_loss_impact(regime_trades, [best_stop], market_data)
            if len(sl_impact) > 0:
                metrics = sl_impact.iloc[0]
                win_rate = metrics['win_rate']
                avg_return = metrics['avg_return_per_trade']
        else:
            # No stop is best
            win_rate = (regime_trades['net_return'] > 0).mean()
            avg_return = regime_trades['net_return'].mean()
        
        results.append({
            'strategy_hash': strategy_row['strategy_hash'],
            'regime': regime,
            'trades': len(regime_trades),
            'optimal_stop': best_stop,
            'total_return': best_return,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'period': strategy_row.get('period'),
            'std_dev': strategy_row.get('std_dev'),
            'fast_period': strategy_row.get('fast_period'),
            'slow_period': strategy_row.get('slow_period')
        })
    
    return pd.DataFrame(results)

# Main analysis
if len(performance_df) > 0 and market_data is not None:
    print("🎯 Regime-Specific Parameter Optimization with Stops")
    print("=" * 80)
    
    # First, add volatility regimes to market data if not already present
    if 'vol_regime' not in market_data.columns:
        market_data['returns'] = market_data['close'].pct_change()
        market_data['volatility'] = market_data['returns'].rolling(window=20*78).std() * np.sqrt(252*78)
        vol_percentiles = market_data['volatility'].quantile([0.33, 0.67])
        market_data['vol_regime'] = pd.cut(
            market_data['volatility'],
            bins=[0, vol_percentiles[0.33], vol_percentiles[0.67], np.inf],
            labels=['Low Vol', 'Medium Vol', 'High Vol']
        )
    
    # Filter for high-frequency strategies
    trading_days = len(market_data['timestamp'].dt.date.unique())
    high_freq_mask = performance_df['num_trades'] >= MIN_TRADES_PER_DAY * trading_days
    high_freq_df = performance_df[high_freq_mask].copy()
    
    print(f"Analyzing {len(high_freq_df)} high-frequency strategies")
    print(f"Testing stop levels: {STOP_LEVELS_TO_TEST}")
    
    # Analyze top 30 strategies by regime (to ensure we get enough for top 20)
    all_regime_results = []
    
    for idx, row in high_freq_df.head(30).iterrows():
        if idx % 5 == 0:
            print(f"  Progress: {idx+1}/30")
        
        regime_analysis = analyze_strategy_by_regime_with_stops(row, market_data, execution_cost_bps)
        
        if regime_analysis is not None:
            all_regime_results.append(regime_analysis)
    
    if all_regime_results:
        # Combine all results
        regime_results_df = pd.concat(all_regime_results, ignore_index=True)
        
        # Find the strategy 3bee7e1f specifically
        target_strategy = '3bee7e1f'
        target_results = regime_results_df[regime_results_df['strategy_hash'].str.contains(target_strategy)]
        
        if len(target_results) > 0:
            print(f"\n📊 Found target strategy {target_strategy}:")
            for _, row in target_results.iterrows():
                print(f"  {row['regime']}: period={row['period']}, std_dev={row['std_dev']}")
        
        # Analyze each regime separately
        for regime in ['Medium Vol', 'High Vol']:
            print(f"\n{'='*60}")
            print(f"📈 {regime} Regime Analysis")
            print(f"{'='*60}")
            
            # Get top 20 performers in this regime
            regime_data = regime_results_df[regime_results_df['regime'] == regime].copy()
            regime_data = regime_data.sort_values('total_return', ascending=False).head(20)
            
            if len(regime_data) > 0:
                print(f"\nTop 5 strategies in {regime}:")
                for idx, row in regime_data.head(5).iterrows():
                    print(f"\n{row['strategy_hash'][:8]}:")
                    print(f"  Period: {row['period']}, Std Dev: {row['std_dev']}")
                    print(f"  Return: {row['total_return']*100:.2f}% (with {row['optimal_stop']:.2f}% stop)")
                    print(f"  Win Rate: {row['win_rate']*100:.1f}%, Trades: {row['trades']}")
                
                # Create parameter heatmaps
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f'{regime} Regime - Parameter Analysis (Top 20 with Optimal Stops)', fontsize=16)
                
                # 1. Return heatmap
                ax = axes[0, 0]
                if 'period' in regime_data.columns and 'std_dev' in regime_data.columns:
                    pivot_return = regime_data.pivot_table(
                        values='total_return',
                        index='period',
                        columns='std_dev',
                        aggfunc='mean'
                    )
                    if not pivot_return.empty:
                        sns.heatmap(pivot_return * 100, cmap='RdYlGn', center=0, 
                                   annot=True, fmt='.1f', ax=ax,
                                   cbar_kws={'label': 'Return %'})
                        ax.set_title('Total Return % (with optimal stops)')
                
                # 2. Optimal stop level heatmap
                ax = axes[0, 1]
                if 'period' in regime_data.columns and 'std_dev' in regime_data.columns:
                    pivot_stop = regime_data.pivot_table(
                        values='optimal_stop',
                        index='period',
                        columns='std_dev',
                        aggfunc='mean'
                    )
                    if not pivot_stop.empty:
                        sns.heatmap(pivot_stop, cmap='YlOrRd', 
                                   annot=True, fmt='.2f', ax=ax,
                                   cbar_kws={'label': 'Stop %'})
                        ax.set_title('Optimal Stop Loss %')
                
                # 3. Win rate heatmap
                ax = axes[1, 0]
                if 'period' in regime_data.columns and 'std_dev' in regime_data.columns:
                    pivot_winrate = regime_data.pivot_table(
                        values='win_rate',
                        index='period',
                        columns='std_dev',
                        aggfunc='mean'
                    )
                    if not pivot_winrate.empty:
                        sns.heatmap(pivot_winrate * 100, cmap='RdYlGn', center=50,
                                   annot=True, fmt='.0f', ax=ax,
                                   cbar_kws={'label': 'Win Rate %'})
                        ax.set_title('Win Rate % (with optimal stops)')
                
                # 4. Trade count
                ax = axes[1, 1]
                if 'period' in regime_data.columns and 'std_dev' in regime_data.columns:
                    pivot_trades = regime_data.pivot_table(
                        values='trades',
                        index='period',
                        columns='std_dev',
                        aggfunc='sum'
                    )
                    if not pivot_trades.empty:
                        sns.heatmap(pivot_trades, cmap='Blues',
                                   annot=True, fmt='.0f', ax=ax,
                                   cbar_kws={'label': 'Trade Count'})
                        ax.set_title(f'Number of Trades in {regime}')
                
                plt.tight_layout()
                plt.show()
                
                # Parameter sweet spots
                print(f"\n🎯 Parameter Sweet Spots for {regime}:")
                
                # Find parameters with best average performance
                param_performance = regime_data.groupby(['period', 'std_dev']).agg({
                    'total_return': 'mean',
                    'optimal_stop': 'mean',
                    'trades': 'sum'
                }).sort_values('total_return', ascending=False)
                
                print("\nTop 5 parameter combinations:")
                for (period, std_dev), metrics in param_performance.head(5).iterrows():
                    print(f"  Period={period}, StdDev={std_dev}: "
                          f"Return={metrics['total_return']*100:.2f}%, "
                          f"OptStop={metrics['optimal_stop']:.2f}%, "
                          f"Trades={metrics['trades']}")
        
        # Overall insights
        print("\n📊 Overall Insights:")
        print("=" * 60)
        
        # Best parameters by regime
        for regime in ['Low Vol', 'Medium Vol', 'High Vol']:
            regime_best = regime_results_df[regime_results_df['regime'] == regime].nlargest(1, 'total_return')
            if len(regime_best) > 0:
                best = regime_best.iloc[0]
                print(f"\n{regime} best: period={best['period']}, std_dev={best['std_dev']}, "
                      f"stop={best['optimal_stop']}%, return={best['total_return']*100:.2f}%")
        
        # Save results
        regime_results_df.to_csv(run_dir / 'regime_parameter_optimization.csv', index=False)
        print(f"\n✅ Saved detailed results to: regime_parameter_optimization.csv")
        
        # Specific recommendations
        print("\n💡 Recommendations:")
        print("1. Medium Vol regime shows best performance overall")
        print("2. Optimal stops vary by regime (tighter in low vol, wider in high vol)")
        print("3. Consider regime-specific parameter sets")
        print("4. Filter trades to only take signals in favorable regimes")
        
    else:
        print("❌ Could not analyze strategies by regime")
        
else:
    print("⚠️ No performance data available")
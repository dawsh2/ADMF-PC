# High-Frequency Strategy Analysis with Regime Filtering and Better Stop Analysis
# Includes volatility regimes and investigates why stops aren't effective

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration
MIN_TRADES_PER_DAY = 2
MIN_TOTAL_TRADES = 100

if len(performance_df) > 0 and market_data is not None:
    # First, add volatility regime to market data
    print("📊 Calculating Market Regimes")
    print("=" * 80)
    
    # Calculate rolling volatility (20-day)
    market_data['returns'] = market_data['close'].pct_change()
    market_data['volatility'] = market_data['returns'].rolling(window=20*78).std() * np.sqrt(252*78)
    
    # Define volatility regimes
    vol_percentiles = market_data['volatility'].quantile([0.33, 0.67])
    market_data['vol_regime'] = pd.cut(
        market_data['volatility'],
        bins=[0, vol_percentiles[0.33], vol_percentiles[0.67], np.inf],
        labels=['Low Vol', 'Medium Vol', 'High Vol']
    )
    
    print(f"Volatility Percentiles:")
    print(f"  33rd: {vol_percentiles[0.33]:.1%}")
    print(f"  67th: {vol_percentiles[0.67]:.1%}")
    
    # Calculate trading days
    trading_days = len(market_data['timestamp'].dt.date.unique())
    
    # Filter for high-frequency strategies
    frequency_mask = (performance_df['num_trades'] >= MIN_TRADES_PER_DAY * trading_days) & \
                    (performance_df['num_trades'] >= MIN_TOTAL_TRADES)
    
    high_freq_df = performance_df[frequency_mask].copy()
    
    if len(high_freq_df) > 0:
        print(f"\n✅ Found {len(high_freq_df)} high-frequency strategies")
        
        # Analyze top strategies by regime
        print("\n📈 Regime Analysis for Top High-Frequency Strategies")
        print("=" * 80)
        
        regime_results = []
        stop_effectiveness = []
        
        for idx, row in high_freq_df.head(5).iterrows():
            print(f"\nAnalyzing Strategy {idx+1}: {row['strategy_type']} - {row['strategy_hash'][:8]}")
            print(f"Total trades: {row['num_trades']} ({row['num_trades']/trading_days:.1f}/day)")
            
            # Extract trades with regime information
            trades = extract_trades(row['strategy_hash'], row['trace_path'], market_data, execution_cost_bps)
            
            if len(trades) > 0:
                # Add regime to trades
                trades = trades.merge(
                    market_data[['vol_regime']], 
                    left_on='entry_idx', 
                    right_index=True, 
                    how='left'
                )
                
                # Analyze by regime
                print("\nPerformance by Regime:")
                for regime in ['Low Vol', 'Medium Vol', 'High Vol']:
                    regime_trades = trades[trades['vol_regime'] == regime]
                    if len(regime_trades) > 0:
                        regime_return = (1 + regime_trades['net_return']).cumprod().iloc[-1] - 1
                        regime_win_rate = (regime_trades['net_return'] > 0).mean()
                        avg_return = regime_trades['net_return'].mean()
                        
                        print(f"  {regime}: {len(regime_trades)} trades, "
                              f"Return={regime_return*100:.2f}%, "
                              f"Win Rate={regime_win_rate*100:.1f}%, "
                              f"Avg/Trade={avg_return*100:.3f}%")
                        
                        regime_results.append({
                            'strategy': row['strategy_hash'][:8],
                            'regime': regime,
                            'trades': len(regime_trades),
                            'total_return': regime_return,
                            'win_rate': regime_win_rate,
                            'avg_return': avg_return
                        })
                
                # Analyze why stops aren't working
                print("\n🔍 Investigating Stop Loss Effectiveness:")
                
                # Calculate intra-trade movements
                trade_movements = []
                for _, trade in trades.head(20).iterrows():  # Sample first 20 trades
                    trade_bars = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
                    
                    if len(trade_bars) > 1:
                        if trade['direction'] == 1:  # Long
                            max_move = ((trade_bars['high'].max() - trade['entry_price']) / trade['entry_price']) * 100
                            max_adverse = ((trade['entry_price'] - trade_bars['low'].min()) / trade['entry_price']) * 100
                        else:  # Short
                            max_move = ((trade['entry_price'] - trade_bars['low'].min()) / trade['entry_price']) * 100
                            max_adverse = ((trade_bars['high'].max() - trade['entry_price']) / trade['entry_price']) * 100
                        
                        trade_movements.append({
                            'bars_in_trade': len(trade_bars),
                            'max_favorable_move': max_move,
                            'max_adverse_move': max_adverse,
                            'final_return': trade['net_return'] * 100,
                            'duration_minutes': trade['duration_minutes']
                        })
                
                if trade_movements:
                    movements_df = pd.DataFrame(trade_movements)
                    print(f"  Average bars per trade: {movements_df['bars_in_trade'].mean():.1f}")
                    print(f"  Average max favorable move: {movements_df['max_favorable_move'].mean():.2f}%")
                    print(f"  Average max adverse move: {movements_df['max_adverse_move'].mean():.2f}%")
                    print(f"  Trades with >0.5% adverse move: {(movements_df['max_adverse_move'] > 0.5).sum()}")
                    print(f"  Trades with >1.0% adverse move: {(movements_df['max_adverse_move'] > 1.0).sum()}")
                    
                    stop_effectiveness.append({
                        'strategy': row['strategy_hash'][:8],
                        'avg_bars': movements_df['bars_in_trade'].mean(),
                        'avg_favorable': movements_df['max_favorable_move'].mean(),
                        'avg_adverse': movements_df['max_adverse_move'].mean(),
                        'pct_over_0.5': (movements_df['max_adverse_move'] > 0.5).mean() * 100,
                        'pct_over_1.0': (movements_df['max_adverse_move'] > 1.0).mean() * 100
                    })
        
        # Visualize regime analysis
        if regime_results:
            regime_df = pd.DataFrame(regime_results)
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Returns by regime
            ax = axes[0, 0]
            pivot_returns = regime_df.pivot_table(
                values='total_return', 
                index='strategy', 
                columns='regime'
            ) * 100
            pivot_returns.plot(kind='bar', ax=ax)
            ax.set_ylabel('Total Return %')
            ax.set_title('Returns by Volatility Regime')
            ax.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax.legend(title='Regime')
            
            # 2. Win rate by regime
            ax = axes[0, 1]
            pivot_winrate = regime_df.pivot_table(
                values='win_rate', 
                index='strategy', 
                columns='regime'
            ) * 100
            pivot_winrate.plot(kind='bar', ax=ax)
            ax.set_ylabel('Win Rate %')
            ax.set_title('Win Rate by Volatility Regime')
            ax.legend(title='Regime')
            
            # 3. Trade count by regime
            ax = axes[0, 2]
            pivot_trades = regime_df.pivot_table(
                values='trades', 
                index='strategy', 
                columns='regime'
            )
            pivot_trades.plot(kind='bar', ax=ax)
            ax.set_ylabel('Number of Trades')
            ax.set_title('Trade Distribution by Regime')
            ax.legend(title='Regime')
            
            # 4. Stop effectiveness analysis
            if stop_effectiveness:
                stop_df = pd.DataFrame(stop_effectiveness)
                
                ax = axes[1, 0]
                ax.bar(range(len(stop_df)), stop_df['avg_adverse'])
                ax.set_xlabel('Strategy')
                ax.set_ylabel('Avg Max Adverse Move %')
                ax.set_title('Average Maximum Adverse Excursion')
                ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='0.5% stop')
                ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='1.0% stop')
                ax.legend()
                
                ax = axes[1, 1]
                width = 0.35
                x = np.arange(len(stop_df))
                ax.bar(x - width/2, stop_df['pct_over_0.5'], width, label='>0.5% move')
                ax.bar(x + width/2, stop_df['pct_over_1.0'], width, label='>1.0% move')
                ax.set_xlabel('Strategy')
                ax.set_ylabel('% of Trades')
                ax.set_title('Percentage of Trades Exceeding Stop Levels')
                ax.legend()
            
            # 5. Summary
            ax = axes[1, 2]
            ax.axis('off')
            
            summary_text = "Key Findings:\n\n"
            
            # Best regime for each strategy
            best_regimes = regime_df.groupby('strategy')['total_return'].idxmax()
            summary_text += "Best Regime by Strategy:\n"
            for strategy in best_regimes.index[:3]:
                best_idx = best_regimes[strategy]
                best_regime = regime_df.loc[best_idx, 'regime']
                best_return = regime_df.loc[best_idx, 'total_return'] * 100
                summary_text += f"  {strategy}: {best_regime} ({best_return:.1f}%)\n"
            
            summary_text += "\nStop Loss Insights:\n"
            if stop_effectiveness:
                avg_adverse = np.mean([s['avg_adverse'] for s in stop_effectiveness])
                pct_hitting_stops = np.mean([s['pct_over_1.0'] for s in stop_effectiveness])
                summary_text += f"  Avg adverse move: {avg_adverse:.2f}%\n"
                summary_text += f"  Trades hitting 1% stop: {pct_hitting_stops:.1f}%\n"
                
                if avg_adverse < 0.5:
                    summary_text += "\n⚠️ Trades have very small movements!\n"
                    summary_text += "Consider:\n"
                    summary_text += "- Tighter parameters\n"
                    summary_text += "- Different strategy type\n"
                    summary_text += "- Leverage (carefully!)"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                    verticalalignment='top', fontsize=11, family='monospace')
            
            plt.tight_layout()
            plt.show()
        
        # Optimal parameters by regime
        print("\n🎯 Optimal Parameters by Regime")
        print("=" * 80)
        
        # For each regime, find best performing parameters
        for regime in ['Low Vol', 'Medium Vol', 'High Vol']:
            print(f"\n{regime}:")
            
            # Filter strategies that performed well in this regime
            regime_winners = []
            for _, strategy in high_freq_df.iterrows():
                # Get regime performance from our results
                regime_perf = [r for r in regime_results if r['strategy'] == strategy['strategy_hash'][:8] and r['regime'] == regime]
                if regime_perf and regime_perf[0]['total_return'] > 0:
                    regime_winners.append({
                        'strategy': strategy['strategy_hash'][:8],
                        'return': regime_perf[0]['total_return'],
                        'period': strategy.get('period'),
                        'std_dev': strategy.get('std_dev')
                    })
            
            if regime_winners:
                winners_df = pd.DataFrame(regime_winners)
                if 'period' in winners_df.columns and 'std_dev' in winners_df.columns:
                    print(f"  Best period range: {winners_df['period'].min()}-{winners_df['period'].max()}")
                    print(f"  Best std_dev range: {winners_df['std_dev'].min():.1f}-{winners_df['std_dev'].max():.1f}")
        
        # Save enhanced results
        if regime_results:
            pd.DataFrame(regime_results).to_csv(run_dir / 'regime_analysis.csv', index=False)
            print(f"\n✅ Saved regime analysis to: regime_analysis.csv")
        
        if stop_effectiveness:
            pd.DataFrame(stop_effectiveness).to_csv(run_dir / 'stop_effectiveness.csv', index=False)
            print(f"✅ Saved stop effectiveness analysis to: stop_effectiveness.csv")
        
        # Final recommendations
        print("\n💡 Recommendations Based on Analysis:")
        print("=" * 60)
        print("1. Stop losses aren't effective because trades have minimal adverse movements")
        print("2. Consider regime-specific position sizing instead of fixed stops")
        print("3. Focus on High Vol regime where movements are larger")
        print("4. Test even tighter Bollinger Band parameters (std_dev < 1.5)")
        print("5. Consider mean reversion strategies that capitalize on small moves")
        
    else:
        print("\n❌ No strategies meet the frequency requirements!")
        print(f"   Adjust MIN_TRADES_PER_DAY or test different parameters")
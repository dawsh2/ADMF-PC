# Diagnose Why Trades Have Small Movements
# Analyzes trade characteristics to understand limited stop effectiveness

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_trade_characteristics(strategy_hash, trace_path, market_data, execution_cost_bps=1.0):
    """Deep dive into why trades have small movements"""
    
    # Extract trades
    trades = extract_trades(strategy_hash, trace_path, market_data, execution_cost_bps)
    
    if len(trades) == 0:
        return None
    
    # Analyze each trade in detail
    detailed_trades = []
    
    for idx, trade in trades.iterrows():
        trade_bars = market_data.iloc[int(trade['entry_idx']):int(trade['exit_idx'])+1]
        
        if len(trade_bars) < 2:
            continue
        
        entry_price = trade['entry_price']
        
        # Calculate various metrics
        if trade['direction'] == 1:  # Long
            # Price movements
            high_pct = ((trade_bars['high'].max() - entry_price) / entry_price) * 100
            low_pct = ((entry_price - trade_bars['low'].min()) / entry_price) * 100
            
            # When did max/min occur?
            high_bar = (trade_bars['high'] == trade_bars['high'].max()).idxmax()
            low_bar = (trade_bars['low'] == trade_bars['low'].min()).idxmax()
            
            # Path efficiency
            total_movement = trade_bars['high'].diff().abs().sum() + trade_bars['low'].diff().abs().sum()
            direct_movement = abs(trade['exit_price'] - entry_price)
            path_efficiency = direct_movement / total_movement if total_movement > 0 else 0
            
        else:  # Short
            high_pct = ((trade_bars['high'].max() - entry_price) / entry_price) * 100
            low_pct = ((entry_price - trade_bars['low'].min()) / entry_price) * 100
            
            high_bar = (trade_bars['high'] == trade_bars['high'].max()).idxmax()
            low_bar = (trade_bars['low'] == trade_bars['low'].min()).idxmax()
            
            total_movement = trade_bars['high'].diff().abs().sum() + trade_bars['low'].diff().abs().sum()
            direct_movement = abs(entry_price - trade['exit_price'])
            path_efficiency = direct_movement / total_movement if total_movement > 0 else 0
        
        # Bar-by-bar volatility during trade
        trade_volatility = trade_bars['close'].pct_change().std() * 100
        
        # Range of each bar
        bar_ranges = ((trade_bars['high'] - trade_bars['low']) / trade_bars['close'] * 100).mean()
        
        detailed_trades.append({
            'trade_idx': idx,
            'direction': 'Long' if trade['direction'] == 1 else 'Short',
            'duration_bars': len(trade_bars),
            'duration_minutes': trade['duration_minutes'],
            'max_favorable_pct': low_pct if trade['direction'] == -1 else high_pct,
            'max_adverse_pct': high_pct if trade['direction'] == -1 else low_pct,
            'final_return_pct': trade['net_return'] * 100,
            'path_efficiency': path_efficiency,
            'trade_volatility': trade_volatility,
            'avg_bar_range': bar_ranges,
            'favorable_bar_position': (high_bar - trade['entry_idx']) / len(trade_bars) if trade['direction'] == 1 else (low_bar - trade['entry_idx']) / len(trade_bars),
            'adverse_bar_position': (low_bar - trade['entry_idx']) / len(trade_bars) if trade['direction'] == 1 else (high_bar - trade['entry_idx']) / len(trade_bars)
        })
    
    return pd.DataFrame(detailed_trades)

# Main diagnostic analysis
if len(top_overall) > 0:
    print("🔬 Diagnosing Small Trade Movements")
    print("=" * 80)
    
    # Focus on most active strategy
    most_active = performance_df.nlargest(1, 'num_trades').iloc[0]
    
    print(f"Analyzing: {most_active['strategy_type']} - {most_active['strategy_hash'][:8]}")
    print(f"Total trades: {most_active['num_trades']}")
    print(f"Parameters: period={most_active.get('period')}, std_dev={most_active.get('std_dev')}")
    
    # Get detailed analysis
    detailed_analysis = analyze_trade_characteristics(
        most_active['strategy_hash'],
        most_active['trace_path'],
        market_data,
        execution_cost_bps
    )
    
    if detailed_analysis is not None and len(detailed_analysis) > 0:
        print(f"\n📊 Trade Characteristics (first 50 trades):")
        print("=" * 60)
        
        analysis_sample = detailed_analysis.head(50)
        
        print(f"Average duration: {analysis_sample['duration_bars'].mean():.1f} bars ({analysis_sample['duration_minutes'].mean():.1f} minutes)")
        print(f"Average max favorable move: {analysis_sample['max_favorable_pct'].mean():.3f}%")
        print(f"Average max adverse move: {analysis_sample['max_adverse_pct'].mean():.3f}%")
        print(f"Average final return: {analysis_sample['final_return_pct'].mean():.3f}%")
        print(f"Average bar range: {analysis_sample['avg_bar_range'].mean():.3f}%")
        print(f"Average trade volatility: {analysis_sample['trade_volatility'].mean():.3f}%")
        
        # Key insights
        print("\n🔍 Key Insights:")
        
        # 1. How many trades would hit various stop levels?
        print("\nStop Loss Hit Rates:")
        for stop_level in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
            hit_rate = (analysis_sample['max_adverse_pct'] > stop_level).mean() * 100
            print(f"  {stop_level:>4.2f}% stop: {hit_rate:>5.1f}% of trades")
        
        # 2. When do extremes occur?
        print(f"\nTiming of Price Extremes:")
        print(f"  Favorable extreme occurs at: {analysis_sample['favorable_bar_position'].mean()*100:.0f}% through trade")
        print(f"  Adverse extreme occurs at: {analysis_sample['adverse_bar_position'].mean()*100:.0f}% through trade")
        
        # 3. Path efficiency
        print(f"\nPath Efficiency: {analysis_sample['path_efficiency'].mean():.1%}")
        print("  (Low efficiency = price meandering, not trending)")
        
        # Visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribution of max adverse moves
        ax = axes[0, 0]
        ax.hist(analysis_sample['max_adverse_pct'], bins=30, alpha=0.7, color='red')
        ax.axvline(0.5, color='black', linestyle='--', label='0.5% stop')
        ax.axvline(1.0, color='black', linestyle='--', label='1.0% stop')
        ax.set_xlabel('Max Adverse Move %')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Maximum Adverse Excursions')
        ax.legend()
        
        # 2. Favorable vs Adverse moves
        ax = axes[0, 1]
        ax.scatter(analysis_sample['max_adverse_pct'], 
                  analysis_sample['max_favorable_pct'],
                  c=analysis_sample['final_return_pct'],
                  cmap='RdYlGn', alpha=0.6)
        ax.plot([0, 5], [0, 5], 'k--', alpha=0.3)  # 1:1 line
        ax.set_xlabel('Max Adverse Move %')
        ax.set_ylabel('Max Favorable Move %')
        ax.set_title('Risk vs Reward (color = final return)')
        plt.colorbar(ax.collections[0], ax=ax, label='Final Return %')
        
        # 3. Trade duration vs movement
        ax = axes[0, 2]
        ax.scatter(analysis_sample['duration_bars'], 
                  analysis_sample['max_adverse_pct'] + analysis_sample['max_favorable_pct'],
                  alpha=0.6)
        ax.set_xlabel('Trade Duration (bars)')
        ax.set_ylabel('Total Movement %')
        ax.set_title('Movement vs Duration')
        
        # 4. Trade volatility distribution
        ax = axes[1, 0]
        ax.hist(analysis_sample['trade_volatility'], bins=20, alpha=0.7)
        ax.set_xlabel('Intra-trade Volatility %')
        ax.set_ylabel('Count')
        ax.set_title('Volatility During Trades')
        
        # 5. Path efficiency
        ax = axes[1, 1]
        ax.hist(analysis_sample['path_efficiency'], bins=20, alpha=0.7, color='orange')
        ax.set_xlabel('Path Efficiency')
        ax.set_ylabel('Count')
        ax.set_title('How Direct Are Price Movements?')
        ax.axvline(0.5, color='red', linestyle='--', label='50% efficient')
        ax.legend()
        
        # 6. Summary
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = "Diagnosis Summary:\n\n"
        
        if analysis_sample['max_adverse_pct'].mean() < 0.5:
            summary_text += "✓ Very tight trades with minimal drawdown\n"
            summary_text += "  → Stops are ineffective\n\n"
        
        if analysis_sample['duration_bars'].mean() < 10:
            summary_text += "✓ Short duration trades\n"
            summary_text += "  → Limited price movement\n\n"
        
        if analysis_sample['path_efficiency'].mean() < 0.3:
            summary_text += "✓ Low path efficiency\n"
            summary_text += "  → Price meandering, not trending\n\n"
        
        summary_text += "Recommendations:\n"
        summary_text += "1. Use tighter BB (std_dev ≤ 1.5)\n"
        summary_text += "2. Consider profit targets\n"
        summary_text += "3. Trade volatility breakouts\n"
        summary_text += "4. Use time-based exits"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=12, family='monospace')
        
        plt.tight_layout()
        plt.show()
        
        # Compare with market volatility
        print("\n📈 Market Context:")
        market_returns = market_data['close'].pct_change() * 100
        print(f"Market avg bar movement: {market_returns.std():.3f}%")
        print(f"Market avg bar range: {((market_data['high'] - market_data['low']) / market_data['close'] * 100).mean():.3f}%")
        print(f"Trade movement vs market: {analysis_sample['avg_bar_range'].mean() / (((market_data['high'] - market_data['low']) / market_data['close'] * 100).mean()):.1f}x")
        
        # Save detailed analysis
        detailed_analysis.to_csv(run_dir / 'trade_movement_analysis.csv', index=False)
        print(f"\n✅ Saved detailed analysis to: trade_movement_analysis.csv")
        
    else:
        print("❌ Could not analyze trades")
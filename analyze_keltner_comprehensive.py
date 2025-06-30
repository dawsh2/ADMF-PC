#!/usr/bin/env python3
"""Comprehensive analysis of Keltner bands strategy results"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the results
results_dir = Path("config/keltner/results/20250622_180858")
metadata_path = results_dir / "metadata.json"
traces_dir = results_dir / "traces" / "keltner_bands"

# Load metadata
logger.info("Loading metadata...")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Extract strategy information
strategies = []
for comp_name, comp_data in metadata['components'].items():
    if comp_data['component_type'] == 'strategy':
        strategies.append({
            'name': comp_name,
            'signal_changes': comp_data['signal_changes'],
            'total_bars': comp_data['total_bars'],
            'signal_frequency': comp_data['signal_frequency'],
            'file_path': traces_dir / Path(comp_data['signal_file_path']).name
        })

logger.info(f"Found {len(strategies)} strategies to analyze")

# Sort strategies by signal frequency for better sampling
strategies_sorted = sorted(strategies, key=lambda x: x['signal_frequency'])

# Sample strategies: take every 25th strategy to get ~110 samples across the range
sample_indices = list(range(0, len(strategies), 25))
sampled_strategies = [strategies_sorted[i] for i in sample_indices if i < len(strategies)]

logger.info(f"Analyzing {len(sampled_strategies)} sampled strategies...")

# Analyze signal patterns
results = []

for i, strategy in enumerate(sampled_strategies):
    if i % 10 == 0:
        logger.info(f"Processing strategy {i}/{len(sampled_strategies)}")
    
    try:
        # Load the trace file
        df = pd.read_parquet(strategy['file_path'])
        
        # Basic metrics
        total_changes = len(df)
        
        # Signal distribution (val column contains signal values)
        long_signals = (df['val'] > 0).sum()
        short_signals = (df['val'] < 0).sum()
        neutral_signals = (df['val'] == 0).sum()
        
        # Calculate actual signal counts (need to expand sparse representation)
        # Each row represents a signal change that persists until the next change
        total_bars = strategy['total_bars']
        
        # Trade analysis
        signal_values = df['val'].values
        trade_count = 0
        trade_durations = []
        
        # Count trades and durations
        current_trade_start = None
        current_signal = 0
        
        for idx in range(len(df)):
            new_signal = signal_values[idx]
            
            # Check if we're starting a new trade
            if current_signal == 0 and new_signal != 0:
                current_trade_start = idx
                trade_count += 1
            # Check if we're ending a trade
            elif current_signal != 0 and new_signal == 0:
                if current_trade_start is not None:
                    # Calculate duration in bars
                    if idx < len(df) - 1:
                        duration = df.iloc[idx]['idx'] - df.iloc[current_trade_start]['idx']
                    else:
                        duration = total_bars - df.iloc[current_trade_start]['idx']
                    trade_durations.append(duration)
                current_trade_start = None
            
            current_signal = new_signal
        
        # Average trade duration
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        
        # Win/loss analysis (based on price movements during trades)
        wins = 0
        losses = 0
        returns = []
        
        for i in range(len(df) - 1):
            if df.iloc[i]['val'] != 0:  # In a trade
                entry_price = df.iloc[i]['px']
                exit_price = df.iloc[i + 1]['px']
                
                if df.iloc[i]['val'] > 0:  # Long trade
                    ret = (exit_price - entry_price) / entry_price
                else:  # Short trade
                    ret = (entry_price - exit_price) / entry_price
                
                returns.append(ret)
                if ret > 0:
                    wins += 1
                else:
                    losses += 1
        
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        avg_return = np.mean(returns) if returns else 0
        
        # Extract strategy parameters from name (e.g., "SPY_5m_compiled_strategy_0")
        strategy_id = int(strategy['name'].split('_')[-1])
        
        results.append({
            'strategy': strategy['name'],
            'strategy_id': strategy_id,
            'total_changes': total_changes,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'neutral_signals': neutral_signals,
            'signal_frequency': strategy['signal_frequency'],
            'trade_count': trade_count,
            'avg_trade_duration_bars': avg_trade_duration,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return * 100,  # Convert to percentage
            'total_bars': total_bars
        })
        
    except Exception as e:
        logger.warning(f"Error processing {strategy['name']}: {e}")
        continue

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)

# Create comprehensive report
logger.info("\n" + "="*80)
logger.info("COMPREHENSIVE KELTNER BANDS STRATEGY ANALYSIS")
logger.info("="*80)

logger.info(f"\nDataset Overview:")
logger.info(f"- Total strategies tested: {len(strategies)}")
logger.info(f"- Strategies analyzed: {len(results_df)}")
logger.info(f"- Total bars in dataset: {metadata['total_bars']:,}")
logger.info(f"- Timeframe: 5-minute bars")
logger.info(f"- Total duration: ~{metadata['total_bars'] * 5 / 60 / 6.5:.1f} trading days")

logger.info("\n--- Signal Activity ---")
logger.info(f"Signal frequency range: {results_df['signal_frequency'].min():.1%} to {results_df['signal_frequency'].max():.1%}")
logger.info(f"Average signal frequency: {results_df['signal_frequency'].mean():.1%}")
logger.info(f"Median signal frequency: {results_df['signal_frequency'].median():.1%}")

# Categorize strategies by activity level
low_activity = results_df[results_df['signal_frequency'] < 0.10]
medium_activity = results_df[(results_df['signal_frequency'] >= 0.10) & (results_df['signal_frequency'] < 0.20)]
high_activity = results_df[results_df['signal_frequency'] >= 0.20]

logger.info(f"\nActivity Distribution:")
logger.info(f"- Low activity (<10%): {len(low_activity)} strategies ({len(low_activity)/len(results_df)*100:.1f}%)")
logger.info(f"- Medium activity (10-20%): {len(medium_activity)} strategies ({len(medium_activity)/len(results_df)*100:.1f}%)")
logger.info(f"- High activity (>20%): {len(high_activity)} strategies ({len(high_activity)/len(results_df)*100:.1f}%)")

logger.info("\n--- Trading Metrics ---")
logger.info(f"Average trades per strategy: {results_df['trade_count'].mean():.1f}")
logger.info(f"Average trade duration: {results_df['avg_trade_duration_bars'].mean():.1f} bars ({results_df['avg_trade_duration_bars'].mean() * 5:.0f} minutes)")

# Win rate analysis
winning_strategies = results_df[results_df['win_rate'] > 0.5]
logger.info(f"\nWin Rate Analysis:")
logger.info(f"- Strategies with >50% win rate: {len(winning_strategies)} ({len(winning_strategies)/len(results_df)*100:.1f}%)")
logger.info(f"- Average win rate: {results_df['win_rate'].mean():.1%}")
logger.info(f"- Best win rate: {results_df['win_rate'].max():.1%}")

# Return analysis
profitable_strategies = results_df[results_df['avg_return_per_trade'] > 0]
logger.info(f"\nReturn Analysis:")
logger.info(f"- Profitable strategies: {len(profitable_strategies)} ({len(profitable_strategies)/len(results_df)*100:.1f}%)")
logger.info(f"- Average return per trade: {results_df['avg_return_per_trade'].mean():.3f}%")
logger.info(f"- Best average return: {results_df['avg_return_per_trade'].max():.3f}%")
logger.info(f"- Worst average return: {results_df['avg_return_per_trade'].min():.3f}%")

# Direction bias
logger.info("\n--- Directional Analysis ---")
results_df['long_bias'] = results_df['long_signals'] / (results_df['long_signals'] + results_df['short_signals'])
results_df['long_bias'] = results_df['long_bias'].fillna(0.5)  # Handle division by zero

logger.info(f"Average long bias: {results_df['long_bias'].mean():.1%}")
logger.info(f"Purely long strategies: {len(results_df[results_df['long_bias'] == 1.0])}")
logger.info(f"Purely short strategies: {len(results_df[results_df['long_bias'] == 0.0])}")
logger.info(f"Balanced strategies (40-60% long): {len(results_df[(results_df['long_bias'] >= 0.4) & (results_df['long_bias'] <= 0.6)])}")

# Top performing strategies
logger.info("\n--- Top 10 Strategies by Average Return ---")
top_return = results_df.nlargest(10, 'avg_return_per_trade')
for idx, row in top_return.iterrows():
    logger.info(f"Strategy {row['strategy_id']:4d}: "
                f"Return: {row['avg_return_per_trade']:6.3f}%, "
                f"Win Rate: {row['win_rate']:5.1%}, "
                f"Trades: {row['trade_count']:3d}, "
                f"Frequency: {row['signal_frequency']:5.1%}")

# Most active strategies
logger.info("\n--- Top 10 Most Active Strategies ---")
top_active = results_df.nlargest(10, 'trade_count')
for idx, row in top_active.iterrows():
    logger.info(f"Strategy {row['strategy_id']:4d}: "
                f"Trades: {row['trade_count']:4d}, "
                f"Frequency: {row['signal_frequency']:5.1%}, "
                f"Avg Duration: {row['avg_trade_duration_bars']:5.1f} bars")

# Parameter insights
logger.info("\n--- Parameter Insights ---")
# Map strategy IDs to parameter combinations
# With 5 periods × 5 multipliers = 25 base combinations
# Strategy ID 0-24 = no filter, 25-49 = first filter set, etc.
results_df['base_param_id'] = results_df['strategy_id'] % 25
results_df['filter_id'] = results_df['strategy_id'] // 25

# Best performing parameter combinations
best_by_param = results_df.groupby('base_param_id')['avg_return_per_trade'].mean().sort_values(ascending=False)
logger.info("\nBest base parameter combinations (by average return):")
for param_id in best_by_param.head(5).index:
    period = [10, 15, 20, 30, 50][param_id % 5]
    multiplier = [1.0, 1.5, 2.0, 2.5, 3.0][param_id // 5]
    avg_return = best_by_param[param_id]
    logger.info(f"  Period: {period}, Multiplier: {multiplier} -> Avg Return: {avg_return:.3f}%")

# Filter effectiveness
filter_performance = results_df.groupby('filter_id').agg({
    'avg_return_per_trade': 'mean',
    'win_rate': 'mean',
    'signal_frequency': 'mean',
    'trade_count': 'mean'
}).sort_values('avg_return_per_trade', ascending=False)

logger.info("\nTop 5 filter configurations (by average return):")
for filter_id in filter_performance.head(5).index:
    row = filter_performance.loc[filter_id]
    logger.info(f"  Filter {filter_id}: "
                f"Return: {row['avg_return_per_trade']:.3f}%, "
                f"Win Rate: {row['win_rate']:.1%}, "
                f"Frequency: {row['signal_frequency']:.1%}")

# Save detailed results
output_file = results_dir / "comprehensive_analysis.csv"
results_df.to_csv(output_file, index=False)
logger.info(f"\nDetailed results saved to: {output_file}")

# Create executive summary
summary = f"""
KELTNER BANDS STRATEGY OPTIMIZATION RESULTS
==========================================

Dataset: SPY 5-minute bars
Duration: ~{metadata['total_bars'] * 5 / 60 / 6.5:.1f} trading days
Total Strategies Tested: {len(strategies):,}

KEY FINDINGS:

1. ACTIVITY LEVELS:
   - Most strategies ({len(low_activity)/len(results_df)*100:.0f}%) have low signal frequency (<10%)
   - Average signal frequency: {results_df['signal_frequency'].mean():.1%}
   - Strategies generate {results_df['trade_count'].mean():.0f} trades on average

2. PERFORMANCE METRICS:
   - Average win rate: {results_df['win_rate'].mean():.1%}
   - Average return per trade: {results_df['avg_return_per_trade'].mean():.3f}%
   - {len(profitable_strategies)/len(results_df)*100:.0f}% of strategies are profitable on average

3. OPTIMAL PARAMETERS:
   Best performing base configurations:
"""

for i, param_id in enumerate(best_by_param.head(3).index):
    period = [10, 15, 20, 30, 50][param_id % 5]
    multiplier = [1.0, 1.5, 2.0, 2.5, 3.0][param_id // 5]
    avg_return = best_by_param[param_id]
    summary += f"   {i+1}. Period: {period}, Multiplier: {multiplier} (Avg Return: {avg_return:.3f}%)\n"

summary += f"""
4. TRADING CHARACTERISTICS:
   - Average trade duration: {results_df['avg_trade_duration_bars'].mean():.0f} bars ({results_df['avg_trade_duration_bars'].mean() * 5:.0f} minutes)
   - Directional bias: {results_df['long_bias'].mean():.0%} long vs {100 - results_df['long_bias'].mean():.0%} short
   - Best single strategy return: {results_df['avg_return_per_trade'].max():.3f}% per trade

5. RECOMMENDATIONS:
   - Focus on strategies with 5-15% signal frequency for quality over quantity
   - Best filters appear to be in the {filter_performance.head(3).index.tolist()} range
   - Consider position sizing based on signal strength and market conditions
   - Implement proper risk management with stop losses

Note: These are raw signal results without transaction costs or slippage.
Further backtesting with realistic execution assumptions is recommended.
"""

summary_file = results_dir / "executive_summary.txt"
with open(summary_file, 'w') as f:
    f.write(summary)

logger.info(f"\nExecutive summary saved to: {summary_file}")
logger.info("\nAnalysis complete!")

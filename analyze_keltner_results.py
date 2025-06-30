#!/usr/bin/env python3
"""Analyze Keltner bands strategy results from the latest run"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging

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

# Analyze signal patterns
results = []

for i, strategy in enumerate(strategies[:100]):  # Analyze first 100 for speed
    if i % 20 == 0:
        logger.info(f"Processing strategy {i}/{min(100, len(strategies))}")
    
    try:
        # Load the trace file
        df = pd.read_parquet(strategy['file_path'])
        
        # Basic metrics
        total_signals = len(df)
        
        # Signal distribution
        long_signals = (df['signal'] > 0).sum() if 'signal' in df.columns else 0
        short_signals = (df['signal'] < 0).sum() if 'signal' in df.columns else 0
        neutral_signals = (df['signal'] == 0).sum() if 'signal' in df.columns else 0
        
        # Trade duration (consecutive non-zero signals)
        if 'signal' in df.columns and len(df) > 0:
            signal_changes = df['signal'].diff().fillna(1) != 0
            trade_groups = signal_changes.cumsum()
            trade_durations = df.groupby(trade_groups)['signal'].count()
            avg_trade_duration = trade_durations[df.groupby(trade_groups)['signal'].first() != 0].mean()
        else:
            avg_trade_duration = 0
        
        results.append({
            'strategy': strategy['name'],
            'total_signals': total_signals,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'neutral_signals': neutral_signals,
            'long_pct': long_signals / total_signals * 100 if total_signals > 0 else 0,
            'short_pct': short_signals / total_signals * 100 if total_signals > 0 else 0,
            'signal_frequency': strategy['signal_frequency'],
            'avg_trade_duration_bars': avg_trade_duration if pd.notna(avg_trade_duration) else 0
        })
        
    except Exception as e:
        logger.warning(f"Error processing {strategy['name']}: {e}")
        continue

# Convert to DataFrame for analysis
results_df = pd.DataFrame(results)

# Summary statistics
logger.info("\n" + "="*60)
logger.info("KELTNER BANDS STRATEGY ANALYSIS")
logger.info("="*60)

logger.info(f"\nTotal strategies analyzed: {len(results_df)}")
logger.info(f"Total strategies in run: {len(strategies)}")

logger.info("\n--- Signal Distribution ---")
logger.info(f"Average long signal %: {results_df['long_pct'].mean():.2f}%")
logger.info(f"Average short signal %: {results_df['short_pct'].mean():.2f}%")
logger.info(f"Average neutral %: {(100 - results_df['long_pct'].mean() - results_df['short_pct'].mean()):.2f}%")

logger.info("\n--- Signal Frequency ---")
logger.info(f"Min signal frequency: {results_df['signal_frequency'].min():.4f}")
logger.info(f"Max signal frequency: {results_df['signal_frequency'].max():.4f}")
logger.info(f"Mean signal frequency: {results_df['signal_frequency'].mean():.4f}")
logger.info(f"Median signal frequency: {results_df['signal_frequency'].median():.4f}")

logger.info("\n--- Trade Duration ---")
avg_duration_5min = results_df['avg_trade_duration_bars'].mean() * 5  # Convert to minutes
logger.info(f"Average trade duration: {results_df['avg_trade_duration_bars'].mean():.1f} bars ({avg_duration_5min:.1f} minutes)")
logger.info(f"Min trade duration: {results_df['avg_trade_duration_bars'].min():.1f} bars")
logger.info(f"Max trade duration: {results_df['avg_trade_duration_bars'].max():.1f} bars")

# Find most active strategies
logger.info("\n--- Most Active Strategies (by signal frequency) ---")
top_active = results_df.nlargest(10, 'signal_frequency')
for idx, row in top_active.iterrows():
    logger.info(f"{row['strategy']}: {row['signal_frequency']:.4f} ({row['long_pct']:.1f}% long, {row['short_pct']:.1f}% short)")

# Find most selective strategies (lowest signal frequency)
logger.info("\n--- Most Selective Strategies (by signal frequency) ---")
top_selective = results_df.nsmallest(10, 'signal_frequency')
for idx, row in top_selective.iterrows():
    logger.info(f"{row['strategy']}: {row['signal_frequency']:.4f} ({row['long_pct']:.1f}% long, {row['short_pct']:.1f}% short)")

# Distribution of signal frequencies
logger.info("\n--- Signal Frequency Distribution ---")
freq_bins = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 1.0]
freq_labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%', '30-35%', '35-40%', '40%+']
results_df['freq_bin'] = pd.cut(results_df['signal_frequency'], bins=freq_bins, labels=freq_labels)
freq_dist = results_df['freq_bin'].value_counts().sort_index()
for bin_label, count in freq_dist.items():
    logger.info(f"{bin_label}: {count} strategies ({count/len(results_df)*100:.1f}%)")

# Save detailed results
output_file = results_dir / "analysis_results.csv"
results_df.to_csv(output_file, index=False)
logger.info(f"\nDetailed results saved to: {output_file}")

# Look for patterns in strategy names (which encode parameters)
logger.info("\n--- Parameter Insights ---")
# Strategy names are like "SPY_5m_compiled_strategy_0", "SPY_5m_compiled_strategy_1", etc.
# The number corresponds to the parameter combination index

# Since we have 2750 strategies from 5 periods × 5 multipliers × many filter combinations
# Let's estimate the parameter grid
n_periods = 5
n_multipliers = 5
n_base_combos = n_periods * n_multipliers  # 25 base combinations
n_filter_variations = len(strategies) // n_base_combos
logger.info(f"Estimated parameter combinations:")
logger.info(f"  Base combinations (period × multiplier): {n_base_combos}")
logger.info(f"  Filter variations per base: ~{n_filter_variations}")
logger.info(f"  Total combinations: {len(strategies)}")

# Create a summary report
summary = f"""
KELTNER BANDS STRATEGY ANALYSIS SUMMARY
======================================

Run Details:
- Timestamp: 2025-06-22 18:08:58
- Total strategies tested: {len(strategies)}
- Total bars processed: {metadata['total_bars']:,}
- Total signals generated: {metadata['total_signals']:,}

Key Findings:
1. Signal Frequency: Strategies generate signals {results_df['signal_frequency'].mean():.1%} of the time on average
2. Directional Bias: {results_df['long_pct'].mean():.1f}% long vs {results_df['short_pct'].mean():.1f}% short signals
3. Trade Duration: Average trade lasts {avg_duration_5min:.0f} minutes ({results_df['avg_trade_duration_bars'].mean():.1f} bars)
4. Most strategies ({freq_dist.iloc[0]} out of {len(results_df)}) have signal frequency in the {freq_dist.index[0]} range

Recommendation:
- Focus on strategies with signal frequency between 10-20% for balanced trading
- Consider the most selective strategies (5-10% frequency) for high-conviction setups
- Investigate parameter combinations of top-performing strategies for further optimization
"""

summary_file = results_dir / "analysis_summary.txt"
with open(summary_file, 'w') as f:
    f.write(summary)

logger.info(f"\nSummary report saved to: {summary_file}")
logger.info("\nAnalysis complete!")
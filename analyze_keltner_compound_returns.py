#!/usr/bin/env python3
"""Calculate compound returns for all Keltner strategies"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start time
start_time = datetime.now()

# Path to results
results_dir = Path("config/keltner/results/20250622_180858")
metadata_path = results_dir / "metadata.json"
traces_dir = results_dir / "traces" / "keltner_bands"

logger.info("Loading metadata...")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Get all strategy files
strategy_files = list(traces_dir.glob("*.parquet"))
logger.info(f"Found {len(strategy_files)} strategy files to analyze")

# Process all strategies
results = []
errors = []

for i, file_path in enumerate(strategy_files):
    if i % 100 == 0:
        logger.info(f"Processing strategy {i}/{len(strategy_files)} ({i/len(strategy_files)*100:.1f}%)")
    
    try:
        # Extract strategy ID from filename
        strategy_name = file_path.stem
        strategy_id = int(strategy_name.split('_')[-1])
        
        # Load trace data
        df = pd.read_parquet(file_path)
        
        if len(df) == 0:
            continue
            
        # Analyze trades with compound returns
        completed_trades = 0
        trade_returns = []
        cumulative_return = 1.0  # Start with $1
        
        # Track trade entry/exit
        in_trade = False
        entry_price = None
        entry_idx = None
        trade_direction = 0
        
        for idx in range(len(df)):
            signal = df.iloc[idx]['val']
            price = df.iloc[idx]['px']
            bar_idx = df.iloc[idx]['idx']
            
            # Check for trade entry
            if not in_trade and signal != 0:
                in_trade = True
                entry_price = price
                entry_idx = bar_idx
                trade_direction = signal
            
            # Check for trade exit
            elif in_trade and signal == 0:
                exit_price = price
                
                # Calculate return
                if trade_direction > 0:  # Long
                    ret = (exit_price - entry_price) / entry_price
                else:  # Short
                    ret = (entry_price - exit_price) / entry_price
                
                trade_returns.append(ret)
                cumulative_return *= (1 + ret)  # Compound the return
                completed_trades += 1
                in_trade = False
                entry_price = None
                trade_direction = 0
        
        # Calculate metrics
        if completed_trades > 0:
            avg_return_pct = np.mean(trade_returns) * 100
            total_return_simple_pct = np.sum(trade_returns) * 100
            total_return_compound_pct = (cumulative_return - 1) * 100
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
            
            # Geometric mean return per trade
            geometric_mean_return = cumulative_return ** (1/completed_trades) - 1
        else:
            avg_return_pct = 0
            total_return_simple_pct = 0
            total_return_compound_pct = 0
            win_rate = 0
            geometric_mean_return = 0
        
        # Get component metadata
        comp_name = f"SPY_5m_compiled_strategy_{strategy_id}"
        if comp_name in metadata['components']:
            total_bars = metadata['components'][comp_name]['total_bars']
            signal_frequency = metadata['components'][comp_name]['signal_frequency']
        else:
            total_bars = metadata['total_bars']
            signal_frequency = 0
        
        # Calculate trading days
        bars_per_day = 78  # 6.5 hours * 60 minutes / 5 minutes
        trading_days = total_bars / bars_per_day
        
        # Annualized returns
        TRADING_DAYS_PER_YEAR = 252
        years = trading_days / TRADING_DAYS_PER_YEAR
        
        # Simple annualization (for comparison)
        annualized_simple_pct = (total_return_simple_pct / trading_days) * TRADING_DAYS_PER_YEAR
        
        # Compound annualization: (final_value)^(1/years) - 1
        if years > 0 and cumulative_return > 0:
            annualized_compound_pct = ((cumulative_return ** (1/years)) - 1) * 100
        else:
            annualized_compound_pct = 0
        
        # Alternative: compound based on trades per year
        trades_per_year = (completed_trades / trading_days) * TRADING_DAYS_PER_YEAR
        if trades_per_year > 0 and geometric_mean_return >= -1:
            annualized_by_trade_pct = ((1 + geometric_mean_return) ** trades_per_year - 1) * 100
        else:
            annualized_by_trade_pct = 0
        
        results.append({
            'strategy_id': strategy_id,
            'completed_trades': completed_trades,
            'trading_days': trading_days,
            'trades_per_day': completed_trades / trading_days if trading_days > 0 else 0,
            'avg_return_pct': avg_return_pct,
            'total_return_simple_pct': total_return_simple_pct,
            'total_return_compound_pct': total_return_compound_pct,
            'win_rate': win_rate,
            'signal_frequency': signal_frequency,
            'annualized_simple_pct': annualized_simple_pct,
            'annualized_compound_pct': annualized_compound_pct,
            'annualized_by_trade_pct': annualized_by_trade_pct,
            'geometric_mean_return_pct': geometric_mean_return * 100
        })
        
    except Exception as e:
        errors.append({'file': str(file_path), 'error': str(e)})
        continue

# Convert to DataFrame
logger.info(f"\nProcessed {len(results)} strategies successfully")
logger.info(f"Errors encountered: {len(errors)}")

results_df = pd.DataFrame(results)

# Sort by compound annualized return
results_df = results_df.sort_values('annualized_compound_pct', ascending=False)

# Display results
logger.info("\n" + "="*80)
logger.info("KELTNER STRATEGIES - COMPOUND RETURNS ANALYSIS")
logger.info("="*80)

logger.info(f"\nTotal strategies analyzed: {len(results_df)}")
logger.info(f"Average trading days: {results_df['trading_days'].mean():.1f}")

logger.info("\n=== TOP 20 BY COMPOUND ANNUAL RETURN ===")
top_20 = results_df.head(20)[['strategy_id', 'completed_trades', 'trades_per_day', 
                              'total_return_simple_pct', 'total_return_compound_pct',
                              'annualized_simple_pct', 'annualized_compound_pct', 
                              'win_rate', 'geometric_mean_return_pct']]
print(top_20.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

# Compare simple vs compound for top strategies
logger.info("\n=== SIMPLE VS COMPOUND COMPARISON (Top 10) ===")
for idx, row in results_df.head(10).iterrows():
    logger.info(f"Strategy {int(row['strategy_id']):4d}:")
    logger.info(f"  Total Return: Simple={row['total_return_simple_pct']:.2f}% vs Compound={row['total_return_compound_pct']:.2f}%")
    logger.info(f"  Annual Return: Simple={row['annualized_simple_pct']:.2f}% vs Compound={row['annualized_compound_pct']:.2f}%")
    logger.info(f"  Difference: {row['total_return_compound_pct'] - row['total_return_simple_pct']:.2f}% total, "
                f"{row['annualized_compound_pct'] - row['annualized_simple_pct']:.2f}% annual")

# Summary statistics
logger.info("\n=== SUMMARY STATISTICS ===")
logger.info(f"Average total return (simple): {results_df['total_return_simple_pct'].mean():.2f}%")
logger.info(f"Average total return (compound): {results_df['total_return_compound_pct'].mean():.2f}%")
logger.info(f"Average annual return (simple): {results_df['annualized_simple_pct'].mean():.2f}%")
logger.info(f"Average annual return (compound): {results_df['annualized_compound_pct'].mean():.2f}%")
logger.info(f"Best annual return (compound): {results_df['annualized_compound_pct'].max():.2f}%")
logger.info(f"Worst annual return (compound): {results_df['annualized_compound_pct'].min():.2f}%")

# Compound effect analysis
compound_effect = results_df['total_return_compound_pct'] - results_df['total_return_simple_pct']
logger.info(f"\nCompound effect on total returns:")
logger.info(f"  Average difference: {compound_effect.mean():.3f}%")
logger.info(f"  Strategies with positive compound effect: {len(compound_effect[compound_effect > 0])} ({len(compound_effect[compound_effect > 0])/len(results_df)*100:.1f}%)")
logger.info(f"  Strategies with negative compound effect: {len(compound_effect[compound_effect < 0])} ({len(compound_effect[compound_effect < 0])/len(results_df)*100:.1f}%)")

# Save results
output_path = results_dir / "compound_returns_analysis.csv"
results_df.to_csv(output_path, index=False)
logger.info(f"\nResults saved to: {output_path}")

# Execution time
elapsed = datetime.now() - start_time
logger.info(f"\nTotal execution time: {elapsed.total_seconds():.1f} seconds")
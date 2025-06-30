#!/usr/bin/env python3
"""Analyze all 2,750 Keltner strategies to verify returns calculations"""

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
            
        # Analyze trades
        completed_trades = 0
        trade_returns = []
        
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
                completed_trades += 1
                in_trade = False
                entry_price = None
                trade_direction = 0
        
        # Calculate metrics
        if completed_trades > 0:
            avg_return_pct = np.mean(trade_returns) * 100
            total_return_pct = np.sum(trade_returns) * 100
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
        else:
            avg_return_pct = 0
            total_return_pct = 0
            win_rate = 0
        
        # Get component metadata
        comp_name = f"SPY_5m_compiled_strategy_{strategy_id}"
        if comp_name in metadata['components']:
            total_bars = metadata['components'][comp_name]['total_bars']
            signal_frequency = metadata['components'][comp_name]['signal_frequency']
        else:
            total_bars = metadata['total_bars']
            signal_frequency = 0
        
        # Calculate trading days (assuming 6.5 hours per day, 5 min bars)
        bars_per_day = 78  # 6.5 hours * 60 minutes / 5 minutes
        trading_days = total_bars / bars_per_day
        
        results.append({
            'strategy_id': strategy_id,
            'completed_trades': completed_trades,
            'trading_days': trading_days,
            'trades_per_day': completed_trades / trading_days if trading_days > 0 else 0,
            'avg_return_pct': avg_return_pct,
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'signal_frequency': signal_frequency
        })
        
    except Exception as e:
        errors.append({'file': str(file_path), 'error': str(e)})
        continue

# Convert to DataFrame
logger.info(f"\nProcessed {len(results)} strategies successfully")
logger.info(f"Errors encountered: {len(errors)}")

results_df = pd.DataFrame(results)

# Calculate annualized returns (simple method to match colleague)
TRADING_DAYS_PER_YEAR = 252
results_df['annualized_return_pct'] = (results_df['total_return_pct'] / results_df['trading_days']) * TRADING_DAYS_PER_YEAR

# Sort by annualized return
results_df = results_df.sort_values('annualized_return_pct', ascending=False)

# Display results
logger.info("\n" + "="*80)
logger.info("ALL KELTNER STRATEGIES ANALYSIS - VERIFICATION")
logger.info("="*80)

logger.info(f"\nTotal strategies analyzed: {len(results_df)}")
logger.info(f"Average trading days: {results_df['trading_days'].mean():.1f}")

logger.info("\n=== SIMPLE PERFORMANCE (Top 20) ===")
print(results_df.head(20).to_string(index=False))

# Compare with colleague's top result
logger.info("\n=== COMPARISON WITH COLLEAGUE'S DATA ===")
colleague_top = {
    'strategy_id': 1029,
    'completed_trades': 401,
    'trading_days': 212.0,
    'avg_return_pct': 0.0334,
    'total_return_pct': 14.09,
    'annualized_return_pct': 16.97
}

our_1029 = results_df[results_df['strategy_id'] == 1029]
if not our_1029.empty:
    our_data = our_1029.iloc[0]
    logger.info(f"\nStrategy 1029 comparison:")
    logger.info(f"Colleague: Trades={colleague_top['completed_trades']}, Total Return={colleague_top['total_return_pct']:.2f}%, Annual={colleague_top['annualized_return_pct']:.2f}%")
    logger.info(f"Our calc:  Trades={int(our_data['completed_trades'])}, Total Return={our_data['total_return_pct']:.2f}%, Annual={our_data['annualized_return_pct']:.2f}%")

# Summary statistics
logger.info("\n=== SUMMARY STATISTICS ===")
logger.info(f"Average trades per strategy: {results_df['completed_trades'].mean():.1f}")
logger.info(f"Average trades per day: {results_df['trades_per_day'].mean():.2f}")
logger.info(f"Average total return: {results_df['total_return_pct'].mean():.2f}%")
logger.info(f"Average annualized return: {results_df['annualized_return_pct'].mean():.2f}%")
logger.info(f"Best annualized return: {results_df['annualized_return_pct'].max():.2f}%")
logger.info(f"Worst annualized return: {results_df['annualized_return_pct'].min():.2f}%")

# Distribution of returns
profitable = len(results_df[results_df['annualized_return_pct'] > 0])
logger.info(f"\nProfitable strategies: {profitable} ({profitable/len(results_df)*100:.1f}%)")

# Save results
output_path = results_dir / "all_strategies_verification.csv"
results_df.to_csv(output_path, index=False)
logger.info(f"\nResults saved to: {output_path}")

# Execution time
elapsed = datetime.now() - start_time
logger.info(f"\nTotal execution time: {elapsed.total_seconds():.1f} seconds")
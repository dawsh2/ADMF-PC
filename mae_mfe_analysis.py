#!/usr/bin/env python3
"""
MAE/MFE Analysis for RSI Composite Strategy

Analyzes Maximum Adverse Excursion (worst drawdown) and Maximum Favorable Excursion 
(best profit) during trades to optimize stop losses and profit targets.
"""
import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_mae_mfe(workspace_path: str, data_path: str):
    """Comprehensive MAE/MFE analysis for composite RSI strategy."""
    
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    print("=== MAE/MFE Analysis for RSI Composite Strategy ===\n")
    
    # 1. Get all composite trades with tick-by-tick price movement
    print("1. Building trade-by-trade price series...")
    
    mae_mfe_query = f"""
    WITH composite_trades AS (
        SELECT 
            e.idx as entry_idx,
            MIN(x.idx) as exit_idx,
            MIN(x.idx) - e.idx as holding_period
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet') e
        JOIN read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet') x
            ON x.idx > e.idx 
            AND x.idx <= e.idx + 20
            AND e.val = 1 
            AND x.val = -1
            AND e.strat LIKE '%_7_%'
            AND (x.strat LIKE '%_14_%' OR x.strat LIKE '%_21_%')
        GROUP BY e.idx
    ),
    trade_prices AS (
        SELECT 
            t.entry_idx,
            t.exit_idx,
            t.holding_period,
            m_entry.close as entry_price,
            m_exit.close as exit_price,
            (m_exit.close - m_entry.close) / m_entry.close * 100 as final_return
        FROM composite_trades t
        JOIN read_parquet('{data_path}') m_entry ON t.entry_idx = m_entry.bar_index
        JOIN read_parquet('{data_path}') m_exit ON t.exit_idx = m_exit.bar_index
    ),
    price_series AS (
        SELECT 
            t.entry_idx,
            t.exit_idx,
            t.holding_period,
            t.entry_price,
            t.exit_price,
            t.final_return,
            m.bar_index,
            m.close,
            (m.close - t.entry_price) / t.entry_price * 100 as unrealized_pnl
        FROM trade_prices t
        JOIN read_parquet('{data_path}') m 
            ON m.bar_index >= t.entry_idx 
            AND m.bar_index <= t.exit_idx
    ),
    mae_mfe_calc AS (
        SELECT 
            entry_idx,
            exit_idx,
            holding_period,
            entry_price,
            exit_price,
            final_return,
            MIN(unrealized_pnl) as mae,  -- Maximum Adverse Excursion (worst drawdown)
            MAX(unrealized_pnl) as mfe   -- Maximum Favorable Excursion (best profit)
        FROM price_series
        GROUP BY entry_idx, exit_idx, holding_period, entry_price, exit_price, final_return
    )
    SELECT 
        entry_idx,
        holding_period,
        ROUND(final_return, 4) as final_return_pct,
        ROUND(mae, 4) as mae_pct,
        ROUND(mfe, 4) as mfe_pct,
        ROUND(mfe - mae, 4) as mfe_mae_spread,
        ROUND(ABS(mae) / final_return, 2) as mae_to_return_ratio,
        ROUND(mfe / final_return, 2) as mfe_to_return_ratio
    FROM mae_mfe_calc
    WHERE final_return > 0  -- Focus on winning trades first
    ORDER BY final_return DESC
    """
    
    mae_mfe_df = con.execute(mae_mfe_query).df()
    print(f"Analyzed {len(mae_mfe_df)} winning trades\n")
    
    # 2. Summary statistics
    print("2. MAE/MFE Summary Statistics:")
    
    summary_stats = {
        'Final Return': mae_mfe_df['final_return_pct'].describe(),
        'MAE (Max Drawdown)': mae_mfe_df['mae_pct'].describe(),
        'MFE (Max Profit)': mae_mfe_df['mfe_pct'].describe(),
        'MFE-MAE Spread': mae_mfe_df['mfe_mae_spread'].describe()
    }
    
    for metric, stats in summary_stats.items():
        print(f"\n{metric}:")
        print(f"  Mean: {stats['mean']:.4f}%")
        print(f"  Median: {stats['50%']:.4f}%")
        print(f"  75th percentile: {stats['75%']:.4f}%")
        print(f"  Max: {stats['max']:.4f}%")
        print(f"  Min: {stats['min']:.4f}%")
    
    # 3. Stop loss analysis
    print("\n\n3. Stop Loss Analysis (based on MAE):")
    
    stop_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    for stop_pct in stop_levels:
        trades_stopped = len(mae_mfe_df[mae_mfe_df['mae_pct'] <= -stop_pct])
        pct_stopped = trades_stopped / len(mae_mfe_df) * 100
        
        avg_return_if_stopped = mae_mfe_df[mae_mfe_df['mae_pct'] <= -stop_pct]['final_return_pct'].mean()
        avg_return_not_stopped = mae_mfe_df[mae_mfe_df['mae_pct'] > -stop_pct]['final_return_pct'].mean()
        
        print(f"Stop at -{stop_pct}%:")
        print(f"  Would stop {trades_stopped}/{len(mae_mfe_df)} trades ({pct_stopped:.1f}%)")
        if trades_stopped > 0:
            print(f"  Avg return of stopped trades: {avg_return_if_stopped:.4f}%")
        if trades_stopped < len(mae_mfe_df):
            print(f"  Avg return of remaining trades: {avg_return_not_stopped:.4f}%")
        print()
    
    # 4. Profit target analysis
    print("4. Profit Target Analysis (based on MFE):")
    
    target_levels = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    
    for target_pct in target_levels:
        trades_hit_target = len(mae_mfe_df[mae_mfe_df['mfe_pct'] >= target_pct])
        pct_hit_target = trades_hit_target / len(mae_mfe_df) * 100
        
        # Calculate what return we'd get if we exited at target
        potential_returns = mae_mfe_df[mae_mfe_df['mfe_pct'] >= target_pct]
        if len(potential_returns) > 0:
            # Assume we exit at target, so return would be target_pct
            total_if_targets = len(potential_returns) * target_pct
            total_actual = potential_returns['final_return_pct'].sum()
            improvement = total_if_targets - total_actual
        else:
            improvement = 0
        
        print(f"Profit target at +{target_pct}%:")
        print(f"  {trades_hit_target}/{len(mae_mfe_df)} trades hit target ({pct_hit_target:.1f}%)")
        if trades_hit_target > 0:
            print(f"  Improvement if exited at target: {improvement:.4f}% total")
        print()
    
    # 5. Efficiency analysis
    print("5. Trade Efficiency Analysis:")
    
    mae_mfe_df['efficiency'] = mae_mfe_df['final_return_pct'] / mae_mfe_df['mfe_pct']
    mae_mfe_df['drawdown_risk'] = abs(mae_mfe_df['mae_pct']) / mae_mfe_df['final_return_pct']
    
    efficiency_buckets = pd.cut(mae_mfe_df['efficiency'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    efficiency_analysis = mae_mfe_df.groupby(efficiency_buckets).agg({
        'final_return_pct': ['count', 'mean'],
        'mae_pct': 'mean',
        'mfe_pct': 'mean',
        'holding_period': 'mean'
    }).round(4)
    
    print("Efficiency (Final Return / MFE) Analysis:")
    print(efficiency_analysis)
    
    # 6. Holding period vs MAE/MFE
    print("\n\n6. Holding Period vs MAE/MFE Analysis:")
    
    period_analysis = mae_mfe_df.groupby(pd.cut(mae_mfe_df['holding_period'], bins=4)).agg({
        'final_return_pct': ['count', 'mean'],
        'mae_pct': 'mean',
        'mfe_pct': 'mean',
        'efficiency': 'mean'
    }).round(4)
    
    print("MAE/MFE by Holding Period:")
    print(period_analysis)
    
    # 7. Best and worst trades analysis
    print("\n\n7. Best and Worst Trade Analysis:")
    
    print("Top 5 Most Efficient Trades (Final Return / MFE):")
    top_efficient = mae_mfe_df.nlargest(5, 'efficiency')[['final_return_pct', 'mae_pct', 'mfe_pct', 'efficiency', 'holding_period']]
    print(top_efficient.to_string(index=False))
    
    print("\nTop 5 Worst Drawdown Trades (MAE vs Final Return):")
    worst_drawdown = mae_mfe_df.nlargest(5, 'drawdown_risk')[['final_return_pct', 'mae_pct', 'mfe_pct', 'drawdown_risk', 'holding_period']]
    print(worst_drawdown.to_string(index=False))
    
    # 8. Include losing trades analysis
    print("\n\n8. Losing Trades Analysis:")
    
    losing_trades_query = f"""
    WITH composite_trades AS (
        SELECT 
            e.idx as entry_idx,
            MIN(x.idx) as exit_idx,
            MIN(x.idx) - e.idx as holding_period
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet') e
        JOIN read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet') x
            ON x.idx > e.idx 
            AND x.idx <= e.idx + 20
            AND e.val = 1 
            AND x.val = -1
            AND e.strat LIKE '%_7_%'
            AND (x.strat LIKE '%_14_%' OR x.strat LIKE '%_21_%')
        GROUP BY e.idx
    ),
    trade_prices AS (
        SELECT 
            t.entry_idx,
            t.exit_idx,
            t.holding_period,
            m_entry.close as entry_price,
            m_exit.close as exit_price,
            (m_exit.close - m_entry.close) / m_entry.close * 100 as final_return
        FROM composite_trades t
        JOIN read_parquet('{data_path}') m_entry ON t.entry_idx = m_entry.bar_index
        JOIN read_parquet('{data_path}') m_exit ON t.exit_idx = m_exit.bar_index
        WHERE (m_exit.close - m_entry.close) / m_entry.close * 100 < 0  -- Losing trades only
    ),
    price_series AS (
        SELECT 
            t.entry_idx,
            t.exit_idx,
            t.holding_period,
            t.entry_price,
            t.exit_price,
            t.final_return,
            m.bar_index,
            m.close,
            (m.close - t.entry_price) / t.entry_price * 100 as unrealized_pnl
        FROM trade_prices t
        JOIN read_parquet('{data_path}') m 
            ON m.bar_index >= t.entry_idx 
            AND m.bar_index <= t.exit_idx
    ),
    losing_mae_mfe AS (
        SELECT 
            entry_idx,
            holding_period,
            final_return,
            MIN(unrealized_pnl) as mae,
            MAX(unrealized_pnl) as mfe
        FROM price_series
        GROUP BY entry_idx, holding_period, final_return
    )
    SELECT 
        COUNT(*) as losing_trades,
        ROUND(AVG(final_return), 4) as avg_loss,
        ROUND(AVG(mae), 4) as avg_mae,
        ROUND(AVG(mfe), 4) as avg_mfe,
        ROUND(MIN(final_return), 4) as worst_loss,
        ROUND(MIN(mae), 4) as worst_mae,
        ROUND(MAX(mfe), 4) as best_mfe_in_losing_trade
    FROM losing_mae_mfe
    """
    
    losing_analysis = con.execute(losing_trades_query).df()
    if len(losing_analysis) > 0 and losing_analysis.iloc[0]['losing_trades'] > 0:
        print("Losing Trades MAE/MFE:")
        print(losing_analysis.to_string(index=False))
    else:
        print("No losing trades found in dataset!")
    
    con.close()
    
    print("\n\n=== Key Insights ===")
    print("1. MAE shows worst-case drawdown during trades")
    print("2. MFE shows profit potential that could be captured")
    print("3. Efficiency = Final Return / MFE (higher is better)")
    print("4. Use MAE for stop loss placement")
    print("5. Use MFE for profit target optimization")
    print("6. Compare efficiency across holding periods")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python mae_mfe_analysis.py <workspace_path> <data_path>")
        sys.exit(1)
    
    analyze_mae_mfe(sys.argv[1], sys.argv[2])
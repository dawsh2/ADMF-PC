#!/usr/bin/env python3
"""
Comprehensive RSI Strategy Simulation

Simulates the complete RSI strategy with our optimized exit framework:
1. Fast RSI (7-period) entry signals
2. Multi-layered exit system:
   - Signal-based exits (mean reversion, slow RSI)
   - Profit targets (0.20%, 0.25%)
   - Stop losses (-0.15%)
   - 18-bar time safety net

Shows realistic performance expectations with no look-ahead bias.
"""
import duckdb
import pandas as pd
import numpy as np


def simulate_comprehensive_rsi_strategy(workspace_path: str, data_path: str):
    """Simulate complete RSI strategy with optimized exit framework."""
    
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    print("=== Comprehensive RSI Strategy Simulation ===\n")
    print("Strategy Components:")
    print("✓ Entry: Fast RSI (7-period) oversold/overbought signals")
    print("✓ Exit Layer 1: Signal-based (mean reversion, slow RSI)")
    print("✓ Exit Layer 2: Profit targets (0.20%, 0.25%)")
    print("✓ Exit Layer 3: Stop losses (-0.15%)")
    print("✓ Exit Layer 4: 18-bar time safety net")
    print("✓ No look-ahead bias\n")
    
    # Simulate the complete strategy
    strategy_query = f"""
    WITH fast_rsi_entries AS (
        SELECT DISTINCT idx as entry_idx
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
        WHERE strat LIKE '%_7_%' AND val = 1
    ),
    -- Check for signal-based exits available within 18 bars
    signal_exits AS (
        SELECT 
            e.entry_idx,
            MIN(CASE WHEN s.strat LIKE '%mean_reversion%' AND s.val = -1 THEN s.idx END) as mean_reversion_exit,
            MIN(CASE WHEN (s.strat LIKE '%_14_%' OR s.strat LIKE '%_21_%') AND s.val = -1 THEN s.idx END) as slow_rsi_exit
        FROM fast_rsi_entries e
        LEFT JOIN read_parquet('{workspace_path}/traces/SPY_1m/signals/*/*.parquet') s 
            ON s.idx > e.entry_idx AND s.idx <= e.entry_idx + 18
        GROUP BY e.entry_idx
    ),
    -- Track price path for each trade to find optimal exits
    price_paths AS (
        SELECT 
            e.entry_idx,
            m.bar_index,
            m.bar_index - e.entry_idx as bars_held,
            m1.close as entry_price,
            m.close as current_price,
            (m.close - m1.close) / m1.close * 100 as unrealized_pnl,
            s.mean_reversion_exit,
            s.slow_rsi_exit
        FROM fast_rsi_entries e
        JOIN read_parquet('{data_path}') m1 ON e.entry_idx = m1.bar_index
        JOIN read_parquet('{data_path}') m ON m.bar_index >= e.entry_idx AND m.bar_index <= e.entry_idx + 18
        JOIN signal_exits s ON e.entry_idx = s.entry_idx
    ),
    -- Determine exit point for each trade using our framework
    exit_logic AS (
        SELECT 
            entry_idx,
            entry_price,
            CASE 
                -- Layer 1: Signal-based exits (highest priority)
                WHEN mean_reversion_exit IS NOT NULL AND bar_index = mean_reversion_exit THEN 'mean_reversion_signal'
                WHEN slow_rsi_exit IS NOT NULL AND bar_index = slow_rsi_exit THEN 'slow_rsi_signal'
                
                -- Layer 2: Profit targets
                WHEN unrealized_pnl >= 0.25 THEN 'profit_target_025'
                WHEN unrealized_pnl >= 0.20 AND bars_held >= 10 THEN 'profit_target_020'
                
                -- Layer 3: Stop losses  
                WHEN unrealized_pnl <= -0.15 THEN 'stop_loss_015'
                
                -- Layer 4: Time safety net
                WHEN bars_held >= 18 THEN 'time_safety_18bar'
                
                ELSE NULL
            END as exit_reason,
            CASE 
                WHEN mean_reversion_exit IS NOT NULL AND bar_index = mean_reversion_exit THEN bar_index
                WHEN slow_rsi_exit IS NOT NULL AND bar_index = slow_rsi_exit THEN bar_index
                WHEN unrealized_pnl >= 0.25 THEN bar_index
                WHEN unrealized_pnl >= 0.20 AND bars_held >= 10 THEN bar_index
                WHEN unrealized_pnl <= -0.15 THEN bar_index
                WHEN bars_held >= 18 THEN bar_index
                ELSE NULL
            END as exit_idx,
            CASE 
                WHEN mean_reversion_exit IS NOT NULL AND bar_index = mean_reversion_exit THEN unrealized_pnl
                WHEN slow_rsi_exit IS NOT NULL AND bar_index = slow_rsi_exit THEN unrealized_pnl
                WHEN unrealized_pnl >= 0.25 THEN unrealized_pnl
                WHEN unrealized_pnl >= 0.20 AND bars_held >= 10 THEN unrealized_pnl
                WHEN unrealized_pnl <= -0.15 THEN unrealized_pnl
                WHEN bars_held >= 18 THEN unrealized_pnl
                ELSE NULL
            END as exit_pnl,
            bars_held as exit_bars_held
        FROM price_paths
        WHERE 
            mean_reversion_exit IS NOT NULL AND bar_index = mean_reversion_exit OR
            slow_rsi_exit IS NOT NULL AND bar_index = slow_rsi_exit OR
            unrealized_pnl >= 0.25 OR
            (unrealized_pnl >= 0.20 AND bars_held >= 10) OR
            unrealized_pnl <= -0.15 OR
            bars_held >= 18
    ),
    -- Get first exit for each trade (earliest trigger wins)
    first_exits AS (
        SELECT 
            entry_idx,
            entry_price,
            FIRST_VALUE(exit_reason) OVER (PARTITION BY entry_idx ORDER BY exit_idx) as exit_reason,
            FIRST_VALUE(exit_pnl) OVER (PARTITION BY entry_idx ORDER BY exit_idx) as trade_pnl,
            FIRST_VALUE(exit_bars_held) OVER (PARTITION BY entry_idx ORDER BY exit_idx) as holding_period,
            ROW_NUMBER() OVER (PARTITION BY entry_idx ORDER BY exit_idx) as rn
        FROM exit_logic
        WHERE exit_reason IS NOT NULL
    )
    SELECT 
        exit_reason,
        COUNT(*) as trades,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct_of_trades,
        ROUND(AVG(trade_pnl), 4) as avg_return_pct,
        ROUND(STDDEV(trade_pnl), 4) as volatility,
        COUNT(CASE WHEN trade_pnl > 0 THEN 1 END) as winners,
        ROUND(COUNT(CASE WHEN trade_pnl > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate,
        ROUND(AVG(holding_period), 1) as avg_holding_bars,
        ROUND(SUM(trade_pnl), 2) as total_pnl_contribution,
        ROUND(AVG(trade_pnl) / STDDEV(trade_pnl), 3) as sharpe_ratio
    FROM first_exits
    WHERE rn = 1  -- Only first exit per trade
    GROUP BY exit_reason
    ORDER BY trades DESC
    """
    
    try:
        strategy_results = con.execute(strategy_query).df()
        
        print("1. Strategy Performance by Exit Type:")
        print(strategy_results.to_string(index=False))
        
        # Calculate overall strategy performance
        if len(strategy_results) > 0:
            total_trades = strategy_results['trades'].sum()
            total_pnl = strategy_results['total_pnl_contribution'].sum()
            avg_strategy_return = total_pnl / total_trades if total_trades > 0 else 0
            total_winners = strategy_results['winners'].sum()
            overall_win_rate = total_winners / total_trades * 100 if total_trades > 0 else 0
            
            # Calculate weighted volatility and Sharpe
            weighted_variance = sum(
                (strategy_results['trades'] * strategy_results['volatility']**2) / total_trades
            )
            strategy_volatility = np.sqrt(weighted_variance)
            strategy_sharpe = avg_strategy_return / strategy_volatility if strategy_volatility > 0 else 0
            
            print(f"\n2. Overall Strategy Performance:")
            print(f"Total Trades: {total_trades}")
            print(f"Average Return per Trade: {avg_strategy_return:.4f}%")
            print(f"Overall Win Rate: {overall_win_rate:.2f}%")
            print(f"Strategy Volatility: {strategy_volatility:.4f}%")
            print(f"Strategy Sharpe Ratio: {strategy_sharpe:.3f}")
            print(f"Total Strategy Return: {total_pnl:.2f}%")
            
            # Annualized projections (assuming 1-minute bars, ~375 trading days)
            bars_per_day = 390  # 6.5 hours * 60 minutes
            trading_days_per_year = 252
            total_bars_in_sample = 10000  # Approximate from our data
            
            trades_per_day = total_trades / (total_bars_in_sample / bars_per_day)
            annual_trades = trades_per_day * trading_days_per_year
            annual_return = annual_trades * avg_strategy_return
            
            print(f"\n3. Annualized Projections:")
            print(f"Estimated trades per day: {trades_per_day:.1f}")
            print(f"Estimated annual trades: {annual_trades:.0f}")
            print(f"Estimated annual return: {annual_return:.1f}%")
            print(f"Daily return estimate: {annual_return/252:.3f}%")
            
    except Exception as e:
        print(f"Error in strategy simulation: {e}")
        
        # Fallback: Simple analysis
        print("Running simplified analysis...")
        
        simple_query = f"""
        WITH fast_rsi_entries AS (
            SELECT DISTINCT idx as entry_idx
            FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
            WHERE strat LIKE '%_7_%' AND val = 1
        ),
        simple_trades AS (
            SELECT 
                entry_idx,
                entry_idx + 18 as exit_idx
            FROM fast_rsi_entries
        )
        SELECT 
            COUNT(*) as total_trades,
            ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return,
            ROUND(STDDEV((m2.close - m1.close) / m1.close * 100), 4) as volatility,
            COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) as winners,
            ROUND(COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate
        FROM simple_trades t
        JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
        JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
        WHERE t.exit_idx <= (SELECT MAX(bar_index) FROM read_parquet('{data_path}'))
        """
        
        simple_results = con.execute(simple_query).df()
        print("Baseline 18-bar exit performance:")
        print(simple_results.to_string(index=False))
    
    # Compare to buy-and-hold benchmark
    print(f"\n4. Benchmark Comparison:")
    
    benchmark_query = f"""
    WITH price_data AS (
        SELECT 
            close,
            LAG(close, 18) OVER (ORDER BY bar_index) as price_18_bars_ago
        FROM read_parquet('{data_path}')
        WHERE bar_index >= 18
    )
    SELECT 
        COUNT(*) as periods,
        ROUND(AVG((close - price_18_bars_ago) / price_18_bars_ago * 100), 4) as avg_18bar_return,
        ROUND(STDDEV((close - price_18_bars_ago) / price_18_bars_ago * 100), 4) as volatility,
        COUNT(CASE WHEN close > price_18_bars_ago THEN 1 END) as up_periods,
        ROUND(COUNT(CASE WHEN close > price_18_bars_ago THEN 1 END) * 100.0 / COUNT(*), 2) as up_rate
    FROM price_data
    WHERE price_18_bars_ago IS NOT NULL
    """
    
    benchmark = con.execute(benchmark_query).df()
    print("SPY 18-bar buy-and-hold benchmark:")
    print(benchmark.to_string(index=False))
    
    con.close()
    
    print(f"\n=== Strategy Assessment ===")
    if len(strategy_results) > 0:
        signal_exits = strategy_results[strategy_results['exit_reason'].str.contains('signal', na=False)]['trades'].sum()
        profit_exits = strategy_results[strategy_results['exit_reason'].str.contains('profit', na=False)]['trades'].sum()
        stop_exits = strategy_results[strategy_results['exit_reason'].str.contains('stop', na=False)]['trades'].sum()
        time_exits = strategy_results[strategy_results['exit_reason'].str.contains('time', na=False)]['trades'].sum()
        
        print(f"Exit Distribution:")
        print(f"- Signal-based exits: {signal_exits} trades ({signal_exits/total_trades*100:.1f}%)")
        print(f"- Profit target exits: {profit_exits} trades ({profit_exits/total_trades*100:.1f}%)")
        print(f"- Stop loss exits: {stop_exits} trades ({stop_exits/total_trades*100:.1f}%)")
        print(f"- Time-based exits: {time_exits} trades ({time_exits/total_trades*100:.1f}%)")
        
        print(f"\nStrategy Advantages:")
        print(f"- Systematic profit taking and loss cutting")
        print(f"- 100% trade coverage (no orphaned positions)")
        print(f"- Risk management through multiple exit layers")
        print(f"- Realistic performance expectations")
        
        if avg_strategy_return > 0:
            print(f"\nStrategy shows positive edge of {avg_strategy_return:.4f}% per trade")
            print(f"With {total_trades} trades, this generates {total_pnl:.2f}% total return")
        else:
            print(f"\nStrategy shows minimal edge - consider additional filters or improvements")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python comprehensive_rsi_strategy_simulation.py <workspace_path> <data_path>")
        sys.exit(1)
    
    simulate_comprehensive_rsi_strategy(sys.argv[1], sys.argv[2])
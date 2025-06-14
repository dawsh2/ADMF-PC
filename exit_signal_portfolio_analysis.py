#!/usr/bin/env python3
"""
Exit Signal Portfolio Analysis

Treats fast RSI as ENTRY ONLY signal, then analyzes which combination
of exit signals provides the best performance across all entries.

Key insight: Once entered, ignore fast RSI and listen to exit signal portfolio.
"""
import duckdb
import pandas as pd
import numpy as np


def analyze_exit_signal_portfolio(workspace_path: str, data_path: str):
    """Analyze optimal portfolio of exit signals for fast RSI entries."""
    
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    print("=== Exit Signal Portfolio Analysis ===\n")
    print("Entry: Fast RSI (7-period) oversold/overbought signals")
    print("Exit: Portfolio of different signal types\n")
    
    # 1. Available exit signals analysis
    print("1. Available Exit Signals in Dataset:")
    
    exit_signals_query = f"""
    SELECT 
        CASE 
            WHEN strat LIKE '%_14_%' OR strat LIKE '%_21_%' THEN 'Slow RSI'
            WHEN strat LIKE '%ma_crossover%' THEN 'MA Crossover'
            WHEN strat LIKE '%momentum%' THEN 'Momentum'
            WHEN strat LIKE '%mean_reversion%' THEN 'Mean Reversion'
            WHEN strat LIKE '%breakout%' THEN 'Breakout'
            ELSE 'Other'
        END as signal_family,
        COUNT(DISTINCT strat) as strategies,
        COUNT(*) as total_signals,
        COUNT(CASE WHEN val = -1 THEN 1 END) as exit_signals,
        COUNT(CASE WHEN val = 1 THEN 1 END) as entry_signals
    FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/*/*.parquet')
    GROUP BY signal_family
    ORDER BY exit_signals DESC
    """
    
    available_signals = con.execute(exit_signals_query).df()
    print(available_signals.to_string(index=False))
    
    # 2. Exit signal frequency and timing analysis
    print("\n\n2. Exit Signal Frequency Analysis:")
    
    exit_frequency_query = f"""
    WITH fast_rsi_entries AS (
        SELECT idx as entry_idx
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
        WHERE strat LIKE '%_7_%' AND val = 1
    ),
    exit_signal_types AS (
        SELECT 
            idx as exit_idx,
            CASE 
                WHEN strat LIKE '%_14_%' OR strat LIKE '%_21_%' THEN 'slow_rsi'
                WHEN strat LIKE '%ma_crossover%' THEN 'ma_crossover'
                WHEN strat LIKE '%momentum%' THEN 'momentum'
                WHEN strat LIKE '%mean_reversion%' THEN 'mean_reversion'
                WHEN strat LIKE '%breakout%' THEN 'breakout'
                ELSE 'other'
            END as exit_type,
            strat as exit_strategy
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/*/*.parquet')
        WHERE val = -1  -- Exit signals only
    ),
    exit_opportunities AS (
        SELECT 
            e.entry_idx,
            x.exit_idx,
            x.exit_type,
            x.exit_strategy,
            x.exit_idx - e.entry_idx as bars_to_exit
        FROM fast_rsi_entries e
        CROSS JOIN exit_signal_types x
        WHERE x.exit_idx > e.entry_idx 
            AND x.exit_idx <= e.entry_idx + 30  -- Look ahead 30 bars max
    )
    SELECT 
        exit_type,
        COUNT(*) as opportunities,
        COUNT(DISTINCT entry_idx) as entries_covered,
        ROUND(AVG(bars_to_exit), 1) as avg_bars_to_exit,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM fast_rsi_entries), 2) as coverage_pct
    FROM exit_opportunities
    GROUP BY exit_type
    ORDER BY opportunities DESC
    """
    
    exit_frequency = con.execute(exit_frequency_query).df()
    print(exit_frequency.to_string(index=False))
    
    # 3. Performance by exit signal type
    print("\n\n3. Performance Analysis by Exit Signal Type:")
    
    performance_by_exit_query = f"""
    WITH fast_rsi_entries AS (
        SELECT idx as entry_idx
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
        WHERE strat LIKE '%_7_%' AND val = 1
    ),
    exit_signal_types AS (
        SELECT 
            idx as exit_idx,
            CASE 
                WHEN strat LIKE '%_14_%' OR strat LIKE '%_21_%' THEN 'slow_rsi'
                WHEN strat LIKE '%ma_crossover%' THEN 'ma_crossover'
                WHEN strat LIKE '%momentum%' THEN 'momentum'
                WHEN strat LIKE '%mean_reversion%' THEN 'mean_reversion'
                WHEN strat LIKE '%breakout%' THEN 'breakout'
                ELSE 'other'
            END as exit_type
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/*/*.parquet')
        WHERE val = -1
    ),
    matched_trades AS (
        SELECT 
            e.entry_idx,
            MIN(x.exit_idx) as exit_idx,
            FIRST(x.exit_type) as exit_type_used,
            MIN(x.exit_idx) - e.entry_idx as holding_period
        FROM fast_rsi_entries e
        JOIN exit_signal_types x ON x.exit_idx > e.entry_idx AND x.exit_idx <= e.entry_idx + 20
        GROUP BY e.entry_idx
    )
    SELECT 
        t.exit_type_used,
        COUNT(*) as trades,
        ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return_pct,
        ROUND(STDDEV((m2.close - m1.close) / m1.close * 100), 4) as volatility,
        COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) as winners,
        ROUND(COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate,
        ROUND(AVG(holding_period), 1) as avg_holding_bars,
        ROUND(AVG((m2.close - m1.close) / m1.close * 100) / STDDEV((m2.close - m1.close) / m1.close * 100), 3) as sharpe
    FROM matched_trades t
    JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
    JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
    GROUP BY t.exit_type_used
    ORDER BY sharpe DESC
    """
    
    performance_by_exit = con.execute(performance_by_exit_query).df()
    print(performance_by_exit.to_string(index=False))
    
    # 4. Multi-exit signal portfolio analysis
    print("\n\n4. Multi-Exit Signal Portfolio Analysis:")
    print("Testing combinations of exit signals with priority ordering...")
    
    portfolio_strategies = [
        ("Slow RSI Only", ["slow_rsi"]),
        ("MA Crossover Only", ["ma_crossover"]),
        ("Momentum Only", ["momentum"]),
        ("RSI + MA", ["slow_rsi", "ma_crossover"]),
        ("RSI + Momentum", ["slow_rsi", "momentum"]),
        ("MA + Momentum", ["ma_crossover", "momentum"]),
        ("RSI + MA + Momentum", ["slow_rsi", "ma_crossover", "momentum"]),
        ("All Signals", ["slow_rsi", "ma_crossover", "momentum", "mean_reversion", "breakout"])
    ]
    
    portfolio_results = []
    
    for portfolio_name, signal_types in portfolio_strategies:
        signal_filter = " OR ".join([f"x.exit_type = '{sig}'" for sig in signal_types])
        
        portfolio_query = f"""
        WITH fast_rsi_entries AS (
            SELECT idx as entry_idx
            FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
            WHERE strat LIKE '%_7_%' AND val = 1
        ),
        exit_signal_types AS (
            SELECT 
                idx as exit_idx,
                CASE 
                    WHEN strat LIKE '%_14_%' OR strat LIKE '%_21_%' THEN 'slow_rsi'
                    WHEN strat LIKE '%ma_crossover%' THEN 'ma_crossover'
                    WHEN strat LIKE '%momentum%' THEN 'momentum'
                    WHEN strat LIKE '%mean_reversion%' THEN 'mean_reversion'
                    WHEN strat LIKE '%breakout%' THEN 'breakout'
                    ELSE 'other'
                END as exit_type
            FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/*/*.parquet')
            WHERE val = -1 AND ({signal_filter})
        ),
        portfolio_trades AS (
            SELECT 
                e.entry_idx,
                MIN(x.exit_idx) as exit_idx,
                MIN(x.exit_idx) - e.entry_idx as holding_period
            FROM fast_rsi_entries e
            JOIN exit_signal_types x ON x.exit_idx > e.entry_idx AND x.exit_idx <= e.entry_idx + 20
            GROUP BY e.entry_idx
        )
        SELECT 
            COUNT(*) as trades,
            ROUND(AVG((m2.close - m1.close) / m1.close * 100), 4) as avg_return_pct,
            ROUND(STDDEV((m2.close - m1.close) / m1.close * 100), 4) as volatility,
            COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) as winners,
            ROUND(COUNT(CASE WHEN (m2.close - m1.close) / m1.close > 0 THEN 1 END) * 100.0 / COUNT(*), 2) as win_rate,
            ROUND(AVG(holding_period), 1) as avg_holding_bars,
            ROUND(AVG((m2.close - m1.close) / m1.close * 100) / STDDEV((m2.close - m1.close) / m1.close * 100), 3) as sharpe
        FROM portfolio_trades t
        JOIN read_parquet('{data_path}') m1 ON t.entry_idx = m1.bar_index
        JOIN read_parquet('{data_path}') m2 ON t.exit_idx = m2.bar_index
        """
        
        try:
            result = con.execute(portfolio_query).df()
            if len(result) > 0 and result.iloc[0]['trades'] > 0:
                result['portfolio'] = portfolio_name
                portfolio_results.append(result.iloc[0])
        except Exception as e:
            print(f"Error in {portfolio_name}: {e}")
    
    if portfolio_results:
        portfolio_df = pd.DataFrame(portfolio_results)
        portfolio_df = portfolio_df.sort_values('sharpe', ascending=False)
        print(portfolio_df[['portfolio', 'trades', 'avg_return_pct', 'win_rate', 'sharpe', 'avg_holding_bars']].to_string(index=False))
    
    # 5. Coverage analysis - what % of entries get exits
    print("\n\n5. Entry Coverage Analysis:")
    print("What percentage of fast RSI entries get exit signals?")
    
    coverage_query = f"""
    WITH fast_rsi_entries AS (
        SELECT idx as entry_idx
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
        WHERE strat LIKE '%_7_%' AND val = 1
    ),
    all_exit_signals AS (
        SELECT 
            idx as exit_idx,
            CASE 
                WHEN strat LIKE '%_14_%' OR strat LIKE '%_21_%' THEN 'slow_rsi'
                WHEN strat LIKE '%ma_crossover%' THEN 'ma_crossover'
                WHEN strat LIKE '%momentum%' THEN 'momentum'
                WHEN strat LIKE '%mean_reversion%' THEN 'mean_reversion'
                WHEN strat LIKE '%breakout%' THEN 'breakout'
                ELSE 'other'
            END as exit_type
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/*/*.parquet')
        WHERE val = -1
    ),
    coverage_analysis AS (
        SELECT 
            e.entry_idx,
            COUNT(DISTINCT x.exit_type) as exit_types_available,
            MIN(x.exit_idx) as first_exit,
            CASE WHEN MIN(x.exit_idx) IS NOT NULL THEN 1 ELSE 0 END as has_exit
        FROM fast_rsi_entries e
        LEFT JOIN all_exit_signals x ON x.exit_idx > e.entry_idx AND x.exit_idx <= e.entry_idx + 20
        GROUP BY e.entry_idx
    )
    SELECT 
        (SELECT COUNT(*) FROM fast_rsi_entries) as total_entries,
        SUM(has_exit) as entries_with_exits,
        ROUND(SUM(has_exit) * 100.0 / COUNT(*), 2) as coverage_pct,
        ROUND(AVG(exit_types_available), 1) as avg_exit_types_per_entry,
        COUNT(CASE WHEN exit_types_available >= 3 THEN 1 END) as entries_with_3plus_exits
    FROM coverage_analysis
    """
    
    coverage = con.execute(coverage_query).df()
    print(coverage.to_string(index=False))
    
    # 6. Optimal exit timing analysis
    print("\n\n6. Optimal Exit Timing Analysis:")
    print("When do different exit signals typically fire relative to entry?")
    
    timing_analysis_query = f"""
    WITH fast_rsi_entries AS (
        SELECT idx as entry_idx
        FROM read_parquet('{workspace_path}/traces/SPY_1m/signals/rsi_grid/*.parquet')
        WHERE strat LIKE '%_7_%' AND val = 1
    ),
    exit_timing AS (
        SELECT 
            e.entry_idx,
            x.idx as exit_idx,
            x.idx - e.entry_idx as bars_to_exit,
            CASE 
                WHEN x.strat LIKE '%_14_%' OR x.strat LIKE '%_21_%' THEN 'slow_rsi'
                WHEN x.strat LIKE '%ma_crossover%' THEN 'ma_crossover'
                WHEN x.strat LIKE '%momentum%' THEN 'momentum'
                WHEN x.strat LIKE '%mean_reversion%' THEN 'mean_reversion'
                WHEN x.strat LIKE '%breakout%' THEN 'breakout'
                ELSE 'other'
            END as exit_type
        FROM fast_rsi_entries e
        JOIN read_parquet('{workspace_path}/traces/SPY_1m/signals/*/*.parquet') x 
            ON x.idx > e.entry_idx 
            AND x.idx <= e.entry_idx + 20
            AND x.val = -1
    )
    SELECT 
        exit_type,
        COUNT(*) as signals,
        ROUND(AVG(bars_to_exit), 1) as avg_bars_to_exit,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY bars_to_exit), 1) as q25_bars,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY bars_to_exit), 1) as median_bars,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY bars_to_exit), 1) as q75_bars,
        MIN(bars_to_exit) as min_bars,
        MAX(bars_to_exit) as max_bars
    FROM exit_timing
    GROUP BY exit_type
    ORDER BY avg_bars_to_exit
    """
    
    timing_analysis = con.execute(timing_analysis_query).df()
    print(timing_analysis.to_string(index=False))
    
    con.close()
    
    print("\n\n=== Strategic Insights ===")
    print("1. ENTRY SIGNAL: Fast RSI (7-period) provides entry timing")
    print("2. EXIT PORTFOLIO: Listen to multiple exit signal types after entry")
    print("3. SIGNAL PRIORITY: Use first available exit signal (or weighted combination)")
    print("4. COVERAGE OPTIMIZATION: Combine signals to maximize entry coverage")
    print("5. TIMING OPTIMIZATION: Different signals have different optimal timing")
    
    print("\n=== Next Steps ===")
    print("1. Implement exit signal portfolio in strategy")
    print("2. Test signal priority rules (first-exit vs weighted)")
    print("3. Add fallback exits for uncovered entries")
    print("4. Optimize signal combination weights")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python exit_signal_portfolio_analysis.py <workspace_path> <data_path>")
        sys.exit(1)
    
    analyze_exit_signal_portfolio(sys.argv[1], sys.argv[2])
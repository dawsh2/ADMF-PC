#!/usr/bin/env python3
"""
Check if classifiers are properly stored in sparse format.
"""
import duckdb
import pandas as pd
from pathlib import Path


def check_classifier_storage(workspace_path: str):
    """Verify classifier sparse storage implementation."""
    
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    print("=== Classifier Storage Analysis ===\n")
    
    # 1. Check raw classifier data
    print("1. Raw Classifier Data Sample:")
    
    sample_query = f"""
    SELECT 
        strat,
        idx,
        val,
        ts
    FROM read_parquet('{workspace_path}/traces/*/classifiers/*/*.parquet')
    WHERE strat = 'SPY_momentum_regime_grid_70_30_01'
    ORDER BY idx
    LIMIT 20
    """
    
    sample_df = con.execute(sample_query).df()
    print(sample_df.to_string(index=False))
    
    # 2. Check for consecutive indices (should have gaps if sparse)
    print("\n\n2. Checking for Sparse Storage (gaps between indices):")
    
    gap_query = f"""
    WITH classifier_indices AS (
        SELECT 
            strat,
            idx,
            LAG(idx) OVER (PARTITION BY strat ORDER BY idx) as prev_idx,
            idx - LAG(idx) OVER (PARTITION BY strat ORDER BY idx) as gap
        FROM read_parquet('{workspace_path}/traces/*/classifiers/*/*.parquet')
    )
    SELECT 
        strat,
        COUNT(*) as total_entries,
        MIN(gap) as min_gap,
        MAX(gap) as max_gap,
        AVG(gap) as avg_gap,
        COUNT(CASE WHEN gap = 1 THEN 1 END) as consecutive_count,
        COUNT(CASE WHEN gap > 1 THEN 1 END) as gap_count
    FROM classifier_indices
    WHERE gap IS NOT NULL
    GROUP BY strat
    ORDER BY avg_gap DESC
    LIMIT 10
    """
    
    gap_df = con.execute(gap_query).df()
    print(gap_df.to_string(index=False))
    
    # 3. Check actual state changes
    print("\n\n3. Actual State Changes Analysis:")
    
    state_change_query = f"""
    WITH ordered_states AS (
        SELECT 
            strat,
            idx,
            val,
            LAG(val) OVER (PARTITION BY strat ORDER BY idx) as prev_val,
            idx - LAG(idx) OVER (PARTITION BY strat ORDER BY idx) as idx_gap
        FROM read_parquet('{workspace_path}/traces/*/classifiers/*/*.parquet')
    ),
    state_changes AS (
        SELECT 
            strat,
            idx,
            val,
            prev_val,
            idx_gap,
            CASE WHEN val != prev_val OR prev_val IS NULL THEN 1 ELSE 0 END as is_change
        FROM ordered_states
    )
    SELECT 
        strat,
        COUNT(*) as total_records,
        SUM(is_change) as state_changes,
        ROUND(AVG(idx_gap), 1) as avg_idx_gap,
        MAX(idx_gap) as max_idx_gap
    FROM state_changes
    WHERE prev_val IS NOT NULL
    GROUP BY strat
    HAVING COUNT(*) > 100
    ORDER BY avg_idx_gap DESC
    LIMIT 10
    """
    
    changes_df = con.execute(state_change_query).df()
    print(changes_df.to_string(index=False))
    
    # 4. Check if we're storing redundant data
    print("\n\n4. Redundant Data Check (consecutive same values):")
    
    redundancy_query = f"""
    WITH consecutive_values AS (
        SELECT 
            strat,
            idx,
            val,
            LAG(val) OVER (PARTITION BY strat ORDER BY idx) as prev_val
        FROM read_parquet('{workspace_path}/traces/*/classifiers/*/*.parquet')
    )
    SELECT 
        strat,
        COUNT(*) as total_entries,
        COUNT(CASE WHEN val = prev_val THEN 1 END) as redundant_entries,
        ROUND(COUNT(CASE WHEN val = prev_val THEN 1 END) * 100.0 / COUNT(*), 2) as redundancy_pct
    FROM consecutive_values
    WHERE prev_val IS NOT NULL
    GROUP BY strat
    HAVING COUNT(*) > 100
    ORDER BY redundancy_pct DESC
    LIMIT 10
    """
    
    redundancy_df = con.execute(redundancy_query).df()
    print(redundancy_df.to_string(index=False))
    
    # 5. Compare with signal storage pattern
    print("\n\n5. Compare with Signal Storage (should be similar if both sparse):")
    
    signal_comparison_query = f"""
    WITH signal_gaps AS (
        SELECT 
            'signals' as type,
            COUNT(*) as total_entries,
            AVG(idx - LAG(idx) OVER (PARTITION BY strat ORDER BY idx)) as avg_gap
        FROM read_parquet('{workspace_path}/traces/*/signals/*/*.parquet')
        WHERE strat LIKE 'SPY_rsi%'
        GROUP BY type
    ),
    classifier_gaps AS (
        SELECT 
            'classifiers' as type,
            COUNT(*) as total_entries,
            AVG(idx - LAG(idx) OVER (PARTITION BY strat ORDER BY idx)) as avg_gap
        FROM read_parquet('{workspace_path}/traces/*/classifiers/*/*.parquet')
        WHERE strat LIKE 'SPY_momentum_regime%'
        GROUP BY type
    )
    SELECT * FROM signal_gaps
    UNION ALL
    SELECT * FROM classifier_gaps
    """
    
    comparison_df = con.execute(signal_comparison_query).df()
    print(comparison_df.to_string(index=False))
    
    # 6. Look at a specific classifier's state transitions
    print("\n\n6. Detailed State Transitions for SPY_momentum_regime_grid_70_30_01:")
    
    transition_detail_query = f"""
    WITH transitions AS (
        SELECT 
            idx,
            val,
            LAG(idx) OVER (ORDER BY idx) as prev_idx,
            LAG(val) OVER (ORDER BY idx) as prev_val
        FROM read_parquet('{workspace_path}/traces/*/classifiers/*/*.parquet')
        WHERE strat = 'SPY_momentum_regime_grid_70_30_01'
        ORDER BY idx
    )
    SELECT 
        idx,
        val,
        idx - prev_idx as gap,
        CASE WHEN val != prev_val THEN 'CHANGE' ELSE 'SAME' END as transition
    FROM transitions
    WHERE prev_idx IS NOT NULL
    ORDER BY idx
    LIMIT 30
    """
    
    transitions_df = con.execute(transition_detail_query).df()
    print(transitions_df.to_string(index=False))
    
    con.close()
    
    print("\n\n=== Analysis Summary ===")
    print("1. If classifiers are sparse, we should see large gaps between indices")
    print("2. Redundancy should be 0% in sparse storage (no consecutive same values)")
    print("3. Average gap should be >> 1 if properly sparse")
    print("4. Signal and classifier storage patterns should be similar")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python check_classifier_storage.py <workspace_path>")
        sys.exit(1)
    
    check_classifier_storage(sys.argv[1])
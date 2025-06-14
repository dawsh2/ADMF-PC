"""
Fast correlation filter using only sparse signal data.
No price joins needed - just signal overlap analysis.
"""
import duckdb
import pandas as pd
import numpy as np
from typing import List, Set, Dict, Tuple


def fast_correlation_filter(workspace_path: str, strategies: List[str], 
                          max_correlation: float = 0.7) -> List[str]:
    """
    Ultra-fast correlation filter using only signal indices.
    
    Two strategies are correlated if they have similar signal timing.
    We use Jaccard similarity on signal indices as a proxy for correlation.
    """
    con = duckdb.connect()
    
    # Get signal indices for each strategy (tiny data)
    query = f"""
    SELECT 
        strat,
        LIST(idx) as signal_indices,
        COUNT(*) as signal_count
    FROM read_parquet('{workspace_path}/traces/*/signals/*/*.parquet')
    WHERE val != 0 AND strat IN ('{"','".join(strategies)}')
    GROUP BY strat
    """
    
    df = con.execute(query).df()
    
    if df.empty:
        return []
    
    # Convert to dict for fast lookup
    strategy_signals = {}
    for _, row in df.iterrows():
        strategy_signals[row['strat']] = set(row['signal_indices'])
    
    # Calculate pairwise Jaccard similarity (intersection / union)
    correlations = {}
    strategies_list = list(strategy_signals.keys())
    
    for i, strat1 in enumerate(strategies_list):
        correlations[strat1] = {}
        signals1 = strategy_signals[strat1]
        
        for j, strat2 in enumerate(strategies_list):
            if i == j:
                correlations[strat1][strat2] = 1.0
            elif j > i:  # Only calculate upper triangle
                signals2 = strategy_signals[strat2]
                
                # Jaccard similarity
                intersection = len(signals1 & signals2)
                union = len(signals1 | signals2)
                
                similarity = intersection / union if union > 0 else 0
                correlations[strat1][strat2] = similarity
                correlations[strat2][strat1] = similarity  # Symmetric
    
    # Now select uncorrelated strategies
    # Get average returns for tie-breaking (small query)
    returns_query = f"""
    SELECT 
        strat,
        AVG(CASE 
            WHEN val = 1 THEN 0.01  -- Dummy positive return for ranking
            WHEN val = -1 THEN 0.01
        END) * COUNT(*) as score  -- Favor strategies with more signals
    FROM read_parquet('{workspace_path}/traces/*/signals/*/*.parquet')
    WHERE val != 0 AND strat IN ('{"','".join(strategies)}')
    GROUP BY strat
    """
    
    scores_df = con.execute(returns_query).df()
    strategy_scores = dict(zip(scores_df['strat'], scores_df['score']))
    
    # Greedy selection of uncorrelated strategies
    selected = []
    remaining = set(strategies_list)
    
    while remaining:
        # Get best scoring remaining strategy
        best_strat = max(remaining, key=lambda s: strategy_scores.get(s, 0))
        selected.append(best_strat)
        remaining.remove(best_strat)
        
        # Remove correlated strategies
        to_remove = set()
        for strat in remaining:
            if correlations[best_strat][strat] > max_correlation:
                to_remove.add(strat)
        remaining -= to_remove
    
    con.close()
    return selected


def ultra_fast_correlation_filter(workspace_path: str, strategies: List[str],
                                max_overlap_pct: float = 30.0) -> List[str]:
    """
    Even faster version - just check signal overlap percentage.
    
    If two strategies have >30% of their signals on the same bars,
    they're considered correlated.
    """
    con = duckdb.connect()
    
    # Single query to get all we need
    query = f"""
    WITH strategy_signals AS (
        SELECT 
            strat,
            ARRAY_AGG(idx) as indices,
            COUNT(*) as count
        FROM read_parquet('{workspace_path}/traces/*/signals/*/*.parquet')
        WHERE val != 0 AND strat IN ('{"','".join(strategies)}')
        GROUP BY strat
    ),
    overlap_matrix AS (
        SELECT 
            s1.strat as strat1,
            s2.strat as strat2,
            s1.count as count1,
            s2.count as count2,
            -- Count overlapping indices
            CARDINALITY(LIST_INTERSECT(s1.indices, s2.indices)) as overlap_count
        FROM strategy_signals s1
        CROSS JOIN strategy_signals s2
        WHERE s1.strat < s2.strat  -- Only upper triangle
    )
    SELECT 
        strat1,
        strat2,
        overlap_count,
        count1,
        count2,
        -- Overlap percentage relative to smaller strategy
        ROUND(overlap_count * 100.0 / LEAST(count1, count2), 1) as overlap_pct
    FROM overlap_matrix
    WHERE overlap_count * 100.0 / LEAST(count1, count2) > {max_overlap_pct}
    ORDER BY overlap_pct DESC
    """
    
    # Get correlated pairs
    correlated_pairs = con.execute(query).df()
    
    # Build correlation graph
    correlated_with = {}
    for strat in strategies:
        correlated_with[strat] = set()
    
    for _, row in correlated_pairs.iterrows():
        correlated_with[row['strat1']].add(row['strat2'])
        correlated_with[row['strat2']].add(row['strat1'])
    
    # Get strategy scores for selection
    score_query = f"""
    SELECT 
        strat,
        COUNT(*) as signal_count
    FROM read_parquet('{workspace_path}/traces/*/signals/*/*.parquet')
    WHERE val != 0 AND strat IN ('{"','".join(strategies)}')
    GROUP BY strat
    ORDER BY signal_count DESC
    """
    
    scores = con.execute(score_query).df()
    strategy_priority = list(scores['strat'])  # Ordered by signal count
    
    # Greedy selection
    selected = []
    excluded = set()
    
    for strat in strategy_priority:
        if strat not in excluded:
            selected.append(strat)
            # Exclude correlated strategies
            excluded.update(correlated_with[strat])
    
    con.close()
    return selected


if __name__ == "__main__":
    import sys
    import time
    
    if len(sys.argv) < 3:
        print("Usage: python fast_correlation_filter.py <workspace_path> <strategy1,strategy2,...>")
        sys.exit(1)
    
    workspace = sys.argv[1]
    strategies = sys.argv[2].split(',')
    
    print(f"Testing correlation filter on {len(strategies)} strategies...")
    
    # Time the ultra-fast version
    start = time.time()
    selected = ultra_fast_correlation_filter(workspace, strategies)
    elapsed = time.time() - start
    
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Selected {len(selected)} uncorrelated strategies:")
    for s in selected:
        print(f"  - {s}")
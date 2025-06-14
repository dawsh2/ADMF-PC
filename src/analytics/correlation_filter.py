"""
Pure correlation filter - takes strategies with their scores/returns
and filters out correlated ones, keeping the best performers.
"""
import duckdb
import pandas as pd
from typing import List, Dict, Tuple


def filter_correlated_strategies(
    workspace_path: str,
    strategies_with_scores: pd.DataFrame,
    max_overlap_pct: float = 30.0,
    score_column: str = 'avg_return'
) -> pd.DataFrame:
    """
    Filter correlated strategies, keeping the best performer from each group.
    
    Args:
        workspace_path: Path to workspace with signal data
        strategies_with_scores: DataFrame with 'strat' and score column
        max_overlap_pct: Maximum signal overlap percentage to consider uncorrelated
        score_column: Column name to use for selecting best strategy
        
    Returns:
        DataFrame with uncorrelated strategies
    """
    if strategies_with_scores.empty:
        return strategies_with_scores
    
    strategies = strategies_with_scores['strat'].tolist()
    con = duckdb.connect()
    
    # Calculate signal overlaps in one efficient query
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
            -- Count overlapping indices
            LENGTH(LIST_INTERSECT(s1.indices, s2.indices)) as overlap_count,
            LEAST(s1.count, s2.count) as min_count
        FROM strategy_signals s1
        CROSS JOIN strategy_signals s2
        WHERE s1.strat < s2.strat  -- Only upper triangle
    )
    SELECT 
        strat1,
        strat2,
        overlap_count * 100.0 / min_count as overlap_pct
    FROM overlap_matrix
    WHERE overlap_count * 100.0 / min_count > {max_overlap_pct}
    """
    
    correlated_pairs = con.execute(query).df()
    con.close()
    
    # Build correlation sets
    correlated_groups = []
    strategy_to_group = {}
    
    for _, row in correlated_pairs.iterrows():
        s1, s2 = row['strat1'], row['strat2']
        
        # Find which groups these strategies belong to
        g1 = strategy_to_group.get(s1)
        g2 = strategy_to_group.get(s2)
        
        if g1 is None and g2 is None:
            # Create new group
            new_group = {s1, s2}
            correlated_groups.append(new_group)
            strategy_to_group[s1] = new_group
            strategy_to_group[s2] = new_group
        elif g1 is None:
            # Add s1 to s2's group
            g2.add(s1)
            strategy_to_group[s1] = g2
        elif g2 is None:
            # Add s2 to s1's group
            g1.add(s2)
            strategy_to_group[s2] = g1
        elif g1 != g2:
            # Merge groups
            g1.update(g2)
            for s in g2:
                strategy_to_group[s] = g1
            correlated_groups.remove(g2)
    
    # Create score lookup
    score_lookup = dict(zip(
        strategies_with_scores['strat'], 
        strategies_with_scores[score_column]
    ))
    
    # Select best from each correlated group
    selected_strategies = []
    
    # First, add all uncorrelated strategies
    all_correlated = set()
    for group in correlated_groups:
        all_correlated.update(group)
    
    for strat in strategies:
        if strat not in all_correlated:
            selected_strategies.append(strat)
    
    # Then add best from each correlated group
    for group in correlated_groups:
        best_strat = max(group, key=lambda s: score_lookup.get(s, float('-inf')))
        selected_strategies.append(best_strat)
    
    # Return filtered dataframe maintaining original structure
    return strategies_with_scores[
        strategies_with_scores['strat'].isin(selected_strategies)
    ].copy()


def get_signal_overlap_matrix(workspace_path: str, strategies: List[str]) -> pd.DataFrame:
    """
    Get detailed overlap matrix for analysis.
    """
    con = duckdb.connect()
    
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
    overlap_details AS (
        SELECT 
            s1.strat as strat1,
            s2.strat as strat2,
            s1.count as count1,
            s2.count as count2,
            LENGTH(LIST_INTERSECT(s1.indices, s2.indices)) as overlap_count,
            LENGTH(LIST_INTERSECT(s1.indices, s2.indices)) * 100.0 / 
                LEAST(s1.count, s2.count) as overlap_pct
        FROM strategy_signals s1
        CROSS JOIN strategy_signals s2
    )
    SELECT * FROM overlap_details
    ORDER BY overlap_pct DESC
    """
    
    result = con.execute(query).df()
    con.close()
    return result
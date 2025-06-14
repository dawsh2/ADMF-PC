#!/usr/bin/env python3
"""
Analyze classifier regime distribution for balance.
Good classifiers should have balanced distribution between states.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path


def analyze_classifier_balance(workspace_path: str, data_path: str):
    """Analyze classifier balance and identify good vs biased classifiers."""
    
    con = duckdb.connect(f'{workspace_path}/analytics.duckdb')
    
    print("=== Classifier Balance Analysis ===\n")
    
    # 1. Get all classifiers and their distributions
    print("1. Classifier State Distributions:")
    
    classifier_dist_query = f"""
    WITH classifier_data AS (
        SELECT 
            strat as classifier,
            val as state,
            COUNT(*) as occurrences,
            MIN(idx) as first_idx,
            MAX(idx) as last_idx
        FROM read_parquet('{workspace_path}/traces/*/classifiers/*/*.parquet')
        GROUP BY strat, val
    ),
    total_bars AS (
        SELECT 
            classifier,
            SUM(occurrences) as total
        FROM classifier_data
        GROUP BY classifier
    )
    SELECT 
        c.classifier,
        c.state,
        c.occurrences,
        ROUND(c.occurrences * 100.0 / t.total, 2) as pct,
        t.total as total_predictions
    FROM classifier_data c
    JOIN total_bars t ON c.classifier = t.classifier
    ORDER BY c.classifier, c.state
    """
    
    dist_df = con.execute(classifier_dist_query).df()
    
    # Calculate balance metrics
    balance_metrics = []
    
    for classifier in dist_df['classifier'].unique():
        clf_data = dist_df[dist_df['classifier'] == classifier]
        
        # Skip classifiers with only one state (not useful)
        if len(clf_data) <= 1:
            continue
            
        # Calculate distribution metrics
        percentages = clf_data['pct'].values
        occurrences = clf_data['occurrences'].values
        
        # Entropy (higher = more balanced)
        probs = percentages / 100
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(probs))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Imbalance ratio (lower = more balanced)
        imbalance = max(percentages) / min(percentages) if min(percentages) > 0 else float('inf')
        
        # Coefficient of variation (lower = more balanced)
        cv = np.std(percentages) / np.mean(percentages) if np.mean(percentages) > 0 else float('inf')
        
        balance_metrics.append({
            'classifier': classifier,
            'n_states': len(clf_data),
            'total_predictions': clf_data['total_predictions'].iloc[0],
            'entropy': normalized_entropy,
            'imbalance_ratio': imbalance,
            'cv': cv,
            'state_dist': ', '.join([f"S{row['state']}:{row['pct']}%" for _, row in clf_data.iterrows()])
        })
    
    balance_df = pd.DataFrame(balance_metrics)
    
    # Sort by entropy (most balanced first)
    balance_df = balance_df.sort_values('entropy', ascending=False)
    
    print("\nBalanced Classifiers (entropy > 0.8):")
    balanced = balance_df[balance_df['entropy'] > 0.8]
    if len(balanced) > 0:
        print(balanced[['classifier', 'n_states', 'entropy', 'state_dist']].to_string(index=False))
    else:
        print("No well-balanced classifiers found!")
    
    print("\n\nBiased Classifiers (entropy < 0.5 or imbalance > 10):")
    biased = balance_df[(balance_df['entropy'] < 0.5) | (balance_df['imbalance_ratio'] > 10)]
    if len(biased) > 0:
        print(biased[['classifier', 'n_states', 'entropy', 'imbalance_ratio', 'state_dist']].head(10).to_string(index=False))
    
    # 2. Test strategy performance with balanced vs biased classifiers
    print("\n\n2. Strategy Performance by Classifier Type:")
    
    if len(balanced) > 0 and len(biased) > 0:
        # Pick best balanced and most biased classifier
        best_balanced = balanced.iloc[0]['classifier']
        most_biased = biased.iloc[-1]['classifier']
        
        performance_query = f"""
        WITH strategy_performance AS (
            -- Performance with balanced classifier
            SELECT 
                'balanced' as classifier_type,
                s.strat,
                c.val as regime,
                COUNT(*) as trades,
                AVG(CASE 
                    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
                END) as avg_return
            FROM read_parquet('{workspace_path}/traces/*/signals/*/*.parquet') s
            JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
            JOIN read_parquet('{workspace_path}/traces/*/classifiers/*/*.parquet') c 
                ON s.idx = c.idx AND c.strat = '{best_balanced}'
            WHERE s.val != 0
            GROUP BY s.strat, c.val
            
            UNION ALL
            
            -- Performance with biased classifier
            SELECT 
                'biased' as classifier_type,
                s.strat,
                c.val as regime,
                COUNT(*) as trades,
                AVG(CASE 
                    WHEN s.val = 1 THEN (m2.close - m1.close) / m1.close * 100
                    WHEN s.val = -1 THEN (m1.close - m2.close) / m1.close * 100
                END) as avg_return
            FROM read_parquet('{workspace_path}/traces/*/signals/*/*.parquet') s
            JOIN read_parquet('{data_path}') m1 ON s.idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON s.idx + 1 = m2.bar_index
            JOIN read_parquet('{workspace_path}/traces/*/classifiers/*/*.parquet') c 
                ON s.idx = c.idx AND c.strat = '{most_biased}'
            WHERE s.val != 0
            GROUP BY s.strat, c.val
        )
        SELECT 
            classifier_type,
            COUNT(DISTINCT strat) as strategies,
            SUM(trades) as total_trades,
            ROUND(AVG(avg_return), 4) as avg_return
        FROM strategy_performance
        WHERE trades >= 10
        GROUP BY classifier_type
        """
        
        perf_comparison = con.execute(performance_query).df()
        print(f"\nComparing {best_balanced} (balanced) vs {most_biased} (biased):")
        print(perf_comparison.to_string(index=False))
    
    # 3. Find regime transitions
    print("\n\n3. Regime Transition Analysis (for balanced classifiers):")
    
    if len(balanced) > 0:
        transition_classifier = balanced.iloc[0]['classifier']
        
        transition_query = f"""
        WITH regime_sequence AS (
            SELECT 
                idx,
                val as regime,
                LAG(val) OVER (ORDER BY idx) as prev_regime
            FROM read_parquet('{workspace_path}/traces/*/classifiers/*/*.parquet')
            WHERE strat = '{transition_classifier}'
        ),
        transitions AS (
            SELECT 
                prev_regime,
                regime,
                COUNT(*) as transitions
            FROM regime_sequence
            WHERE prev_regime IS NOT NULL AND prev_regime != regime
            GROUP BY prev_regime, regime
        )
        SELECT 
            prev_regime as from_state,
            regime as to_state,
            transitions,
            ROUND(transitions * 100.0 / SUM(transitions) OVER (), 2) as pct
        FROM transitions
        ORDER BY transitions DESC
        """
        
        transitions = con.execute(transition_query).df()
        print(f"\nRegime transitions for {transition_classifier}:")
        print(transitions.to_string(index=False))
        
        # Average regime duration
        duration_query = f"""
        WITH regime_changes AS (
            SELECT 
                idx,
                val as regime,
                CASE WHEN val != LAG(val) OVER (ORDER BY idx) THEN 1 ELSE 0 END as is_change
            FROM read_parquet('{workspace_path}/traces/*/classifiers/*/*.parquet')
            WHERE strat = '{transition_classifier}'
        ),
        regime_blocks AS (
            SELECT 
                idx,
                regime,
                SUM(is_change) OVER (ORDER BY idx) as block_id
            FROM regime_changes
        ),
        regime_durations AS (
            SELECT 
                regime,
                block_id,
                COUNT(*) as duration
            FROM regime_blocks
            GROUP BY regime, block_id
        )
        SELECT 
            regime,
            COUNT(*) as n_periods,
            ROUND(AVG(duration), 1) as avg_duration,
            MIN(duration) as min_duration,
            MAX(duration) as max_duration
        FROM regime_durations
        GROUP BY regime
        """
        
        durations = con.execute(duration_query).df()
        print(f"\nRegime durations for {transition_classifier}:")
        print(durations.to_string(index=False))
    
    con.close()
    
    print("\n\n=== Recommendations ===")
    print("1. Use classifiers with entropy > 0.8 for regime filtering")
    print("2. Avoid classifiers that spend >90% time in one state")
    print("3. Momentum regime classifiers appear most balanced")
    print("4. Consider regime duration when setting holding periods")
    print("5. Test strategies across all regime states before deployment")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python analyze_classifier_balance.py <workspace_path> <data_path>")
        sys.exit(1)
    
    analyze_classifier_balance(sys.argv[1], sys.argv[2])
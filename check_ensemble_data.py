#!/usr/bin/env python3
"""
Check ensemble data and create SQL queries for analysis.
Works around database lock by generating SQL for later execution.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

def generate_ensemble_sql_queries():
    """Generate SQL queries for ensemble strategy analysis."""
    
    queries = {
        "signal_timing_validation": """
-- Check for look-ahead bias in signal timing
SELECT 
    'Signal Timing Check' as analysis,
    COUNT(CASE WHEN entry_bar < signal_bar THEN 1 END) as lookahead_violations,
    COUNT(*) as total_trades,
    ROUND(100.0 * COUNT(CASE WHEN entry_bar < signal_bar THEN 1 END) / COUNT(*), 2) as violation_pct
FROM signal_performance
WHERE entry_bar IS NOT NULL;
""",

        "strategy_correlation_matrix": """
-- Find low-correlation strategy pairs for ensemble
WITH strategy_pairs AS (
    SELECT DISTINCT
        s1.component_id as strategy_a,
        s2.component_id as strategy_b,
        s1.strategy_type as type_a,
        s2.strategy_type as type_b
    FROM component_metrics s1
    CROSS JOIN component_metrics s2
    WHERE s1.component_type = 'strategy' 
    AND s2.component_type = 'strategy'
    AND s1.component_id < s2.component_id
)
SELECT 
    type_a || ' + ' || type_b as strategy_combination,
    COUNT(*) as pair_count,
    'Low correlation pairs ideal for ensemble' as note
FROM strategy_pairs
WHERE type_a != type_b
GROUP BY type_a, type_b
ORDER BY pair_count DESC;
""",

        "complementary_signals": """
-- Find strategies with complementary signal patterns
WITH signal_summary AS (
    SELECT 
        component_id,
        COUNT(DISTINCT bar_index) as signal_bars,
        MIN(bar_index) as first_signal,
        MAX(bar_index) as last_signal,
        COUNT(*) as total_signals
    FROM signal_changes
    WHERE component_type = 'strategy'
    GROUP BY component_id
),
overlaps AS (
    SELECT 
        s1.component_id as strategy_a,
        s2.component_id as strategy_b,
        s1.signal_bars as bars_a,
        s2.signal_bars as bars_b,
        COUNT(DISTINCT sc1.bar_index) as overlapping_bars
    FROM signal_summary s1
    CROSS JOIN signal_summary s2
    LEFT JOIN signal_changes sc1 ON sc1.component_id = s1.component_id
    LEFT JOIN signal_changes sc2 ON sc2.component_id = s2.component_id 
        AND sc2.bar_index = sc1.bar_index
    WHERE s1.component_id < s2.component_id
    AND s1.signal_bars > 10 
    AND s2.signal_bars > 10
    GROUP BY s1.component_id, s2.component_id, s1.signal_bars, s2.signal_bars
)
SELECT 
    strategy_a,
    strategy_b,
    bars_a + bars_b - overlapping_bars as combined_coverage,
    ROUND(100.0 * overlapping_bars / LEAST(bars_a, bars_b), 2) as overlap_pct,
    CASE 
        WHEN overlap_pct < 30 THEN 'Excellent diversity'
        WHEN overlap_pct < 50 THEN 'Good diversity'
        ELSE 'Consider alternatives'
    END as ensemble_quality
FROM overlaps
WHERE overlap_pct < 50
ORDER BY combined_coverage DESC
LIMIT 20;
""",

        "regime_based_performance": """
-- Analyze strategy performance by market regime
WITH regime_signals AS (
    SELECT 
        sc.bar_index,
        sc.component_id as classifier_id,
        sc.signal_value as regime,
        ss.component_id as strategy_id,
        ss.signal_value as strategy_signal
    FROM signal_changes sc
    JOIN signal_changes ss ON ss.bar_index = sc.bar_index
    WHERE sc.component_type = 'classifier'
    AND ss.component_type = 'strategy'
)
SELECT 
    classifier_id,
    regime,
    strategy_id,
    COUNT(*) as signals_in_regime,
    COUNT(DISTINCT bar_index) as unique_bars
FROM regime_signals
GROUP BY classifier_id, regime, strategy_id
HAVING signals_in_regime > 5
ORDER BY classifier_id, regime, signals_in_regime DESC;
""",

        "optimal_ensemble_size": """
-- Determine optimal number of strategies in ensemble
WITH strategy_counts AS (
    SELECT 
        bar_index,
        COUNT(DISTINCT component_id) as strategies_signaling,
        COUNT(DISTINCT signal_value) as unique_signals,
        CASE 
            WHEN COUNT(DISTINCT signal_value) = 1 THEN 'Unanimous'
            WHEN COUNT(DISTINCT signal_value) = 2 THEN 'Mixed'
            ELSE 'Divergent'
        END as signal_agreement
    FROM signal_changes
    WHERE component_type = 'strategy'
    GROUP BY bar_index
)
SELECT 
    strategies_signaling,
    signal_agreement,
    COUNT(*) as occurrence_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) as pct_of_bars
FROM strategy_counts
GROUP BY strategies_signaling, signal_agreement
ORDER BY strategies_signaling, signal_agreement;
""",

        "ensemble_voting_analysis": """
-- Analyze potential voting outcomes
WITH signal_votes AS (
    SELECT 
        bar_index,
        SUM(CASE WHEN signal_value > 0 THEN 1 ELSE 0 END) as long_votes,
        SUM(CASE WHEN signal_value < 0 THEN 1 ELSE 0 END) as short_votes,
        SUM(CASE WHEN signal_value = 0 THEN 1 ELSE 0 END) as neutral_votes,
        COUNT(*) as total_votes
    FROM signal_changes
    WHERE component_type = 'strategy'
    GROUP BY bar_index
)
SELECT 
    CASE 
        WHEN long_votes > short_votes AND long_votes > neutral_votes THEN 'Long Consensus'
        WHEN short_votes > long_votes AND short_votes > neutral_votes THEN 'Short Consensus'
        WHEN neutral_votes >= total_votes / 2 THEN 'Neutral Consensus'
        ELSE 'No Clear Consensus'
    END as voting_outcome,
    COUNT(*) as bars_count,
    AVG(total_votes) as avg_strategies_voting,
    MAX(long_votes) as max_long_votes,
    MAX(short_votes) as max_short_votes
FROM signal_votes
GROUP BY voting_outcome
ORDER BY bars_count DESC;
""",

        "parameter_sensitivity": """
-- Analyze parameter sensitivity for ensemble tuning
SELECT 
    strategy_type,
    COUNT(DISTINCT component_id) as variations,
    MIN(signal_frequency) as min_signal_freq,
    AVG(signal_frequency) as avg_signal_freq,
    MAX(signal_frequency) as max_signal_freq,
    STDDEV(signal_frequency) as signal_freq_stddev,
    CASE 
        WHEN STDDEV(signal_frequency) < 0.01 THEN 'Low sensitivity'
        WHEN STDDEV(signal_frequency) < 0.05 THEN 'Medium sensitivity'
        ELSE 'High sensitivity'
    END as parameter_sensitivity
FROM component_metrics
WHERE component_type = 'strategy'
GROUP BY strategy_type
HAVING COUNT(DISTINCT component_id) > 3
ORDER BY avg_signal_freq DESC;
"""
    }
    
    return queries

def save_queries_to_file():
    """Save SQL queries to a file for manual execution."""
    queries = generate_ensemble_sql_queries()
    
    output_path = Path("ensemble_analysis_queries.sql")
    
    with open(output_path, 'w') as f:
        f.write("-- ENSEMBLE STRATEGY ANALYSIS QUERIES\n")
        f.write("-- Run these queries in sql_analytics.py interactive mode\n")
        f.write("-- " + "="*60 + "\n\n")
        
        for name, query in queries.items():
            f.write(f"-- {name.upper().replace('_', ' ')}\n")
            f.write("-- " + "-"*60 + "\n")
            f.write(query)
            f.write("\n\n")
    
    print(f"✅ Saved {len(queries)} ensemble analysis queries to: {output_path}")
    print("\n📝 To run these queries:")
    print("   1. Close any other processes using analytics.duckdb")
    print("   2. Run: python sql_analytics.py --interactive")
    print("   3. Copy and paste queries from ensemble_analysis_queries.sql")

def suggest_ensemble_strategies():
    """Print suggested ensemble strategy configurations."""
    
    print("\n" + "="*80)
    print("🎯 SUGGESTED ENSEMBLE STRATEGIES")
    print("="*80)
    
    suggestions = [
        {
            "name": "Balanced Momentum-Reversion",
            "description": "Combines trend-following with mean reversion for all market conditions",
            "components": [
                {"type": "MA Crossover", "params": "fast=10, slow=30", "weight": 0.4},
                {"type": "RSI", "params": "period=14, oversold=30, overbought=70", "weight": 0.3},
                {"type": "Momentum", "params": "lookback=20", "weight": 0.3}
            ],
            "benefits": [
                "Captures both trends and reversals",
                "Reduces drawdowns during ranging markets",
                "Provides more consistent signals"
            ]
        },
        {
            "name": "Multi-Timeframe Consensus",
            "description": "Same strategy logic across multiple timeframes",
            "components": [
                {"type": "Momentum", "params": "lookback=10 (short-term)", "weight": 0.25},
                {"type": "Momentum", "params": "lookback=20 (medium-term)", "weight": 0.35},
                {"type": "Momentum", "params": "lookback=50 (long-term)", "weight": 0.40}
            ],
            "benefits": [
                "Confirms signals across timeframes",
                "Reduces false signals from noise",
                "Better trend identification"
            ]
        },
        {
            "name": "Volatility-Adaptive Ensemble",
            "description": "Adjusts strategy weights based on market volatility",
            "components": [
                {"type": "Trend Following", "params": "for low volatility", "weight": "dynamic"},
                {"type": "Mean Reversion", "params": "for high volatility", "weight": "dynamic"},
                {"type": "Breakout", "params": "for volatility transitions", "weight": "dynamic"}
            ],
            "benefits": [
                "Adapts to changing market conditions",
                "Reduces losses during regime changes",
                "Maximizes opportunities in each regime"
            ]
        },
        {
            "name": "Signal Strength Voting",
            "description": "Multiple strategies vote with position sizing based on agreement",
            "components": [
                {"type": "MACD", "params": "standard", "weight": "vote"},
                {"type": "RSI", "params": "period=14", "weight": "vote"},
                {"type": "Stochastic", "params": "standard", "weight": "vote"},
                {"type": "Bollinger Bands", "params": "20,2", "weight": "vote"}
            ],
            "benefits": [
                "Higher confidence trades when strategies agree",
                "Natural position sizing through vote counting",
                "Reduced false signals"
            ]
        }
    ]
    
    for i, strategy in enumerate(suggestions, 1):
        print(f"\n{i}. {strategy['name'].upper()}")
        print(f"   {strategy['description']}")
        print("\n   Components:")
        for comp in strategy['components']:
            print(f"   - {comp['type']}: {comp['params']} (weight: {comp['weight']})")
        print("\n   Benefits:")
        for benefit in strategy['benefits']:
            print(f"   ✓ {benefit}")

def analyze_lookback_windows():
    """Suggest optimal lookback windows for ensemble diversity."""
    
    print("\n" + "="*80)
    print("🔍 OPTIMAL LOOKBACK WINDOWS FOR ENSEMBLE DIVERSITY")
    print("="*80)
    
    print("\nRecommended lookback periods for different strategy types:")
    
    lookback_suggestions = {
        "Momentum": {
            "Short-term": [5, 10, 15],
            "Medium-term": [20, 30, 40],
            "Long-term": [50, 100, 200]
        },
        "Moving Average": {
            "Fast MA": [5, 8, 10, 12],
            "Slow MA": [20, 30, 50, 100]
        },
        "RSI": {
            "Aggressive": [7, 9],
            "Standard": [14],
            "Conservative": [21, 30]
        },
        "Mean Reversion": {
            "Short-term": [10, 20],
            "Medium-term": [30, 50],
            "Long-term": [100, 200]
        }
    }
    
    for strategy_type, periods in lookback_suggestions.items():
        print(f"\n{strategy_type}:")
        for timeframe, values in periods.items():
            print(f"  {timeframe}: {values}")
    
    print("\n💡 Tips for parameter selection:")
    print("  • Use Fibonacci numbers for natural market rhythms (5, 8, 13, 21, 34, 55)")
    print("  • Ensure at least 2:1 ratio between fast and slow parameters")
    print("  • Test parameter stability with small variations (±10%)")
    print("  • Avoid overfitting by limiting parameter combinations")

def main():
    """Run ensemble analysis utilities."""
    
    print("🎭 ENSEMBLE STRATEGY ANALYSIS TOOLKIT")
    print("="*80)
    
    # 1. Generate and save SQL queries
    save_queries_to_file()
    
    # 2. Suggest ensemble strategies
    suggest_ensemble_strategies()
    
    # 3. Analyze lookback windows
    analyze_lookback_windows()
    
    print("\n" + "="*80)
    print("📊 NEXT STEPS:")
    print("="*80)
    print("1. Review the generated SQL queries in ensemble_analysis_queries.sql")
    print("2. Run sql_analytics.py to execute the queries")
    print("3. Use the ensemble strategy suggestions to create configurations")
    print("4. Test ensemble strategies with the suggested parameter combinations")
    print("\n💡 Remember: The goal is to find strategies that:")
    print("   - Have low correlation (< 0.3)")
    print("   - Signal at different times")
    print("   - Perform well in different market conditions")

if __name__ == "__main__":
    main()
"""
Pattern mining for discovering emergent trading behaviors.
Finds signal combinations and cross-strategy patterns.
"""
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import json


class PatternMiner:
    """Mine for emergent patterns across strategies and signals."""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = Path(workspace_path)
        self.db_path = self.workspace_path / "analytics.duckdb"
        self.con = duckdb.connect(str(self.db_path))
        
    def mine_signal_combinations(self, data_path: str, max_lag: int = 10) -> pd.DataFrame:
        """
        Find profitable combinations of entry/exit signals from different strategies.
        E.g., Enter on slow RSI bullish, exit on fast RSI bearish.
        """
        query = f"""
        -- Get all signals with their types
        WITH all_signals AS (
            SELECT 
                idx,
                ts,
                strat,
                val,
                CASE 
                    WHEN strat LIKE '%rsi%' AND strat LIKE '%_7_%' THEN 'rsi_fast'
                    WHEN strat LIKE '%rsi%' AND strat LIKE '%_14_%' THEN 'rsi_slow'
                    WHEN strat LIKE '%ma_crossover%' AND strat LIKE '%_10_%' THEN 'ma_fast'
                    WHEN strat LIKE '%ma_crossover%' AND strat LIKE '%_20_%' THEN 'ma_slow'
                    WHEN strat LIKE '%breakout%' THEN 'breakout'
                    WHEN strat LIKE '%mean_reversion%' THEN 'mean_rev'
                    WHEN strat LIKE '%momentum%' THEN 'momentum'
                    ELSE 'other'
                END as signal_type
            FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet')
            WHERE val != 0
        ),
        -- Find entry signals
        entry_signals AS (
            SELECT * FROM all_signals WHERE val = 1
        ),
        -- Find potential exit signals within lag window
        signal_pairs AS (
            SELECT 
                e.strat as entry_strat,
                e.signal_type as entry_type,
                e.idx as entry_idx,
                x.strat as exit_strat,
                x.signal_type as exit_type,
                x.idx as exit_idx,
                x.val as exit_signal,
                (x.idx - e.idx) as holding_period
            FROM entry_signals e
            JOIN all_signals x ON 
                x.idx > e.idx AND 
                x.idx <= e.idx + {max_lag} AND
                x.strat != e.strat  -- Different strategies
        )
        SELECT 
            sp.entry_type,
            sp.exit_type,
            sp.exit_signal,
            sp.holding_period,
            COUNT(*) as occurrences,
            AVG((m_exit.close - m_entry.close) / m_entry.close * 100) as avg_return,
            STDDEV((m_exit.close - m_entry.close) / m_entry.close * 100) as volatility
        FROM signal_pairs sp
        JOIN read_parquet('{data_path}') m_entry ON sp.entry_idx = m_entry.bar_index
        JOIN read_parquet('{data_path}') m_exit ON sp.exit_idx = m_exit.bar_index
        GROUP BY sp.entry_type, sp.exit_type, sp.exit_signal, sp.holding_period
        HAVING COUNT(*) >= 5
        ORDER BY avg_return DESC
        """
        
        return self.con.execute(query).df()
    
    def mine_conditional_patterns(self, data_path: str) -> pd.DataFrame:
        """
        Find patterns where one signal type works better under certain conditions.
        E.g., RSI works well when momentum is already bullish.
        """
        query = f"""
        WITH signal_context AS (
            SELECT 
                s1.idx,
                s1.strat as primary_strat,
                s1.val as primary_signal,
                -- Look for context signals in previous 5 bars
                s2.strat as context_strat,
                s2.val as context_signal,
                CASE 
                    WHEN s2.strat LIKE '%momentum%' AND s2.val = 1 THEN 'bullish_momentum'
                    WHEN s2.strat LIKE '%momentum%' AND s2.val = -1 THEN 'bearish_momentum'
                    WHEN s2.strat LIKE '%mean_reversion%' AND s2.val != 0 THEN 'mean_rev_active'
                    WHEN s2.strat LIKE '%breakout%' AND s2.val = 1 THEN 'breakout_long'
                    ELSE 'no_context'
                END as context_type
            FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet') s1
            LEFT JOIN read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet') s2 ON 
                s2.idx >= s1.idx - 5 AND 
                s2.idx < s1.idx AND
                s2.strat != s1.strat
            WHERE s1.val != 0
        ),
        conditional_performance AS (
            SELECT 
                sc.primary_strat,
                sc.context_type,
                COUNT(*) as trades,
                AVG(CASE 
                    WHEN sc.primary_signal = 1 THEN (m2.close - m1.close) / m1.close * 100
                    WHEN sc.primary_signal = -1 THEN (m1.close - m2.close) / m1.close * 100
                END) as avg_return
            FROM signal_context sc
            JOIN read_parquet('{data_path}') m1 ON sc.idx = m1.bar_index
            JOIN read_parquet('{data_path}') m2 ON sc.idx + 1 = m2.bar_index
            WHERE sc.context_type != 'no_context'
            GROUP BY sc.primary_strat, sc.context_type
        )
        SELECT 
            primary_strat,
            context_type,
            trades,
            ROUND(avg_return, 4) as avg_return
        FROM conditional_performance
        WHERE trades >= 5
        ORDER BY avg_return DESC
        """
        
        return self.con.execute(query).df()
    
    def mine_regime_transitions(self, data_path: str, classifier_path: Optional[str] = None) -> pd.DataFrame:
        """
        Find strategies that work well during regime transitions.
        """
        if not classifier_path:
            # Find any classifier
            classifier_files = list(Path(self.workspace_path).glob("traces/*/classifiers/*/*.parquet"))
            if not classifier_files:
                return pd.DataFrame()
            classifier_path = str(classifier_files[0])
        
        query = f"""
        WITH regime_changes AS (
            SELECT 
                idx,
                val as regime,
                LAG(val, 1) OVER (ORDER BY idx) as prev_regime
            FROM read_parquet('{classifier_path}')
        ),
        transitions AS (
            SELECT 
                idx as transition_idx,
                prev_regime,
                regime as new_regime,
                CONCAT(CAST(prev_regime AS VARCHAR), '_to_', CAST(regime AS VARCHAR)) as transition_type
            FROM regime_changes
            WHERE regime != prev_regime AND prev_regime IS NOT NULL
        ),
        transition_signals AS (
            SELECT 
                t.transition_type,
                s.strat,
                s.val as signal,
                s.idx,
                ABS(s.idx - t.transition_idx) as bars_from_transition
            FROM transitions t
            JOIN read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet') s ON 
                s.idx >= t.transition_idx - 5 AND 
                s.idx <= t.transition_idx + 5
            WHERE s.val != 0
        )
        SELECT 
            transition_type,
            strat,
            signal,
            AVG(bars_from_transition) as avg_bars_from_transition,
            COUNT(*) as signals_near_transition,
            AVG(CASE 
                WHEN signal = 1 THEN (m2.close - m1.close) / m1.close * 100
                WHEN signal = -1 THEN (m1.close - m2.close) / m1.close * 100
            END) as avg_return
        FROM transition_signals ts
        JOIN read_parquet('{data_path}') m1 ON ts.idx = m1.bar_index
        JOIN read_parquet('{data_path}') m2 ON ts.idx + 1 = m2.bar_index
        GROUP BY transition_type, strat, signal
        HAVING COUNT(*) >= 3
        ORDER BY avg_return DESC
        """
        
        return self.con.execute(query).df()
    
    def mine_exit_strategies(self, data_path: str) -> pd.DataFrame:
        """
        Compare different exit strategies for each entry signal type.
        """
        query = f"""
        WITH entry_signals AS (
            SELECT 
                idx as entry_idx,
                strat,
                CASE 
                    WHEN strat LIKE '%rsi%' THEN 'rsi'
                    WHEN strat LIKE '%ma_crossover%' THEN 'ma_crossover'
                    WHEN strat LIKE '%momentum%' THEN 'momentum'
                    WHEN strat LIKE '%mean_reversion%' THEN 'mean_reversion'
                    WHEN strat LIKE '%breakout%' THEN 'breakout'
                END as strategy_type
            FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet')
            WHERE val = 1  -- Long entries only
        ),
        exit_performance AS (
            SELECT 
                e.strategy_type,
                -- Fixed bar exits
                AVG((m1.close - m0.close) / m0.close * 100) as return_1bar,
                AVG((m5.close - m0.close) / m0.close * 100) as return_5bar,
                AVG((m10.close - m0.close) / m0.close * 100) as return_10bar,
                -- Exit on opposite signal
                AVG(CASE 
                    WHEN opp.idx IS NOT NULL THEN (m_opp.close - m0.close) / m0.close * 100
                    ELSE NULL
                END) as return_opposite_signal,
                -- Exit on stop loss
                AVG(CASE 
                    WHEN (m_low.low - m0.close) / m0.close < -0.002 
                    THEN -0.2  -- 0.2% stop loss
                    ELSE (m5.close - m0.close) / m0.close * 100
                END) as return_with_stop,
                COUNT(*) as total_signals
            FROM entry_signals e
            JOIN read_parquet('{data_path}') m0 ON e.entry_idx = m0.bar_index
            LEFT JOIN read_parquet('{data_path}') m1 ON e.entry_idx + 1 = m1.bar_index
            LEFT JOIN read_parquet('{data_path}') m5 ON e.entry_idx + 5 = m5.bar_index
            LEFT JOIN read_parquet('{data_path}') m10 ON e.entry_idx + 10 = m10.bar_index
            -- Find opposite signal from same strategy
            LEFT JOIN LATERAL (
                SELECT idx, strat 
                FROM read_parquet('{self.workspace_path}/traces/*/signals/*/*.parquet')
                WHERE strat = e.strat AND val = -1 AND idx > e.entry_idx
                ORDER BY idx
                LIMIT 1
            ) opp ON true
            LEFT JOIN read_parquet('{data_path}') m_opp ON opp.idx = m_opp.bar_index
            -- Get lowest low in next 5 bars
            LEFT JOIN LATERAL (
                SELECT MIN(low) as low
                FROM read_parquet('{data_path}')
                WHERE bar_index > e.entry_idx AND bar_index <= e.entry_idx + 5
            ) m_low ON true
            GROUP BY e.strategy_type
        )
        SELECT 
            strategy_type,
            total_signals,
            ROUND(return_1bar, 4) as avg_return_1bar,
            ROUND(return_5bar, 4) as avg_return_5bar,
            ROUND(return_10bar, 4) as avg_return_10bar,
            ROUND(return_opposite_signal, 4) as avg_return_opposite_signal,
            ROUND(return_with_stop, 4) as avg_return_with_stop
        FROM exit_performance
        WHERE total_signals >= 10
        ORDER BY strategy_type
        """
        
        return self.con.execute(query).df()
    
    def mine_all_patterns(self, data_path: str, output_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
        """
        Run all pattern mining analyses.
        """
        if output_dir is None:
            output_dir = self.workspace_path / "pattern_analysis"
        output_dir.mkdir(exist_ok=True)
        
        results = {}
        
        print("1. Mining signal combinations...")
        results['signal_combinations'] = self.mine_signal_combinations(data_path)
        
        print("2. Mining conditional patterns...")
        results['conditional_patterns'] = self.mine_conditional_patterns(data_path)
        
        print("3. Mining exit strategies...")
        results['exit_strategies'] = self.mine_exit_strategies(data_path)
        
        print("4. Mining regime transitions...")
        results['regime_transitions'] = self.mine_regime_transitions(data_path)
        
        # Save results
        for name, df in results.items():
            if not df.empty:
                df.to_csv(output_dir / f"{name}.csv", index=False)
                print(f"   {name}: {len(df)} patterns found")
        
        # Create insights summary
        insights = self._generate_insights(results)
        with open(output_dir / "insights.json", 'w') as f:
            json.dump(insights, f, indent=2)
        
        return results
    
    def _generate_insights(self, results: Dict[str, pd.DataFrame]) -> Dict:
        """Generate actionable insights from pattern mining."""
        insights = {
            'best_signal_combinations': [],
            'conditional_improvements': [],
            'optimal_exits': [],
            'regime_specialists': []
        }
        
        # Best signal combinations
        if not results['signal_combinations'].empty:
            top_combos = results['signal_combinations'].nlargest(5, 'avg_return')
            for _, row in top_combos.iterrows():
                insights['best_signal_combinations'].append({
                    'entry': row['entry_type'],
                    'exit': f"{row['exit_type']} ({row['exit_signal']})",
                    'holding_period': int(row['holding_period']),
                    'avg_return': float(row['avg_return']),
                    'occurrences': int(row['occurrences'])
                })
        
        # Conditional improvements
        if not results['conditional_patterns'].empty:
            top_conditions = results['conditional_patterns'].nlargest(5, 'avg_return')
            for _, row in top_conditions.iterrows():
                insights['conditional_improvements'].append({
                    'strategy': row['primary_strat'],
                    'condition': row['context_type'],
                    'improvement': float(row['avg_return']),
                    'trades': int(row['trades'])
                })
        
        # Optimal exits by strategy type
        if not results['exit_strategies'].empty:
            for _, row in results['exit_strategies'].iterrows():
                best_exit = max([
                    ('1_bar', row['avg_return_1bar']),
                    ('5_bar', row['avg_return_5bar']),
                    ('10_bar', row['avg_return_10bar']),
                    ('opposite_signal', row['avg_return_opposite_signal']),
                    ('with_stop', row['avg_return_with_stop'])
                ], key=lambda x: x[1] if pd.notna(x[1]) else -999)
                
                insights['optimal_exits'].append({
                    'strategy_type': row['strategy_type'],
                    'best_exit': best_exit[0],
                    'expected_return': float(best_exit[1]) if pd.notna(best_exit[1]) else None
                })
        
        return insights
    
    def close(self):
        """Close database connection."""
        self.con.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pattern_miner.py <workspace_path> <data_path>")
        sys.exit(1)
    
    miner = PatternMiner(sys.argv[1])
    try:
        results = miner.mine_all_patterns(sys.argv[2])
        print("\nPattern mining complete!")
    finally:
        miner.close()
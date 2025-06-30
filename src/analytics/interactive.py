#!/usr/bin/env python3
"""
Interactive analysis tools for ADMF-PC.

Provides a Python-first interface for exploratory analysis of backtest results,
focusing on reusable queries and analysis patterns.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class BacktestRun:
    """Represents a backtest run with convenient access to data."""
    run_dir: Path
    config: Dict[str, Any]
    
    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        if not self.run_dir.exists():
            raise ValueError(f"Run directory {self.run_dir} does not exist")
        
        # Load config if not provided
        if not self.config:
            config_path = self.run_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    self.config = json.load(f)
        
        # Initialize DuckDB connection
        self.db = duckdb.connect(":memory:")
        self._setup_views()
    
    def _setup_views(self):
        """Create DuckDB views for easy querying."""
        # Create view for strategy index
        strategy_index = self.run_dir / "strategy_index.parquet"
        if strategy_index.exists():
            self.db.execute(f"""
                CREATE OR REPLACE VIEW strategies AS 
                SELECT * FROM read_parquet('{strategy_index}')
            """)
        
        # Create view for all trace files
        trace_pattern = str(self.run_dir / "traces/*/*.parquet")
        self.db.execute(f"""
            CREATE OR REPLACE VIEW signals AS
            SELECT * FROM read_parquet('{trace_pattern}')
        """)
    
    @property
    def strategies(self) -> pd.DataFrame:
        """Get all strategies in this run."""
        return self.db.execute("SELECT * FROM strategies").df()
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Get run summary statistics."""
        strategies_df = self.strategies
        return {
            'run_id': self.run_dir.name,
            'config_name': self.config.get('name', 'unnamed'),
            'total_strategies': len(strategies_df),
            'strategy_types': strategies_df['strategy_type'].unique().tolist(),
            'timeframe': self.config.get('timeframe', 'unknown'),
            'symbols': self.config.get('symbols', []),
            'date_range': self.config.get('date_range', {}),
            'best_sharpe': strategies_df['sharpe_ratio'].max() if 'sharpe_ratio' in strategies_df else None,
            'avg_sharpe': strategies_df['sharpe_ratio'].mean() if 'sharpe_ratio' in strategies_df else None
        }
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute arbitrary SQL query."""
        return self.db.execute(sql).df()


class AnalysisWorkspace:
    """
    Main workspace for interactive analysis.
    
    Example:
        workspace = AnalysisWorkspace()
        run = workspace.load_run("results/run_20250623_143030")
        
        # Get top strategies
        top = run.top_strategies(10)
        
        # Analyze correlations
        corr = workspace.correlation_matrix(run, top)
        
        # Find ensemble candidates
        ensemble = workspace.find_ensemble(run, size=5)
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.query_library = QueryLibrary()
        self.pattern_library = PatternLibrary()
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all available backtest runs."""
        runs = []
        for run_dir in sorted(self.results_dir.glob("run_*"), reverse=True):
            if (run_dir / "strategy_index.parquet").exists():
                try:
                    run = BacktestRun(run_dir, {})
                    runs.append(run.summary)
                except Exception as e:
                    print(f"Failed to load {run_dir}: {e}")
        return runs
    
    def load_run(self, run_path: Union[str, Path]) -> BacktestRun:
        """Load a specific backtest run."""
        return BacktestRun(run_path, {})
    
    def compare_runs(self, run_paths: List[Union[str, Path]]) -> pd.DataFrame:
        """Compare multiple backtest runs."""
        comparisons = []
        for path in run_paths:
            run = self.load_run(path)
            summary = run.summary
            comparisons.append(summary)
        return pd.DataFrame(comparisons)
    
    def top_strategies(self, run: BacktestRun, n: int = 10, metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """Get top N strategies by a metric."""
        strategies = run.strategies
        if metric not in strategies.columns:
            raise ValueError(f"Metric {metric} not found in strategies")
        return strategies.nlargest(n, metric)
    
    def correlation_matrix(self, run: BacktestRun, strategies: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for given strategies."""
        # Load signals for each strategy
        signals_dict = {}
        for _, strategy in strategies.iterrows():
            trace_path = run.run_dir / strategy['trace_path']
            if trace_path.exists():
                signals = pd.read_parquet(trace_path)
                signals['ts'] = pd.to_datetime(signals['ts'])
                signals = signals.set_index('ts')['val']
                signals_dict[strategy['strategy_hash']] = signals
        
        # Create aligned DataFrame
        signals_df = pd.DataFrame(signals_dict)
        signals_df = signals_df.fillna(method='ffill').fillna(0)
        
        return signals_df.corr()
    
    def find_ensemble(self, run: BacktestRun, size: int = 5, 
                     correlation_threshold: float = 0.7) -> Dict[str, Any]:
        """Find optimal ensemble of uncorrelated strategies."""
        # Get top performers
        top = self.top_strategies(run, n=20)
        
        # Calculate correlations
        corr_matrix = self.correlation_matrix(run, top)
        
        # Find uncorrelated strategies
        selected = []
        candidates = top['strategy_hash'].tolist()
        
        while len(selected) < size and candidates:
            # Add first candidate
            if not selected:
                selected.append(candidates.pop(0))
                continue
            
            # Find least correlated candidate
            min_corr = 1.0
            best_candidate = None
            
            for candidate in candidates:
                max_corr_with_selected = max(
                    abs(corr_matrix.loc[candidate, s]) for s in selected
                )
                if max_corr_with_selected < min_corr:
                    min_corr = max_corr_with_selected
                    best_candidate = candidate
            
            if best_candidate and min_corr < correlation_threshold:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                break
        
        # Return ensemble info
        ensemble_strategies = top[top['strategy_hash'].isin(selected)]
        return {
            'strategies': ensemble_strategies,
            'avg_sharpe': ensemble_strategies['sharpe_ratio'].mean(),
            'max_correlation': corr_matrix.loc[selected, selected].values[
                ~np.eye(len(selected), dtype=bool)
            ].max() if len(selected) > 1 else 0
        }
    
    def analyze_parameters(self, run: BacktestRun, strategy_type: str) -> pd.DataFrame:
        """Analyze parameter sensitivity for a strategy type."""
        strategies = run.strategies[run.strategies['strategy_type'] == strategy_type]
        
        # Extract parameter columns
        param_cols = [col for col in strategies.columns if col.startswith('param_')]
        
        analysis = []
        for param in param_cols:
            if strategies[param].dtype in ['int64', 'float64']:
                analysis.append({
                    'parameter': param,
                    'min': strategies[param].min(),
                    'max': strategies[param].max(),
                    'correlation_with_sharpe': strategies[[param, 'sharpe_ratio']].corr().iloc[0, 1],
                    'best_value': strategies.loc[strategies['sharpe_ratio'].idxmax(), param]
                })
        
        return pd.DataFrame(analysis)


class QueryLibrary:
    """Library of reusable analysis queries."""
    
    @staticmethod
    def signal_frequency(run: BacktestRun) -> pd.DataFrame:
        """Analyze signal frequency by strategy."""
        return run.query("""
            WITH signal_stats AS (
                SELECT 
                    strategy_hash,
                    COUNT(*) as total_signals,
                    COUNT(DISTINCT DATE(ts)) as trading_days,
                    SUM(CASE WHEN val > 0 THEN 1 ELSE 0 END) as long_signals,
                    SUM(CASE WHEN val < 0 THEN 1 ELSE 0 END) as short_signals
                FROM signals
                WHERE val != 0
                GROUP BY strategy_hash
            )
            SELECT 
                s.*,
                st.strategy_type,
                st.sharpe_ratio,
                s.total_signals::FLOAT / s.trading_days as signals_per_day
            FROM signal_stats s
            JOIN strategies st ON s.strategy_hash = st.strategy_hash
            ORDER BY st.sharpe_ratio DESC
        """)
    
    @staticmethod
    def intraday_patterns(run: BacktestRun) -> pd.DataFrame:
        """Find intraday trading patterns."""
        return run.query("""
            SELECT 
                EXTRACT(HOUR FROM ts) as hour,
                COUNT(*) as signal_count,
                COUNT(DISTINCT strategy_hash) as active_strategies,
                AVG(CASE WHEN val > 0 THEN 1 WHEN val < 0 THEN -1 ELSE 0 END) as avg_direction
            FROM signals
            WHERE val != 0
            GROUP BY hour
            ORDER BY hour
        """)
    
    @staticmethod
    def regime_performance(run: BacktestRun, volatility_threshold: float = 0.02) -> pd.DataFrame:
        """Analyze performance in different volatility regimes."""
        # This is a simplified version - in practice you'd join with market data
        return run.query(f"""
            WITH hourly_vol AS (
                SELECT 
                    DATE_TRUNC('hour', ts) as hour,
                    STDDEV(val) as volatility
                FROM signals
                GROUP BY hour
            ),
            regime_signals AS (
                SELECT 
                    s.*,
                    CASE 
                        WHEN hv.volatility > {volatility_threshold} THEN 'high_vol'
                        ELSE 'low_vol'
                    END as regime
                FROM signals s
                JOIN hourly_vol hv ON DATE_TRUNC('hour', s.ts) = hv.hour
            )
            SELECT 
                strategy_hash,
                regime,
                COUNT(*) as signals_in_regime,
                AVG(val) as avg_signal_strength
            FROM regime_signals
            WHERE val != 0
            GROUP BY strategy_hash, regime
        """)


class PatternLibrary:
    """Library for saving and retrieving discovered patterns."""
    
    def __init__(self, library_path: str = "analytics_patterns.json"):
        self.library_path = Path(library_path)
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, Any]:
        """Load saved patterns."""
        if self.library_path.exists():
            with open(self.library_path) as f:
                return json.load(f)
        return {}
    
    def save_pattern(self, name: str, pattern: Dict[str, Any]):
        """Save a discovered pattern."""
        self.patterns[name] = {
            'pattern': pattern,
            'discovered': datetime.now().isoformat(),
            'usage_count': 0
        }
        self._save_patterns()
    
    def get_pattern(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a saved pattern."""
        if name in self.patterns:
            self.patterns[name]['usage_count'] += 1
            self._save_patterns()
            return self.patterns[name]['pattern']
        return None
    
    def list_patterns(self) -> List[Dict[str, Any]]:
        """List all saved patterns."""
        return [
            {
                'name': name,
                'discovered': data['discovered'],
                'usage_count': data['usage_count']
            }
            for name, data in self.patterns.items()
        ]
    
    def _save_patterns(self):
        """Save patterns to disk."""
        with open(self.library_path, 'w') as f:
            json.dump(self.patterns, f, indent=2)


def quick_analysis(run_path: str):
    """Quick analysis function for command line usage."""
    workspace = AnalysisWorkspace()
    run = workspace.load_run(run_path)
    
    print(f"\nRun Summary:")
    for key, value in run.summary.items():
        print(f"  {key}: {value}")
    
    print(f"\nTop 5 Strategies:")
    top = workspace.top_strategies(run, n=5)
    print(top[['strategy_type', 'strategy_hash', 'sharpe_ratio', 'total_return']])
    
    print(f"\nOptimal Ensemble:")
    ensemble = workspace.find_ensemble(run, size=5)
    print(f"  Average Sharpe: {ensemble['avg_sharpe']:.2f}")
    print(f"  Max Correlation: {ensemble['max_correlation']:.2f}")
    print("\nEnsemble Strategies:")
    print(ensemble['strategies'][['strategy_type', 'sharpe_ratio']])


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        quick_analysis(sys.argv[1])
    else:
        print("Usage: python interactive.py <run_directory>")

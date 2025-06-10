"""Two-layer mining implementation for optimization analysis."""

from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
import pandas as pd
import sqlite3
import json
import logging

from src.core.events.types import Event
from src.core.events.protocols import EventStorageProtocol
from .models import OptimizationRun
from .query import EventQueryInterface

logger = logging.getLogger(__name__)


class TwoLayerMiningSystem:
    """
    Two-layer data mining system for optimization analysis.
    
    Layer 1: SQL database for metrics and optimization results
    Layer 2: Event traces for detailed analysis
    
    This implements the architecture described in docs/architecture/data-mining-architecture.md
    """
    
    def __init__(self, db_path: Path, event_storage: EventStorageProtocol):
        self.db_path = db_path
        self.event_storage = event_storage
        self.query_interface = EventQueryInterface(event_storage)
        self._init_database()
        
    def _init_database(self) -> None:
        """Initialize SQL database schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Optimization runs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS optimization_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp DATETIME,
                    parameters TEXT,  -- JSON
                    objective_value REAL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    num_trades INTEGER
                )
            """)
            
            # Parameter performance table for quick lookups
            conn.execute("""
                CREATE TABLE IF NOT EXISTS parameter_performance (
                    parameter_name TEXT,
                    parameter_value TEXT,
                    avg_objective REAL,
                    num_runs INTEGER,
                    best_objective REAL,
                    worst_objective REAL,
                    PRIMARY KEY (parameter_name, parameter_value)
                )
            """)
            
            # Pattern discovery table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS discovered_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    description TEXT,
                    frequency INTEGER,
                    avg_return REAL,
                    confidence REAL,
                    discovered_at DATETIME
                )
            """)
            
    def record_optimization_run(self, run: OptimizationRun) -> None:
        """Record optimization run in SQL database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO optimization_runs 
                (run_id, timestamp, parameters, objective_value, 
                 total_return, sharpe_ratio, max_drawdown, win_rate, num_trades)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.run_id,
                run.timestamp,
                json.dumps(run.parameters),
                run.objective_value,
                run.metrics.get('total_return', 0),
                run.metrics.get('sharpe_ratio', 0),
                run.metrics.get('max_drawdown', 0),
                run.metrics.get('win_rate', 0),
                run.metrics.get('num_trades', 0)
            ))
            
            # Update parameter performance
            for param_name, param_value in run.parameters.items():
                self._update_parameter_performance(conn, param_name, 
                                                 str(param_value), 
                                                 run.objective_value)
                                                 
    def _update_parameter_performance(self, conn: sqlite3.Connection,
                                    param_name: str, param_value: str,
                                    objective_value: float) -> None:
        """Update parameter performance statistics."""
        # Check if exists
        cursor = conn.execute("""
            SELECT avg_objective, num_runs, best_objective, worst_objective
            FROM parameter_performance
            WHERE parameter_name = ? AND parameter_value = ?
        """, (param_name, param_value))
        
        row = cursor.fetchone()
        if row:
            # Update existing
            avg_obj, num_runs, best_obj, worst_obj = row
            new_avg = (avg_obj * num_runs + objective_value) / (num_runs + 1)
            new_best = max(best_obj, objective_value)
            new_worst = min(worst_obj, objective_value)
            
            conn.execute("""
                UPDATE parameter_performance
                SET avg_objective = ?, num_runs = ?, 
                    best_objective = ?, worst_objective = ?
                WHERE parameter_name = ? AND parameter_value = ?
            """, (new_avg, num_runs + 1, new_best, new_worst, 
                  param_name, param_value))
        else:
            # Insert new
            conn.execute("""
                INSERT INTO parameter_performance
                (parameter_name, parameter_value, avg_objective, 
                 num_runs, best_objective, worst_objective)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (param_name, param_value, objective_value, 1, 
                  objective_value, objective_value))
                  
    def find_best_parameters(self, top_n: int = 10) -> pd.DataFrame:
        """Find best performing parameter combinations."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT run_id, parameters, objective_value, 
                       total_return, sharpe_ratio, num_trades
                FROM optimization_runs
                ORDER BY objective_value DESC
                LIMIT ?
            """, conn, params=(top_n,))
            
        # Parse parameters JSON
        df['parameters'] = df['parameters'].apply(json.loads)
        return df
        
    def analyze_parameter_sensitivity(self) -> pd.DataFrame:
        """Analyze sensitivity of objective to each parameter."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("""
                SELECT parameter_name, 
                       COUNT(DISTINCT parameter_value) as num_values,
                       MAX(avg_objective) - MIN(avg_objective) as objective_range,
                       AVG(avg_objective) as avg_objective
                FROM parameter_performance
                GROUP BY parameter_name
                ORDER BY objective_range DESC
            """, conn)
            
        return df
        
    def discover_patterns(self, min_frequency: int = 5) -> List[Dict[str, Any]]:
        """
        Discover patterns by analyzing event traces.
        
        This bridges Layer 1 (metrics) with Layer 2 (events).
        """
        # Get top performing runs
        best_runs = self.find_best_parameters(top_n=20)
        patterns = []
        
        for _, run in best_runs.iterrows():
            run_id = run['run_id']
            
            # Query events for this run
            events = self.event_storage.query({'run_id': run_id})
            
            # Look for specific patterns
            signal_to_fill_times = self._analyze_signal_to_fill(events)
            if len(signal_to_fill_times) >= min_frequency:
                avg_time = sum(signal_to_fill_times) / len(signal_to_fill_times)
                patterns.append({
                    'pattern_type': 'signal_to_fill_latency',
                    'run_id': run_id,
                    'frequency': len(signal_to_fill_times),
                    'avg_latency_ms': avg_time,
                    'parameters': run['parameters']
                })
                
        return patterns
        
    def _analyze_signal_to_fill(self, events: List[Event]) -> List[float]:
        """Analyze signal to fill latencies."""
        latencies = []
        
        # Group by correlation ID
        from collections import defaultdict
        correlated = defaultdict(list)
        for event in events:
            if event.correlation_id:
                correlated[event.correlation_id].append(event)
                
        # Find signal → fill patterns
        for correlation_id, event_group in correlated.items():
            event_group.sort(key=lambda e: e.timestamp)
            
            signal_event = None
            fill_event = None
            
            for event in event_group:
                if event.event_type == 'SIGNAL' and not signal_event:
                    signal_event = event
                elif event.event_type == 'FILL' and signal_event:
                    fill_event = event
                    break
                    
            if signal_event and fill_event:
                latency = (fill_event.timestamp - signal_event.timestamp).total_seconds() * 1000
                latencies.append(latency)
                
        return latencies
        
    def create_optimization_report(self) -> Dict[str, Any]:
        """Create comprehensive optimization report."""
        report = {
            'summary': {},
            'best_parameters': {},
            'sensitivity_analysis': {},
            'discovered_patterns': []
        }
        
        # Summary statistics
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) as total_runs,
                       MAX(objective_value) as best_objective,
                       AVG(objective_value) as avg_objective,
                       MIN(objective_value) as worst_objective
                FROM optimization_runs
            """)
            report['summary'] = dict(cursor.fetchone())
            
        # Best parameters
        best_params_df = self.find_best_parameters(top_n=5)
        report['best_parameters'] = best_params_df.to_dict('records')
        
        # Sensitivity analysis
        sensitivity_df = self.analyze_parameter_sensitivity()
        report['sensitivity_analysis'] = sensitivity_df.to_dict('records')
        
        # Pattern discovery
        report['discovered_patterns'] = self.discover_patterns()
        
        return report
#!/usr/bin/env python3
"""
Extract performance metrics from Parquet signal traces into DuckDB for analysis.

This script:
1. Scans all Parquet files in a workspace
2. Extracts signal data and metadata
3. Loads everything into DuckDB tables
4. Provides SQL interface for performance analysis
"""

import os
import json
import duckdb
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Any
import argparse

class PerformanceMetricsExtractor:
    def __init__(self, db_path: str = "analytics.duckdb"):
        """Initialize the extractor with DuckDB connection."""
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._setup_tables()
    
    def _setup_tables(self):
        """Create the analytics tables."""
        # Main signals table - all signal changes
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_changes (
                workspace_id VARCHAR,
                component_id VARCHAR,
                component_type VARCHAR,  -- 'strategy' or 'classifier'
                strategy_type VARCHAR,   -- 'rsi_strategy', 'trend_classifier', etc.
                symbol VARCHAR,
                bar_index INTEGER,
                timestamp TIMESTAMP,
                signal_value VARCHAR,    -- Can be numeric (-1,0,1) or categorical ('ranging', 'trending')
                price DOUBLE,
                signal_file_path VARCHAR
            )
        """)
        
        # Component metadata table - performance stats per component
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS component_metrics (
                workspace_id VARCHAR,
                component_id VARCHAR,
                component_type VARCHAR,
                strategy_type VARCHAR,
                symbol VARCHAR,
                total_bars INTEGER,
                signal_changes INTEGER,
                compression_ratio DOUBLE,
                signal_frequency DOUBLE,
                
                -- Strategy-specific metrics
                total_positions INTEGER,
                avg_position_duration DOUBLE,
                long_positions INTEGER,
                short_positions INTEGER,
                flat_positions INTEGER,
                
                -- Classifier-specific metrics
                regime_classifications JSON,
                
                -- File metadata
                signal_file_path VARCHAR,
                created_at TIMESTAMP
            )
        """)
        
        # Workspace summary table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS workspace_summary (
                workspace_id VARCHAR PRIMARY KEY,
                config_name VARCHAR,
                total_strategies INTEGER,
                total_classifiers INTEGER,
                total_components INTEGER,
                total_signals INTEGER,
                total_classifications INTEGER,
                stored_changes INTEGER,
                overall_compression_ratio DOUBLE,
                created_at TIMESTAMP
            )
        """)
        
        print(f"Analytics database initialized: {self.db_path}")
    
    def extract_workspace(self, workspace_path: str) -> str:
        """Extract all performance metrics from a workspace."""
        workspace_path = Path(workspace_path)
        workspace_id = workspace_path.name
        
        print(f"Extracting workspace: {workspace_id}")
        
        # Read workspace metadata
        metadata_path = workspace_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata.json found in {workspace_path}")
        
        with open(metadata_path, 'r') as f:
            workspace_metadata = json.load(f)
        
        # Process all Parquet files
        traces_dir = workspace_path / "traces"
        if not traces_dir.exists():
            raise FileNotFoundError(f"No traces directory found in {workspace_path}")
        
        signal_changes_data = []
        component_metrics_data = []
        
        # Scan all Parquet files
        for parquet_file in traces_dir.rglob("*.parquet"):
            try:
                # Read the signal data
                df = pd.read_parquet(parquet_file)
                
                # Read metadata
                table = pq.read_table(parquet_file)
                metadata = {}
                if table.schema.metadata:
                    for key, value in table.schema.metadata.items():
                        key_str = key.decode() if isinstance(key, bytes) else str(key)
                        value_str = value.decode() if isinstance(value, bytes) else str(value)
                        metadata[key_str] = value_str
                
                # Determine component type and strategy type from file path
                relative_path = parquet_file.relative_to(traces_dir)
                if relative_path.parts[0] == "signals":
                    component_type = "strategy"
                    strategy_type = relative_path.parts[1] if len(relative_path.parts) > 1 else "unknown"
                elif relative_path.parts[0] == "classifiers":
                    component_type = "classifier"
                    strategy_type = relative_path.parts[1] if len(relative_path.parts) > 1 else "unknown"
                else:
                    component_type = "unknown"
                    strategy_type = "unknown"
                
                component_id = parquet_file.stem  # Filename without extension
                
                # Extract signal changes
                for _, row in df.iterrows():
                    signal_changes_data.append({
                        'workspace_id': workspace_id,
                        'component_id': component_id,
                        'component_type': component_type,
                        'strategy_type': strategy_type,
                        'symbol': row.get('sym', 'unknown'),
                        'bar_index': row.get('idx', 0),
                        'timestamp': pd.to_datetime(row.get('ts')),
                        'signal_value': str(row.get('val', '')),
                        'price': float(row.get('px', 0.0)),
                        'signal_file_path': str(parquet_file)
                    })
                
                # Extract component metrics
                signal_stats = json.loads(metadata.get('signal_statistics', '{}'))
                strategies_info = json.loads(metadata.get('strategies', '{}'))
                
                # Get the first (and usually only) strategy info
                first_strategy_key = list(strategies_info.keys())[0] if strategies_info else component_id
                symbol = first_strategy_key.split('_')[0] if '_' in first_strategy_key else 'SPY'
                
                component_metrics_data.append({
                    'workspace_id': workspace_id,
                    'component_id': component_id,
                    'component_type': component_type,
                    'strategy_type': strategy_type,
                    'symbol': symbol,
                    'total_bars': int(metadata.get('total_bars', 0)),
                    'signal_changes': int(metadata.get('total_changes', 0)),
                    'compression_ratio': float(metadata.get('compression_ratio', 0.0)),
                    'signal_frequency': signal_stats.get('signal_frequency', 0.0),
                    
                    # Strategy metrics
                    'total_positions': signal_stats.get('total_positions', 0),
                    'avg_position_duration': signal_stats.get('avg_position_duration', 0.0),
                    'long_positions': signal_stats.get('position_breakdown', {}).get('long', 0),
                    'short_positions': signal_stats.get('position_breakdown', {}).get('short', 0),
                    'flat_positions': signal_stats.get('position_breakdown', {}).get('flat', 0),
                    
                    # Classifier metrics
                    'regime_classifications': json.dumps(signal_stats.get('regime_breakdown', {})),
                    
                    'signal_file_path': str(parquet_file),
                    'created_at': pd.to_datetime(metadata.get('created_at', '2025-01-01'))
                })
                
            except Exception as e:
                print(f"Error processing {parquet_file}: {e}")
                continue
        
        # Insert signal changes
        if signal_changes_data:
            signal_changes_df = pd.DataFrame(signal_changes_data)
            self.conn.execute("DELETE FROM signal_changes WHERE workspace_id = ?", [workspace_id])
            self.conn.execute("INSERT INTO signal_changes SELECT * FROM signal_changes_df")
            print(f"Inserted {len(signal_changes_data)} signal changes")
        
        # Insert component metrics
        if component_metrics_data:
            component_metrics_df = pd.DataFrame(component_metrics_data)
            self.conn.execute("DELETE FROM component_metrics WHERE workspace_id = ?", [workspace_id])
            self.conn.execute("INSERT INTO component_metrics SELECT * FROM component_metrics_df")
            print(f"Inserted {len(component_metrics_data)} component metrics")
        
        # Insert workspace summary
        workspace_summary = {
            'workspace_id': workspace_id,
            'config_name': workspace_metadata.get('workflow_id', 'unknown'),
            'total_strategies': len([c for c in component_metrics_data if c['component_type'] == 'strategy']),
            'total_classifiers': len([c for c in component_metrics_data if c['component_type'] == 'classifier']),
            'total_components': len(component_metrics_data),
            'total_signals': workspace_metadata.get('total_signals', 0),
            'total_classifications': workspace_metadata.get('total_classifications', 0),
            'stored_changes': workspace_metadata.get('stored_changes', 0),
            'overall_compression_ratio': workspace_metadata.get('compression_ratio', 0.0),
            'created_at': pd.to_datetime('now')
        }
        
        workspace_summary_df = pd.DataFrame([workspace_summary])
        self.conn.execute("DELETE FROM workspace_summary WHERE workspace_id = ?", [workspace_id])
        self.conn.execute("INSERT INTO workspace_summary SELECT * FROM workspace_summary_df")
        
        print(f"Workspace {workspace_id} extraction complete!")
        return workspace_id
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        return self.conn.execute(sql).df()
    
    def get_top_strategies(self, limit: int = 10) -> pd.DataFrame:
        """Get top performing strategies by signal frequency."""
        return self.query(f"""
            SELECT 
                component_id,
                strategy_type,
                signal_frequency,
                total_positions,
                avg_position_duration,
                compression_ratio,
                long_positions,
                short_positions,
                flat_positions
            FROM component_metrics 
            WHERE component_type = 'strategy'
            ORDER BY signal_frequency DESC
            LIMIT {limit}
        """)
    
    def get_classifier_analysis(self) -> pd.DataFrame:
        """Analyze classifier regime detection patterns."""
        return self.query("""
            SELECT 
                component_id,
                strategy_type,
                signal_frequency,
                regime_classifications,
                total_bars,
                signal_changes
            FROM component_metrics 
            WHERE component_type = 'classifier'
            ORDER BY signal_frequency DESC
        """)
    
    def get_strategy_type_summary(self) -> pd.DataFrame:
        """Summarize performance by strategy type."""
        return self.query("""
            SELECT 
                strategy_type,
                COUNT(*) as count,
                AVG(signal_frequency) as avg_signal_frequency,
                AVG(compression_ratio) as avg_compression_ratio,
                AVG(total_positions) as avg_positions,
                AVG(avg_position_duration) as avg_duration,
                SUM(long_positions) as total_long,
                SUM(short_positions) as total_short,
                SUM(flat_positions) as total_flat
            FROM component_metrics 
            WHERE component_type = 'strategy'
            GROUP BY strategy_type
            ORDER BY avg_signal_frequency DESC
        """)
    
    def close(self):
        """Close the database connection."""
        self.conn.close()

def main():
    parser = argparse.ArgumentParser(description='Extract performance metrics from signal traces')
    parser.add_argument('workspace_path', help='Path to workspace directory')
    parser.add_argument('--db', default='analytics.duckdb', help='DuckDB database path')
    parser.add_argument('--query', help='SQL query to execute after extraction')
    
    args = parser.parse_args()
    
    # Extract metrics
    extractor = PerformanceMetricsExtractor(args.db)
    workspace_id = extractor.extract_workspace(args.workspace_path)
    
    print(f"\n📊 Extraction Summary:")
    summary = extractor.query(f"SELECT * FROM workspace_summary WHERE workspace_id = '{workspace_id}'")
    print(summary.to_string(index=False))
    
    # Show top strategies
    print(f"\n🏆 Top Strategies by Signal Frequency:")
    top_strategies = extractor.get_top_strategies(5)
    print(top_strategies.to_string(index=False))
    
    # Show strategy type summary
    print(f"\n📈 Strategy Type Performance Summary:")
    type_summary = extractor.get_strategy_type_summary()
    print(type_summary.to_string(index=False))
    
    # Custom query if provided
    if args.query:
        print(f"\n🔍 Custom Query Results:")
        results = extractor.query(args.query)
        print(results.to_string(index=False))
    
    extractor.close()
    print(f"\n✅ Analytics database saved to: {args.db}")

if __name__ == "__main__":
    main()
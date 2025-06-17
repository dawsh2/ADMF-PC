#!/usr/bin/env python3
"""
Comprehensive analysis of two-layer ensemble strategy performance.
Adapts to the actual database structure.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

class TwoLayerEnsembleAnalyzer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = duckdb.connect(str(db_path), read_only=True)
        
    def explore_database_structure(self):
        """Explore the actual database structure."""
        print("=== DETAILED DATABASE EXPLORATION ===\n")
        
        # Get all tables
        tables = self.conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
            ORDER BY table_name
        """).fetchall()
        
        for table in tables:
            table_name = table[0]
            print(f"Table: {table_name}")
            
            # Get columns
            columns = self.conn.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """).fetchall()
            
            print("  Columns:")
            for col_name, col_type in columns:
                print(f"    - {col_name}: {col_type}")
            
            # Get row count
            count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"  Rows: {count:,}")
            
            # Show sample data for small tables
            if count <= 10:
                print("  Sample data:")
                sample = self.conn.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchdf()
                for _, row in sample.iterrows():
                    print(f"    {dict(row)}")
            else:
                print("  Sample data (first 3 rows):")
                sample = self.conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchdf()
                for _, row in sample.iterrows():
                    print(f"    {dict(row)}")
            
            print()
        
    def analyze_strategies_table(self):
        """Analyze the strategies table."""
        print("=== STRATEGIES TABLE ANALYSIS ===\n")
        
        try:
            strategies = self.conn.execute("SELECT * FROM strategies").fetchdf()
            
            print(f"Total strategies: {len(strategies)}")
            print("\nStrategy details:")
            for _, row in strategies.iterrows():
                print(f"  - Strategy ID: {row.get('strategy_id', 'N/A')}")
                for col in strategies.columns:
                    if col != 'strategy_id':
                        print(f"    {col}: {row[col]}")
                print()
                
        except Exception as e:
            print(f"Error analyzing strategies table: {e}")
            
    def analyze_classifiers_table(self):
        """Analyze the classifiers table."""
        print("=== CLASSIFIERS TABLE ANALYSIS ===\n")
        
        try:
            classifiers = self.conn.execute("SELECT * FROM classifiers").fetchdf()
            
            print(f"Total classifiers: {len(classifiers)}")
            print("\nClassifier details:")
            for _, row in classifiers.iterrows():
                print(f"  - Classifier: {row.get('classifier_id', 'N/A')}")
                for col in classifiers.columns:
                    if col != 'classifier_id':
                        print(f"    {col}: {row[col]}")
                print()
                
        except Exception as e:
            print(f"Error analyzing classifiers table: {e}")
            
    def analyze_runs_table(self):
        """Analyze the runs table."""
        print("=== RUNS TABLE ANALYSIS ===\n")
        
        try:
            runs = self.conn.execute("SELECT * FROM runs").fetchdf()
            
            print(f"Total runs: {len(runs)}")
            print("\nRun details:")
            for _, row in runs.iterrows():
                print(f"  - Run ID: {row.get('run_id', 'N/A')}")
                for col in runs.columns:
                    if col != 'run_id':
                        value = row[col]
                        if isinstance(value, str) and len(value) > 100:
                            value = value[:100] + "..."
                        print(f"    {col}: {value}")
                print()
                
        except Exception as e:
            print(f"Error analyzing runs table: {e}")
            
    def analyze_metadata_table(self):
        """Analyze the analytics metadata table."""
        print("=== ANALYTICS METADATA ===\n")
        
        try:
            metadata = self.conn.execute("SELECT * FROM _analytics_metadata").fetchdf()
            
            print(f"Metadata entries: {len(metadata)}")
            for _, row in metadata.iterrows():
                for col in metadata.columns:
                    value = row[col]
                    if isinstance(value, str) and len(value) > 200:
                        value = value[:200] + "..."
                    print(f"  {col}: {value}")
                print()
                
        except Exception as e:
            print(f"Error analyzing metadata table: {e}")
            
    def check_for_time_series_data(self):
        """Look for actual performance/signal data in the database."""
        print("=== SEARCHING FOR TIME SERIES DATA ===\n")
        
        # Check if there are any other tables or views
        all_objects = self.conn.execute("""
            SELECT 
                table_name,
                table_type
            FROM information_schema.tables 
            WHERE table_schema = 'main'
            UNION ALL
            SELECT 
                table_name,
                'VIEW' as table_type
            FROM information_schema.views
            WHERE table_schema = 'main'
            ORDER BY table_name
        """).fetchall()
        
        print("All database objects:")
        for obj_name, obj_type in all_objects:
            print(f"  - {obj_name} ({obj_type})")
            
        # Check if strategies table contains time series data
        try:
            strategies = self.conn.execute("SELECT * FROM strategies LIMIT 1").fetchdf()
            
            for col in strategies.columns:
                sample_value = strategies[col].iloc[0]
                if isinstance(sample_value, str):
                    try:
                        # Try to parse as JSON to see if it contains time series
                        if sample_value.startswith('[') or sample_value.startswith('{'):
                            print(f"\nColumn '{col}' appears to contain structured data:")
                            if len(sample_value) > 500:
                                print(f"  Sample (first 500 chars): {sample_value[:500]}...")
                            else:
                                print(f"  Sample: {sample_value}")
                    except:
                        pass
                        
        except Exception as e:
            print(f"Error examining strategies for time series data: {e}")
            
        print()
        
    def extract_performance_data(self):
        """Try to extract performance data from available tables."""
        print("=== EXTRACTING PERFORMANCE DATA ===\n")
        
        try:
            # Get the strategies data and see if we can extract performance info
            strategies = self.conn.execute("SELECT * FROM strategies").fetchdf()
            
            for idx, row in strategies.iterrows():
                strategy_id = row.get('strategy_id', f'strategy_{idx}')
                print(f"Strategy: {strategy_id}")
                
                # Look for columns that might contain performance data
                performance_cols = []
                for col in strategies.columns:
                    if any(keyword in col.lower() for keyword in ['signal', 'return', 'pnl', 'performance', 'result']):
                        performance_cols.append(col)
                
                if performance_cols:
                    print(f"  Performance-related columns: {performance_cols}")
                    
                    for col in performance_cols:
                        value = row[col]
                        if pd.notna(value):
                            print(f"    {col}: {type(value)} - ", end="")
                            if isinstance(value, (int, float)):
                                print(f"{value}")
                            elif isinstance(value, str):
                                if len(value) > 100:
                                    print(f"{value[:100]}...")
                                else:
                                    print(f"{value}")
                            else:
                                print(f"{value}")
                else:
                    print("  No obvious performance columns found")
                    
                print()
                
        except Exception as e:
            print(f"Error extracting performance data: {e}")
            
    def analyze_specific_columns(self):
        """Analyze specific columns that might contain useful data."""
        print("=== DETAILED COLUMN ANALYSIS ===\n")
        
        try:
            # Check each table for interesting columns
            tables_to_analyze = ['strategies', 'classifiers', 'runs']
            
            for table_name in tables_to_analyze:
                print(f"Table: {table_name}")
                
                # Get all data
                data = self.conn.execute(f"SELECT * FROM {table_name}").fetchdf()
                
                for col in data.columns:
                    print(f"  Column: {col}")
                    
                    # Get some statistics
                    non_null_count = data[col].notna().sum()
                    null_count = data[col].isna().sum()
                    unique_count = data[col].nunique()
                    
                    print(f"    Non-null: {non_null_count}, Null: {null_count}, Unique: {unique_count}")
                    
                    # Show sample values
                    sample_values = data[col].dropna().head(3).tolist()
                    for i, val in enumerate(sample_values):
                        if isinstance(val, str) and len(val) > 150:
                            val = val[:150] + "..."
                        print(f"    Sample {i+1}: {val}")
                    
                    print()
                print()
                    
        except Exception as e:
            print(f"Error in detailed column analysis: {e}")
        
    def run_full_analysis(self):
        """Run complete analysis adapted to actual database structure."""
        print(f"Two-Layer Ensemble Strategy Analysis")
        print(f"Database: {self.db_path}")
        print(f"Analysis Time: {datetime.now()}")
        print("=" * 80 + "\n")
        
        self.explore_database_structure()
        self.analyze_strategies_table()
        self.analyze_classifiers_table()
        self.analyze_runs_table()
        self.analyze_metadata_table()
        self.check_for_time_series_data()
        self.extract_performance_data()
        self.analyze_specific_columns()
        
        print("\n" + "=" * 80)
        print("Analysis complete!")
        
    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    # Path to the analytics database
    db_path = Path("workspaces/two_layer_regime_ensemble_v1_4f71d9e1/analytics.duckdb")
    
    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return
        
    analyzer = TwoLayerEnsembleAnalyzer(db_path)
    
    try:
        analyzer.run_full_analysis()
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
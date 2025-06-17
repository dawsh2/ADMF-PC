#!/usr/bin/env python3
"""
Comprehensive analysis of two-layer ensemble strategy performance.
Examines signal frequency, regime distribution, and performance metrics.
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
        
    def get_table_info(self):
        """Get information about available tables."""
        print("=== DATABASE STRUCTURE ===\n")
        
        # Get all tables
        tables = self.conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
            ORDER BY table_name
        """).fetchall()
        
        print(f"Available tables: {len(tables)}")
        for table in tables:
            table_name = table[0]
            count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"  - {table_name}: {count:,} rows")
            
            # Get columns for key tables
            if table_name in ['signals', 'regime_states', 'strategy_metadata']:
                print(f"    Columns:")
                columns = self.conn.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position
                """).fetchall()
                for col_name, col_type in columns:
                    print(f"      - {col_name}: {col_type}")
        print()
        
    def analyze_regime_distribution(self):
        """Analyze regime distribution and transitions."""
        print("=== REGIME ANALYSIS ===\n")
        
        # Check if regime_states table exists
        try:
            # Get regime distribution
            regime_dist = self.conn.execute("""
                SELECT 
                    regime,
                    COUNT(*) as count,
                    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage,
                    MIN(timestamp) as first_seen,
                    MAX(timestamp) as last_seen
                FROM regime_states
                GROUP BY regime
                ORDER BY count DESC
            """).fetchdf()
            
            print("Regime Distribution:")
            for _, row in regime_dist.iterrows():
                print(f"  - {row['regime']}: {row['count']:,} ({row['percentage']:.2f}%)")
                print(f"    First: {row['first_seen']}, Last: {row['last_seen']}")
            
            # Analyze regime transitions
            print("\nRegime Transitions:")
            transitions = self.conn.execute("""
                WITH regime_changes AS (
                    SELECT 
                        timestamp,
                        regime,
                        LAG(regime) OVER (ORDER BY timestamp) as prev_regime
                    FROM regime_states
                )
                SELECT 
                    prev_regime,
                    regime,
                    COUNT(*) as transition_count
                FROM regime_changes
                WHERE prev_regime IS NOT NULL AND prev_regime != regime
                GROUP BY prev_regime, regime
                ORDER BY transition_count DESC
                LIMIT 10
            """).fetchdf()
            
            for _, row in transitions.iterrows():
                print(f"  - {row['prev_regime']} → {row['regime']}: {row['transition_count']} times")
                
        except Exception as e:
            print(f"No regime_states table found or error: {e}")
            
            # Try to extract regime info from signals
            print("\nExtracting regime info from signals table...")
            try:
                regime_info = self.conn.execute("""
                    SELECT 
                        strategy_id,
                        COUNT(DISTINCT timestamp) as total_bars,
                        COUNT(*) as total_signals,
                        SUM(CASE WHEN signal != 0 THEN 1 ELSE 0 END) as active_signals
                    FROM signals
                    WHERE strategy_id LIKE '%classifier%'
                    GROUP BY strategy_id
                """).fetchdf()
                
                if not regime_info.empty:
                    print("\nClassifier Activity:")
                    for _, row in regime_info.iterrows():
                        print(f"  - {row['strategy_id']}:")
                        print(f"    Total bars: {row['total_bars']:,}")
                        print(f"    Active signals: {row['active_signals']:,}")
            except:
                pass
        
        print()
        
    def analyze_strategy_performance(self):
        """Analyze performance metrics for each strategy."""
        print("=== STRATEGY PERFORMANCE ===\n")
        
        # Get unique strategies
        strategies = self.conn.execute("""
            SELECT DISTINCT strategy_id 
            FROM signals 
            ORDER BY strategy_id
        """).fetchall()
        
        print(f"Total strategies: {len(strategies)}\n")
        
        # Analyze each strategy
        for strategy_id in strategies:
            strategy_id = strategy_id[0]
            print(f"Strategy: {strategy_id}")
            
            # Get signal statistics
            signal_stats = self.conn.execute(f"""
                SELECT 
                    COUNT(*) as total_signals,
                    SUM(CASE WHEN signal = 1 THEN 1 ELSE 0 END) as long_signals,
                    SUM(CASE WHEN signal = -1 THEN 1 ELSE 0 END) as short_signals,
                    SUM(CASE WHEN signal = 0 THEN 1 ELSE 0 END) as flat_signals,
                    COUNT(DISTINCT timestamp) as unique_timestamps
                FROM signals
                WHERE strategy_id = '{strategy_id}'
            """).fetchone()
            
            total, long, short, flat, timestamps = signal_stats
            print(f"  - Total signals: {total:,}")
            print(f"  - Long: {long:,} ({long/total*100:.1f}%)")
            print(f"  - Short: {short:,} ({short/total*100:.1f}%)")
            print(f"  - Flat: {flat:,} ({flat/total*100:.1f}%)")
            print(f"  - Unique timestamps: {timestamps:,}")
            
            # Calculate signal changes
            signal_changes = self.conn.execute(f"""
                WITH signal_changes AS (
                    SELECT 
                        timestamp,
                        signal,
                        LAG(signal) OVER (ORDER BY timestamp) as prev_signal
                    FROM signals
                    WHERE strategy_id = '{strategy_id}'
                )
                SELECT COUNT(*) as changes
                FROM signal_changes
                WHERE prev_signal IS NOT NULL AND signal != prev_signal
            """).fetchone()[0]
            
            print(f"  - Signal changes: {signal_changes:,}")
            print(f"  - Change frequency: {signal_changes/timestamps*100:.2f}% of bars\n")
            
    def analyze_ensemble_performance(self):
        """Analyze ensemble-specific performance."""
        print("=== ENSEMBLE PERFORMANCE ===\n")
        
        # Identify ensemble strategies
        ensemble_strategies = self.conn.execute("""
            SELECT DISTINCT strategy_id 
            FROM signals 
            WHERE strategy_id LIKE '%ensemble%'
            ORDER BY strategy_id
        """).fetchall()
        
        if ensemble_strategies:
            print(f"Found {len(ensemble_strategies)} ensemble strategies\n")
            
            for strategy_id in ensemble_strategies:
                strategy_id = strategy_id[0]
                print(f"Ensemble: {strategy_id}")
                
                # Get performance metrics if available
                try:
                    # Check for returns data
                    returns_data = self.conn.execute(f"""
                        SELECT 
                            timestamp,
                            signal,
                            LAG(signal) OVER (ORDER BY timestamp) as prev_signal
                        FROM signals
                        WHERE strategy_id = '{strategy_id}'
                        ORDER BY timestamp
                    """).fetchdf()
                    
                    # Calculate basic metrics
                    signal_changes = (returns_data['signal'] != returns_data['prev_signal']).sum()
                    total_bars = len(returns_data)
                    
                    print(f"  - Total bars: {total_bars:,}")
                    print(f"  - Signal changes: {signal_changes:,}")
                    print(f"  - Avg holding period: {total_bars/max(signal_changes, 1):.1f} bars")
                    
                    # Signal distribution
                    signal_dist = returns_data['signal'].value_counts()
                    for signal, count in signal_dist.items():
                        print(f"  - Signal {signal}: {count:,} ({count/total_bars*100:.1f}%)")
                    
                except Exception as e:
                    print(f"  - Error analyzing ensemble: {e}")
                
                print()
        else:
            print("No ensemble strategies found in signals table")
            
    def analyze_baseline_comparison(self):
        """Compare performance across baseline strategies."""
        print("=== BASELINE STRATEGY COMPARISON ===\n")
        
        baseline_strategies = [
            'dema_crossover',
            'elder_ray', 
            'sma_crossover',
            'stochastic_crossover',
            'pivot_channel_bounces'
        ]
        
        comparison_data = []
        
        for base_strategy in baseline_strategies:
            # Find all instances of this baseline
            instances = self.conn.execute(f"""
                SELECT DISTINCT strategy_id 
                FROM signals 
                WHERE strategy_id LIKE '%{base_strategy}%'
                AND strategy_id NOT LIKE '%ensemble%'
                AND strategy_id NOT LIKE '%classifier%'
            """).fetchall()
            
            if instances:
                print(f"\n{base_strategy.upper()} ({len(instances)} instances):")
                
                for strategy_id in instances:
                    strategy_id = strategy_id[0]
                    
                    stats = self.conn.execute(f"""
                        WITH signal_data AS (
                            SELECT 
                                timestamp,
                                signal,
                                LAG(signal) OVER (ORDER BY timestamp) as prev_signal
                            FROM signals
                            WHERE strategy_id = '{strategy_id}'
                        )
                        SELECT 
                            COUNT(*) as total_bars,
                            SUM(CASE WHEN signal != prev_signal THEN 1 ELSE 0 END) as signal_changes,
                            SUM(CASE WHEN signal = 1 THEN 1 ELSE 0 END) as long_bars,
                            SUM(CASE WHEN signal = -1 THEN 1 ELSE 0 END) as short_bars,
                            SUM(CASE WHEN signal = 0 THEN 1 ELSE 0 END) as flat_bars
                        FROM signal_data
                    """).fetchone()
                    
                    total_bars, changes, long_bars, short_bars, flat_bars = stats
                    
                    print(f"  - {strategy_id}:")
                    print(f"    Total bars: {total_bars:,}")
                    print(f"    Signal changes: {changes:,} ({changes/total_bars*100:.2f}% of bars)")
                    print(f"    Long: {long_bars:,} ({long_bars/total_bars*100:.1f}%)")
                    print(f"    Short: {short_bars:,} ({short_bars/total_bars*100:.1f}%)")
                    print(f"    Flat: {flat_bars:,} ({flat_bars/total_bars*100:.1f}%)")
                    
                    comparison_data.append({
                        'base_strategy': base_strategy,
                        'strategy_id': strategy_id,
                        'total_bars': total_bars,
                        'signal_changes': changes,
                        'change_pct': changes/total_bars*100,
                        'long_pct': long_bars/total_bars*100,
                        'short_pct': short_bars/total_bars*100,
                        'flat_pct': flat_bars/total_bars*100
                    })
        
        # Summary statistics
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print("\n=== BASELINE SUMMARY ===")
            summary = df.groupby('base_strategy').agg({
                'signal_changes': 'mean',
                'change_pct': 'mean',
                'long_pct': 'mean',
                'short_pct': 'mean',
                'flat_pct': 'mean'
            }).round(2)
            
            print("\nAverage metrics by baseline strategy:")
            print(summary)
            
    def analyze_metadata(self):
        """Analyze strategy metadata if available."""
        print("\n=== STRATEGY METADATA ===\n")
        
        try:
            metadata = self.conn.execute("""
                SELECT * FROM strategy_metadata
                LIMIT 5
            """).fetchdf()
            
            if not metadata.empty:
                print("Sample metadata entries:")
                print(metadata)
                
                # Get summary statistics
                summary = self.conn.execute("""
                    SELECT 
                        COUNT(DISTINCT strategy_id) as unique_strategies,
                        SUM(total_bars) as total_bars_processed,
                        SUM(total_signals) as total_signals_generated,
                        SUM(signal_changes) as total_signal_changes,
                        SUM(classifier_changes) as total_classifier_changes,
                        AVG(compression_ratio) as avg_compression_ratio
                    FROM strategy_metadata
                """).fetchone()
                
                print("\nMetadata Summary:")
                print(f"  - Unique strategies: {summary[0]}")
                print(f"  - Total bars processed: {summary[1]:,}")
                print(f"  - Total signals: {summary[2]:,}")
                print(f"  - Total signal changes: {summary[3]:,}")
                print(f"  - Total classifier changes: {summary[4]:,}")
                print(f"  - Avg compression ratio: {summary[5]:.2f}x")
                
        except Exception as e:
            print(f"No metadata table found or error: {e}")
            
    def run_full_analysis(self):
        """Run complete analysis."""
        print(f"Analyzing Two-Layer Ensemble Strategy")
        print(f"Database: {self.db_path}")
        print(f"Analysis Time: {datetime.now()}")
        print("=" * 80 + "\n")
        
        self.get_table_info()
        self.analyze_regime_distribution()
        self.analyze_strategy_performance()
        self.analyze_ensemble_performance()
        self.analyze_baseline_comparison()
        self.analyze_metadata()
        
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
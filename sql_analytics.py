#!/usr/bin/env python3
"""
Interactive SQL analytics interface for signal performance data.
"""

import duckdb
import pandas as pd
import sys

class SignalAnalytics:
    def __init__(self, db_path: str = "analytics.duckdb"):
        self.conn = duckdb.connect(db_path)
        self.conn.execute("SET memory_limit='1GB'")
        
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return DataFrame."""
        try:
            return self.conn.execute(sql).df()
        except Exception as e:
            print(f"SQL Error: {e}")
            return pd.DataFrame()
    
    def show_tables(self):
        """Show available tables."""
        tables = self.query("SHOW TABLES")
        print("📋 Available Tables:")
        for table in tables['name']:
            print(f"  • {table}")
            
    def describe_table(self, table_name: str):
        """Describe a table schema."""
        schema = self.query(f"DESCRIBE {table_name}")
        print(f"\n📊 {table_name} Schema:")
        print(schema.to_string(index=False))
        
        # Show sample data
        sample = self.query(f"SELECT * FROM {table_name} LIMIT 3")
        print(f"\n🔍 Sample Data:")
        print(sample.to_string(index=False))
    
    def strategy_performance_report(self):
        """Generate comprehensive strategy performance report."""
        print("=" * 80)
        print("🏆 STRATEGY PERFORMANCE REPORT")
        print("=" * 80)
        
        # Overall strategy stats
        overall = self.query("""
            SELECT 
                COUNT(*) as total_strategies,
                AVG(signal_frequency) as avg_signal_freq,
                AVG(compression_ratio) as avg_compression,
                AVG(total_positions) as avg_positions,
                MAX(signal_frequency) as max_signal_freq,
                MIN(signal_frequency) as min_signal_freq
            FROM component_metrics 
            WHERE component_type = 'strategy'
        """)
        
        print("\n📈 Overall Statistics:")
        print(overall.to_string(index=False))
        
        # Top performers
        print("\n🥇 Top 5 Most Active Strategies:")
        top_active = self.query("""
            SELECT 
                component_id,
                strategy_type,
                signal_frequency,
                total_positions,
                avg_position_duration,
                ROUND(100.0 * compression_ratio, 2) as compression_pct
            FROM component_metrics 
            WHERE component_type = 'strategy'
            ORDER BY signal_frequency DESC
            LIMIT 5
        """)
        print(top_active.to_string(index=False))
        
        # RSI parameter analysis
        print("\n📊 RSI Strategy Parameter Analysis:")
        rsi_analysis = self.query("""
            SELECT 
                CASE 
                    WHEN component_id LIKE '%_7_%' THEN 'RSI_7'
                    WHEN component_id LIKE '%_14_%' THEN 'RSI_14' 
                    WHEN component_id LIKE '%_21_%' THEN 'RSI_21'
                    ELSE 'Other'
                END as rsi_period,
                COUNT(*) as strategies,
                AVG(signal_frequency) as avg_signal_freq,
                AVG(total_positions) as avg_positions,
                SUM(long_positions) as total_longs,
                SUM(short_positions) as total_shorts
            FROM component_metrics 
            WHERE component_type = 'strategy' AND strategy_type = 'rsi'
            GROUP BY 1
            ORDER BY avg_signal_freq DESC
        """)
        print(rsi_analysis.to_string(index=False))
        
    def classifier_analysis_report(self):
        """Generate classifier analysis report."""
        print("\n" + "=" * 80)
        print("🔍 CLASSIFIER ANALYSIS REPORT") 
        print("=" * 80)
        
        # Classifier overview
        overview = self.query("""
            SELECT 
                strategy_type,
                COUNT(*) as count,
                AVG(signal_frequency) as avg_signal_freq,
                SUM(signal_changes) as total_regime_changes
            FROM component_metrics 
            WHERE component_type = 'classifier'
            GROUP BY strategy_type
            ORDER BY avg_signal_freq DESC
        """)
        
        print("\n📊 Classifier Type Summary:")
        print(overview.to_string(index=False))
        
        # Regime breakdown analysis
        print("\n🎯 Regime Classifications:")
        regime_data = self.query("""
            SELECT 
                strategy_type,
                component_id,
                regime_classifications
            FROM component_metrics 
            WHERE component_type = 'classifier'
            AND regime_classifications != '{}'
            LIMIT 10
        """)
        
        for _, row in regime_data.iterrows():
            print(f"  {row['strategy_type']}: {row['regime_classifications']}")
    
    def signal_timeline_analysis(self):
        """Analyze signal timing patterns."""
        print("\n" + "=" * 80)
        print("⏱️  SIGNAL TIMELINE ANALYSIS")
        print("=" * 80)
        
        # Signal distribution by bar index
        timeline = self.query("""
            SELECT 
                bar_index,
                COUNT(*) as signal_count,
                COUNT(DISTINCT component_id) as active_components,
                GROUP_CONCAT(DISTINCT signal_value) as signal_types
            FROM signal_changes 
            GROUP BY bar_index
            ORDER BY bar_index
        """)
        
        print("\n📅 Signal Activity by Bar Index:")
        print(timeline.to_string(index=False))
        
    def interactive_mode(self):
        """Start interactive SQL mode."""
        print("\n" + "=" * 80)
        print("💻 INTERACTIVE SQL MODE")
        print("=" * 80)
        print("Available tables: signal_changes, component_metrics, workspace_summary")
        print("Type 'exit' to quit, 'tables' to show tables, 'help' for examples")
        print()
        
        while True:
            try:
                query = input("SQL> ").strip()
                
                if query.lower() == 'exit':
                    break
                elif query.lower() == 'tables':
                    self.show_tables()
                elif query.lower() == 'help':
                    self._show_sql_examples()
                elif query:
                    result = self.query(query)
                    if not result.empty:
                        print(result.to_string(index=False))
                    else:
                        print("No results returned.")
                        
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_sql_examples(self):
        """Show example SQL queries."""
        examples = [
            "-- Find most active strategies",
            "SELECT component_id, signal_frequency FROM component_metrics WHERE component_type = 'strategy' ORDER BY signal_frequency DESC LIMIT 5;",
            "",
            "-- Compare RSI periods",
            "SELECT SUBSTR(component_id, 13, 2) as rsi_period, AVG(signal_frequency) FROM component_metrics WHERE strategy_type = 'rsi' GROUP BY 1;",
            "",
            "-- Signal changes over time",
            "SELECT bar_index, COUNT(*) as changes FROM signal_changes GROUP BY bar_index ORDER BY bar_index;",
            "",
            "-- Regime detection summary", 
            "SELECT strategy_type, COUNT(*) FROM component_metrics WHERE component_type = 'classifier' GROUP BY strategy_type;"
        ]
        
        for example in examples:
            print(example)
    
    def close(self):
        self.conn.close()

def main():
    analytics = SignalAnalytics()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        analytics.interactive_mode()
    else:
        # Run standard reports
        analytics.strategy_performance_report()
        analytics.classifier_analysis_report() 
        analytics.signal_timeline_analysis()
        
        # Offer interactive mode
        choice = input("\n🔧 Enter interactive SQL mode? (y/n): ").strip().lower()
        if choice == 'y':
            analytics.interactive_mode()
    
    analytics.close()

if __name__ == "__main__":
    main()
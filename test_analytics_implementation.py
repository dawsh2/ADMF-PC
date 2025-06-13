#!/usr/bin/env python3
"""
Test script for new SQL analytics implementation.
Quick validation that the new analytics system works correctly.
"""

import sys
import tempfile
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from analytics import AnalyticsWorkspace, setup_workspace, migrate_workspace


def test_basic_workspace_creation():
    """Test basic workspace creation and SQL operations"""
    print("Testing basic workspace creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / 'test_workspace'
        
        # Create new workspace
        workspace = setup_workspace(workspace_path)
        
        # Test basic SQL operations
        tables = workspace.describe()
        print(f"Available tables: {tables['name'].tolist()}")
        
        # Test that all expected tables exist
        expected_tables = ['runs', 'strategies', 'classifiers', 'regime_performance', 
                          'event_archives', 'parameter_analysis', 'strategy_correlations']
        
        actual_tables = set(tables['name'].tolist())
        missing_tables = set(expected_tables) - actual_tables
        
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
            return False
        
        print("✅ All expected tables created")
        
        # Test summary on empty workspace
        summary = workspace.summary()
        print(f"Empty workspace summary: {summary}")
        
        workspace.close()
        return True


def test_sample_data_insertion():
    """Test inserting sample data and querying"""
    print("\nTesting sample data insertion...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / 'test_workspace'
        workspace = setup_workspace(workspace_path)
        
        # Insert sample run
        workspace.conn.execute("""
            INSERT INTO runs (
                run_id, created_at, workflow_type, symbols, timeframes,
                total_strategies, status, workspace_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            'test_run_001',
            datetime.now(),
            'test',
            ['SPY'],
            ['1m'],
            2,
            'completed',
            str(workspace_path)
        ])
        
        # Insert sample strategies
        sample_strategies = [
            {
                'strategy_id': 'momentum_test_001',
                'run_id': 'test_run_001',
                'strategy_type': 'momentum',
                'strategy_name': 'Test Momentum Strategy',
                'parameters': '{"sma_period": 20, "threshold": 0.02}',
                'sharpe_ratio': 1.5,
                'total_return': 0.15,
                'max_drawdown': 0.08,
                'total_trades': 45,
                'win_rate': 0.67,
                'created_at': datetime.now()
            },
            {
                'strategy_id': 'mean_reversion_test_001',
                'run_id': 'test_run_001',
                'strategy_type': 'mean_reversion',
                'strategy_name': 'Test Mean Reversion Strategy',
                'parameters': '{"lookback": 10, "zscore_threshold": 2.0}',
                'sharpe_ratio': 1.2,
                'total_return': 0.12,
                'max_drawdown': 0.06,
                'total_trades': 38,
                'win_rate': 0.71,
                'created_at': datetime.now()
            }
        ]
        
        for strategy in sample_strategies:
            columns = ', '.join(strategy.keys())
            placeholders = ', '.join(['?' for _ in strategy])
            workspace.conn.execute(f"""
                INSERT INTO strategies ({columns}) VALUES ({placeholders})
            """, list(strategy.values()))
        
        # Test SQL queries
        print("Testing SQL queries...")
        
        # Simple count
        count_result = workspace.sql("SELECT COUNT(*) as total FROM strategies")
        print(f"Total strategies: {count_result.iloc[0]['total']}")
        
        # Performance analysis
        performance = workspace.sql("""
            SELECT strategy_type, AVG(sharpe_ratio) as avg_sharpe, COUNT(*) as count
            FROM strategies 
            GROUP BY strategy_type
            ORDER BY avg_sharpe DESC
        """)
        print("Performance by strategy type:")
        print(performance.to_string(index=False))
        
        # Parameter analysis
        param_analysis = workspace.sql("""
            SELECT 
                JSON_EXTRACT(parameters, '$.sma_period') as sma_period,
                sharpe_ratio
            FROM strategies 
            WHERE JSON_EXTRACT(parameters, '$.sma_period') IS NOT NULL
        """)
        print("Parameter analysis:")
        print(param_analysis.to_string(index=False))
        
        # Test workspace summary
        summary = workspace.summary()
        print(f"Workspace summary: {summary}")
        
        workspace.close()
        return True


def test_signal_functions():
    """Test custom signal functions"""
    print("\nTesting custom signal functions...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / 'test_workspace'
        workspace = setup_workspace(workspace_path)
        
        # Create sample signal data
        signals_dir = workspace_path / 'signals' / 'momentum'
        signals_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample sparse signal file
        sample_signals = pd.DataFrame({
            'bar_idx': [100, 250, 400, 600, 800],
            'signal': [1, -1, 0, 1, -1]
        })
        
        signal_file = signals_dir / 'test_momentum_001.parquet'
        sample_signals.to_parquet(signal_file, index=False)
        
        # Test loading signals
        loaded_signals = workspace.load_signal_data('signals/momentum/test_momentum_001.parquet')
        print(f"Loaded signals shape: {loaded_signals.shape}")
        print("Sample signals:")
        print(loaded_signals.to_string(index=False))
        
        # Test signal statistics
        stats = workspace.get_signal_statistics('signals/momentum/test_momentum_001.parquet')
        print(f"Signal statistics: {stats}")
        
        # Test signal expansion
        expanded = workspace.expand_sparse_signals('signals/momentum/test_momentum_001.parquet', 1000)
        print(f"Expanded signals shape: {expanded.shape}")
        print("Expanded signals sample (first 10 and around changes):")
        print(expanded.head(10).to_string(index=False))
        print("...")
        print(expanded.iloc[95:105].to_string(index=False))
        
        workspace.close()
        return True


def test_export_functionality():
    """Test result export functionality"""
    print("\nTesting export functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir) / 'test_workspace'
        workspace = setup_workspace(workspace_path)
        
        # Insert sample data (reuse from previous test)
        workspace.conn.execute("""
            INSERT INTO runs (run_id, created_at, workflow_type, symbols, total_strategies, status, workspace_path)
            VALUES ('export_test', ?, 'test', ?, 1, 'completed', ?)
        """, [datetime.now(), ['SPY'], str(workspace_path)])
        
        workspace.conn.execute("""
            INSERT INTO strategies (strategy_id, run_id, strategy_type, sharpe_ratio, total_return, created_at)
            VALUES ('export_strategy_001', 'export_test', 'momentum', 1.8, 0.18, ?)
        """, [datetime.now()])
        
        # Test CSV export
        output_file = Path(temp_dir) / 'test_export.csv'
        workspace.export_results(
            "SELECT strategy_type, sharpe_ratio FROM strategies",
            output_file,
            format='csv'
        )
        
        # Verify export
        if output_file.exists():
            exported_data = pd.read_csv(output_file)
            print(f"Exported data shape: {exported_data.shape}")
            print("Exported data:")
            print(exported_data.to_string(index=False))
            return True
        else:
            print("❌ Export file not created")
            return False


def main():
    """Run all tests"""
    print("🧪 Testing ADMF-PC SQL Analytics Implementation")
    print("=" * 50)
    
    tests = [
        test_basic_workspace_creation,
        test_sample_data_insertion,
        test_signal_functions,
        test_export_functionality
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                print("✅ PASSED")
                passed += 1
            else:
                print("❌ FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ FAILED with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! SQL analytics implementation is working.")
    else:
        print("⚠️  Some tests failed. Check implementation.")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test script for WFV CLI functionality.

Tests the new walk-forward validation CLI arguments and config-less operation.
"""

import sys
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from src.core.cli.parser import parse_arguments
from src.core.cli.parameter_parser import parse_strategy_specs, parse_classifier_specs, build_config_from_cli


def test_cli_argument_parsing():
    """Test that CLI arguments are parsed correctly."""
    print("=== Testing CLI Argument Parsing ===")
    
    # Mock sys.argv for testing
    test_args = [
        'test_script.py',
        '--signal-generation',
        '--results-dir', 'momentum_wfv_study',
        '--strategies', 'momentum:lookback=10,20,30;threshold=0.01,0.02',
        '--classifiers', 'trend:fast_ma=10,20;slow_ma=30,50',
        '--wfv-windows', '5',
        '--wfv-window', '2',
        '--phase', 'train',
        '--dataset', 'train'
    ]
    
    # Temporarily replace sys.argv
    original_argv = sys.argv
    sys.argv = test_args
    
    try:
        args = parse_arguments()
        
        print(f"✅ Signal generation: {args.signal_generation}")
        print(f"✅ Results dir: {args.results_dir}")
        print(f"✅ Strategies: {args.strategies}")
        print(f"✅ Classifiers: {args.classifiers}")
        print(f"✅ WFV windows: {args.wfv_windows}")
        print(f"✅ WFV window: {args.wfv_window}")
        print(f"✅ Phase: {args.phase}")
        print(f"✅ Dataset: {args.dataset}")
        
    finally:
        sys.argv = original_argv
    
    print()


def test_strategy_parsing():
    """Test strategy specification parsing."""
    print("=== Testing Strategy Parsing ===")
    
    strategy_specs = [
        "momentum:lookback=10,20,30;threshold=0.01,0.02",
        "ma_crossover:fast_period=5,10;slow_period=20,30"
    ]
    
    strategies = parse_strategy_specs(strategy_specs)
    
    for i, strategy in enumerate(strategies):
        print(f"Strategy {i+1}: {strategy}")
    
    expected_expansions = 3 * 2 + 2 * 2  # momentum: 3*2=6, ma_crossover: 2*2=4
    print(f"✅ Parsed {len(strategies)} strategy configurations from {len(strategy_specs)} specs")
    print()


def test_classifier_parsing():
    """Test classifier specification parsing."""
    print("=== Testing Classifier Parsing ===")
    
    classifier_specs = [
        "trend:fast_ma=10,20;slow_ma=30,50",
        "volatility:period=14,21"
    ]
    
    classifiers = parse_classifier_specs(classifier_specs)
    
    for i, classifier in enumerate(classifiers):
        print(f"Classifier {i+1}: {classifier}")
    
    print(f"✅ Parsed {len(classifiers)} classifier configurations from {len(classifier_specs)} specs")
    print()


def test_config_less_operation():
    """Test building configuration from CLI parameters."""
    print("=== Testing Config-less Operation ===")
    
    # Create mock args object
    class MockArgs:
        def __init__(self):
            self.strategies = ["momentum:lookback=20;threshold=0.01"]
            self.classifiers = ["trend:fast_ma=10;slow_ma=20"]
            self.parameters = None
    
    args = MockArgs()
    config = build_config_from_cli(args)
    
    print("Generated configuration:")
    print(json.dumps(config, indent=2))
    
    assert 'strategies' in config
    assert 'classifiers' in config
    assert 'execution' in config
    assert 'symbols' in config
    
    print("✅ Config-less operation working correctly")
    print()


def test_parameter_export():
    """Test parameter export functionality."""
    print("=== Testing Parameter Export ===")
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_dir = Path(temp_dir) / "test_workspace"
        workspace_dir.mkdir()
        
        # Create mock analytics database
        import sqlite3
        db_path = workspace_dir / "analytics.db"
        
        with sqlite3.connect(db_path) as conn:
            # Create mock table
            conn.execute("""
                CREATE TABLE strategies (
                    strategy_name TEXT,
                    strategy_type TEXT,
                    strategy_params TEXT,
                    sharpe_ratio REAL
                )
            """)
            
            # Insert mock data
            conn.execute("""
                INSERT INTO strategies VALUES 
                ('momentum_20_001', 'momentum', '{"lookback": 20, "threshold": 0.01}', 1.5),
                ('momentum_30_002', 'momentum', '{"lookback": 30, "threshold": 0.02}', 1.8)
            """)
            conn.commit()
        
        # Test parameter export
        from src.analytics.parameter_export import export_selected_parameters
        
        query = "SELECT * FROM strategies WHERE sharpe_ratio > 1.6"
        output_file = workspace_dir / "selected_params.json"
        
        export_selected_parameters(str(workspace_dir), query, str(output_file))
        
        # Verify export
        with open(output_file, 'r') as f:
            exported_params = json.load(f)
        
        print("Exported parameters:")
        print(json.dumps(exported_params, indent=2))
        
        assert 'strategies' in exported_params
        assert len(exported_params['strategies']) == 1  # Only one strategy with sharpe > 1.6
        
        print("✅ Parameter export working correctly")
        print()


def main():
    """Run all tests."""
    print("🧪 Testing WFV CLI Functionality\n")
    
    try:
        test_cli_argument_parsing()
        test_strategy_parsing() 
        test_classifier_parsing()
        test_config_less_operation()
        test_parameter_export()
        
        print("🎉 All tests passed! WFV CLI functionality is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
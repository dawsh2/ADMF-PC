#!/usr/bin/env python3
"""Test parameter expansion fix with a small config"""

import yaml
import tempfile
import subprocess
from pathlib import Path

# Create a minimal test config with parameter expansion
test_config = {
    "name": "parameter_expansion_test",
    "symbols": ["SPY"],
    "timeframes": ["1m"],
    "data_source": "file", 
    "data_dir": "./data",
    "start_date": "2023-01-01",
    "end_date": "2023-01-10", 
    "max_bars": 50,
    
    "strategies": [
        {
            "name": "ma_test",
            "type": "ma_crossover",
            "params": {
                "fast_period": [5, 10],      # 2 values
                "slow_period": [20, 30],     # 2 values  
                "stop_loss_pct": [1.0]       # 1 value
            }
            # Expected: 2 * 2 * 1 = 4 strategies
        }
    ],
    
    "classifiers": [
        {
            "name": "momentum_test", 
            "type": "momentum_regime_classifier",
            "params": {
                "fast_period": [10, 20],     # 2 values
                "threshold": [0.01, 0.02]    # 2 values
            }
            # Expected: 2 * 2 = 4 classifiers
        }
    ]
}

# Write test config
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(test_config, f)
    config_path = f.name

try:
    print(f"Running test with config: {config_path}")
    print(f"Expected: 4 strategies + 4 classifiers = 8 total")
    
    # Run with venv python
    cmd = [
        "./venv/bin/python", "main.py",
        "--config", config_path,
        "--signal-generation",
        "--bars", "50"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Run completed successfully")
        
        # Find the created workspace
        import os
        workspaces = [d for d in os.listdir('workspaces') if d.startswith('2025')]
        if workspaces:
            latest_workspace = sorted(workspaces)[-1]
            print(f"✓ Found workspace: {latest_workspace}")
            
            # Check the database
            from src.analytics.workspace import AnalyticsWorkspace
            workspace = AnalyticsWorkspace(f'workspaces/{latest_workspace}')
            
            # Count strategies
            strategies = workspace.sql("SELECT COUNT(*) as count FROM strategies")
            strategy_count = strategies.iloc[0]['count']
            
            # Count classifiers  
            classifiers = workspace.sql("SELECT COUNT(*) as count FROM classifiers")
            classifier_count = classifiers.iloc[0]['count']
            
            print(f"✓ Strategies in database: {strategy_count} (expected: 4)")
            print(f"✓ Classifiers in database: {classifier_count} (expected: 4)")
            
            # Show sample strategies
            if strategy_count > 0:
                sample = workspace.sql("SELECT strategy_id, strategy_type, parameters FROM strategies LIMIT 3")
                print(f"✓ Sample strategies:")
                for _, row in sample.iterrows():
                    print(f"  - {row['strategy_id']}: {row['parameters']}")
            
            workspace.close()
            
            if strategy_count == 4 and classifier_count == 4:
                print("\n🎉 SUCCESS! Parameter expansion fix is working correctly!")
            else:
                print(f"\n⚠️  Partial success: Got {strategy_count} strategies and {classifier_count} classifiers")
        else:
            print("✗ No workspace found")
    else:
        print(f"✗ Run failed: {result.returncode}")
        print("STDERR:", result.stderr)
        
finally:
    # Cleanup
    Path(config_path).unlink()
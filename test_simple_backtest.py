#!/usr/bin/env python3
"""
Simple test script to run a basic backtest.

This tests the complete integration from main.py through the coordinator
to the actual backtest execution.
"""
import subprocess
import os
import sys
import yaml
from pathlib import Path


def create_test_config():
    """Create a minimal test configuration."""
    config = {
        'workflow_type': 'backtest',
        
        # Minimal data config
        'data': {
            'start_date': '2023-01-01',
            'end_date': '2023-01-31',  # Just one month for testing
            'symbols': ['AAPL']
        },
        
        # Simple strategy
        'strategies': [{
            'name': 'test_momentum',
            'class': 'MomentumStrategy',
            'parameters': {
                'fast_period': 5,
                'slow_period': 10
            }
        }],
        
        # Basic risk settings
        'risk': {
            'max_position_size': 0.02,
            'max_total_exposure': 0.1
        },
        
        # Portfolio
        'backtest': {
            'initial_capital': 100000
        },
        
        # Output
        'parameters': {
            'output_dir': 'output/test_backtest'
        }
    }
    
    # Save config
    config_path = 'configs/test_simple.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config_path


def run_backtest(config_path):
    """Run the backtest using main.py."""
    cmd = [
        sys.executable,
        'main.py',
        '--config', config_path,
        '--mode', 'backtest',
        '--verbose'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 80)
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
    
    return result.returncode


def test_signal_generation(config_path):
    """Test signal generation mode."""
    cmd = [
        sys.executable,
        'main.py',
        '--config', config_path,
        '--mode', 'signal-generation',
        '--signal-output', 'output/test_signals.json',
        '--verbose'
    ]
    
    print(f"\nTesting signal generation mode...")
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 80)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
    
    return result.returncode


def main():
    """Run the test."""
    print("ADMF-PC Simple Backtest Test")
    print("=" * 80)
    
    # Create output directory
    Path('output/test_backtest').mkdir(parents=True, exist_ok=True)
    
    # Create test config
    print("Creating test configuration...")
    config_path = create_test_config()
    print(f"Config saved to: {config_path}")
    
    # Run backtest
    print("\nRunning backtest...")
    exit_code = run_backtest(config_path)
    
    if exit_code == 0:
        print("\n✓ Backtest completed successfully!")
    else:
        print(f"\n✗ Backtest failed with exit code: {exit_code}")
        return exit_code
    
    # Test signal generation
    print("\n" + "=" * 80)
    print("Testing signal generation mode...")
    
    exit_code = test_signal_generation(config_path)
    
    if exit_code == 0:
        print("\n✓ Signal generation completed successfully!")
        
        # Check if signal file was created
        if Path('output/test_signals.json').exists():
            print("✓ Signal file created")
            with open('output/test_signals.json', 'r') as f:
                import json
                signals = json.load(f)
                print(f"✓ Generated {len(signals)} signals")
    else:
        print(f"\n✗ Signal generation failed with exit code: {exit_code}")
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
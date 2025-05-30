#!/usr/bin/env python3
"""
Example workflow showing how to use the three-pattern architecture.

This demonstrates:
1. Signal generation for analysis
2. Signal replay for ensemble optimization
3. Full backtest with optimal parameters
"""
import subprocess
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def run_command(cmd: List[str]) -> int:
    """Run a command and return exit code."""
    print(f"\nRunning: {' '.join(cmd)}")
    print("-" * 80)
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def phase1_signal_generation(config_file: str, output_dir: str) -> str:
    """
    Phase 1: Generate signals for analysis.
    
    Returns path to signal log file.
    """
    print("\n" + "="*80)
    print("PHASE 1: Signal Generation and Analysis")
    print("="*80)
    
    signal_log = os.path.join(output_dir, "signals_phase1.json")
    
    cmd = [
        "python", "main.py",
        "--config", config_file,
        "--mode", "signal-generation",
        "--signal-output", signal_log,
        "--output-dir", output_dir
    ]
    
    exit_code = run_command(cmd)
    if exit_code != 0:
        raise RuntimeError("Signal generation failed")
        
    print(f"\n✓ Signals saved to: {signal_log}")
    return signal_log


def phase2_ensemble_optimization(
    config_file: str,
    signal_log: str,
    output_dir: str
) -> Dict[str, float]:
    """
    Phase 2: Optimize ensemble weights using signal replay.
    
    Returns optimal weights.
    """
    print("\n" + "="*80)
    print("PHASE 2: Ensemble Weight Optimization")
    print("="*80)
    
    # Test different weight combinations
    weight_combinations = [
        {"momentum": 0.5, "mean_reversion": 0.5},
        {"momentum": 0.7, "mean_reversion": 0.3},
        {"momentum": 0.3, "mean_reversion": 0.7},
        {"momentum": 0.6, "mean_reversion": 0.2, "breakout": 0.2},
        {"momentum": 0.4, "mean_reversion": 0.4, "breakout": 0.2},
        {"momentum": 0.33, "mean_reversion": 0.33, "breakout": 0.34},
    ]
    
    best_sharpe = -np.inf
    best_weights = {}
    
    for i, weights in enumerate(weight_combinations):
        print(f"\nTesting weight combination {i+1}/{len(weight_combinations)}: {weights}")
        
        weights_file = os.path.join(output_dir, f"weights_{i}.json")
        with open(weights_file, 'w') as f:
            json.dump(weights, f)
        
        cmd = [
            "python", "main.py",
            "--config", config_file,
            "--mode", "signal-replay",
            "--signal-log", signal_log,
            "--weights", weights_file,
            "--output-dir", output_dir
        ]
        
        # Run and capture output to parse Sharpe ratio
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Parse Sharpe ratio from output
            for line in result.stdout.split('\n'):
                if "Sharpe ratio:" in line:
                    sharpe = float(line.split(':')[1].strip())
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_weights = weights
                    break
                    
    print(f"\n✓ Best weights found: {best_weights}")
    print(f"✓ Best Sharpe ratio: {best_sharpe:.2f}")
    
    # Save best weights
    best_weights_file = os.path.join(output_dir, "optimal_weights.json")
    with open(best_weights_file, 'w') as f:
        json.dump(best_weights, f, indent=2)
        
    return best_weights


def phase3_full_backtest(
    config_file: str,
    optimal_weights: Dict[str, float],
    output_dir: str
) -> None:
    """
    Phase 3: Run full backtest with optimal parameters.
    """
    print("\n" + "="*80)
    print("PHASE 3: Full Backtest with Optimal Parameters")
    print("="*80)
    
    # For a full backtest, we would update the config with optimal weights
    # and run a standard backtest
    
    cmd = [
        "python", "main.py",
        "--config", config_file,
        "--mode", "backtest",
        "--output-dir", output_dir
    ]
    
    print(f"Using optimal ensemble weights: {optimal_weights}")
    
    exit_code = run_command(cmd)
    if exit_code != 0:
        raise RuntimeError("Full backtest failed")
        
    print("\n✓ Full backtest completed successfully")


def main():
    """Run complete signal-based workflow."""
    # Configuration
    config_file = "configs/example_backtest.yaml"
    output_dir = "output/signal_workflow"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Phase 1: Generate and analyze signals
        signal_log = phase1_signal_generation(config_file, output_dir)
        
        # Phase 2: Optimize ensemble weights
        optimal_weights = phase2_ensemble_optimization(
            config_file, signal_log, output_dir
        )
        
        # Phase 3: Run full backtest
        phase3_full_backtest(config_file, optimal_weights, output_dir)
        
        print("\n" + "="*80)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nResults saved to: {output_dir}")
        print(f"- Signal log: {signal_log}")
        print(f"- Optimal weights: {os.path.join(output_dir, 'optimal_weights.json')}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
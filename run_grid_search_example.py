#!/usr/bin/env python3
"""
Example: Run Complete Grid Search Workflow

This script demonstrates the full workflow:
1. Generate expanded configurations from grid
2. Run signal generation with all combinations
3. Analyze results to find optimal pairings
"""

import subprocess
import time
from pathlib import Path
import json


def run_grid_search_workflow():
    """Run the complete grid search workflow."""
    
    print("="*80)
    print("PARAMETER GRID SEARCH WORKFLOW")
    print("="*80)
    
    # Step 1: Generate expanded configuration
    print("\n1. Generating configurations from grid...")
    print("-"*80)
    
    grid_config = "config/comprehensive_grid_search.yaml"
    expanded_config = "config/expanded_grid_search.yaml"
    manifest_dir = "./grid_manifests"
    
    cmd = [
        "python", "generate_grid_configs.py",
        "--grid-config", grid_config,
        "--output-config", expanded_config,
        "--manifest-dir", manifest_dir,
        "--estimate-runtime"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
        
    # Load manifest to see what was generated
    with open(f"{manifest_dir}/expansion_summary.json", 'r') as f:
        summary = json.load(f)
        
    print(f"\nGenerated {summary['total_strategies']} strategies and {summary['total_classifiers']} classifiers")
    
    # Step 2: Run signal generation (with a smaller subset for demo)
    print("\n2. Running signal generation...")
    print("-"*80)
    print("NOTE: For demo, we'll run with fewer bars and a subset of configurations")
    
    # Create a demo config with just a few combinations
    create_demo_config(expanded_config, "config/demo_grid_search.yaml", max_strategies=3, max_classifiers=3)
    
    cmd = [
        "python", "main.py",
        "--config", "config/demo_grid_search.yaml",
        "--signal-generation",
        "--bars", "1000"  # Fewer bars for demo
    ]
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    elapsed = time.time() - start_time
    print(f"Signal generation completed in {elapsed:.1f} seconds")
    
    if result.returncode != 0:
        print("Error running signal generation:", result.stderr)
        return
        
    # Step 3: Analyze results
    print("\n3. Analyzing grid search results...")
    print("-"*80)
    
    # Find the results directory
    workspaces_dir = Path("./workspaces")
    latest_dir = max(workspaces_dir.glob("*/"), key=lambda p: p.stat().st_mtime)
    
    cmd = [
        "python", "analyze_grid_search_results.py",
        "--results-dir", str(latest_dir),
        "--top-n", "5"
    ]
    
    print(f"Analyzing results in: {latest_dir}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    
    # Step 4: Generate recommendations
    print("\n4. RECOMMENDATIONS")
    print("-"*80)
    
    # Load best configurations
    best_configs_path = latest_dir / "best_configurations.json"
    if best_configs_path.exists():
        with open(best_configs_path, 'r') as f:
            best = json.load(f)
            
        print("\nBest Strategy Configuration:")
        if best['best_strategies']:
            top_strat = best['best_strategies'][0]
            print(f"  Type: {top_strat.get('strategy_type')}")
            print(f"  Parameters: {json.dumps({k: v for k, v in top_strat.items() 
                                             if k not in ['strategy_id', 'strategy_type', 'composite_score', 
                                                         'sharpe_ratio', 'win_rate', 'max_drawdown', 
                                                         'profit_factor', 'total_trades', 'avg_return']}, indent=4)}")
            print(f"  Sharpe Ratio: {top_strat.get('sharpe_ratio', 0):.3f}")
            print(f"  Win Rate: {top_strat.get('win_rate', 0):.1%}")
            
        print("\nBest Classifier Configuration:")
        if best['best_classifiers']:
            top_class = best['best_classifiers'][0]
            print(f"  Type: {top_class.get('classifier_type')}")
            print(f"  Stability Score: {top_class.get('stability_score', 0):.3f}")
            
        print("\nBest Strategy-Classifier Pairing:")
        if best['best_pairings']:
            top_pair = best['best_pairings'][0]
            print(f"  Strategy: {top_pair.get('strategy_type')}")
            print(f"  Classifier: {top_pair.get('classifier_type')}")
            print(f"  Pairing Score: {top_pair.get('pairing_score', 0):.3f}")
            
    print("\n" + "="*80)
    print("Grid search workflow complete!")
    print("="*80)


def create_demo_config(full_config_path: str, demo_config_path: str, 
                      max_strategies: int = 3, max_classifiers: int = 3):
    """Create a smaller demo configuration for testing."""
    import yaml
    
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Limit strategies and classifiers
    if 'strategies' in config:
        config['strategies'] = config['strategies'][:max_strategies]
        
    if 'classifiers' in config:
        config['classifiers'] = config['classifiers'][:max_classifiers]
        
    # Update metadata
    config['metadata']['demo_mode'] = True
    config['metadata']['original_strategies'] = len(config.get('strategies', []))
    config['metadata']['original_classifiers'] = len(config.get('classifiers', []))
    
    # Save demo config
    with open(demo_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print(f"Created demo config with {len(config.get('strategies', []))} strategies and {len(config.get('classifiers', []))} classifiers")


if __name__ == "__main__":
    run_grid_search_workflow()
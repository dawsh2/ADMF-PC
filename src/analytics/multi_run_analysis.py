#!/usr/bin/env python3
"""
Multi-run analysis tool for combining results from multiple parameter sweeps.

Usage:
    # Analyze specific runs
    python -m src.analytics.multi_run_analysis results/run_1 results/run_2 results/run_3
    
    # Analyze all runs matching a pattern
    python -m src.analytics.multi_run_analysis --pattern "results/run_2025*"
    
    # Analyze runs from specific configs
    python -m src.analytics.multi_run_analysis --configs config/bollinger config/rsi config/momentum
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import json
from datetime import datetime
import glob


def find_runs_from_configs(config_dirs: List[str], results_dir: str = "results") -> List[Path]:
    """Find run directories that used specific configs."""
    runs = []
    results_path = Path(results_dir)
    
    for run_dir in sorted(results_path.glob("run_*"), reverse=True):
        config_file = run_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                config_name = config.get('name', '')
                
                # Check if this run matches any of our target configs
                for config_dir in config_dirs:
                    if config_name and (config_name in config_dir or config_dir in config_name):
                        runs.append(run_dir)
                        print(f"Found run for {config_dir}: {run_dir.name}")
                        break
    
    return runs


def find_latest_runs_per_config(results_dir: str = "results", limit: int = 10) -> Dict[str, Path]:
    """Find the latest run for each unique config name."""
    results_path = Path(results_dir)
    config_runs = {}
    
    for run_dir in sorted(results_path.glob("run_*"), reverse=True):
        if len(config_runs) >= limit:
            break
            
        config_file = run_dir / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                config_name = config.get('name', 'unnamed')
                
                if config_name not in config_runs:
                    config_runs[config_name] = run_dir
    
    return config_runs


def analyze_runs(run_dirs: List[Path]) -> Dict[str, Any]:
    """Quick analysis of multiple runs."""
    total_strategies = 0
    strategy_types = set()
    run_summaries = []
    
    for run_dir in run_dirs:
        index_path = run_dir / "strategy_index.parquet"
        if index_path.exists():
            strategies = pd.read_parquet(index_path)
            total_strategies += len(strategies)
            strategy_types.update(strategies['strategy_type'].unique())
            
            # Get config info
            config_path = run_dir / "config.json"
            config_name = "unknown"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    config_name = config.get('name', 'unknown')
            
            run_summaries.append({
                'run_id': run_dir.name,
                'config_name': config_name,
                'num_strategies': len(strategies),
                'strategy_types': list(strategies['strategy_type'].unique())
            })
    
    return {
        'total_runs': len(run_dirs),
        'total_strategies': total_strategies,
        'unique_strategy_types': list(strategy_types),
        'run_summaries': run_summaries
    }


def create_multi_run_notebook(run_dirs: List[Path], output_path: Path = None):
    """Create a parameterized notebook for multi-run analysis."""
    try:
        from src.analytics.papermill_runner import PapermillNotebookRunner, PAPERMILL_AVAILABLE
        
        if not PAPERMILL_AVAILABLE:
            print("❌ Papermill not installed. Install with: pip install papermill")
            return None
            
    except ImportError:
        print("❌ Could not import papermill runner")
        return None
    
    # Default output path
    if output_path is None:
        output_path = Path(f"multi_run_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb")
    
    # Prepare parameters
    params = {
        'run_dirs': [str(run_dir) for run_dir in run_dirs],
        'output_name': output_path.stem
    }
    
    # Run the multi-run template
    runner = PapermillNotebookRunner()
    template = Path("src/analytics/templates/multi_run_analysis.ipynb")
    
    if not template.exists():
        print(f"❌ Template not found: {template}")
        return None
    
    import papermill as pm
    
    try:
        pm.execute_notebook(
            str(template),
            str(output_path),
            parameters=params,
            kernel_name='python3'
        )
        print(f"✅ Created multi-run analysis notebook: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Failed to create notebook: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Multi-run analysis for ensemble building")
    parser.add_argument("run_dirs", nargs="*", help="Specific run directories to analyze")
    parser.add_argument("--pattern", help="Glob pattern to find runs (e.g., 'results/run_2025*')")
    parser.add_argument("--configs", nargs="+", help="Find runs from specific config directories")
    parser.add_argument("--latest", type=int, help="Use latest N runs per config")
    parser.add_argument("--output", help="Output notebook path")
    parser.add_argument("--no-notebook", action="store_true", help="Just print summary, don't create notebook")
    
    args = parser.parse_args()
    
    # Collect run directories
    run_dirs = []
    
    # From explicit paths
    if args.run_dirs:
        run_dirs.extend([Path(d) for d in args.run_dirs])
    
    # From pattern
    if args.pattern:
        matched = glob.glob(args.pattern)
        run_dirs.extend([Path(d) for d in matched if Path(d).is_dir()])
        print(f"Found {len(matched)} runs matching pattern '{args.pattern}'")
    
    # From configs
    if args.configs:
        config_runs = find_runs_from_configs(args.configs)
        run_dirs.extend(config_runs)
    
    # From latest
    if args.latest:
        latest_runs = find_latest_runs_per_config(limit=args.latest)
        run_dirs.extend(latest_runs.values())
        print(f"Found latest runs for {len(latest_runs)} unique configs")
    
    # Remove duplicates
    run_dirs = list(set(run_dirs))
    
    if not run_dirs:
        print("❌ No run directories specified or found")
        print("\nUsage examples:")
        print("  python -m src.analytics.multi_run_analysis results/run_1 results/run_2")
        print("  python -m src.analytics.multi_run_analysis --pattern 'results/run_2025*'")
        print("  python -m src.analytics.multi_run_analysis --configs config/bollinger config/rsi")
        print("  python -m src.analytics.multi_run_analysis --latest 5")
        sys.exit(1)
    
    # Sort by date (newest first)
    run_dirs.sort(reverse=True)
    
    print(f"\nAnalyzing {len(run_dirs)} runs:")
    for run_dir in run_dirs:
        print(f"  - {run_dir}")
    
    # Quick analysis
    summary = analyze_runs(run_dirs)
    
    print(f"\nSummary:")
    print(f"  Total strategies: {summary['total_strategies']:,}")
    print(f"  Strategy types: {', '.join(summary['unique_strategy_types'])}")
    
    # Create notebook unless disabled
    if not args.no_notebook:
        output_path = Path(args.output) if args.output else None
        notebook_path = create_multi_run_notebook(run_dirs, output_path)
        
        if notebook_path:
            print(f"\n🚀 To continue analysis:")
            print(f"  jupyter lab {notebook_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
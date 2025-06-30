#!/usr/bin/env python3
"""Generate a comprehensive checklist of all strategies and their configs."""

import os
import sys
import yaml
from pathlib import Path
from collections import defaultdict

sys.path.append('.')

from src.core.components.discovery import discover_components_in_module, get_component_registry

def discover_all_strategies():
    """Discover all indicator strategies."""
    strategies = defaultdict(list)
    
    # Discover each indicator module
    modules = {
        'crossover': 'src.strategy.strategies.indicators.crossover',
        'divergence': 'src.strategy.strategies.indicators.divergence', 
        'momentum': 'src.strategy.strategies.indicators.momentum',
        'oscillators': 'src.strategy.strategies.indicators.oscillators',
        'structure': 'src.strategy.strategies.indicators.structure',
        'trend': 'src.strategy.strategies.indicators.trend',
        'volatility': 'src.strategy.strategies.indicators.volatility',
        'volume': 'src.strategy.strategies.indicators.volume'
    }
    
    for category, module in modules.items():
        try:
            discover_components_in_module(module)
        except Exception as e:
            print(f"Error discovering {category}: {e}")
    
    # Get all strategies from registry
    registry = get_component_registry()
    all_strategies = registry.get_components_by_type('strategy')
    
    # Organize by category
    for strategy in all_strategies:
        # Determine category from module path
        if hasattr(strategy.factory, '__module__'):
            module = strategy.factory.__module__
            for category in modules.keys():
                if f'.{category}' in module or module.endswith(category):
                    strategies[category].append(strategy.name)
                    break
    
    return strategies

def find_config_files():
    """Find all config files in the indicators directory."""
    configs = defaultdict(list)
    config_root = Path("config/indicators")
    
    if not config_root.exists():
        return configs
    
    for config_file in config_root.rglob("*.yaml"):
        # Get category from path
        category = config_file.parent.name
        
        # Load config to get strategy name
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            # Extract strategy names from config
            if 'strategy' in config:
                for strategy_name in config['strategy'].keys():
                    configs[category].append({
                        'strategy': strategy_name,
                        'file': str(config_file.relative_to(Path("config"))),
                        'path': config_file
                    })
        except Exception as e:
            print(f"Error loading {config_file}: {e}")
    
    return configs

def test_config(config_path):
    """Test if a config runs successfully."""
    import subprocess
    
    cmd = [
        sys.executable,
        "main.py",
        "--config", str(config_path),
        "--signal-generation",
        "--bars", "50"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Check for success
        if result.returncode == 0:
            # Check for signals
            has_signals = '📡 SIGNAL' in result.stdout
            return 'signals' if has_signals else 'success'
        else:
            return 'failed'
    except subprocess.TimeoutExpired:
        return 'timeout'
    except Exception:
        return 'error'

def main():
    """Generate comprehensive strategy checklist."""
    print("Discovering all strategies...")
    strategies = discover_all_strategies()
    
    print("\nFinding all config files...")
    configs = find_config_files()
    
    # Create checklist
    print("\n" + "=" * 80)
    print("STRATEGY CHECKLIST")
    print("=" * 80)
    
    total_strategies = 0
    total_configs = 0
    missing_configs = []
    
    for category in sorted(strategies.keys()):
        print(f"\n## {category.upper()}")
        print("-" * 40)
        
        category_strategies = sorted(strategies[category])
        category_configs = {c['strategy']: c for c in configs.get(category, [])}
        
        for strategy in category_strategies:
            total_strategies += 1
            
            if strategy in category_configs:
                config = category_configs[strategy]
                print(f"✅ {strategy:<30} → {config['file']}")
                total_configs += 1
            else:
                # Try to find in other categories
                found = False
                for other_cat, other_configs in configs.items():
                    for c in other_configs:
                        if c['strategy'] == strategy:
                            print(f"⚠️  {strategy:<30} → {c['file']} (in {other_cat})")
                            found = True
                            total_configs += 1
                            break
                    if found:
                        break
                
                if not found:
                    print(f"❌ {strategy:<30} → NO CONFIG FOUND")
                    missing_configs.append((category, strategy))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total strategies: {total_strategies}")
    print(f"Strategies with configs: {total_configs}")
    print(f"Missing configs: {len(missing_configs)}")
    
    if missing_configs:
        print("\nStrategies missing configs:")
        for category, strategy in missing_configs:
            print(f"  - {category}/{strategy}")
    
    # Test configs
    print("\n" + "=" * 80)
    print("TESTING CONFIGS")
    print("=" * 80)
    
    test_all = input("\nTest all configs? (y/n): ").lower() == 'y'
    
    if test_all:
        results = defaultdict(list)
        
        for category, config_list in configs.items():
            print(f"\nTesting {category} configs...")
            
            for config in config_list:
                print(f"  Testing {config['strategy']}...", end=" ", flush=True)
                result = test_config(config['path'])
                results[result].append(config['file'])
                
                if result == 'signals':
                    print("✅ (with signals)")
                elif result == 'success':
                    print("✅ (no signals)")
                elif result == 'failed':
                    print("❌ (failed)")
                elif result == 'timeout':
                    print("⏱️  (timeout)")
                else:
                    print("❓ (error)")
        
        # Test summary
        print("\n" + "-" * 40)
        print("TEST RESULTS")
        print("-" * 40)
        print(f"With signals: {len(results['signals'])}")
        print(f"No signals: {len(results['success'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Timeout: {len(results['timeout'])}")
        print(f"Error: {len(results['error'])}")
        
        if results['failed']:
            print("\nFailed configs:")
            for config in results['failed']:
                print(f"  - {config}")

if __name__ == "__main__":
    main()
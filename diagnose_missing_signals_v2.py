#!/usr/bin/env python3
"""
Simplified diagnostic script to analyze why strategies aren't generating signals.
"""

import os
import sys
import yaml
import importlib
import inspect
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_missing_strategies():
    """Analyze which strategies are missing and why."""
    
    # Load config
    with open('config/expansive_grid_search.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get expected strategies
    expected_strategies = {}
    for strategy in config['strategies']:
        expected_strategies[strategy['name']] = {
            'type': strategy['type'],
            'params': strategy['params']
        }
    
    # Get actual strategies from workspace
    workspace_dir = 'workspaces/expansive_grid_search_8c6c181f/traces/SPY_1m/signals'
    actual_strategies = set()
    if os.path.exists(workspace_dir):
        actual_strategies = set(os.listdir(workspace_dir))
    
    # Find missing strategies
    missing_strategies = set(expected_strategies.keys()) - actual_strategies
    
    print(f"Expected strategies from config: {len(expected_strategies)}")
    print(f"Actual strategies in workspace: {len(actual_strategies)}")
    print(f"\nMissing strategies ({len(missing_strategies)}):\n")
    
    # Try to import and analyze each missing strategy
    results = {}
    
    # Import all indicator modules
    modules_to_check = [
        'src.strategy.strategies.indicators.crossovers',
        'src.strategy.strategies.indicators.oscillators',
        'src.strategy.strategies.indicators.volatility',
        'src.strategy.strategies.indicators.trend',
        'src.strategy.strategies.indicators.volume',
        'src.strategy.strategies.indicators.structure'
    ]
    
    available_functions = {}
    for module_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and not name.startswith('_'):
                    available_functions[name] = {
                        'module': module_name,
                        'function': obj,
                        'source': inspect.getsource(obj) if hasattr(inspect, 'getsource') else 'N/A'
                    }
        except Exception as e:
            print(f"Error importing {module_name}: {e}")
    
    print(f"Found {len(available_functions)} strategy functions\n")
    
    # Analyze each missing strategy
    for strategy_name in sorted(missing_strategies):
        strategy_info = expected_strategies[strategy_name]
        strategy_type = strategy_info['type']
        
        result = {
            'name': strategy_name,
            'type': strategy_type,
            'status': 'unknown',
            'reason': '',
            'params': strategy_info['params']
        }
        
        # Check if function exists
        if strategy_type in available_functions:
            result['status'] = 'found'
            func_info = available_functions[strategy_type]
            
            # Check if it looks for specific features
            source = func_info['source']
            if 'features.get' in source:
                # Extract feature lookups
                import re
                feature_pattern = r"features\.get\(['\"]([^'\"]+)['\"]\)"
                features_used = re.findall(feature_pattern, source)
                result['features_needed'] = features_used
                
                # Check for complex features that might need more warmup
                complex_indicators = ['ichimoku', 'adx', 'aroon', 'ultosc', 'vortex', 'sar', 'linearreg']
                for indicator in complex_indicators:
                    if any(indicator in f for f in features_used):
                        result['reason'] = f'Uses complex indicator ({indicator}) that may need >1000 bars warmup'
                        break
                
                if not result['reason']:
                    result['reason'] = 'Unknown - check feature availability and signal logic'
                    
        else:
            result['status'] = 'not_found' 
            result['reason'] = f'Strategy function "{strategy_type}" not found in any module'
        
        results[strategy_name] = result
    
    # Print analysis
    by_status = defaultdict(list)
    for name, info in results.items():
        by_status[info['status']].append(name)
    
    print("=== ANALYSIS BY STATUS ===\n")
    for status, strategies in by_status.items():
        print(f"{status.upper()} ({len(strategies)} strategies):")
        for s in sorted(strategies):
            info = results[s]
            print(f"  - {s} ({info['type']})")
            if info['reason']:
                print(f"    Reason: {info['reason']}")
            if 'features_needed' in info and len(info['features_needed']) > 0:
                print(f"    Features: {', '.join(info['features_needed'][:5])}")
        print()
    
    # Group by reason
    print("=== ANALYSIS BY REASON ===\n")
    by_reason = defaultdict(list)
    for name, info in results.items():
        reason = info['reason'] or 'No specific reason'
        by_reason[reason].append(name)
    
    for reason, strategies in sorted(by_reason.items()):
        print(f"{reason}:")
        for s in strategies[:5]:  # Show first 5
            print(f"  - {s}")
        if len(strategies) > 5:
            print(f"  ... and {len(strategies) - 5} more")
        print()
    
    # Summary recommendations
    print("=== RECOMMENDATIONS ===\n")
    print("1. For complex indicators (ADX, Aroon, Ichimoku, etc.):")
    print("   - Increase --bars to 3000+ to ensure sufficient warmup")
    print("   - These indicators often need 100-500 bars before generating first signal\n")
    
    print("2. For missing strategy functions:")
    print("   - Check if the strategy type name in config matches function name")
    print("   - Verify the strategy is properly decorated with @strategy\n")
    
    print("3. For feature-dependent strategies:")
    print("   - Ensure FeatureHub is configured to compute required features")
    print("   - Check feature naming consistency (e.g., 'sma_10' vs 'SMA_10')\n")
    
    # Save detailed results
    import json
    with open('strategy_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Detailed results saved to: strategy_analysis_results.json")


if __name__ == "__main__":
    analyze_missing_strategies()
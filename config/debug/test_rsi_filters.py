#!/usr/bin/env python3
"""
Test different RSI filter expressions to debug the issue.
"""

import sys
sys.path.append('.')

from src.core.coordinator.config.clean_syntax_parser import CleanSyntaxParser
import yaml

# Test different filter expressions
test_filters = [
    # Test 1: Clean syntax
    {"rsi_below": {"threshold": 50}},
    
    # Test 2: With period specification
    {"rsi_below": {"threshold": 50, "period": 14}},
    
    # Test 3: Raw expression with rsi(14)
    "signal == 0 or rsi(14) < 50",
    
    # Test 4: Raw expression with rsi_14
    "signal == 0 or rsi_14 < 50",
]

parser = CleanSyntaxParser()

print("🔍 Testing RSI Filter Expressions:\n")

for i, filter_spec in enumerate(test_filters):
    print(f"Test {i+1}: {filter_spec}")
    
    try:
        if isinstance(filter_spec, str):
            # Direct expression
            expr = filter_spec
            params = {}
        else:
            # Parse clean syntax
            expr, params = parser._parse_filter(filter_spec)
            if expr:
                expr = f"signal == 0 or ({expr})"
        
        print(f"  Expression: {expr}")
        print(f"  Parameters: {params}")
        
        # Check the template
        if isinstance(filter_spec, dict) and 'rsi_below' in filter_spec:
            template = parser.FILTER_TEMPLATES.get('rsi_below', 'Not found')
            defaults = parser.FILTER_DEFAULTS.get('rsi_below', {})
            print(f"  Template: {template}")
            print(f"  Defaults: {defaults}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
    
    print()

# Now test the full config parsing
print("\n📊 Testing Full Config Parsing:\n")

with open('config/debug/config.yaml') as f:
    config = yaml.safe_load(f)

# Check if clean syntax is detected
if 'strategy' in config:
    from src.core.coordinator.config.clean_syntax_parser import parse_clean_config
    parsed = parse_clean_config(config)
    
    strategies = parsed.get('parameter_space', {}).get('strategies', [])
    print(f"Total strategies generated: {len(strategies)}")
    
    for i, strategy in enumerate(strategies):
        print(f"\nStrategy {i}:")
        print(f"  Type: {strategy.get('type')}")
        print(f"  Params: {strategy.get('param_overrides')}")
        if 'filter' in strategy:
            print(f"  Filter: {strategy['filter'][:80]}...")
        else:
            print(f"  Filter: None")
        if 'filter_params' in strategy:
            print(f"  Filter params: {strategy['filter_params']}")

print("\n💡 Key findings:")
print("1. Check if 'rsi(14)' is converted to 'rsi_14' in expressions")
print("2. Verify filter parameters are properly expanded")
print("3. Ensure features are defined in the config")
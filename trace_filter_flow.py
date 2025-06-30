#!/usr/bin/env python3
"""
Trace the complete filter flow from config to execution.
"""

import yaml
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from src.core.coordinator.config.clean_syntax_parser import CleanSyntaxParser

def trace_flow():
    print("=== TRACING FILTER FLOW ===\n")
    
    # Load config
    with open('config/keltner/config_2826/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("1. RAW CONFIG:")
    print("-" * 50)
    print(yaml.dump(config, default_flow_style=False))
    
    # Parse with clean syntax parser
    parser = CleanSyntaxParser()
    parsed_config = parser.parse_config(config)
    
    print("\n2. PARSED CONFIG (parameter_space):")
    print("-" * 50)
    if 'parameter_space' in parsed_config:
        print(yaml.dump(parsed_config['parameter_space'], default_flow_style=False))
    
    # The issue might be that filters in list format need to be converted
    print("\n3. FILTER CONVERSION ISSUE:")
    print("-" * 50)
    
    # Current filter in config
    filter_spec = config['strategy'][0]['keltner_bands']['filter']
    print(f"Current filter spec: {filter_spec}")
    print(f"Type: {type(filter_spec)}")
    
    # This is a list with dict, but wrap_strategy_with_filter expects a string expression
    # The filter needs to be converted from list format to expression string
    
    if isinstance(filter_spec, list) and len(filter_spec) > 0:
        # Parse the filter list
        expr, params = parser._parse_combined_filter(filter_spec)
        print(f"\nParsed expression: {expr}")
        print(f"Parameters: {params}")
        
        # This is the expression that should be used
        print(f"\nThe filter wrapper expects this expression string,")
        print(f"but it's getting the raw list instead!")
    
    print("\n" + "="*60)
    print("ROOT CAUSE:")
    print("="*60)
    print("The filter list [{volatility_above: {threshold: X}}] needs")
    print("to be converted to expression 'signal != 0 and atr_14 > atr_sma_50 * X'")
    print("BEFORE being passed to wrap_strategy_with_filter().")
    print("\nThe clean_syntax_parser does this conversion, but it's not")
    print("being called in the right place in the pipeline.")

if __name__ == "__main__":
    trace_flow()
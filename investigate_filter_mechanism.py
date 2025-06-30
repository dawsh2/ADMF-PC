#!/usr/bin/env python3
"""
Investigate how filters work in the system by examining the code and configs.
"""

import yaml
from pathlib import Path
import re

def investigate_filters():
    print("=== INVESTIGATING FILTER MECHANISM ===\n")
    
    # Check parameter expander to see how filters are handled
    print("1. HOW FILTERS ARE EXPANDED:")
    print("-" * 60)
    
    # The key insight from our tests
    print("From our tests:")
    print("- Main config with filters: Expands to 2750 strategies")
    print("- Multiple filter test: Expands to 3 strategies") 
    print("- Each strategy should apply different filter")
    print("\nBut ALL strategies produce identical signals!")
    
    # Check if filters are part of strategy compilation
    print("\n2. FILTER IN STRATEGY COMPILATION:")
    print("-" * 60)
    
    print("The clean_syntax_parser.py shows filters are converted to:")
    print("  'filter': 'signal == 0 or (expression)'")
    print("\nThis suggests filters should be expression strings,")
    print("not dictionaries in the compiled strategy.")
    
    # Look for where signals might be filtered
    print("\n3. WHERE FILTERS COULD BE APPLIED:")
    print("-" * 60)
    
    print("Option A: In strategy function itself")
    print("  - Strategy would need to receive filter expression")
    print("  - keltner_bands() doesn't have filter parameter")
    print("\nOption B: In a wrapper around strategy")
    print("  - ConfigSignalFilter could wrap strategies")
    print("  - But not integrated in signal generation")
    print("\nOption C: Post-processing signals")
    print("  - After strategies generate signals")
    print("  - Before writing to parquet")
    
    # The real issue
    print("\n4. THE REAL ISSUE:")
    print("-" * 60)
    
    print("Signal generation mode (`--signal-generation`) appears to:")
    print("1. Generate raw strategy signals")
    print("2. Store them to parquet files")
    print("3. NOT apply any filters")
    print("\nThe filters might only work in:")
    print("- Backtest mode")
    print("- Analysis pipelines")
    print("- Post-processing steps")
    
    # How the original analysis likely worked
    print("\n5. HOW ORIGINAL ANALYSIS WORKED:")
    print("-" * 60)
    
    print("The original analysis that found 2826 signals likely:")
    print("1. Generated all signals without filters")
    print("2. Applied filters during analysis/backtesting")
    print("3. Counted filtered results")
    print("\nEvidence:")
    print("- workspace_analysis files show filtered results")
    print("- Signal files contain raw signals")
    print("- Filters reduce signal counts during analysis")
    
    # Solution
    print("\n" + "="*60)
    print("SOLUTION:")
    print("="*60)
    
    print("To apply filters, we need to either:")
    print("\n1. Use backtest mode instead of signal generation:")
    print("   python main.py --config config.yaml --backtest")
    print("\n2. Apply filters during analysis:")
    print("   Load signals → Apply filter expressions → Analyze")
    print("\n3. Find the original analysis script that applies filters")
    print("\nThe filter mechanism EXISTS but is NOT integrated")
    print("into the signal generation pipeline.")

if __name__ == "__main__":
    investigate_filters()
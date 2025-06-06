#!/usr/bin/env python3
"""
Test script to demonstrate data path inference capabilities.

This shows how the system can automatically find data files based on
symbol and timeframe without explicit path configuration.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.path_resolver import DataPathResolver


def main():
    """Test path resolution."""
    print("Data Path Inference Demo")
    print("=" * 50)
    
    # Create resolver
    resolver = DataPathResolver(base_dir="data")
    
    # List available data
    print("\nAvailable Data:")
    available = resolver.list_available_data()
    for symbol, timeframes in sorted(available.items()):
        print(f"  {symbol}: {', '.join(sorted(timeframes))}")
    
    print("\n" + "=" * 50)
    print("Path Resolution Tests:")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ("SPY", "1m"),   # SPY 1-minute data
        ("SPY", "1d"),   # SPY daily data
        ("SPY", "1M"),   # SPY monthly data
        ("QQQ", "5m"),   # QQQ 5-minute data
        ("AAPL", "1h"),  # AAPL hourly data
        ("TSLA", "15m"), # TSLA 15-minute data
        ("SPY", "daily"), # Alternative daily notation
        ("SPY", "minute"), # Alternative minute notation
    ]
    
    for symbol, timeframe in test_cases:
        path = resolver.resolve_path(symbol, timeframe)
        if path:
            print(f"✓ {symbol} {timeframe:8} -> {path}")
        else:
            print(f"✗ {symbol} {timeframe:8} -> Not found")
            
            # Show suggestions
            suggestions = resolver.suggest_alternatives(symbol, timeframe)
            if suggestions:
                print(f"  Suggestions:")
                for suggestion in suggestions[:3]:
                    if 'timeframe' in suggestion:
                        print(f"    - Try timeframe '{suggestion['timeframe']}' at {suggestion['path']}")
                    else:
                        print(f"    - Try symbol '{suggestion['symbol']}' ({suggestion['available_files']} files)")
    
    print("\n" + "=" * 50)
    print("Naming Convention Examples:")
    print("=" * 50)
    print("Supported patterns:")
    print("  - SPY_1m.csv    -> 1-minute data")
    print("  - SPY_5m.csv    -> 5-minute data") 
    print("  - SPY_1h.csv    -> Hourly data")
    print("  - SPY_1d.csv    -> Daily data")
    print("  - SPY_daily.csv -> Daily data (alternative)")
    print("  - SPY.csv       -> Daily data (default)")
    print("  - SPY_1M.csv    -> Monthly data (uppercase M)")
    print("  - SPY_1w.csv    -> Weekly data")


if __name__ == "__main__":
    main()
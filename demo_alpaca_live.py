#!/usr/bin/env python3
"""
Demo: Using ADMF-PC with Alpaca Live Data

This demonstrates how to use the --alpaca flag to stream live market data
and run strategies in real-time.

Prerequisites:
1. Install dependencies: pip install -r requirements-alpaca.txt
2. Set Alpaca API credentials:
   export ALPACA_API_KEY='your_api_key'
   export ALPACA_SECRET_KEY='your_secret_key'
"""

import os
import subprocess
import sys

def check_prerequisites():
    """Check if prerequisites are met."""
    # Check API credentials
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
    
    if not api_key or not secret_key:
        print("❌ Alpaca API credentials not found!")
        print("\nPlease set environment variables:")
        print("  export ALPACA_API_KEY='your_api_key'")
        print("  export ALPACA_SECRET_KEY='your_secret_key'")
        return False
    
    print("✅ Alpaca API credentials found")
    
    # Check if websockets is installed
    try:
        import websockets
        print("✅ websockets module installed")
    except ImportError:
        print("❌ websockets module not found")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements-alpaca.txt")
        return False
    
    return True


def demo_alpaca_live():
    """Run live trading demo with Alpaca."""
    
    print("=" * 60)
    print("🚀 ADMF-PC Live Trading Demo with Alpaca")
    print("=" * 60)
    print()
    
    if not check_prerequisites():
        return
    
    print("\n📊 Example 1: Simple Bollinger Bands with Live Data")
    print("-" * 60)
    print("Command: python main.py --config config/bollinger/config.yaml --alpaca --bars 20")
    print("\nThis will:")
    print("• Connect to Alpaca WebSocket API")
    print("• Stream live SPY data")
    print("• Run Bollinger Bands strategy in real-time")
    print("• Use universal topology (signals → portfolio → execution)")
    print()
    
    print("📊 Example 2: Multiple Symbols")
    print("-" * 60)
    print("First, create a config with multiple symbols...")
    
    # Create a multi-symbol config
    multi_symbol_config = """
# Multi-symbol live trading config
name: "multi_symbol_live"
description: "Live trading multiple symbols with Alpaca"

# Data section will be overridden by --alpaca flag
# But we can specify symbols here for reference
data:
  symbols: ["SPY", "QQQ", "AAPL"]

# Simple moving average crossover strategy
strategies:
  - type: ma_crossover
    params:
      fast_period: 5
      slow_period: 20

# Risk management
risk:
  max_position_size: 100
  stop_loss: 0.01

# Portfolio settings
portfolio:
  initial_capital: 100000
"""
    
    config_path = "config/examples/multi_symbol_live.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(multi_symbol_config)
    
    print(f"Created config: {config_path}")
    print(f"\nCommand: python main.py --config {config_path} --alpaca")
    print()
    
    print("📊 Example 3: Live Trading with Specific Parameters")
    print("-" * 60)
    print("You can also run without a config file:")
    print('Command: python main.py --strategies "bollinger_bands:period=20,std_dev=2.0" --alpaca')
    print()
    
    print("🎯 Key Features of --alpaca Flag:")
    print("-" * 60)
    print("• Automatically uses 'universal' topology for complete trading")
    print("• Configures Alpaca WebSocket data streaming")
    print("• Extracts symbols from config or defaults to SPY")
    print("• Uses paper trading API for safety")
    print("• Streams real-time bars during market hours")
    print()
    
    print("💡 Tips:")
    print("-" * 60)
    print("• Use --bars N to limit the number of bars for testing")
    print("• Add --verbose for detailed logging")
    print("• Outside market hours, you'll see connection messages but no data")
    print("• The system will pre-warm indicators with historical data if configured")
    print()
    
    # Ask if user wants to run a demo
    response = input("Would you like to run a quick demo? (y/n): ")
    if response.lower() == 'y':
        print("\n🚀 Running live demo...")
        cmd = [
            sys.executable,
            "main.py",
            "--config", "config/bollinger/config.yaml",
            "--alpaca",
            "--bars", "10",
            "--verbose"
        ]
        print(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd)


if __name__ == "__main__":
    demo_alpaca_live()
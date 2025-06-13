#!/usr/bin/env python3
"""
Direct test of strategy signal generation to debug why signals aren't being produced.
"""

import sys
import pandas as pd
import logging

# Add src to path
sys.path.insert(0, 'src')

from src.strategy.strategies.momentum import momentum_strategy

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load some data
df = pd.read_csv('data/SPY_1m.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Calculate simple features
df['sma_10'] = df['Close'].rolling(10).mean()
df['sma_20'] = df['Close'].rolling(20).mean()
df['rsi_14'] = 50.0  # Fake RSI for testing

print(f"Loaded {len(df)} bars")
print(f"First few bars with features:")
print(df[['timestamp', 'Close', 'sma_10', 'sma_20']].head(25))

# Test strategy on some bars
params = {
    'sma_period': 10,
    'rsi_period': 14,
    'rsi_threshold_long': 30,
    'rsi_threshold_short': 70
}

print("\n" + "="*60)
print("Testing momentum strategy")
print("="*60)

signal_count = 0
for i in range(20, min(50, len(df))):  # Start after we have enough data for indicators
    bar = {
        'symbol': 'SPY',
        'timestamp': df.iloc[i]['timestamp'],
        'open': df.iloc[i]['Open'],
        'high': df.iloc[i]['High'],
        'low': df.iloc[i]['Low'],
        'close': df.iloc[i]['Close'],
        'volume': df.iloc[i]['Volume']
    }
    
    features = {
        'sma_10': df.iloc[i]['sma_10'],
        'sma_20': df.iloc[i]['sma_20'],
        'rsi_14': 25 if i % 5 == 0 else 75 if i % 7 == 0 else 50  # Vary RSI for testing
    }
    
    print(f"\nBar {i}: close={bar['close']:.2f}, sma_10={features['sma_10']:.2f}, sma_20={features['sma_20']:.2f}, rsi={features['rsi_14']}")
    
    signal = momentum_strategy(features, bar, params)
    
    if signal:
        signal_count += 1
        print(f"  *** SIGNAL: {signal['direction']} at {signal['price']:.2f}")
        print(f"      Reason: {signal['reason']}")

print(f"\n\nTotal signals generated: {signal_count}")
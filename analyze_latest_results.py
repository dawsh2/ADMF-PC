#!/usr/bin/env python3
"""Analyze the latest results from position open/close events."""

import pandas as pd
from pathlib import Path

print("=== Analyzing Latest Results ===")

results_dir = Path("config/bollinger/results/latest")

# Load position events
pos_open_file = results_dir / "traces/portfolio/positions_open/positions_open.parquet"
pos_close_file = results_dir / "traces/portfolio/positions_close/positions_close.parquet"

if pos_open_file.exists() and pos_close_file.exists():
    opens = pd.read_parquet(pos_open_file)
    closes = pd.read_parquet(pos_close_file)
    
    print(f"\nPosition opens: {len(opens)}")
    print(f"Position closes: {len(closes)}")
    
    # The number of trades is the number of closed positions
    print(f"\nTotal trades (closed positions): {len(closes)}")
    
    if len(closes) > 0:
        # Check exit types
        if 'exit_type' in closes.columns:
            print("\nExit type breakdown:")
            print(closes['exit_type'].value_counts())
        
        # Calculate returns if price data available
        if 'entry_price' in closes.columns and 'exit_price' in closes.columns:
            closes['return_pct'] = ((closes['exit_price'] - closes['entry_price']) / closes['entry_price']) * 100
            
            # Check for stop losses with gains
            if 'exit_type' in closes.columns:
                stop_losses = closes[closes['exit_type'] == 'stop_loss']
                if len(stop_losses) > 0:
                    sl_with_gains = stop_losses[stop_losses['return_pct'] > 0]
                    print(f"\n⚠️ Stop losses with GAINS: {len(sl_with_gains)} out of {len(stop_losses)}")
                    
                    if len(sl_with_gains) > 0:
                        print("\nFirst few stop losses with gains:")
                        for i, (_, trade) in enumerate(sl_with_gains.head(3).iterrows()):
                            print(f"  {trade.get('timestamp', 'N/A')}: {trade['return_pct']:.4f}%")
            
            # Overall performance
            total_return = closes['return_pct'].sum()
            win_rate = (closes['return_pct'] > 0).mean() * 100
            print(f"\nPerformance:")
            print(f"  Total return: {total_return:.2f}%")
            print(f"  Win rate: {win_rate:.1f}%")
        
        # Check for strategy_id
        if 'strategy_id' in closes.columns:
            missing_strategy = closes['strategy_id'].isna().sum()
            print(f"\nTrades missing strategy_id: {missing_strategy}")
        else:
            print("\n❌ No strategy_id column!")

# Also check signals for OHLC
signals_file = results_dir / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet"
if signals_file.exists():
    signals = pd.read_parquet(signals_file)
    print(f"\n\nSignal analysis:")
    print(f"Total signals: {len(signals)}")
    
    # Check first signal for OHLC
    if len(signals) > 0:
        first_signal = signals.iloc[0]
        print("\nChecking for OHLC in signals...")
        
        # The signal data might be in different formats
        if 'metadata' in first_signal:
            metadata = first_signal['metadata']
            if isinstance(metadata, dict):
                has_ohlc = all(k in metadata for k in ['open', 'high', 'low', 'close'])
                print(f"  Has OHLC data: {has_ohlc}")
                if has_ohlc:
                    print(f"  Sample - High: {metadata['high']}, Low: {metadata['low']}")
            elif isinstance(metadata, str):
                print("  Metadata is string, checking contents...")
                print(f"  {metadata[:100]}...")
        
        # Check columns directly
        for col in ['open', 'high', 'low', 'close']:
            if col in signals.columns:
                print(f"  ✓ Found '{col}' column")

print("\n\n=== Summary ===")
print("This shows the actual results from your latest run.")
print("If the trade count doesn't match 463, you may be looking at old results.")
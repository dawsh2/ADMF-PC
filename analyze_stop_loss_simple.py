#!/usr/bin/env python3
"""
Simple analysis of stop loss behavior for Bollinger Band trades.
Directly reads the parquet files to understand what's happening.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def main():
    workspace = Path("config/bollinger/results/latest")
    
    # Load metadata
    with open(workspace / "metadata.json") as f:
        metadata = json.load(f)
    
    print(f"\n=== Stop Loss Analysis ===")
    print(f"Workspace: {workspace}")
    print(f"Total bars: {metadata['total_bars']}")
    print(f"Total signals: {metadata['total_signals']}")
    print(f"Total orders: {metadata['total_orders']}")
    
    # Extract risk configuration
    strategy_config = metadata['components']['SPY_5m_strategy_0']
    risk_params = strategy_config['parameters'].get('_risk', {})
    print(f"\nRisk Parameters:")
    print(f"  Stop Loss: {risk_params.get('stop_loss', 'Not set')} ({risk_params.get('stop_loss', 0) * 100:.1f}%)")
    print(f"  Take Profit: {risk_params.get('take_profit', 'Not set')} ({risk_params.get('take_profit', 0) * 100:.2f}%)")
    print(f"  Trailing Stop: {risk_params.get('trailing_stop', 'Not set')} ({risk_params.get('trailing_stop', 0) * 100:.2f}%)")
    
    # Load signals
    signals_path = workspace / "traces/signals/bollinger_bands/SPY_5m_strategy_0.parquet"
    signals = pd.read_parquet(signals_path)
    print(f"\n=== Signal Analysis ===")
    print(f"Total signal changes: {len(signals)}")
    print(f"Signal values: {signals['val'].unique()}")
    
    # Load positions
    positions_open_path = workspace / "traces/portfolio/positions_open/positions_open.parquet"
    positions_close_path = workspace / "traces/portfolio/positions_close/positions_close.parquet"
    
    if positions_open_path.exists():
        positions_open = pd.read_parquet(positions_open_path)
        print(f"\nPositions opened: {len(positions_open)}")
    else:
        print("\nNo positions open file found")
        positions_open = pd.DataFrame()
    
    if positions_close_path.exists():
        positions_close = pd.read_parquet(positions_close_path)
        print(f"Positions closed: {len(positions_close)}")
    else:
        print("No positions close file found")
        positions_close = pd.DataFrame()
    
    # Analyze each position
    if not positions_open.empty and not positions_close.empty:
        print(f"\n=== Position Analysis ===")
        
        # Match opens with closes
        for i, open_row in positions_open.iterrows():
            open_metadata = json.loads(open_row['metadata'])
            symbol = open_metadata['symbol']
            quantity = open_metadata['quantity']
            entry_price = open_metadata['entry_price']
            
            print(f"\nPosition {i+1}:")
            print(f"  Symbol: {symbol}")
            print(f"  Quantity: {quantity}")
            print(f"  Entry Price: ${entry_price}")
            
            # Find corresponding close
            if i < len(positions_close):
                close_row = positions_close.iloc[i]
                close_metadata = json.loads(close_row['metadata'])
                exit_price = close_metadata['exit_price']
                realized_pnl = close_metadata['realized_pnl']
                bars_held = close_metadata['metadata'].get('bars_held', 0)
                exit_type = close_metadata.get('exit_type', 'signal_change')
                exit_reason = close_metadata.get('exit_reason', 'N/A')
                
                # Calculate returns
                if quantity > 0:  # Long
                    pct_return = (exit_price - entry_price) / entry_price * 100
                else:  # Short
                    pct_return = (entry_price - exit_price) / entry_price * 100
                
                print(f"  Exit Price: ${exit_price}")
                print(f"  Return: {pct_return:.3f}%")
                print(f"  Realized P&L: ${realized_pnl:.2f}")
                print(f"  Bars Held: {bars_held}")
                print(f"  Exit Type: {exit_type}")
                print(f"  Exit Reason: {exit_reason}")
                
                # Check if stop loss should have triggered
                stop_loss_pct = risk_params.get('stop_loss', 0) * 100
                if stop_loss_pct > 0:
                    if pct_return < -stop_loss_pct:
                        print(f"  ⚠️ ISSUE: Return ({pct_return:.3f}%) exceeded stop loss (-{stop_loss_pct:.1f}%)")
                        print(f"     Stop should have triggered at ${entry_price * (1 - risk_params['stop_loss']):.2f}")
    
    # Load the actual bar data to understand price movements
    print(f"\n=== Looking for bar data ===")
    data_path = Path("./data/SPY_5m.csv")
    if data_path.exists():
        print(f"Found data file: {data_path}")
        bars = pd.read_csv(data_path)
        bars['timestamp'] = pd.to_datetime(bars['timestamp'])
        print(f"Total bars in data: {len(bars)}")
        print(f"Date range: {bars['timestamp'].min()} to {bars['timestamp'].max()}")
    else:
        print(f"No data file found at {data_path}")

if __name__ == "__main__":
    main()
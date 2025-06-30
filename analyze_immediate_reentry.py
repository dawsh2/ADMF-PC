#!/usr/bin/env python3
"""Analyze immediate re-entries after risk exits to debug exit memory."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_trade_data(workspace_dir: str):
    """Load trade data from the most recent workspace."""
    workspace_path = Path(workspace_dir)
    
    # Find the most recent analysis directory
    result_dirs = sorted([d for d in workspace_path.glob("*/") if d.is_dir()], 
                        key=lambda x: x.name, reverse=True)
    
    if not result_dirs:
        raise ValueError(f"No result directories found in {workspace_dir}")
    
    latest_dir = result_dirs[0]
    logger.info(f"Loading data from: {latest_dir}")
    
    # Load position events from traces directory
    positions_open = latest_dir / "traces/portfolio/positions_open/positions_open.parquet"
    positions_close = latest_dir / "traces/portfolio/positions_close/positions_close.parquet"
    fills = latest_dir / "traces/execution/fills/execution_fills.parquet"
    
    # Combine position events
    traces_list = []
    
    if positions_open.exists():
        logger.info(f"Loading position opens from: {positions_open}")
        open_df = pd.read_parquet(positions_open)
        open_df['event_type'] = 'POSITION_OPEN'
        traces_list.append(open_df)
    
    if positions_close.exists():
        logger.info(f"Loading position closes from: {positions_close}")
        close_df = pd.read_parquet(positions_close)
        close_df['event_type'] = 'POSITION_CLOSE'
        traces_list.append(close_df)
    
    if not traces_list:
        raise ValueError(f"No position event files found in {latest_dir}")
    
    traces = pd.concat(traces_list, ignore_index=True)
    
    # Load signals from traces/signals directory
    signal_dir = latest_dir / "traces/signals/bollinger_bands"
    signals = None
    
    if signal_dir.exists():
        signal_files = list(signal_dir.glob("*.parquet"))
        if signal_files:
            signal_file = signal_files[0]  # Take first signal file
            logger.info(f"Loading signals from: {signal_file}")
            signals = pd.read_parquet(signal_file)
        else:
            logger.warning("No signal files found")
    else:
        logger.warning(f"Signal directory not found: {signal_dir}")
    
    return traces, signals

def analyze_immediate_reentries(traces_df: pd.DataFrame, signals_df: pd.DataFrame = None):
    """Analyze immediate re-entries after risk exits."""
    
    # Filter for position events
    position_events = traces_df[traces_df['event_type'].isin(['POSITION_OPEN', 'POSITION_CLOSE'])].copy()
    
    # Rename columns to standard names
    position_events = position_events.rename(columns={'ts': 'timestamp', 'sym': 'symbol'})
    position_events = position_events.sort_values('timestamp')
    
    # Parse metadata column - it contains the event payload
    def parse_metadata(x):
        if isinstance(x, str):
            return json.loads(x)
        elif isinstance(x, dict):
            return x
        else:
            return {}
    
    position_events['parsed_metadata'] = position_events['metadata'].apply(parse_metadata)
    
    # Extract fields from metadata
    position_events['exit_type'] = position_events['parsed_metadata'].apply(lambda x: x.get('exit_type'))
    position_events['exit_reason'] = position_events['parsed_metadata'].apply(lambda x: x.get('exit_reason'))
    position_events['strategy_id'] = position_events['parsed_metadata'].apply(lambda x: x.get('strategy_id', 'strategy_0'))
    position_events['quantity'] = position_events['parsed_metadata'].apply(lambda x: x.get('quantity', 0))
    
    # Find immediate re-entries
    immediate_reentries = []
    
    for i in range(len(position_events) - 1):
        current = position_events.iloc[i]
        next_event = position_events.iloc[i + 1]
        
        # Check if this is a risk exit followed by an open
        if (current['event_type'] == 'POSITION_CLOSE' and 
            next_event['event_type'] == 'POSITION_OPEN' and
            current['exit_type'] in ['stop_loss', 'trailing_stop', 'take_profit'] and
            current['symbol'] == next_event['symbol'] and
            current['strategy_id'] == next_event['strategy_id']):
            
            # Calculate time difference
            time_diff = (next_event['timestamp'] - current['timestamp']).total_seconds()
            
            # Get bar numbers
            current_bar = current.get('bar_number', 'N/A')
            next_bar = next_event.get('bar_number', 'N/A')
            
            immediate_reentries.append({
                'exit_timestamp': current['timestamp'],
                'exit_bar': current_bar,
                'exit_type': current['exit_type'],
                'exit_reason': current['exit_reason'],
                'reentry_timestamp': next_event['timestamp'],
                'reentry_bar': next_bar,
                'time_diff_seconds': time_diff,
                'time_diff_minutes': time_diff / 60,
                'symbol': current['symbol'],
                'strategy_id': current['strategy_id']
            })
    
    reentries_df = pd.DataFrame(immediate_reentries)
    
    # Print summary
    print("\n=== IMMEDIATE RE-ENTRY ANALYSIS ===")
    print(f"Total position closes: {len(position_events[position_events['event_type'] == 'POSITION_CLOSE'])}")
    print(f"Risk-based exits: {len(position_events[(position_events['event_type'] == 'POSITION_CLOSE') & (position_events['exit_type'].notna())])}")
    print(f"Immediate re-entries found: {len(reentries_df)}")
    
    if len(reentries_df) > 0:
        print(f"\nAverage time to re-entry: {reentries_df['time_diff_minutes'].mean():.2f} minutes")
        print(f"Minimum time to re-entry: {reentries_df['time_diff_minutes'].min():.2f} minutes")
        print(f"Maximum time to re-entry: {reentries_df['time_diff_minutes'].max():.2f} minutes")
        
        print("\n=== RE-ENTRIES BY EXIT TYPE ===")
        for exit_type in reentries_df['exit_type'].unique():
            count = len(reentries_df[reentries_df['exit_type'] == exit_type])
            print(f"{exit_type}: {count} re-entries")
        
        print("\n=== FIRST 10 IMMEDIATE RE-ENTRIES ===")
        for idx, row in reentries_df.head(10).iterrows():
            print(f"\n{idx + 1}. {row['symbol']} - {row['strategy_id']}")
            print(f"   Exit: {row['exit_timestamp']} (bar {row['exit_bar']}) - {row['exit_type']}: {row['exit_reason']}")
            print(f"   Re-entry: {row['reentry_timestamp']} (bar {row['reentry_bar']}) - {row['time_diff_minutes']:.2f} minutes later")
    
    # If we have signals, check signal values at re-entry times
    if signals_df is not None and len(reentries_df) > 0:
        print("\n=== SIGNAL ANALYSIS AT RE-ENTRY ===")
        
        # Rename columns for signals
        signals_df = signals_df.rename(columns={'ts': 'timestamp', 'sym': 'symbol', 'val': 'direction', 'strat': 'strategy_id'})
        
        # Convert signals timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(signals_df['timestamp']):
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
        
        # Sort signals by timestamp
        signals_df = signals_df.sort_values('timestamp')
        
        # For each re-entry, find the signal state
        for idx, row in reentries_df.head(5).iterrows():
            print(f"\n{idx + 1}. Re-entry at {row['reentry_timestamp']}")
            
            # Find signals around the exit and re-entry time
            exit_time = row['exit_timestamp']
            reentry_time = row['reentry_timestamp']
            strategy_id = row['strategy_id']
            symbol = row['symbol']
            
            # Get signals for this strategy/symbol around the times
            strategy_signals = signals_df[
                (signals_df['strategy_id'] == strategy_id) & 
                (signals_df['symbol'] == symbol)
            ]
            
            # Find signal at exit
            exit_signal = strategy_signals[strategy_signals['timestamp'] <= exit_time].tail(1)
            if not exit_signal.empty:
                print(f"   Signal at exit: {exit_signal.iloc[0]['direction']} (strength: {exit_signal.iloc[0].get('strength', 'N/A')})")
            
            # Find signal at re-entry
            reentry_signal = strategy_signals[strategy_signals['timestamp'] <= reentry_time].tail(1)
            if not reentry_signal.empty:
                print(f"   Signal at re-entry: {reentry_signal.iloc[0]['direction']} (strength: {reentry_signal.iloc[0].get('strength', 'N/A')})")
                
                # Check if signal changed
                if not exit_signal.empty:
                    exit_dir = exit_signal.iloc[0]['direction']
                    reentry_dir = reentry_signal.iloc[0]['direction']
                    if exit_dir == reentry_dir:
                        print(f"   ⚠️  SIGNAL UNCHANGED - Exit memory should have blocked this!")
                    else:
                        print(f"   ✓ Signal changed from {exit_dir} to {reentry_dir}")
    
    return reentries_df

def main():
    """Main analysis function."""
    # Use the most recent Bollinger results
    workspace_dir = "config/bollinger/results"
    
    try:
        traces, signals = load_trade_data(workspace_dir)
        
        print(f"\nLoaded {len(traces)} trace events")
        if signals is not None:
            print(f"Loaded {len(signals)} signal events")
        
        # Analyze immediate re-entries
        reentries_df = analyze_immediate_reentries(traces, signals)
        
        # Save analysis results
        if len(reentries_df) > 0:
            output_file = "immediate_reentries_analysis.csv"
            reentries_df.to_csv(output_file, index=False)
            print(f"\n✅ Saved detailed analysis to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
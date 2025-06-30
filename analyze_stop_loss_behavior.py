#!/usr/bin/env python3
"""
Analyze why stop losses aren't triggering in Bollinger Band trades.
This script examines trace data to understand stop loss behavior.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_traces(workspace_path: Path) -> pd.DataFrame:
    """Load and combine all trace parquet files."""
    traces_dir = workspace_path / "traces"
    if not traces_dir.exists():
        raise ValueError(f"No traces directory found at {traces_dir}")
    
    all_traces = []
    for strategy_dir in traces_dir.iterdir():
        if strategy_dir.is_dir():
            for parquet_file in strategy_dir.glob("*.parquet"):
                df = pd.read_parquet(parquet_file)
                all_traces.append(df)
    
    if not all_traces:
        raise ValueError("No trace files found")
    
    combined = pd.concat(all_traces, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    return combined.sort_values('timestamp')


def analyze_position_lifecycle(traces: pd.DataFrame, strategy_id: int) -> List[Dict]:
    """Analyze complete lifecycle of positions for a strategy."""
    strategy_traces = traces[traces['strategy_id'] == strategy_id].copy()
    
    positions = []
    current_position = None
    
    for _, row in strategy_traces.iterrows():
        event_type = row.get('event_type', '')
        
        if 'signal_generated' in event_type and row.get('signal', 0) != 0:
            # New position
            if current_position is not None:
                positions.append(current_position)
            
            current_position = {
                'entry_time': row['timestamp'],
                'entry_price': row.get('price', row.get('close', 0)),
                'signal': row.get('signal', 0),
                'strategy_id': strategy_id,
                'exit_time': None,
                'exit_price': None,
                'exit_reason': None,
                'min_price': row.get('price', row.get('close', 0)),
                'max_price': row.get('price', row.get('close', 0)),
                'price_history': [(row['timestamp'], row.get('price', row.get('close', 0)))],
                'events': [event_type]
            }
        
        elif current_position is not None:
            # Update price tracking
            current_price = row.get('price', row.get('close', 0))
            if current_price > 0:
                current_position['price_history'].append((row['timestamp'], current_price))
                current_position['min_price'] = min(current_position['min_price'], current_price)
                current_position['max_price'] = max(current_position['max_price'], current_price)
                current_position['events'].append(event_type)
            
            # Check for exit
            if 'position_closed' in event_type or 'order_executed' in event_type:
                current_position['exit_time'] = row['timestamp']
                current_position['exit_price'] = current_price
                current_position['exit_reason'] = event_type
                positions.append(current_position)
                current_position = None
    
    # Add any open position
    if current_position is not None:
        positions.append(current_position)
    
    return positions


def calculate_drawdowns(position: Dict) -> Dict:
    """Calculate maximum adverse excursion for a position."""
    if position['signal'] > 0:  # Long position
        entry_price = position['entry_price']
        min_price = position['min_price']
        max_drawdown_pct = (min_price - entry_price) / entry_price * 100
        max_drawdown_price = min_price
    else:  # Short position
        entry_price = position['entry_price']
        max_price = position['max_price']
        max_drawdown_pct = (max_price - entry_price) / entry_price * 100
        max_drawdown_price = max_price
    
    return {
        'max_drawdown_pct': max_drawdown_pct,
        'max_drawdown_price': max_drawdown_price,
        'should_stop_trigger': abs(max_drawdown_pct) >= 0.1
    }


def analyze_risk_rules(traces: pd.DataFrame) -> Dict:
    """Analyze risk rule setup and behavior."""
    risk_events = traces[traces['event_type'].str.contains('risk', case=False, na=False)]
    
    risk_analysis = {
        'total_risk_events': len(risk_events),
        'risk_rules_by_strategy': {},
        'stop_loss_triggers': 0,
        'position_limit_triggers': 0
    }
    
    for _, event in risk_events.iterrows():
        strategy_id = event.get('strategy_id', 'unknown')
        if strategy_id not in risk_analysis['risk_rules_by_strategy']:
            risk_analysis['risk_rules_by_strategy'][strategy_id] = {
                'events': [],
                'stop_loss_configured': False,
                'stop_loss_value': None
            }
        
        risk_analysis['risk_rules_by_strategy'][strategy_id]['events'].append({
            'timestamp': event['timestamp'],
            'type': event['event_type'],
            'details': event.to_dict()
        })
        
        # Check for stop loss configuration
        if 'stop_loss' in str(event.get('event_type', '')).lower():
            risk_analysis['risk_rules_by_strategy'][strategy_id]['stop_loss_configured'] = True
            if 'threshold' in event or 'stop_loss' in event:
                risk_analysis['risk_rules_by_strategy'][strategy_id]['stop_loss_value'] = 0.001  # 0.1%
        
        # Count triggers
        if 'stop_loss' in str(event.get('event_type', '')).lower() and 'triggered' in str(event.get('event_type', '')).lower():
            risk_analysis['stop_loss_triggers'] += 1
    
    return risk_analysis


def analyze_stop_loss_behavior(workspace_path: Path):
    """Main analysis function."""
    print(f"\n=== Stop Loss Behavior Analysis ===")
    print(f"Workspace: {workspace_path}")
    print(f"Timestamp: {datetime.now()}")
    
    # Load traces
    try:
        traces = load_traces(workspace_path)
        print(f"\nLoaded {len(traces)} trace events")
    except Exception as e:
        print(f"Error loading traces: {e}")
        return
    
    # Get unique strategies
    strategies = traces['strategy_id'].unique()
    print(f"Found {len(strategies)} strategies")
    
    # Analyze risk rules
    print("\n=== Risk Rule Analysis ===")
    risk_analysis = analyze_risk_rules(traces)
    print(f"Total risk events: {risk_analysis['total_risk_events']}")
    print(f"Stop loss triggers: {risk_analysis['stop_loss_triggers']}")
    
    # Analyze positions for each strategy
    all_positions = []
    for strategy_id in strategies:
        positions = analyze_position_lifecycle(traces, strategy_id)
        all_positions.extend(positions)
    
    print(f"\n=== Position Analysis ===")
    print(f"Total positions analyzed: {len(all_positions)}")
    
    # Analyze stop loss behavior
    positions_that_should_stop = []
    positions_that_did_stop = []
    
    for pos in all_positions:
        if pos['exit_time'] is None:
            continue  # Skip open positions
        
        drawdown_info = calculate_drawdowns(pos)
        pos.update(drawdown_info)
        
        if drawdown_info['should_stop_trigger']:
            positions_that_should_stop.append(pos)
            
            if 'stop' in str(pos.get('exit_reason', '')).lower():
                positions_that_did_stop.append(pos)
    
    print(f"\nPositions that should have stopped: {len(positions_that_should_stop)}")
    print(f"Positions that actually stopped: {len(positions_that_did_stop)}")
    
    # Detailed analysis of positions that should have stopped
    if positions_that_should_stop:
        print("\n=== Positions That Should Have Triggered Stop Loss ===")
        for i, pos in enumerate(positions_that_should_stop[:10]):  # Show first 10
            duration = (pos['exit_time'] - pos['entry_time']).total_seconds() / 60
            print(f"\nPosition {i+1}:")
            print(f"  Strategy ID: {pos['strategy_id']}")
            print(f"  Entry: {pos['entry_time']} @ ${pos['entry_price']:.2f}")
            print(f"  Exit: {pos['exit_time']} @ ${pos['exit_price']:.2f}")
            print(f"  Duration: {duration:.1f} minutes")
            print(f"  Signal: {'LONG' if pos['signal'] > 0 else 'SHORT'}")
            print(f"  Max Drawdown: {pos['max_drawdown_pct']:.3f}%")
            print(f"  Exit Reason: {pos['exit_reason']}")
            print(f"  Should Stop: {'YES' if pos['should_stop_trigger'] else 'NO'}")
            print(f"  Did Stop: {'YES' if 'stop' in str(pos.get('exit_reason', '')).lower() else 'NO'}")
    
    # Analyze price movements during positions
    print("\n=== Price Movement Analysis ===")
    
    long_positions = [p for p in all_positions if p['signal'] > 0 and p['exit_time'] is not None]
    short_positions = [p for p in all_positions if p['signal'] < 0 and p['exit_time'] is not None]
    
    if long_positions:
        long_drawdowns = [calculate_drawdowns(p)['max_drawdown_pct'] for p in long_positions]
        print(f"\nLong Positions ({len(long_positions)} total):")
        print(f"  Average Max Drawdown: {np.mean(long_drawdowns):.3f}%")
        print(f"  Worst Drawdown: {np.min(long_drawdowns):.3f}%")
        print(f"  Positions exceeding 0.1% drawdown: {sum(1 for d in long_drawdowns if d <= -0.1)}")
    
    if short_positions:
        short_drawdowns = [calculate_drawdowns(p)['max_drawdown_pct'] for p in short_positions]
        print(f"\nShort Positions ({len(short_positions)} total):")
        print(f"  Average Max Drawdown: {np.mean(short_drawdowns):.3f}%")
        print(f"  Worst Drawdown: {np.max(short_drawdowns):.3f}%")
        print(f"  Positions exceeding 0.1% drawdown: {sum(1 for d in short_drawdowns if d >= 0.1)}")
    
    # Check event flow for sample positions
    if positions_that_should_stop and len(positions_that_should_stop) > 0:
        print("\n=== Event Flow Analysis (Sample Position) ===")
        sample_pos = positions_that_should_stop[0]
        print(f"\nAnalyzing position from {sample_pos['entry_time']} to {sample_pos['exit_time']}")
        print(f"Events during position:")
        for event in sample_pos['events'][:20]:  # Show first 20 events
            print(f"  - {event}")
    
    # Summary statistics
    print("\n=== Summary Statistics ===")
    total_completed = len([p for p in all_positions if p['exit_time'] is not None])
    print(f"Total completed positions: {total_completed}")
    
    if total_completed > 0:
        stop_loss_rate = len(positions_that_did_stop) / total_completed * 100
        should_stop_rate = len(positions_that_should_stop) / total_completed * 100
        
        print(f"Positions that should have stopped: {should_stop_rate:.1f}%")
        print(f"Positions that actually stopped: {stop_loss_rate:.1f}%")
        print(f"Stop loss effectiveness: {stop_loss_rate/should_stop_rate*100 if should_stop_rate > 0 else 0:.1f}%")
    
    # Look for specific stop loss events
    print("\n=== Stop Loss Event Search ===")
    stop_events = traces[traces['event_type'].str.contains('stop', case=False, na=False)]
    print(f"Found {len(stop_events)} events containing 'stop'")
    
    if len(stop_events) > 0:
        print("\nSample stop events:")
        for _, event in stop_events.head(5).iterrows():
            print(f"  {event['timestamp']}: {event['event_type']} (Strategy: {event.get('strategy_id', 'N/A')})")
    
    # Save detailed results
    results_df = pd.DataFrame(all_positions)
    if len(results_df) > 0:
        results_df['duration_minutes'] = results_df.apply(
            lambda x: (x['exit_time'] - x['entry_time']).total_seconds() / 60 if x['exit_time'] else None, 
            axis=1
        )
        results_df.to_csv('stop_loss_analysis_results.csv', index=False)
        print(f"\nDetailed results saved to: stop_loss_analysis_results.csv")


if __name__ == "__main__":
    # Find the latest Bollinger Band results
    bollinger_results = Path("config/bollinger/results")
    
    if bollinger_results.exists():
        # Get the most recent results directory
        result_dirs = [d for d in bollinger_results.iterdir() if d.is_dir()]
        if result_dirs:
            latest_dir = max(result_dirs, key=lambda d: d.stat().st_mtime)
            print(f"Analyzing results from: {latest_dir}")
            analyze_stop_loss_behavior(latest_dir)
        else:
            print("No result directories found in config/bollinger/results/")
    else:
        # Try the pattern used in the workspace structure
        workspace_dirs = list(Path(".").glob("bollinger_*"))
        if workspace_dirs:
            latest_workspace = max(workspace_dirs, key=lambda d: d.stat().st_mtime)
            print(f"Analyzing workspace: {latest_workspace}")
            analyze_stop_loss_behavior(latest_workspace)
        else:
            print("No Bollinger Band results found. Please specify a workspace path.")
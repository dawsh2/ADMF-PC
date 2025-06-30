"""
Compare trades between universal analysis and execution engine to understand profit target differences.
"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from analytics.universal_analysis import extract_trades
from analytics.functions import apply_stop_target

def load_universal_trades(workspace_dir: str, strategy_id: str):
    """Load trades from universal analysis with stop/target applied."""
    conn = duckdb.connect()
    
    # Load position data
    query = f"""
    SELECT 
        p.*,
        s.timestamp as signal_timestamp,
        s.signal_type,
        s.signal_strength
    FROM read_parquet('{workspace_dir}/sparse_signals/*/*.parquet') s
    LEFT JOIN read_parquet('{workspace_dir}/position_data/*/*.parquet') p 
        ON s.source_file = p.source_file 
        AND s.timestamp = p.timestamp
    WHERE s.strategy_id = '{strategy_id}'
        AND s.signal_type != 'NEUTRAL'
    ORDER BY s.timestamp
    """
    
    df = conn.execute(query).df()
    
    # Extract trades with re-entry prevention
    trades = extract_trades(df, prevent_reentry=True)
    
    # Apply stop/target exits
    trades_with_exits = apply_stop_target(
        trades, 
        stop_loss=0.01,  # 1% stop loss
        profit_target=0.02  # 2% profit target
    )
    
    return trades_with_exits

def load_execution_trades(workspace_dir: str, strategy_id: str):
    """Load trades from execution engine position traces."""
    conn = duckdb.connect()
    
    # Load position traces
    query = f"""
    WITH position_changes AS (
        SELECT 
            *,
            LAG(position_size, 1, 0) OVER (ORDER BY timestamp) as prev_position
        FROM read_parquet('{workspace_dir}/position_traces/*/*.parquet')
        WHERE strategy_id = '{strategy_id}'
    ),
    trades AS (
        SELECT 
            timestamp as entry_time,
            entry_price,
            position_size,
            exit_time,
            exit_price,
            realized_pnl,
            exit_reason,
            LEAD(timestamp) OVER (ORDER BY timestamp) as next_entry_time
        FROM position_changes
        WHERE position_size != 0 AND prev_position = 0  -- Entry points
    )
    SELECT 
        t.*,
        -- Calculate returns
        CASE 
            WHEN t.position_size > 0 THEN (t.exit_price - t.entry_price) / t.entry_price
            ELSE (t.entry_price - t.exit_price) / t.entry_price
        END as return
    FROM trades t
    WHERE t.exit_time IS NOT NULL
    ORDER BY t.entry_time
    """
    
    trades = conn.execute(query).df()
    return trades

def find_matching_trades(universal_trades, execution_trades, time_tolerance_seconds=1):
    """Find trades that match between systems based on entry time."""
    matches = []
    
    for _, u_trade in universal_trades.iterrows():
        u_entry = pd.to_datetime(u_trade['entry_time'])
        
        # Find execution trades within time tolerance
        for _, e_trade in execution_trades.iterrows():
            e_entry = pd.to_datetime(e_trade['entry_time'])
            
            if abs((u_entry - e_entry).total_seconds()) <= time_tolerance_seconds:
                matches.append({
                    'entry_time': u_entry,
                    'universal': u_trade,
                    'execution': e_trade
                })
                break
    
    return matches

def analyze_divergent_trades(matches, price_data):
    """Analyze trades where exit types differ between systems."""
    divergent = []
    
    for match in matches:
        u_trade = match['universal']
        e_trade = match['execution']
        
        # Check if exit types differ
        u_exit_type = u_trade.get('exit_type', 'signal')
        e_exit_type = e_trade.get('exit_reason', 'signal')
        
        if u_exit_type != e_exit_type:
            # Get price data during trade
            entry_time = pd.to_datetime(match['entry_time'])
            u_exit_time = pd.to_datetime(u_trade['exit_time'])
            e_exit_time = pd.to_datetime(e_trade['exit_time'])
            
            # Get price range during both exit windows
            max_exit = max(u_exit_time, e_exit_time)
            
            trade_prices = price_data[
                (price_data['timestamp'] >= entry_time) & 
                (price_data['timestamp'] <= max_exit)
            ]
            
            divergent.append({
                'entry_time': entry_time,
                'entry_price': u_trade['entry_price'],
                'universal_exit_time': u_exit_time,
                'universal_exit_price': u_trade['exit_price'],
                'universal_exit_type': u_exit_type,
                'universal_return': u_trade['return'],
                'execution_exit_time': e_exit_time,
                'execution_exit_price': e_trade['exit_price'],
                'execution_exit_type': e_exit_type,
                'execution_return': e_trade['return'],
                'price_high': trade_prices['high'].max() if len(trade_prices) > 0 else np.nan,
                'price_low': trade_prices['low'].min() if len(trade_prices) > 0 else np.nan,
                'price_data': trade_prices
            })
    
    return divergent

def main():
    # Configuration
    workspace_dir = "workspaces/bollinger_rsi_workspace_20250118_165732"
    strategy_id = "bollinger_10"
    
    print("Loading universal analysis trades...")
    universal_trades = load_universal_trades(workspace_dir, strategy_id)
    print(f"Universal trades: {len(universal_trades)}")
    print(f"Universal profit targets hit: {(universal_trades['exit_type'] == 'target').sum()}")
    print(f"Universal stops hit: {(universal_trades['exit_type'] == 'stop').sum()}")
    print(f"Universal signal exits: {(universal_trades['exit_type'] == 'signal').sum()}")
    
    print("\nLoading execution engine trades...")
    execution_trades = load_execution_trades(workspace_dir, strategy_id)
    print(f"Execution trades: {len(execution_trades)}")
    print(f"Execution profit targets hit: {(execution_trades['exit_reason'] == 'target').sum()}")
    print(f"Execution stops hit: {(execution_trades['exit_reason'] == 'stop').sum()}")
    print(f"Execution signal exits: {(execution_trades['exit_reason'] == 'signal').sum()}")
    
    print("\nFinding matching trades...")
    matches = find_matching_trades(universal_trades, execution_trades)
    print(f"Found {len(matches)} matching trades")
    
    # Load price data for analysis
    print("\nLoading price data...")
    conn = duckdb.connect()
    price_data = conn.execute(f"""
        SELECT timestamp, open, high, low, close
        FROM read_parquet('{workspace_dir}/position_data/*/*.parquet')
        ORDER BY timestamp
    """).df()
    
    print("\nAnalyzing divergent trades...")
    divergent = analyze_divergent_trades(matches, price_data)
    print(f"Found {len(divergent)} trades with different exit types")
    
    # Display first 10 divergent trades
    print("\nFirst 10 divergent trades:")
    print("-" * 100)
    
    for i, trade in enumerate(divergent[:10]):
        print(f"\nTrade {i+1}:")
        print(f"  Entry: {trade['entry_time']} @ ${trade['entry_price']:.2f}")
        print(f"  Universal: Exit {trade['universal_exit_type']} @ {trade['universal_exit_time']} "
              f"Price ${trade['universal_exit_price']:.2f} Return {trade['universal_return']*100:.2f}%")
        print(f"  Execution: Exit {trade['execution_exit_type']} @ {trade['execution_exit_time']} "
              f"Price ${trade['execution_exit_price']:.2f} Return {trade['execution_return']*100:.2f}%")
        print(f"  Price range during trade: Low ${trade['price_low']:.2f}, High ${trade['price_high']:.2f}")
        
        # Check if profit target should have been hit
        if trade['universal_exit_type'] == 'target' and trade['execution_exit_type'] != 'target':
            target_price = trade['entry_price'] * 1.02  # 2% target
            if trade['price_high'] >= target_price:
                print(f"  ⚠️  Price DID reach target ${target_price:.2f} - execution should have exited!")
            else:
                print(f"  ❓ Price never reached target ${target_price:.2f} - universal exit may be wrong")
    
    # Summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    # Count exit type mismatches
    exit_mismatches = {}
    for trade in divergent:
        key = f"{trade['universal_exit_type']} -> {trade['execution_exit_type']}"
        exit_mismatches[key] = exit_mismatches.get(key, 0) + 1
    
    print("\nExit type transitions (Universal -> Execution):")
    for transition, count in sorted(exit_mismatches.items(), key=lambda x: x[1], reverse=True):
        print(f"  {transition}: {count} trades")
    
    # Check specific case: Universal hits target, Execution doesn't
    target_to_other = [t for t in divergent if t['universal_exit_type'] == 'target' and t['execution_exit_type'] != 'target']
    print(f"\nUniversal hits target but Execution doesn't: {len(target_to_other)} trades")
    
    if len(target_to_other) > 0:
        # Verify with price data
        verified_misses = 0
        for trade in target_to_other:
            target_price = trade['entry_price'] * 1.02
            if trade['price_high'] >= target_price:
                verified_misses += 1
        
        print(f"  Of these, {verified_misses} actually reached the target price based on high prices")
        print(f"  This suggests execution engine may be missing profit targets!")

if __name__ == "__main__":
    main()
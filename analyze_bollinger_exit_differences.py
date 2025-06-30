"""
Analyze why execution engine hits fewer profit targets than universal analysis.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

def load_bollinger_results():
    """Load Bollinger strategy results from recent runs."""
    # Find workspaces with Bollinger strategies
    conn = duckdb.connect()
    
    query = """
    WITH workspace_files AS (
        SELECT 
            filename,
            regexp_extract(filename, 'workspaces/([^/]+)/', 1) as workspace,
            regexp_extract(filename, 'strategies/([^/]+)/', 1) as strategy_type
        FROM glob('workspaces/*/analytics.duckdb')
    ),
    bollinger_workspaces AS (
        SELECT DISTINCT w.workspace
        FROM workspace_files w
        WHERE EXISTS (
            SELECT 1 
            FROM duckdb_scan(w.filename, 'strategies') s
            WHERE s.strategy_id LIKE '%bollinger%'
        )
    )
    SELECT * FROM bollinger_workspaces
    ORDER BY workspace DESC
    LIMIT 10
    """
    
    try:
        workspaces = conn.execute(query).df()
        return workspaces
    except:
        # Manually list some known workspaces
        return pd.DataFrame({
            'workspace': [
                'signal_generation_7112529b',  # Has bollinger_rsi_dependent
                'signal_generation_2434c159',  # Has bollinger_rsi_confirmed
                'signal_generation_cc86ee82',  # Has bollinger_rsi_simple_signals
            ]
        })

def analyze_workspace_trades(workspace_name):
    """Analyze trades from a specific workspace."""
    conn = duckdb.connect(f'workspaces/{workspace_name}/analytics.duckdb')
    
    # Get strategies
    strategies = conn.execute("""
        SELECT * FROM strategies 
        WHERE strategy_id LIKE '%bollinger%'
    """).df()
    
    print(f"\nWorkspace: {workspace_name}")
    print(f"Bollinger strategies found: {len(strategies)}")
    
    if len(strategies) == 0:
        return None
    
    # For each strategy, analyze the results
    for _, strategy in strategies.iterrows():
        print(f"\n{'='*80}")
        print(f"Strategy: {strategy['strategy_id']}")
        print(f"Parameters: {strategy['parameters']}")
        print(f"Total trades: {strategy['total_trades']}")
        print(f"Win rate: {strategy['win_rate']:.2%}")
        print(f"Avg return: {strategy['avg_return']:.4f}")
        
        # Try to get more detailed metrics if available
        if 'exit_metrics' in strategy and strategy['exit_metrics'] is not None:
            print(f"\nExit metrics: {strategy['exit_metrics']}")

def compare_universal_vs_execution():
    """Compare universal analysis vs execution engine results."""
    
    # First, let's examine a specific Bollinger strategy file
    signal_file = "workspaces/signal_generation_7112529b/traces/SPY_1m/signals/bollinger_rsi_dependent/SPY_compiled_strategy_0.parquet"
    
    if Path(signal_file).exists():
        print(f"\nAnalyzing signal file: {signal_file}")
        
        conn = duckdb.connect()
        
        # Load the signal data
        signals = conn.execute(f"""
            SELECT * FROM read_parquet('{signal_file}')
            ORDER BY ts
            LIMIT 20
        """).df()
        
        print("\nSignal data structure:")
        print(signals.columns.tolist())
        print("\nFirst few signals:")
        print(signals.head(10))
        
        # Count signal types
        signal_counts = conn.execute(f"""
            SELECT 
                val as signal_value,
                COUNT(*) as count
            FROM read_parquet('{signal_file}')
            GROUP BY val
            ORDER BY count DESC
        """).df()
        
        print("\nSignal distribution:")
        print(signal_counts)
        
        # Now let's manually apply stop/target logic to understand the difference
        print("\n" + "="*80)
        print("MANUAL TRADE EXTRACTION WITH STOP/TARGET")
        print("="*80)
        
        # Extract trades manually
        trades = extract_trades_with_stops(signal_file)
        
        if trades is not None:
            print(f"\nTotal trades extracted: {len(trades)}")
            print(f"Trades hitting profit target: {(trades['exit_type'] == 'target').sum()}")
            print(f"Trades hitting stop loss: {(trades['exit_type'] == 'stop').sum()}")
            print(f"Trades exiting on signal: {(trades['exit_type'] == 'signal').sum()}")
            
            # Show first few trades that hit targets
            target_trades = trades[trades['exit_type'] == 'target'].head(5)
            if len(target_trades) > 0:
                print("\nExample trades hitting profit target:")
                for _, trade in target_trades.iterrows():
                    print(f"  Entry: {trade['entry_time']} @ ${trade['entry_price']:.2f}")
                    print(f"  Exit:  {trade['exit_time']} @ ${trade['exit_price']:.2f}")
                    print(f"  Return: {trade['return']*100:.2f}%")
                    print(f"  Target price was: ${trade['target_price']:.2f}")
                    print(f"  High during trade: ${trade['high_during_trade']:.2f}")
                    print()

def extract_trades_with_stops(signal_file, stop_loss=0.01, profit_target=0.02):
    """Extract trades and apply stop/target logic, showing the process."""
    
    conn = duckdb.connect()
    
    # Load signals with associated price data
    # We need to match signals with price data from the same time period
    base_query = f"""
    WITH signal_data AS (
        SELECT 
            ts as timestamp,
            val as signal_value,
            strategy_hash
        FROM read_parquet('{signal_file}')
    ),
    -- Find corresponding price data file
    price_files AS (
        SELECT filename 
        FROM glob('data/1m/SPY/*.parquet')
        UNION ALL
        SELECT filename
        FROM glob('data/SPY_1m.parquet')
    ),
    price_data AS (
        SELECT 
            timestamp,
            open,
            high,
            low,
            close
        FROM read_parquet((SELECT filename FROM price_files LIMIT 1))
    )
    SELECT 
        s.*,
        p.open,
        p.high,
        p.low,
        p.close
    FROM signal_data s
    LEFT JOIN price_data p ON DATE_TRUNC('minute', s.timestamp) = DATE_TRUNC('minute', p.timestamp)
    WHERE p.close IS NOT NULL
    ORDER BY s.timestamp
    """
    
    try:
        df = conn.execute(base_query).df()
    except:
        print("Could not match signals with price data")
        return None
    
    print(f"Loaded {len(df)} signals with price data")
    
    # Extract trades with detailed tracking
    trades = []
    position = 0
    entry_time = None
    entry_price = None
    entry_idx = None
    
    for idx, row in df.iterrows():
        signal = row['signal_value']
        
        if position == 0 and signal != 0:
            # Enter position
            position = signal
            entry_time = row['timestamp']
            entry_price = row['close']
            entry_idx = idx
            print(f"\nEntering {'LONG' if signal > 0 else 'SHORT'} at {entry_time} @ ${entry_price:.2f}")
            
        elif position != 0 and signal != position:
            # Exit position
            exit_time = row['timestamp']
            exit_price = row['close']
            
            # Get price data during the trade
            trade_data = df.iloc[entry_idx:idx+1]
            
            # Check if stop or target was hit
            exit_type = 'signal'
            actual_exit_price = exit_price
            actual_exit_time = exit_time
            
            if position > 0:  # Long position
                stop_price = entry_price * (1 - stop_loss)
                target_price = entry_price * (1 + profit_target)
                
                # Check each bar
                for _, bar in trade_data.iterrows():
                    if bar['low'] <= stop_price:
                        exit_type = 'stop'
                        actual_exit_price = stop_price
                        actual_exit_time = bar['timestamp']
                        print(f"  Stop hit at {actual_exit_time} (low: ${bar['low']:.2f})")
                        break
                    elif bar['high'] >= target_price:
                        exit_type = 'target'
                        actual_exit_price = target_price
                        actual_exit_time = bar['timestamp']
                        print(f"  Target hit at {actual_exit_time} (high: ${bar['high']:.2f})")
                        break
            
            else:  # Short position
                stop_price = entry_price * (1 + stop_loss)
                target_price = entry_price * (1 - profit_target)
                
                for _, bar in trade_data.iterrows():
                    if bar['high'] >= stop_price:
                        exit_type = 'stop'
                        actual_exit_price = stop_price
                        actual_exit_time = bar['timestamp']
                        print(f"  Stop hit at {actual_exit_time} (high: ${bar['high']:.2f})")
                        break
                    elif bar['low'] <= target_price:
                        exit_type = 'target'
                        actual_exit_price = target_price
                        actual_exit_time = bar['timestamp']
                        print(f"  Target hit at {actual_exit_time} (low: ${bar['low']:.2f})")
                        break
            
            # Calculate return
            if position > 0:
                ret = (actual_exit_price - entry_price) / entry_price
            else:
                ret = (entry_price - actual_exit_price) / entry_price
            
            print(f"Exiting at {actual_exit_time} @ ${actual_exit_price:.2f} ({exit_type})")
            print(f"Return: {ret*100:.2f}%")
            
            trades.append({
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': actual_exit_time,
                'exit_price': actual_exit_price,
                'exit_type': exit_type,
                'return': ret,
                'position': position,
                'target_price': target_price,
                'stop_price': stop_price,
                'high_during_trade': trade_data['high'].max(),
                'low_during_trade': trade_data['low'].min()
            })
            
            # Reset position
            position = 0
            if signal != 0:
                # Re-enter if signal is opposite direction
                position = signal
                entry_time = row['timestamp']
                entry_price = row['close']
                entry_idx = idx
                print(f"\nRe-entering {'LONG' if signal > 0 else 'SHORT'} at {entry_time} @ ${entry_price:.2f}")
    
    return pd.DataFrame(trades) if trades else None

def main():
    print("ANALYZING BOLLINGER STRATEGY EXIT DIFFERENCES")
    print("=" * 80)
    
    # First, find workspaces with Bollinger strategies
    workspaces = load_bollinger_results()
    print(f"Found {len(workspaces)} workspaces with Bollinger strategies")
    
    # Analyze each workspace
    for _, row in workspaces.iterrows():
        workspace = row['workspace']
        try:
            analyze_workspace_trades(workspace)
        except Exception as e:
            print(f"Error analyzing {workspace}: {e}")
    
    # Then do detailed comparison
    print("\n" + "="*80)
    print("DETAILED UNIVERSAL VS EXECUTION COMPARISON")
    print("="*80)
    
    compare_universal_vs_execution()

if __name__ == "__main__":
    main()
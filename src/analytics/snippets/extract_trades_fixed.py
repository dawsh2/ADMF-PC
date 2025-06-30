def extract_trades_fixed(strategy_hash, trace_path, market_data, execution_cost_bps=1.0):
    """Extract individual trades from strategy signals with proper timezone handling."""
    trace_df = pd.read_parquet(trace_path)
    strategy_signals = trace_df[trace_df['strategy_hash'] == strategy_hash].copy()
    
    if len(strategy_signals) == 0:
        return pd.DataFrame()
    
    strategy_signals = strategy_signals.sort_values('ts')
    
    # Work with a copy to avoid modifying the original
    market_data_copy = market_data.copy()
    
    # Ensure both timestamps are timezone-naive datetime64[ns]
    # Convert signal timestamps (they come as strings from parquet)
    strategy_signals['ts'] = pd.to_datetime(strategy_signals['ts'])
    if hasattr(strategy_signals['ts'].dtype, 'tz'):
        strategy_signals['ts'] = strategy_signals['ts'].dt.tz_localize(None)
    
    # Ensure market data timestamp is also timezone-naive
    if not pd.api.types.is_datetime64_any_dtype(market_data_copy['timestamp']):
        market_data_copy['timestamp'] = pd.to_datetime(market_data_copy['timestamp'])
    if hasattr(market_data_copy['timestamp'].dtype, 'tz') and market_data_copy['timestamp'].dt.tz is not None:
        market_data_copy['timestamp'] = market_data_copy['timestamp'].dt.tz_localize(None)
    
    trades = []
    current_position = 0
    entry_idx = None
    entry_price = None
    entry_time = None
    
    for idx, signal in strategy_signals.iterrows():
        signal_value = signal['val']
        timestamp = signal['ts']
        
        # Find closest timestamp match (within 1 minute tolerance)
        time_diff = (market_data_copy['timestamp'] - timestamp).abs()
        closest_idx = time_diff.idxmin()
        
        if time_diff[closest_idx] <= pd.Timedelta(minutes=1):
            market_idx = closest_idx
        else:
            continue
        
        # Handle both 'close' and 'Close' column names
        close_col = 'Close' if 'Close' in market_data_copy.columns else 'close'
        current_price = market_data_copy.loc[market_idx, close_col]
        
        if current_position == 0 and signal_value != 0:
            current_position = signal_value
            entry_idx = market_idx
            entry_price = current_price
            entry_time = timestamp
            
        elif current_position != 0 and signal_value != current_position:
            if entry_idx is not None:
                exit_idx = market_idx
                exit_price = current_price
                exit_time = timestamp
                
                direction = 1 if current_position > 0 else -1
                
                if direction == 1:
                    raw_return = (exit_price - entry_price) / entry_price
                else:
                    raw_return = (entry_price - exit_price) / entry_price
                
                execution_cost = execution_cost_bps / 10000
                net_return = raw_return - execution_cost * 2
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'entry_idx': entry_idx,
                    'exit_idx': exit_idx,
                    'direction': direction,
                    'raw_return': raw_return,
                    'execution_cost': execution_cost * 2,
                    'net_return': net_return,
                    'duration_bars': exit_idx - entry_idx
                })
            
            current_position = signal_value
            if signal_value != 0:
                entry_idx = market_idx
                entry_price = current_price
                entry_time = timestamp
            else:
                entry_idx = None
    
    return pd.DataFrame(trades)

# Replace the extract_trades function with the fixed version
extract_trades = extract_trades_fixed
print("✅ extract_trades function has been fixed to handle timezone issues!")
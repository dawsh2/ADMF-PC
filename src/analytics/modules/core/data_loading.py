"""Data loading utilities for traces, signals, and market data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import json


def load_global_traces(
    traces_dir: str = '/Users/daws/ADMF-PC/traces',
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    strategy: Optional[str] = None,
    execution_timeframe: Optional[str] = None
) -> pd.DataFrame:
    """
    Load traces from global traces directory with optional filters.
    
    Parameters
    ----------
    traces_dir : str
        Path to global traces directory
    symbol : str, optional
        Filter by symbol (e.g., 'SPY')
    timeframe : str, optional
        Filter by data timeframe (e.g., '5m')
    strategy : str, optional
        Filter by strategy name (e.g., 'bollinger_bands')
    execution_timeframe : str, optional
        Filter by execution timeframe (e.g., '1m')
    
    Returns
    -------
    pd.DataFrame
        Combined signal data with metadata columns
        
    Examples
    --------
    >>> # Load all SPY 5m bollinger signals
    >>> signals = load_global_traces(symbol='SPY', timeframe='5m', strategy='bollinger_bands')
    
    >>> # Load all 1m execution signals
    >>> signals = load_global_traces(execution_timeframe='1m')
    """
    traces_path = Path(traces_dir)
    all_traces = []
    
    # Build path pattern based on filters
    if symbol and timeframe and execution_timeframe and strategy:
        pattern = f"{symbol}_{timeframe}/{execution_timeframe}/signals/{strategy}/*.parquet"
    elif symbol and timeframe and execution_timeframe:
        pattern = f"{symbol}_{timeframe}/{execution_timeframe}/signals/*/*.parquet"
    elif symbol and timeframe:
        pattern = f"{symbol}_{timeframe}/*/signals/*/*.parquet"
    elif symbol:
        pattern = f"{symbol}_*/*/signals/*/*.parquet"
    else:
        pattern = "*/*/signals/*/*.parquet"
    
    print(f"Loading traces matching pattern: {pattern}")
    
    for trace_file in traces_path.glob(pattern):
        try:
            df = pd.read_parquet(trace_file)
            
            # Extract metadata from path
            parts = trace_file.parts
            sym_tf = parts[-5]  # e.g., SPY_5m
            exec_tf = parts[-4]  # e.g., 1m
            strat = parts[-2]   # e.g., bollinger_bands
            
            # Add metadata columns
            df['symbol'] = sym_tf.split('_')[0]
            df['data_timeframe'] = sym_tf.split('_')[1]
            df['execution_timeframe'] = exec_tf
            df['strategy'] = strat
            df['trace_file'] = str(trace_file)
            df['strategy_hash'] = trace_file.stem
            
            all_traces.append(df)
            
        except Exception as e:
            print(f"Error loading {trace_file}: {e}")
    
    if all_traces:
        combined = pd.concat(all_traces, ignore_index=True)
        print(f"Loaded {len(combined):,} signal records from {len(all_traces)} trace files")
        return combined
    else:
        print("No traces found matching criteria")
        return pd.DataFrame()


def load_workspace_traces(
    workspace_dir: str,
    strategy_filter: Optional[str] = None
) -> pd.DataFrame:
    """
    Load traces from a workspace directory.
    
    Parameters
    ----------
    workspace_dir : str
        Path to workspace directory
    strategy_filter : str, optional
        Filter by strategy name pattern
        
    Returns
    -------
    pd.DataFrame
        Combined trace data
    """
    workspace_path = Path(workspace_dir)
    traces_dir = workspace_path / 'traces'
    
    if not traces_dir.exists():
        print(f"No traces directory found in workspace: {workspace_dir}")
        return pd.DataFrame()
    
    pattern = f"*/{strategy_filter}*.parquet" if strategy_filter else "*/*.parquet"
    all_traces = []
    
    for trace_file in traces_dir.glob(pattern):
        try:
            df = pd.read_parquet(trace_file)
            df['trace_file'] = str(trace_file)
            df['strategy_name'] = trace_file.parent.name
            all_traces.append(df)
        except Exception as e:
            print(f"Error loading {trace_file}: {e}")
    
    if all_traces:
        return pd.concat(all_traces, ignore_index=True)
    return pd.DataFrame()


def load_market_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_dir: str = 'data'
) -> pd.DataFrame:
    """
    Load market data for a symbol.
    
    Parameters
    ----------
    symbol : str
        Symbol to load (e.g., 'SPY')
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    data_dir : str
        Directory containing market data files
        
    Returns
    -------
    pd.DataFrame
        OHLCV market data with timestamp index
    """
    # Try multiple possible file locations and formats
    possible_paths = [
        Path(data_dir) / f"{symbol}.csv",
        Path(data_dir) / f"{symbol}_5m.csv",
        Path(data_dir) / f"{symbol}_1m.csv",
        Path(f"../{data_dir}") / f"{symbol}.csv",
        Path(f"../{data_dir}") / f"{symbol}_5m.csv",
    ]
    
    for data_path in possible_paths:
        if data_path.exists():
            print(f"Loading market data from: {data_path}")
            df = pd.read_csv(data_path, parse_dates=['timestamp'])
            
            # Set timestamp as index
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            
            # Filter by date range if specified
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            print(f"Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
            return df
    
    print(f"No market data found for {symbol}")
    return pd.DataFrame()


def load_trace_metadata(trace_file: Union[str, Path]) -> Dict:
    """
    Load metadata for a specific trace file.
    
    Parameters
    ----------
    trace_file : str or Path
        Path to trace parquet file
        
    Returns
    -------
    dict
        Metadata dictionary
    """
    trace_path = Path(trace_file)
    
    # Check for metadata.json in same directory
    metadata_file = trace_path.parent / 'metadata.json'
    
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    # Extract basic metadata from filename and path
    parts = trace_path.parts
    metadata = {
        'strategy_hash': trace_path.stem,
        'strategy': parts[-2] if len(parts) >= 2 else 'unknown',
        'execution_timeframe': parts[-4] if len(parts) >= 4 else 'unknown',
        'symbol_timeframe': parts[-5] if len(parts) >= 5 else 'unknown',
    }
    
    # Try to get more info from the parquet file itself
    try:
        df = pd.read_parquet(trace_path)
        if not df.empty:
            metadata.update({
                'total_signals': len(df),
                'start_time': str(df['timestamp'].min()) if 'timestamp' in df else None,
                'end_time': str(df['timestamp'].max()) if 'timestamp' in df else None,
                'signal_counts': df['signal'].value_counts().to_dict() if 'signal' in df else {}
            })
    except:
        pass
    
    return metadata
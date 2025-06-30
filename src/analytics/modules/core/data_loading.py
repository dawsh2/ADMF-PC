"""Data loading utilities for traces, signals, and market data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import json


def load_global_traces(
    traces_dir: str = '/Users/daws/ADMF-PC/traces',
    strategy_hash: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    strategy_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Load traces from global traces directory using flat store structure.
    
    Parameters
    ----------
    traces_dir : str
        Path to global traces directory
    strategy_hash : str, optional
        Load specific strategy by hash
    symbol : str, optional
        Filter by symbol (requires strategy index)
    timeframe : str, optional
        Filter by timeframe (requires strategy index)
    strategy_type : str, optional
        Filter by strategy type (requires strategy index)
    
    Returns
    -------
    pd.DataFrame
        Combined signal data with metadata columns
        
    Examples
    --------
    >>> # Load specific strategy by hash
    >>> signals = load_global_traces(strategy_hash='f0f5699f1791')
    
    >>> # Load all bollinger strategies for SPY
    >>> signals = load_global_traces(symbol='SPY', strategy_type='bollinger_bands')
    """
    traces_path = Path(traces_dir)
    store_path = traces_path / 'store'
    index_path = traces_path / 'strategy_index.parquet'
    
    # If specific hash provided, load directly
    if strategy_hash:
        signal_file = store_path / f"{strategy_hash}.parquet"
        if signal_file.exists():
            df = pd.read_parquet(signal_file)
            df['strategy_hash'] = strategy_hash
            return df
        else:
            raise FileNotFoundError(f"Signal file not found: {signal_file}")
    
    # Otherwise, use strategy index to filter
    if not index_path.exists():
        raise FileNotFoundError(f"Strategy index not found: {index_path}")
    
    # Load strategy index
    strategy_index = pd.read_parquet(index_path)
    
    # Apply filters
    filtered = strategy_index
    if symbol:
        filtered = filtered[filtered['symbol'] == symbol]
    if timeframe:
        filtered = filtered[filtered['timeframe'] == timeframe]
    if strategy_type:
        filtered = filtered[filtered['strategy_type'] == strategy_type]
    
    if filtered.empty:
        return pd.DataFrame()
    
    # Load all matching signal files
    all_traces = []
    for _, row in filtered.iterrows():
        signal_file = store_path / f"{row['strategy_hash']}.parquet"
        if signal_file.exists():
            df = pd.read_parquet(signal_file)
            # Add metadata columns
            df['strategy_hash'] = row['strategy_hash']
            df['strategy_type'] = row['strategy_type']
            df['symbol'] = row['symbol']
            df['timeframe'] = row['timeframe']
            all_traces.append(df)
    
    if all_traces:
        return pd.concat(all_traces, ignore_index=True)
    else:
        return pd.DataFrame()


def load_strategy_index(
    traces_dir: str = '/Users/daws/ADMF-PC/traces'
) -> pd.DataFrame:
    """
    Load the global strategy index.
    
    Parameters
    ----------
    traces_dir : str
        Path to global traces directory
        
    Returns
    -------
    pd.DataFrame
        Strategy index with metadata
    """
    index_path = Path(traces_dir) / 'strategy_index.parquet'
    if index_path.exists():
        return pd.read_parquet(index_path)
    else:
        raise FileNotFoundError(f"Strategy index not found: {index_path}")


def load_execution_trades(
    run_id: Optional[str] = None,
    trades_hash: Optional[str] = None,
    traces_dir: str = '/Users/daws/ADMF-PC/traces'
) -> pd.DataFrame:
    """
    Load unified trades data from execution results.
    
    Parameters
    ----------
    run_id : str, optional
        Specific run ID to load trades for
    trades_hash : str, optional
        Direct trades file hash
    traces_dir : str
        Path to global traces directory
        
    Returns
    -------
    pd.DataFrame
        Trade records with complete lifecycle data
    """
    store_dir = Path(traces_dir) / 'store'
    
    # If trades hash provided, load directly
    if trades_hash:
        trades_path = store_dir / f"T{trades_hash}.parquet"
        if trades_path.exists():
            return pd.read_parquet(trades_path)
        else:
            raise FileNotFoundError(f"Trades file not found: {trades_path}")
    
    # Otherwise, look up in run index
    run_index = load_run_index(traces_dir)
    if run_index.empty:
        raise FileNotFoundError("No runs found in run index")
    
    if run_id:
        run_data = run_index[run_index['run_id'] == run_id]
        if run_data.empty:
            raise FileNotFoundError(f"Run {run_id} not found")
    else:
        # Get most recent run
        run_data = run_index.iloc[-1:]
    
    trades_hash = run_data.iloc[0]['trades_hash']
    if pd.isna(trades_hash):
        raise FileNotFoundError(f"No trades found for run {run_data.iloc[0]['run_id']}")
    
    trades_path = store_dir / f"T{trades_hash}.parquet"
    if trades_path.exists():
        return pd.read_parquet(trades_path)
    else:
        raise FileNotFoundError(f"Trades file not found: {trades_path}")


def load_signal_references(
    run_id: Optional[str] = None,
    traces_dir: str = '/Users/daws/ADMF-PC/traces'
) -> Dict[str, str]:
    """
    Load signal references from a run.
    
    Parameters
    ----------
    run_id : str, optional
        Specific run ID to load. If None, loads most recent
    traces_dir : str
        Path to global traces directory
        
    Returns
    -------
    dict
        Mapping of strategy_id to signal hash
    """
    run_index = load_run_index(traces_dir)
    if run_index.empty:
        return {}
    
    if run_id:
        run_data = run_index[run_index['run_id'] == run_id]
        if run_data.empty:
            return {}
    else:
        # Get most recent run
        run_data = run_index.iloc[-1:]
    
    # Parse signal references from JSON string
    signal_refs_str = run_data.iloc[0]['signal_references']
    if pd.isna(signal_refs_str):
        return {}
    
    try:
        return json.loads(signal_refs_str)
    except:
        return {}


def load_run_index(
    traces_dir: str = '/Users/daws/ADMF-PC/traces'
) -> pd.DataFrame:
    """
    Load the global run index.
    
    Parameters
    ----------
    traces_dir : str
        Path to global traces directory
        
    Returns
    -------
    pd.DataFrame
        Run index with metadata
    """
    index_path = Path(traces_dir) / 'run_index.parquet'
    if index_path.exists():
        return pd.read_parquet(index_path)
    else:
        return pd.DataFrame()  # Empty if no runs yet


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
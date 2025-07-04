# Sparse Storage Handler for Signals and Classifiers
"""
Handles conversion between JSON sparse format and efficient Parquet storage.
Maintains indices for signal changes and classifier regime transitions.
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import json
from dataclasses import dataclass, asdict


@dataclass
class SignalChange:
    """Represents a signal change event"""
    bar_idx: int
    timestamp: datetime
    signal: int  # -1, 0, 1
    symbol: str
    strategy: str
    price: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ClassifierChange:
    """Represents a classifier regime change"""
    bar_idx: int
    timestamp: datetime
    regime: str
    classifier: str
    confidence: float
    previous_regime: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SparseSignalStorage:
    """Handle sparse signal storage with Parquet"""
    
    @staticmethod
    def from_json_changes(
        changes: List[Dict], 
        total_bars: int,
        strategy_info: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Convert JSON sparse format to Parquet DataFrame
        
        Args:
            changes: List of signal change dicts from JSON
            total_bars: Total number of bars in the dataset
            strategy_info: Optional strategy metadata
            
        Returns:
            DataFrame with sparse signal data
        """
        if not changes:
            # Return empty dataframe with correct schema
            return pd.DataFrame(columns=['bar_idx', 'timestamp', 'signal', 'confidence'])
        
        # Convert to DataFrame
        df = pd.DataFrame(changes)
        
        # Rename columns to match schema
        column_mapping = {
            'idx': 'bar_idx',
            'ts': 'timestamp',
            'val': 'signal',
            'sym': 'symbol',
            'strat': 'strategy',
            'px': 'price'
        }
        df = df.rename(columns=column_mapping)
        
        # Convert types
        df['bar_idx'] = df['bar_idx'].astype('int32')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['signal'] = df['signal'].astype('int8')
        
        if 'confidence' in df.columns:
            df['confidence'] = df['confidence'].astype('float32')
        else:
            df['confidence'] = 1.0  # Default confidence
        
        # Add metadata as attributes
        df.attrs['total_bars'] = total_bars
        df.attrs['compression_ratio'] = len(changes) / total_bars if total_bars > 0 else 0
        df.attrs['signal_changes'] = len(changes)
        
        if strategy_info:
            df.attrs['strategy_info'] = strategy_info
        
        # Select columns for storage
        columns = ['bar_idx', 'timestamp', 'signal', 'confidence']
        return df[columns]
    
    @staticmethod
    def to_parquet(
        df: pd.DataFrame, 
        path: Path, 
        compression: str = 'snappy',
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save sparse signals to Parquet with metadata
        
        Args:
            df: DataFrame with sparse signals
            path: Output file path
            compression: Compression algorithm
            metadata: Additional metadata to store
        """
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        table_metadata = {}
        
        # Add DataFrame attributes
        for key, value in df.attrs.items():
            if isinstance(value, (int, float, str, bool)):
                table_metadata[key.encode()] = str(value).encode()
            elif isinstance(value, dict):
                table_metadata[key.encode()] = json.dumps(value).encode()
        
        # Add custom metadata
        if metadata:
            for key, value in metadata.items():
                table_metadata[key.encode()] = str(value).encode()
        
        # Add format version
        table_metadata[b'sparse_format_version'] = b'2.0'
        table_metadata[b'storage_type'] = b'signal'
        
        # Create Arrow table
        table = pa.Table.from_pandas(df, preserve_index=False)
        
        # Add metadata to schema
        existing_metadata = table.schema.metadata or {}
        combined_metadata = {**existing_metadata, **table_metadata}
        table = table.replace_schema_metadata(combined_metadata)
        
        # Write with compression
        pq.write_table(table, path, compression=compression)
    
    @staticmethod
    def from_parquet(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load sparse signals from Parquet
        
        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        # Read parquet file
        table = pq.read_table(path)
        df = table.to_pandas()
        
        # Extract metadata
        metadata = {}
        if table.schema.metadata:
            for key, value in table.schema.metadata.items():
                key_str = key.decode() if isinstance(key, bytes) else key
                value_str = value.decode() if isinstance(value, bytes) else value
                
                # Try to parse JSON values
                try:
                    metadata[key_str] = json.loads(value_str)
                except (json.JSONDecodeError, TypeError):
                    metadata[key_str] = value_str
        
        # Restore DataFrame attributes
        for key, value in metadata.items():
            if key not in ['sparse_format_version', 'storage_type']:
                df.attrs[key] = value
        
        return df, metadata


class SparseClassifierStorage:
    """Handle sparse classifier state storage with Parquet"""
    
    @staticmethod
    def from_regime_changes(
        changes: List[Dict],
        total_bars: int,
        classifier_info: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Convert regime changes to Parquet DataFrame
        
        Args:
            changes: List of regime change events
            total_bars: Total number of bars
            classifier_info: Optional classifier metadata
            
        Returns:
            DataFrame with sparse regime data
        """
        if not changes:
            return pd.DataFrame(columns=['bar_idx', 'timestamp', 'regime', 'confidence'])
        
        # Convert to DataFrame
        df = pd.DataFrame(changes)
        
        # Ensure required columns
        df['bar_idx'] = df.get('idx', df.get('bar_idx', range(len(df)))).astype('int32')
        df['timestamp'] = pd.to_datetime(df.get('ts', df.get('timestamp')))
        df['regime'] = df.get('regime', df.get('state', 'UNKNOWN'))
        df['confidence'] = df.get('confidence', 1.0).astype('float32')
        
        # Add metadata
        df.attrs['total_bars'] = total_bars
        df.attrs['compression_ratio'] = len(changes) / total_bars if total_bars > 0 else 0
        df.attrs['regime_changes'] = len(changes)
        
        if classifier_info:
            df.attrs['classifier_info'] = classifier_info
        
        # Calculate regime durations
        if len(df) > 1:
            df['duration'] = df['bar_idx'].diff().shift(-1).fillna(total_bars - df['bar_idx'].iloc[-1])
            avg_duration = df['duration'].mean()
            df.attrs['avg_regime_duration'] = float(avg_duration)
        
        return df[['bar_idx', 'timestamp', 'regime', 'confidence']]
    
    @staticmethod
    def to_parquet(
        df: pd.DataFrame,
        path: Path,
        compression: str = 'snappy',
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save sparse classifier states to Parquet"""
        # Use same approach as signal storage
        path.parent.mkdir(parents=True, exist_ok=True)
        
        table_metadata = {}
        
        # Add DataFrame attributes
        for key, value in df.attrs.items():
            if isinstance(value, (int, float, str, bool)):
                table_metadata[key.encode()] = str(value).encode()
            elif isinstance(value, dict):
                table_metadata[key.encode()] = json.dumps(value).encode()
        
        # Add custom metadata
        if metadata:
            for key, value in metadata.items():
                table_metadata[key.encode()] = str(value).encode()
        
        # Add format info
        table_metadata[b'sparse_format_version'] = b'2.0'
        table_metadata[b'storage_type'] = b'classifier'
        
        # Create and write table
        table = pa.Table.from_pandas(df, preserve_index=False)
        existing_metadata = table.schema.metadata or {}
        combined_metadata = {**existing_metadata, **table_metadata}
        table = table.replace_schema_metadata(combined_metadata)
        
        pq.write_table(table, path, compression=compression)


class SparseStorageUtils:
    """Utilities for working with sparse storage"""
    
    @staticmethod
    def calculate_storage_stats(
        original_bars: int,
        stored_changes: int,
        file_size_bytes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Calculate storage efficiency statistics"""
        compression_ratio = stored_changes / original_bars if original_bars > 0 else 0
        
        stats = {
            'original_bars': original_bars,
            'stored_changes': stored_changes,
            'compression_ratio': compression_ratio,
            'space_savings_pct': (1 - compression_ratio) * 100
        }
        
        if file_size_bytes:
            # Estimate original size (8 bytes per bar for signal + timestamp)
            estimated_original = original_bars * 8
            actual_compression = file_size_bytes / estimated_original
            stats['file_size_bytes'] = file_size_bytes
            stats['actual_compression_ratio'] = actual_compression
        
        return stats
    
    @staticmethod
    def merge_sparse_signals(
        signals: List[pd.DataFrame],
        strategy_ids: List[str]
    ) -> pd.DataFrame:
        """Merge multiple sparse signal DataFrames"""
        if not signals:
            return pd.DataFrame()
        
        # Combine all signals with strategy identification
        combined = []
        for df, strategy_id in zip(signals, strategy_ids):
            df = df.copy()
            df['strategy_id'] = strategy_id
            combined.append(df)
        
        # Concatenate and sort by bar_idx
        result = pd.concat(combined, ignore_index=True)
        result = result.sort_values('bar_idx')
        
        return result
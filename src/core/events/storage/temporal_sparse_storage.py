"""
Temporal Sparse Signal Storage

Only stores signal changes, not redundant signals.
Tracks bar indices to allow full signal reconstruction from market data.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
import logging
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SignalChange:
    """Represents a signal state change with source data traceability."""
    bar_index: int
    timestamp: str
    symbol: str
    signal_value: Any  # Can be int (-1, 0, 1) or string (regime classification)
    strategy_id: str
    price: float
    
    # Source data metadata for analytics
    timeframe: Optional[str] = None
    source_file_path: Optional[str] = None
    data_source_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {
            'idx': self.bar_index,
            'ts': self.timestamp,
            'sym': self.symbol,
            'val': self.signal_value,
            'strat': self.strategy_id,
            'px': self.price
        }
        
        # Add source metadata if available
        if self.timeframe:
            result['tf'] = self.timeframe
        if self.source_file_path:
            result['src_file'] = self.source_file_path
        if self.data_source_type:
            result['src_type'] = self.data_source_type
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SignalChange':
        """Create from stored dictionary."""
        return cls(
            bar_index=data['idx'],
            timestamp=data['ts'],
            symbol=data['sym'],
            signal_value=data['val'],
            strategy_id=data['strat'],
            price=data['px'],
            timeframe=data.get('tf'),
            source_file_path=data.get('src_file'),
            data_source_type=data.get('src_type')
        )


class TemporalSparseStorage:
    """
    Storage that only records signal changes, not every signal.
    
    For a strategy that stays long for 25 bars then goes short for 25 bars,
    we only store 2-4 events instead of 50.
    """
    
    def __init__(self, base_dir: str = "./sparse_signals", run_id: Optional[str] = None,
                 timeframe: str = "1m", source_file_path: Optional[str] = None, 
                 data_source_type: str = "csv"):
        # Store run_id but don't use it in path - base_dir already includes full path
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run_id = run_id
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Source data metadata for analytics
        self.timeframe = timeframe
        self.source_file_path = source_file_path
        self.data_source_type = data_source_type
        
        # Track current state per strategy
        self._current_state: Dict[str, Dict[str, Any]] = {}
        
        # Buffer for changes
        self._changes: List[SignalChange] = []
        
        # Bar counter
        self._bar_index = 0
        
        # Strategy metadata storage
        self._strategy_metadata: Dict[str, Any] = {}
        
    def process_signal(self, 
                      symbol: str,
                      direction: str,
                      strategy_id: str,
                      timestamp: str,
                      price: float,
                      bar_index: Optional[int] = None) -> bool:
        """
        Process a signal and store only if it represents a change.
        
        Returns:
            True if this was a new/changed signal, False if redundant
        """
        # Convert direction to value for standard strategy signals
        # Keep categorical values for classifier signals
        if direction == 'long':
            signal_value = 1
        elif direction == 'short':
            signal_value = -1
        elif direction == 'flat':
            signal_value = 0
        else:
            # Categorical value (e.g., regime classification)
            signal_value = direction
            
        # Get current state for this strategy
        state_key = f"{symbol}_{strategy_id}"
        current = self._current_state.get(state_key)
        
        # Check if this is a change
        is_change = False
        
        if current is None:
            # First signal for this strategy
            is_change = True
            current_bar_index = bar_index if bar_index is not None else self._bar_index
            logger.info(f"First signal for {state_key}: {direction} at bar {current_bar_index}")
            
        elif current['value'] != signal_value:
            # Signal changed direction
            is_change = True
            current_bar_index = bar_index if bar_index is not None else self._bar_index
            logger.info(f"Signal change for {state_key}: "
                       f"{current['value']} -> {signal_value} at bar {current_bar_index}")
            
        if is_change:
            # Use provided bar_index or fall back to internal counter
            current_bar_index = bar_index if bar_index is not None else self._bar_index
            
            # Record the change with source metadata
            change = SignalChange(
                bar_index=current_bar_index,
                timestamp=timestamp,
                symbol=symbol,
                signal_value=signal_value,
                strategy_id=strategy_id,
                price=price,
                timeframe=self.timeframe,
                source_file_path=self.source_file_path,
                data_source_type=self.data_source_type
            )
            self._changes.append(change)
            
            # Update current state
            self._current_state[state_key] = {
                'value': signal_value,
                'bar_index': current_bar_index,
                'timestamp': timestamp
            }
            
            # Update internal bar index if provided
            if bar_index is not None:
                self._bar_index = bar_index
        
        return is_change
    
    def get_signal_ranges(self) -> List[Dict[str, Any]]:
        """
        Get signal ranges for reconstruction.
        
        Returns list of ranges like:
        [
            {'signal': 1, 'start_bar': 0, 'end_bar': 25},
            {'signal': -1, 'start_bar': 26, 'end_bar': 50}
        ]
        """
        if not self._changes:
            return []
            
        ranges = []
        
        # Group by strategy
        strategy_changes: Dict[str, List[SignalChange]] = {}
        for change in self._changes:
            key = f"{change.symbol}_{change.strategy_id}"
            if key not in strategy_changes:
                strategy_changes[key] = []
            strategy_changes[key].append(change)
        
        # Build ranges for each strategy
        for strategy_key, changes in strategy_changes.items():
            # Sort by bar index
            changes.sort(key=lambda x: x.bar_index)
            
            for i, change in enumerate(changes):
                # Determine end bar
                if i < len(changes) - 1:
                    end_bar = changes[i + 1].bar_index - 1
                else:
                    # Last change - use current bar index
                    end_bar = self._bar_index - 1
                
                ranges.append({
                    'strategy': strategy_key,
                    'signal': change.signal_value,
                    'start_bar': change.bar_index,
                    'end_bar': end_bar,
                    'start_ts': change.timestamp,
                    'start_price': change.price
                })
        
        return ranges
    
    def _calculate_signal_statistics(self) -> Dict[str, Any]:
        """Calculate statistics about signal patterns for performance inference."""
        if not self._changes:
            return {}
            
        stats = {
            'total_positions': len(self._changes),
            'avg_position_duration': 0,
            'position_breakdown': {'long': 0, 'short': 0, 'flat': 0},
            'signal_frequency': len(self._changes) / self._bar_index if self._bar_index > 0 else 0,
            'by_strategy': {}
        }
        
        # Group by strategy
        strategy_changes: Dict[str, List[SignalChange]] = {}
        for change in self._changes:
            key = f"{change.symbol}_{change.strategy_id}"
            if key not in strategy_changes:
                strategy_changes[key] = []
            strategy_changes[key].append(change)
        
        # Calculate per-strategy stats
        total_duration = 0
        position_count = 0
        
        for strategy_key, changes in strategy_changes.items():
            changes.sort(key=lambda x: x.bar_index)
            
            strategy_stats = {
                'positions': len(changes),
                'position_durations': [],
                'signal_breakdown': {'long': 0, 'short': 0, 'flat': 0}
            }
            
            for i, change in enumerate(changes):
                # Count signal types
                if isinstance(change.signal_value, int):
                    # Numeric signals (strategies)
                    if change.signal_value == 1:
                        stats['position_breakdown']['long'] += 1
                        strategy_stats['signal_breakdown']['long'] += 1
                    elif change.signal_value == -1:
                        stats['position_breakdown']['short'] += 1
                        strategy_stats['signal_breakdown']['short'] += 1
                    else:
                        stats['position_breakdown']['flat'] += 1
                        strategy_stats['signal_breakdown']['flat'] += 1
                else:
                    # Categorical signals (classifiers)
                    regime = str(change.signal_value)
                    if 'regime_breakdown' not in stats:
                        stats['regime_breakdown'] = {}
                    if 'regime_breakdown' not in strategy_stats:
                        strategy_stats['regime_breakdown'] = {}
                    
                    stats['regime_breakdown'][regime] = stats['regime_breakdown'].get(regime, 0) + 1
                    strategy_stats['regime_breakdown'][regime] = strategy_stats['regime_breakdown'].get(regime, 0) + 1
                
                # Calculate position duration
                if i < len(changes) - 1:
                    duration = changes[i + 1].bar_index - change.bar_index
                else:
                    duration = self._bar_index - change.bar_index
                
                strategy_stats['position_durations'].append(duration)
                total_duration += duration
                position_count += 1
            
            # Average duration for this strategy
            if strategy_stats['position_durations']:
                strategy_stats['avg_duration'] = sum(strategy_stats['position_durations']) / len(strategy_stats['position_durations'])
            else:
                strategy_stats['avg_duration'] = 0
                
            stats['by_strategy'][strategy_key] = strategy_stats
        
        # Overall average duration
        if position_count > 0:
            stats['avg_position_duration'] = total_duration / position_count
        
        return stats
    
    def set_strategy_metadata(self, strategy_id: str, metadata: Dict[str, Any]) -> None:
        """Store strategy metadata including parameters."""
        self._strategy_metadata[strategy_id] = metadata
    
    def save(self, tag: Optional[str] = None, performance_metrics: Optional[Dict[str, Any]] = None, 
             strategy_params: Optional[Dict[str, Any]] = None) -> str:
        """Save signal changes to Parquet file with optional performance metrics."""
        if not self._changes:
            logger.warning("No signal changes to save")
            return ""
            
        # Create filename with optional tag
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{tag}.parquet" if tag else f"signals_{timestamp}.parquet"
        filepath = self.base_dir / filename
        
        # Calculate signal statistics for performance inference
        signal_stats = self._calculate_signal_statistics()
        
        # Build strategy info with parameters
        strategy_info = {}
        for strategy_key in set(f"{c.symbol}_{c.strategy_id}" for c in self._changes):
            strategy_info[strategy_key] = {
                'strategy_id': strategy_key,
                'metadata': self._strategy_metadata.get(strategy_key.split('_', 1)[1], {})
            }
        
        # Convert changes to DataFrame for Parquet storage
        changes_data = []
        for c in self._changes:
            change_dict = c.to_dict()
            changes_data.append(change_dict)
        
        df = pd.DataFrame(changes_data)
        
        # Save as Parquet with embedded metadata
        metadata_dict = {
            'run_id': self.run_id,
            'total_bars': str(self._bar_index),
            'total_changes': str(len(self._changes)),
            'compression_ratio': str(len(self._changes) / self._bar_index if self._bar_index > 0 else 0),
            'strategies': json.dumps(strategy_info),
            'signal_statistics': json.dumps(signal_stats)
        }
        
        # Add strategy parameters if provided
        if strategy_params:
            metadata_dict['strategy_parameters'] = json.dumps(strategy_params)
        
        # Add performance metrics if available
        if performance_metrics:
            metadata_dict['performance'] = json.dumps(performance_metrics)
        
        # Save as Parquet with custom metadata
        # Note: PyArrow metadata is handled differently than pandas metadata
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            # Convert DataFrame to PyArrow table
            table = pa.Table.from_pandas(df)
            
            # Add custom metadata to schema
            custom_metadata = {k: str(v) for k, v in metadata_dict.items()}
            table = table.replace_schema_metadata(custom_metadata)
            
            # Write to Parquet file
            pq.write_table(table, filepath)
        except ImportError:
            # Fallback to pandas without metadata if PyArrow not available
            logger.warning("PyArrow not available, saving Parquet without metadata")
            df.to_parquet(filepath, engine='pandas', index=False)
            
        logger.info(f"Saved {len(self._changes)} signal changes to {filepath}")
        logger.info(f"Compression: {len(self._changes)} changes for {self._bar_index} bars "
                   f"({len(self._changes)/self._bar_index*100:.1f}% of original)")
        
        return str(filepath)
    
    def load(self, filepath: str) -> Dict[str, Any]:
        """Load signal changes from Parquet file."""
        # Read Parquet file
        df = pd.read_parquet(filepath, engine='pyarrow')
        
        # Get metadata from Parquet file metadata
        parquet_file = pd.io.parquet.ParquetFile(filepath)
        metadata = parquet_file.metadata_path or {}
        
        # Convert DataFrame back to changes
        changes_data = df.to_dict('records')
        self._changes = [SignalChange.from_dict(c) for c in changes_data]
        
        # Reconstruct metadata
        data = {
            'metadata': {
                'run_id': metadata.get('run_id', self.run_id),
                'total_bars': int(metadata.get('total_bars', len(df))),
                'total_changes': int(metadata.get('total_changes', len(df))),
                'compression_ratio': float(metadata.get('compression_ratio', 0)),
                'strategies': json.loads(metadata.get('strategies', '{}')),
                'signal_statistics': json.loads(metadata.get('signal_statistics', '{}'))
            },
            'changes': changes_data
        }
        
        # Add optional metadata
        if 'strategy_parameters' in metadata:
            data['metadata']['strategy_parameters'] = json.loads(metadata['strategy_parameters'])
        if 'performance' in metadata:
            data['performance'] = json.loads(metadata['performance'])
        
        self._bar_index = data['metadata']['total_bars']
        
        return data
    
    def reconstruct_signals(self, bar_indices: List[int]) -> Dict[int, Dict[str, int]]:
        """
        Reconstruct signal values for specific bar indices.
        
        Returns:
            Dict mapping bar_index to {strategy_key: signal_value}
        """
        ranges = self.get_signal_ranges()
        result = {}
        
        for bar_idx in bar_indices:
            result[bar_idx] = {}
            
            # Find which range each bar falls into
            for range_info in ranges:
                if range_info['start_bar'] <= bar_idx <= range_info['end_bar']:
                    strategy = range_info['strategy']
                    result[bar_idx][strategy] = range_info['signal']
        
        return result
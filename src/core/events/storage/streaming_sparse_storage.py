"""
Streaming Sparse Signal Storage with Periodic Writes

Enhanced version of TemporalSparseStorage that:
1. Writes to disk periodically to prevent memory accumulation
2. Only keeps minimal state in memory (last signal per strategy)
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
    """Represents a signal state change."""
    bar_index: int
    timestamp: str
    symbol: str
    signal_value: Any
    strategy_id: str
    price: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'idx': self.bar_index,
            'ts': self.timestamp,
            'sym': self.symbol,
            'val': self.signal_value,
            'strat': self.strategy_id,
            'px': self.price
        }


class StreamingSparseStorage:
    """
    Sparse storage that writes to disk periodically.
    
    Key improvements:
    1. Writes every N bars or M changes (configurable)
    2. Only keeps last signal state in memory
    3. Appends to same file in chunks for efficiency
    """
    
    def __init__(self, 
                 base_dir: str,
                 write_interval: int = 500,
                 write_on_changes: int = 100):
        """
        Initialize streaming storage.
        
        Args:
            base_dir: Directory for storage
            write_interval: Write every N bars
            write_on_changes: Write every M changes
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.write_interval = write_interval
        self.write_on_changes = write_on_changes
        
        # Minimal state tracking - only last signal
        self._last_signals: Dict[str, Any] = {}
        
        # Small buffer for current batch
        self._buffer: List[SignalChange] = []
        
        # Tracking
        self._bar_index = 0
        self._total_changes = 0
        self._last_write_bar = 0
        self._write_count = 0
        
        # Output file parts
        self._parts_written: List[str] = []
        
    def process_signal(self, 
                      symbol: str,
                      direction: str,
                      strategy_id: str,
                      timestamp: str,
                      price: float,
                      bar_index: Optional[int] = None) -> bool:
        """Process signal and write to disk periodically."""
        # Convert direction
        if direction == 'long':
            signal_value = 1
        elif direction == 'short':
            signal_value = -1
        elif direction == 'flat':
            signal_value = 0
        else:
            signal_value = direction
            
        # Update bar index
        if bar_index is not None:
            self._bar_index = bar_index
            
        # Check for change
        state_key = f"{symbol}_{strategy_id}"
        last_value = self._last_signals.get(state_key)
        
        is_change = (last_value is None or last_value != signal_value)
        
        if is_change:
            # Add to buffer
            change = SignalChange(
                bar_index=self._bar_index,
                timestamp=timestamp,
                symbol=symbol,
                signal_value=signal_value,
                strategy_id=strategy_id,
                price=price
            )
            self._buffer.append(change)
            
            # Update last signal
            self._last_signals[state_key] = signal_value
            self._total_changes += 1
            
            # Check if we should write
            self._check_write_conditions()
            
        return is_change
    
    def _check_write_conditions(self) -> None:
        """Check if we should write buffer to disk."""
        bars_since_write = self._bar_index - self._last_write_bar
        buffer_size = len(self._buffer)
        
        should_write = (
            bars_since_write >= self.write_interval or
            buffer_size >= self.write_on_changes
        )
        
        if should_write and buffer_size > 0:
            self._write_buffer()
    
    def _write_buffer(self) -> None:
        """Write current buffer to disk and clear it."""
        if not self._buffer:
            return
            
        # Create part filename
        part_num = self._write_count
        filename = f"signals_part_{part_num:04d}.parquet"
        filepath = self.base_dir / filename
        
        # Convert to DataFrame
        data = [c.to_dict() for c in self._buffer]
        df = pd.DataFrame(data)
        
        # Save as Parquet
        df.to_parquet(filepath, engine='pyarrow', index=False)
        
        self._parts_written.append(str(filepath))
        logger.debug(f"Wrote {len(self._buffer)} changes to {filename}")
        
        # Clear buffer and update tracking
        self._buffer.clear()
        self._last_write_bar = self._bar_index
        self._write_count += 1
    
    def finalize(self) -> Dict[str, Any]:
        """Write any remaining data and return summary."""
        # Write remaining buffer
        if self._buffer:
            self._write_buffer()
            
        # Create summary
        summary = {
            'total_bars': self._bar_index,
            'total_changes': self._total_changes,
            'compression_ratio': self._total_changes / self._bar_index if self._bar_index > 0 else 0,
            'parts_written': self._parts_written,
            'write_count': self._write_count
        }
        
        # Save summary
        summary_path = self.base_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Finalized storage: {self._total_changes} changes in {self._write_count} parts")
        
        return summary
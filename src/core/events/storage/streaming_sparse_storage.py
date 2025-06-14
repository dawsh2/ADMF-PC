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
                 write_interval: int = 0,
                 write_on_changes: int = 0,
                 component_id: Optional[str] = None):
        """
        Initialize streaming storage.
        
        Args:
            base_dir: Directory for storage
            write_interval: Write every N bars (0 = never)
            write_on_changes: Write every M changes (0 = never)
            component_id: Component ID for filename (e.g., SPY_rsi_grid_14_20_70)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.write_interval = write_interval
        self.write_on_changes = write_on_changes
        self.component_id = component_id
        
        # Minimal state tracking - only last signal
        self._last_signals: Dict[str, Any] = {}
        
        # Small buffer for current batch
        self._buffer: List[SignalChange] = []
        
        # Tracking
        self._bar_index = 0
        self._total_changes = 0
        self._last_write_bar = 0
        self._write_count = 0
        
        # Output file (single file, not parts)
        self._output_file: Optional[Path] = None
        
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
        # Skip if both intervals are 0 (write only at end)
        if self.write_interval == 0 and self.write_on_changes == 0:
            return
            
        bars_since_write = self._bar_index - self._last_write_bar
        buffer_size = len(self._buffer)
        
        should_write = (
            (self.write_interval > 0 and bars_since_write >= self.write_interval) or
            (self.write_on_changes > 0 and buffer_size >= self.write_on_changes)
        )
        
        if should_write and buffer_size > 0:
            self._write_buffer()
    
    def _write_buffer(self) -> None:
        """Write current buffer to disk and clear it."""
        if not self._buffer:
            return
            
        # Create filename from component_id if not already set
        if not self._output_file:
            if self.component_id:
                filename = f"{self.component_id}.parquet"
            else:
                filename = "signals.parquet"
            self._output_file = self.base_dir / filename
        
        # Convert to DataFrame
        data = [c.to_dict() for c in self._buffer]
        df = pd.DataFrame(data)
        
        # Write or append to Parquet file
        if self._write_count == 0:
            # First write - create new file
            df.to_parquet(self._output_file, engine='pyarrow', index=False)
        else:
            # Subsequent writes - append (requires reading existing data)
            try:
                existing_df = pd.read_parquet(self._output_file)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_parquet(self._output_file, engine='pyarrow', index=False)
            except Exception as e:
                logger.error(f"Failed to append to {self._output_file}: {e}")
                # Fall back to writing separate file
                alt_file = self.base_dir / f"{self.component_id}_part{self._write_count}.parquet"
                df.to_parquet(alt_file, engine='pyarrow', index=False)
        
        logger.debug(f"Wrote {len(self._buffer)} changes to {self._output_file.name}")
        
        # Clear buffer and update tracking
        self._buffer.clear()
        self._last_write_bar = self._bar_index
        self._write_count += 1
    
    def finalize(self) -> Dict[str, Any]:
        """Write any remaining data and return summary."""
        # Write remaining buffer
        if self._buffer:
            self._write_buffer()
            
        # Create summary for logging/return (but don't save to file)
        summary = {
            'total_bars': self._bar_index,
            'total_changes': self._total_changes,
            'compression_ratio': self._total_changes / self._bar_index if self._bar_index > 0 else 0,
            'output_file': str(self._output_file) if self._output_file else None
        }
        
        logger.info(f"Finalized storage: {self._total_changes} changes for {self.component_id}")
        
        return summary
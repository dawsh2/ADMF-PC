"""
Dense Event Storage

Stores every event without sparse compression.
Used for orders, fills, and position events where we need complete history.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DenseEventStorage:
    """
    Storage that records every event, not just changes.
    
    Unlike TemporalSparseStorage, this stores all events regardless of
    whether they represent a state change. This is essential for:
    - Order tracking
    - Fill recording  
    - Position lifecycle events
    """
    
    def __init__(self, base_dir: str, event_type: str):
        """
        Initialize dense storage.
        
        Args:
            base_dir: Directory to store events
            event_type: Type of events being stored (orders, fills, positions_open, etc)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.event_type = event_type
        
        # Buffer for events
        self._events: List[Dict[str, Any]] = []
        self._max_buffer_size = 10000
        
    def add_event(self,
                  symbol: str,
                  timestamp: str,
                  bar_index: int,
                  metadata: Dict[str, Any]) -> None:
        """
        Add an event to storage.
        
        Args:
            symbol: Trading symbol
            timestamp: Event timestamp
            bar_index: Bar index when event occurred
            metadata: Full event metadata
        """
        # Keep metadata as dict for proper storage
        # Parquet/pandas will handle serialization properly
        event_record = {
            'idx': bar_index,
            'ts': timestamp,
            'sym': symbol,
            'metadata': metadata if metadata else {}
        }
        
        # Extract important fields from metadata for easier access
        if metadata:
            # Extract strategy_id if present (for position events)
            if 'strategy_id' in metadata:
                event_record['strategy_id'] = metadata['strategy_id']
            # Extract exit type and price info for position closes
            if 'exit_type' in metadata:
                event_record['exit_type'] = metadata['exit_type']
            if 'entry_price' in metadata:
                event_record['entry_price'] = float(metadata['entry_price'])
            if 'exit_price' in metadata:
                event_record['exit_price'] = float(metadata['exit_price'])
        
        self._events.append(event_record)
        
        # Log warning if buffer is getting large
        if len(self._events) >= self._max_buffer_size:
            logger.warning(f"Dense storage buffer at {len(self._events)} events for {self.event_type}")
    
    def save(self, filename: Optional[str] = None) -> str:
        """Save all events to Parquet file."""
        if not self._events:
            logger.warning(f"No {self.event_type} events to save")
            return ""
        
        # Create filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.event_type}_{timestamp}.parquet"
        
        filepath = self.base_dir / filename
        
        # Convert to DataFrame
        df = pd.DataFrame(self._events)
        
        # Save as Parquet
        df.to_parquet(filepath, engine='pyarrow', index=False)
        
        logger.info(f"Saved {len(self._events)} {self.event_type} events to {filepath}")
        
        return str(filepath)
"""
Sparse Signal Storage

Minimal storage for signals - stores only what's needed for replay and analysis.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import logging
from datetime import datetime

from ..types import Event, EventType

logger = logging.getLogger(__name__)


class SparseSignalStorage:
    """
    Lightweight signal storage that stores only essential fields.
    
    Stored fields:
    - timestamp
    - symbol
    - direction
    - strength
    - strategy_id
    """
    
    def __init__(self, base_dir: str = "./signals"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._buffer = []
        self._count = 0
        
    def store_signal(self, event: Event) -> None:
        """Store only essential signal data."""
        if event.event_type != EventType.SIGNAL.value:
            return
            
        payload = event.payload
        
        # With fixed signal values (-1, 0, 1), we can encode even more compactly
        # Combine direction and strength into a single value
        direction = payload.get('direction', 'flat')
        strength = payload.get('strength', 1.0)
        
        # Encode as single value: 1=long, -1=short, 0=flat/exit
        if direction == 'long':
            signal_value = 1
        elif direction == 'short':
            signal_value = -1
        else:
            signal_value = 0
        
        # Ultra-compact format
        signal_data = {
            't': int(event.timestamp.timestamp()),  # Unix timestamp (smaller than ISO string)
            's': payload.get('symbol'),
            'v': signal_value,  # Combined direction/strength
            'id': payload.get('strategy_id', '').split('_')[-1]  # Just strategy name, not full ID
        }
        
        self._buffer.append(signal_data)
        self._count += 1
        
        # Auto-flush every 100 signals
        if len(self._buffer) >= 100:
            self.flush()
    
    def flush(self) -> None:
        """Write buffered signals to disk."""
        if not self._buffer:
            return
            
        # Simple date-based partitioning
        date_str = datetime.now().strftime("%Y%m%d")
        filepath = self.base_dir / f"signals_{date_str}.jsonl"
        
        with open(filepath, 'a') as f:
            for signal in self._buffer:
                f.write(json.dumps(signal) + '\n')
        
        logger.info(f"Flushed {len(self._buffer)} signals to {filepath}")
        self._buffer.clear()
    
    def read_signals(self, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read signals from storage."""
        if date_str:
            filepath = self.base_dir / f"signals_{date_str}.jsonl"
            files = [filepath] if filepath.exists() else []
        else:
            files = sorted(self.base_dir.glob("signals_*.jsonl"))
        
        signals = []
        for filepath in files:
            with open(filepath, 'r') as f:
                for line in f:
                    signal = json.loads(line)
                    # Decode compact format
                    signal_value = signal['v']
                    
                    signals.append({
                        'timestamp': datetime.fromtimestamp(signal['t']).isoformat(),
                        'symbol': signal['s'],
                        'direction': 'long' if signal_value == 1 else 'short' if signal_value == -1 else 'flat',
                        'strength': float(signal_value),
                        'strategy_id': signal['id']
                    })
        
        return signals
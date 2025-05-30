"""FIXED: Data handler using Protocol+Composition instead of inheritance"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from ..models import Bar, Timeframe


class HistoricalDataHandler:
    """Data handler using composition - NO INHERITANCE!"""
    
    def __init__(self, handler_id: str = "historical_data"):
        """Initialize without inheritance"""
        self.handler_id = handler_id
        
        # Just the data we need - no base class
        self.symbols: List[str] = []
        self.data: Dict[str, pd.DataFrame] = {}
        self.current_indices: Dict[str, int] = {}
        self.active_split: Optional[str] = None
        
        # Capabilities will be added by factory

    @property
    def name(self) -> str:
        """Implement protocol method"""
        return self.handler_id

    def load_data(self, symbols: List[str]) -> bool:
        """Implement protocol method - no super() calls!"""
        self.symbols = symbols
        
        for symbol in symbols:
            # Load data logic here
            # No inheritance complexity!
            self.data[symbol] = self._load_symbol_data(symbol)
            self.current_indices[symbol] = 0
        
        return True

    def update_bars(self) -> bool:
        """Implement protocol method - simple and direct"""
        # Find next bar across symbols
        next_timestamp = None
        next_symbol = None
        
        for symbol in self.symbols:
            data = self.data[symbol]
            idx = self.current_indices[symbol]
            
            if idx < len(data):
                timestamp = data.index[idx]
                if not next_timestamp or timestamp < next_timestamp:
                    next_timestamp = timestamp
                    next_symbol = symbol
        
        if not next_symbol:
            return False
        
        # Emit bar event (through event capability if added)
        if hasattr(self, 'emit_event'):  # Added by capability
            bar_data = self.data[next_symbol].iloc[self.current_indices[next_symbol]]
            self.emit_event('BAR', {
                'symbol': next_symbol,
                'timestamp': next_timestamp,
                'data': bar_data.to_dict()
            })
        
        # Update index
        self.current_indices[next_symbol] += 1
        return True

    def _load_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Private helper - no inheritance complexity"""
        # Simple CSV loading
        import pandas as pd
        csv_path = f"data/{symbol}.csv"
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)


# Usage with capabilities instead of inheritance
def create_data_handler():
    from src.core.components import ComponentFactory
    
    return ComponentFactory().create_component({
        'class': 'HistoricalDataHandler',
        'params': {'handler_id': 'hist_data'},
        'capabilities': [
            'lifecycle',      # Adds start/stop methods
            'events',         # Adds event emission
            'data_splitting', # Adds train/test splitting
            'logging',        # Adds logging
            'monitoring'      # Adds performance tracking
        ]
    })

# Result: All functionality without inheritance!

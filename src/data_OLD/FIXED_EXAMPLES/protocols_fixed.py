"""FIXED: Data protocols using typing.Protocol instead of ABC"""

from typing import Protocol, runtime_checkable, List, Optional, Dict, Any, Iterator
from datetime import datetime
import pandas as pd


@runtime_checkable
class DataProvider(Protocol):
    """Protocol for components that provide market data - NO INHERITANCE!"""
    
    def load_data(self, symbols: List[str]) -> bool:
        """Load data for specified symbols"""
        ...
    
    def get_next_bar(self) -> Optional[Dict[str, Any]]:
        """Get next available bar across all symbols"""
        ...
    
    def has_more_data(self) -> bool:
        """Check if more data is available"""
        ...


@runtime_checkable
class DataLoader(Protocol):
    """Protocol for loading market data - NO INHERITANCE!"""
    
    def load(self, symbol: str, **kwargs) -> pd.DataFrame:
        """Load market data for a symbol"""
        ...
    
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate loaded data"""
        ...


@runtime_checkable
class BarEmitter(Protocol):
    """Protocol for components that emit bar events - NO INHERITANCE!"""
    
    def update_bars(self) -> bool:
        """Update to next bar and emit event"""
        ...


# Example: Simple class implementing protocol
class SimpleCSVLoader:
    """Simple CSV loader - NO INHERITANCE!"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def load(self, symbol: str, **kwargs) -> pd.DataFrame:
        # Implementation without inheritance
        pass
    
    def validate(self, df: pd.DataFrame) -> bool:
        # Implementation without inheritance
        pass


# Enhanced through capabilities, not inheritance
def create_csv_loader_with_capabilities():
    from src.core.components import ComponentFactory
    
    return ComponentFactory().create_component({
        'class': 'SimpleCSVLoader',
        'params': {'data_dir': 'data'},
        'capabilities': [
            'logging',           # Adds logging capability
            'memory_optimization', # Adds memory optimization
            'data_validation'    # Adds validation capability
        ]
    })

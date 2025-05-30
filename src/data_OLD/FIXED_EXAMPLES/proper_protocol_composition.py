"""
BETTER: Split the current DataHandler into proper protocols and capabilities
"""

from typing import Protocol, runtime_checkable, List, Optional, Dict, Any
from datetime import datetime


# ✅ PROTOCOL 1: Pure data loading interface
@runtime_checkable
class DataLoader(Protocol):
    """Pure protocol for data loading - no implementation"""
    def load_data(self, symbols: List[str]) -> bool: ...
    def get_symbols(self) -> List[str]: ...


# ✅ PROTOCOL 2: Pure bar streaming interface  
@runtime_checkable
class BarStreamer(Protocol):
    """Pure protocol for bar streaming - no implementation"""
    def update_bars(self) -> bool: ...
    def has_more_data(self) -> bool: ...


# ✅ PROTOCOL 3: Pure data access interface
@runtime_checkable
class DataAccessor(Protocol):
    """Pure protocol for data access - no implementation"""
    def get_latest_bar(self, symbol: str) -> Optional[Bar]: ...
    def get_latest_bars(self, symbol: str, n: int) -> List[Bar]: ...


# ✅ SIMPLE IMPLEMENTATION: No inheritance, just protocol compliance
class HistoricalData:
    """Simple data handler - implements protocols through duck typing"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.data = {}
        self.symbols = []
        self.current_indices = {}
    
    # Implements DataLoader protocol
    def load_data(self, symbols: List[str]) -> bool:
        self.symbols = symbols
        for symbol in symbols:
            csv_file = f"{self.data_dir}/{symbol}.csv"
            self.data[symbol] = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            self.current_indices[symbol] = 0
        return True
    
    def get_symbols(self) -> List[str]:
        return self.symbols
    
    # Implements BarStreamer protocol
    def update_bars(self) -> bool:
        # Find next chronological bar
        next_time = None
        next_symbol = None
        
        for symbol in self.symbols:
            idx = self.current_indices[symbol]
            if idx < len(self.data[symbol]):
                timestamp = self.data[symbol].index[idx]
                if not next_time or timestamp < next_time:
                    next_time = timestamp
                    next_symbol = symbol
        
        if next_symbol:
            # Emit bar if we have event capability
            if hasattr(self, 'emit_bar_event'):
                bar_data = self.data[next_symbol].iloc[self.current_indices[next_symbol]]
                self.emit_bar_event(next_symbol, next_time, bar_data)
            
            self.current_indices[next_symbol] += 1
            return True
        
        return False
    
    def has_more_data(self) -> bool:
        return any(
            self.current_indices[symbol] < len(self.data[symbol])
            for symbol in self.symbols
        )
    
    # Implements DataAccessor protocol
    def get_latest_bar(self, symbol: str) -> Optional[Bar]:
        if symbol not in self.data:
            return None
        idx = self.current_indices.get(symbol, 0)
        if idx == 0:
            return None
        
        row = self.data[symbol].iloc[idx - 1]
        return Bar(
            symbol=symbol,
            timestamp=row.name,
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
    
    def get_latest_bars(self, symbol: str, n: int = 1) -> List[Bar]:
        if symbol not in self.data:
            return []
        
        idx = self.current_indices.get(symbol, 0)
        start_idx = max(0, idx - n)
        
        bars = []
        for i in range(start_idx, idx):
            row = self.data[symbol].iloc[i]
            bars.append(Bar(
                symbol=symbol,
                timestamp=row.name,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            ))
        return bars


# ✅ CAPABILITY-BASED ENHANCEMENT: Add features through composition
class DataSplittingCapability:
    """Adds train/test splitting to any data component"""
    
    def apply(self, component: Any, config: Dict[str, Any]) -> Any:
        # Add splitting state
        component.splits = {}
        component.active_split = None
        
        # Add splitting methods
        def setup_train_test_split(method='ratio', train_ratio=0.7, split_date=None):
            # Implementation here - no inheritance needed!
            for symbol, data in component.data.items():
                if method == 'ratio':
                    split_idx = int(len(data) * train_ratio)
                    train_data = data.iloc[:split_idx]
                    test_data = data.iloc[split_idx:]
                elif method == 'date':
                    train_data = data[data.index < split_date] 
                    test_data = data[data.index >= split_date]
                
                component.splits[symbol] = {
                    'train': train_data,
                    'test': test_data,
                    'full': data
                }
        
        component.setup_train_test_split = setup_train_test_split
        
        def set_active_split(split_name):
            component.active_split = split_name
            # Reset indices for new split
            for symbol in component.symbols:
                component.current_indices[symbol] = 0
        
        component.set_active_split = set_active_split
        
        return component


# ✅ USAGE: Create enhanced handler without inheritance
def create_enhanced_historical_handler():
    from src.core.components import ComponentFactory
    
    return ComponentFactory().create_component({
        'class': 'HistoricalData',
        'params': {'data_dir': 'data'},
        'capabilities': [
            'lifecycle',       # start/stop/reset methods
            'events',         # event emission (emit_bar_event method)
            'data_splitting', # train/test split methods  
            'logging',        # logging methods
            'monitoring'      # performance tracking
        ]
    })

# Result: All DataHandler functionality without ANY inheritance!
# - Implements protocols through duck typing
# - Gets enhanced through capabilities
# - Simple, testable, composable

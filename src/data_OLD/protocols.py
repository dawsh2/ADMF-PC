"""Data protocols for the ADMF-PC system."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Iterator
from datetime import datetime
import pandas as pd


class DataLoader(ABC):
    """Protocol for loading market data."""
    
    @abstractmethod
    def load_data(
        self, 
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_bars: Optional[int] = None
    ) -> pd.DataFrame:
        """Load market data for a symbol.
        
        Args:
            symbol: The symbol to load data for
            start_date: Optional start date filter
            end_date: Optional end date filter  
            max_bars: Optional limit on number of bars
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass
        
    @abstractmethod
    def get_data_info(self, symbol: str) -> Dict[str, Any]:
        """Get information about available data for a symbol.
        
        Returns dict with:
            - start_date: First available date
            - end_date: Last available date
            - total_bars: Total number of bars
            - frequency: Data frequency (e.g., '1min', '1D')
        """
        pass


class StreamingDataProvider(ABC):
    """Protocol for streaming market data."""
    
    @abstractmethod
    def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to real-time data for symbols."""
        pass
        
    @abstractmethod
    def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from real-time data."""
        pass
        
    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest bar for a symbol."""
        pass
        
    @abstractmethod
    def stream_bars(self) -> Iterator[Dict[str, Any]]:
        """Stream bars as they arrive."""
        pass


class DataFeed(ABC):
    """Protocol for unified data access (historical + streaming)."""
    
    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        lookback_bars: int
    ) -> pd.DataFrame:
        """Get historical data up to current point."""
        pass
        
    @abstractmethod
    def get_current_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current bar data."""
        pass
        
    @abstractmethod
    def advance(self) -> bool:
        """Advance to next time step (for backtesting).
        
        Returns:
            True if more data available, False if at end
        """
        pass
        
    @abstractmethod
    def reset(self) -> None:
        """Reset data feed to beginning."""
        pass
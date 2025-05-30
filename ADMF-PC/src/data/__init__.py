"""
Data module for ADMF-PC.

This module provides data loading, management, and distribution capabilities
for the containerized backtesting and optimization system.

Example Usage:
    ```python
    from src.data import HistoricalDataHandler, Bar, Timeframe
    
    # Create data handler
    handler = HistoricalDataHandler(
        handler_id="historical_data",
        data_dir="data",
        timeframe=Timeframe.D1
    )
    
    # Initialize with context
    handler.initialize({
        'event_bus': event_bus,
        'container_id': 'backtest_001'
    })
    
    # Load data
    handler.load_data(['AAPL', 'MSFT'])
    
    # Set up train/test split
    handler.setup_train_test_split(method='ratio', train_ratio=0.7)
    
    # Use training data
    handler.set_active_split('train')
    
    # Start emitting bars
    handler.start()
    while handler.update_bars():
        # Events are published to event bus
        pass
    ```
"""

from .models import (
    Timeframe,
    Bar,
    Tick,
    DataView,
    TimeSeriesData
)

from .handlers import (
    DataHandler,
    HistoricalDataHandler,
    DataSplit
)

from .loaders import (
    DataLoader,
    CSVLoader,
    MemoryOptimizedCSVLoader,
    MultiFileLoader
)

from .streamer import (
    DataStreamer,
    HistoricalDataStreamer
)


__all__ = [
    # Models
    "Timeframe",
    "Bar",
    "Tick",
    "DataView",
    "TimeSeriesData",
    
    # Handlers
    "DataHandler",
    "HistoricalDataHandler",
    "DataSplit",
    
    # Loaders
    "DataLoader",
    "CSVLoader",
    "MemoryOptimizedCSVLoader",
    "MultiFileLoader",
    
    # Streamers
    "DataStreamer",
    "HistoricalDataStreamer"
]
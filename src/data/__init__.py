"""
ADMF-PC Data Module - Protocol+Composition Implementation

Architecture Reference: docs/SYSTEM_ARCHITECTURE_V5.MD#data-module
Style Guide: STYLE.md - Canonical data implementations

This module implements data handling using pure Protocol+Composition architecture
with ZERO inheritance. Components implement protocols directly.

Example Usage:
    ```python
    from src.data import create_data_handler
    
    # Create simple data handler
    handler = create_data_handler('historical', handler_id='hist_data', data_dir='data')
    
    # Use the handler directly - no capability enhancement needed
    handler.load_data(['AAPL', 'GOOGL'])
    handler.setup_split(method='ratio', train_ratio=0.7)
    handler.set_active_split('train')
    
    # Stream data
    handler.start()
    while handler.update_bars():
        # Process bars
        pass
    ```
"""

from typing import Any

from .protocols import (
    # Core protocols - NO ABC anywhere!
    DataProvider,
    DataLoader,
    BarStreamer,
    SignalStreamer,
    DataAccessor,
    DataSplitter,
    DataValidator,
    DataTransformer,
    StreamingProvider,
    DataFeed,
    
    # Event protocols
    EventEmitter,
    EventSubscriber,
    
    # Capability detection protocols
    HasLifecycle,
    HasLogging,
    HasMonitoring
)

from .models import (
    # Data structures - simple classes
    Timeframe,
    Bar,
    Tick,
    DataView,
    TimeSeriesData,
    DataSplit,
    ValidationResult
)

from .loaders import (
    # Loaders - NO INHERITANCE!
    SimpleCSVLoader,
    MemoryEfficientCSVLoader,
    MultiFileLoader,
    DatabaseLoader,
    create_data_loader
)

from .handlers import (
    # Handlers - NO INHERITANCE!
    SimpleHistoricalDataHandler,
    StreamingDataHandler,
    SimpleDataValidator,
    create_data_handler
)

from .streamers import (
    # Bar streamers - NO INHERITANCE!
    StreamedBar,
    SimpleBarStreamer,
    MultiAssetStreamer,
    
    # Signal streamers - NO INHERITANCE!
    BoundaryAwareReplay,
    SignalStreamerComponent,
    
    # Temporal events - NO INHERITANCE!
    MarketHours,
    TemporalEventEmitter,
    create_temporal_emitter,
    add_temporal_events_to_streamer,
)

# Capabilities removed - components implement protocols directly


# Enhanced creation functions removed - use direct creation functions
# Components implement protocols directly without capability enhancement


# All exports - clean interface
__all__ = [
    # Protocols
    "DataProvider",
    "DataLoader", 
    "BarStreamer",
    "SignalStreamer",
    "DataAccessor",
    "DataSplitter",
    "DataValidator",
    "DataTransformer",
    "StreamingProvider",
    "DataFeed",
    "EventEmitter",
    "EventSubscriber",
    "HasLifecycle",
    "HasLogging",
    "HasMonitoring",
    
    # Models
    "Timeframe",
    "Bar",
    "Tick",
    "DataView",
    "TimeSeriesData",
    "DataSplit",
    "ValidationResult",
    
    # Loaders
    "SimpleCSVLoader",
    "MemoryEfficientCSVLoader", 
    "MultiFileLoader",
    "DatabaseLoader",
    "create_data_loader",
    
    # Handlers
    "SimpleHistoricalDataHandler",
    "StreamingDataHandler",
    "SimpleDataValidator",
    "create_data_handler",
    
    # Streamers
    "StreamedBar",
    "SimpleBarStreamer",
    "MultiAssetStreamer",
    "BoundaryAwareReplay",
    "SignalStreamerComponent",
    "MarketHours",
    "TemporalEventEmitter",
    "create_temporal_emitter",
    "add_temporal_events_to_streamer",
    
    # Capabilities removed - components implement protocols directly
    # Enhanced functions removed - use direct creation functions
]

"""
ADMF-PC Data Module - Protocol+Composition Implementation

This module implements data handling using pure Protocol+Composition architecture
with ZERO inheritance. All functionality is added through capabilities.

Example Usage:
    ```python
    from src.data import create_data_handler, apply_capabilities
    
    # Create simple data handler
    handler = create_data_handler('historical', handler_id='hist_data', data_dir='data')
    
    # Enhance with capabilities
    enhanced_handler = apply_capabilities(handler, [
        'logging',      # Adds logging methods
        'monitoring',   # Adds metrics tracking
        'events',       # Adds event emission
        'validation',   # Adds data validation
        'splitting',    # Adds train/test splitting
        'memory_optimization'  # Adds memory optimization
    ])
    
    # Use the enhanced handler
    enhanced_handler.load_data(['AAPL', 'GOOGL'])
    enhanced_handler.setup_split(method='ratio', train_ratio=0.7)
    enhanced_handler.set_active_split('train')
    
    # Stream data
    enhanced_handler.start()
    while enhanced_handler.update_bars():
        # Events are emitted automatically
        pass
    ```
"""

from .protocols import (
    # Core protocols - NO ABC anywhere!
    DataProvider,
    DataLoader,
    BarStreamer,
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
    # Streamers - NO INHERITANCE!
    StreamedBar,
    StreamedSignal,
    SimpleHistoricalStreamer,
    SimpleSignalStreamer,
    SimpleRealTimeStreamer,
    MultiSourceStreamer,
    create_streamer
)

from .capabilities import (
    # Capabilities for enhancement
    DataLoggingCapability,
    DataMonitoringCapability,
    DataEventCapability,
    DataValidationCapability,
    DataSplittingCapability,
    MemoryOptimizationCapability,
    DATA_CAPABILITIES,
    apply_capabilities
)


# Convenience functions for common usage patterns

def create_enhanced_data_handler(
    handler_type: str = 'historical',
    capabilities: List[str] = None,
    **config
) -> Any:
    """
    Create a data handler enhanced with capabilities.
    
    Args:
        handler_type: Type of handler ('historical', 'streaming')
        capabilities: List of capabilities to add
        **config: Handler and capability configuration
        
    Returns:
        Enhanced data handler
    """
    # Default capabilities for data handlers
    if capabilities is None:
        capabilities = ['logging', 'monitoring', 'events', 'validation']
    
    # Create base handler
    handler = create_data_handler(handler_type, **config)
    
    # Apply capabilities
    return apply_capabilities(handler, capabilities, config)


def create_enhanced_data_loader(
    loader_type: str = 'csv',
    capabilities: List[str] = None,
    **config
) -> Any:
    """
    Create a data loader enhanced with capabilities.
    
    Args:
        loader_type: Type of loader ('csv', 'memory_csv', 'multi_file')
        capabilities: List of capabilities to add
        **config: Loader and capability configuration
        
    Returns:
        Enhanced data loader
    """
    # Default capabilities for loaders
    if capabilities is None:
        capabilities = ['logging', 'validation', 'memory_optimization']
    
    # Create base loader
    loader = create_data_loader(loader_type, **config)
    
    # Apply capabilities  
    return apply_capabilities(loader, capabilities, config)


def create_enhanced_streamer(
    streamer_type: str = 'historical',
    capabilities: List[str] = None,
    **config
) -> Any:
    """
    Create a data streamer enhanced with capabilities.
    
    Args:
        streamer_type: Type of streamer ('historical', 'signal', 'realtime')
        capabilities: List of capabilities to add
        **config: Streamer and capability configuration
        
    Returns:
        Enhanced data streamer
    """
    # Default capabilities for streamers
    if capabilities is None:
        capabilities = ['logging', 'monitoring', 'events']
    
    # Create base streamer
    streamer = create_streamer(streamer_type, **config)
    
    # Apply capabilities
    return apply_capabilities(streamer, capabilities, config)


# All exports - clean interface
__all__ = [
    # Protocols
    "DataProvider",
    "DataLoader", 
    "BarStreamer",
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
    "StreamedSignal",
    "SimpleHistoricalStreamer",
    "SimpleSignalStreamer", 
    "SimpleRealTimeStreamer",
    "MultiSourceStreamer",
    "create_streamer",
    
    # Capabilities
    "DataLoggingCapability",
    "DataMonitoringCapability",
    "DataEventCapability", 
    "DataValidationCapability",
    "DataSplittingCapability",
    "MemoryOptimizationCapability",
    "DATA_CAPABILITIES",
    "apply_capabilities",
    
    # Convenience functions
    "create_enhanced_data_handler",
    "create_enhanced_data_loader",
    "create_enhanced_streamer"
]

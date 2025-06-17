"""
Data streaming components for ADMF-PC.

This module provides various data streaming capabilities:
- Bar streaming from historical data
- Signal streaming from stored signals  
- Temporal event emission for constraint management

All streamers follow Protocol+Composition architecture.
"""

from .bar_streamer import (
    StreamedBar,
    SimpleHistoricalStreamer as SimpleBarStreamer,
    MultiSourceStreamer as MultiAssetStreamer
)

# Live streaming (optional - requires websockets)
try:
    from .alpaca_streamer import (
        AlpacaWebSocketStreamer,
        AlpacaBarPrinter,
        create_alpaca_streamer,
        demo_alpaca_streaming
    )
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False

from .signal_streamer import (
    BoundaryAwareReplay,
    SignalStreamerComponent
)

from .temporal_events import (
    MarketHours,
    TemporalEventEmitter,
    create_temporal_emitter,
    add_temporal_events_to_streamer
)

__all__ = [
    # Bar streaming
    'StreamedBar',
    'SimpleBarStreamer', 
    'MultiAssetStreamer',
    
    # Signal streaming
    'BoundaryAwareReplay',
    'SignalStreamerComponent',
    
    # Temporal events
    'MarketHours',
    'TemporalEventEmitter',
    'create_temporal_emitter',
    'add_temporal_events_to_streamer',
]

# Add Alpaca streaming to exports if available
if _ALPACA_AVAILABLE:
    __all__.extend([
        'AlpacaWebSocketStreamer',
        'AlpacaBarPrinter', 
        'create_alpaca_streamer',
        'demo_alpaca_streaming'
    ])
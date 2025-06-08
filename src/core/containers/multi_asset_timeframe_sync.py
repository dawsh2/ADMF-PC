"""
Multi-asset and multi-timeframe synchronization components.

Provides the TimeAlignmentBuffer for handling synchronization of bars
from multiple symbols and timeframes before processing.
"""

from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

from ..events import Event, EventType
from .protocols import ContainerComponent


logger = logging.getLogger(__name__)


@dataclass
class TimeAlignmentBuffer(ContainerComponent):
    """
    Buffers and aligns bars from multiple symbols/timeframes.
    
    This component ensures that:
    1. All required symbols have data before processing
    2. Bars are time-aligned across different timeframes
    3. Processing happens in correct chronological order
    4. Missing data is handled gracefully
    
    Usage:
        - Added to Feature Container to handle multi-asset synchronization
        - Buffers incoming BAR events until alignment criteria met
        - Emits synchronized bar sets for processing
    """
    
    # Configuration
    symbols: List[str]  # Expected symbols
    timeframes: List[str]  # Expected timeframes
    alignment_mode: str = 'all'  # 'all' or 'any'
    max_wait_bars: int = 10  # Max bars to wait for alignment
    
    # Optional callback for synchronized bars
    on_synchronized_bars: Optional[Callable[[Dict[str, Any]], None]] = None
    
    # Internal state
    bar_buffers: Dict[str, Dict[str, List[Any]]] = field(default_factory=dict)
    last_processed_time: Optional[datetime] = None
    pending_count: int = 0
    
    def initialize(self, container: 'Container') -> None:
        """Initialize with container reference."""
        self.container = container
        
        # Initialize buffers for each symbol/timeframe combination
        for symbol in self.symbols:
            self.bar_buffers[symbol] = {}
            for timeframe in self.timeframes:
                self.bar_buffers[symbol][timeframe] = []
        
        logger.info(f"TimeAlignmentBuffer initialized for symbols: {self.symbols}, "
                   f"timeframes: {self.timeframes}, mode: {self.alignment_mode}")
    
    def start(self) -> None:
        """Subscribe to BAR events."""
        self.container.event_bus.subscribe(EventType.BAR, self.on_bar)
        logger.info("TimeAlignmentBuffer started")
    
    def stop(self) -> None:
        """Process any remaining bars."""
        if self.pending_count > 0:
            logger.warning(f"TimeAlignmentBuffer stopped with {self.pending_count} pending bars")
    
    def get_state(self) -> Dict[str, Any]:
        """Get buffer state."""
        buffer_sizes = {}
        for symbol, timeframe_buffers in self.bar_buffers.items():
            for timeframe, buffer in timeframe_buffers.items():
                buffer_sizes[f"{symbol}_{timeframe}"] = len(buffer)
        
        return {
            'pending_count': self.pending_count,
            'buffer_sizes': buffer_sizes,
            'last_processed': self.last_processed_time.isoformat() if self.last_processed_time else None
        }
    
    def on_bar(self, event: Event) -> None:
        """Buffer incoming bar and check for alignment."""
        bar = event.payload.get('bar')
        symbol = event.payload.get('symbol')
        timeframe = event.payload.get('timeframe')
        
        if not all([bar, symbol, timeframe]):
            logger.warning("Incomplete BAR event received")
            return
        
        # Check if this is an expected symbol/timeframe
        if symbol not in self.symbols or timeframe not in self.timeframes:
            logger.debug(f"Ignoring bar for {symbol}_{timeframe} - not in expected set")
            return
        
        # Add to buffer
        self.bar_buffers[symbol][timeframe].append(event.payload)
        self.pending_count += 1
        
        # Check if we can process synchronized bars
        self._check_and_process_alignment()
    
    def _check_and_process_alignment(self) -> None:
        """Check if alignment criteria are met and process if so."""
        if self.alignment_mode == 'all':
            # Need at least one bar from each symbol/timeframe
            aligned = self._check_all_alignment()
        else:  # 'any'
            # Process when any symbol has all timeframes
            aligned = self._check_any_alignment()
        
        if aligned:
            self._process_synchronized_bars(aligned)
    
    def _check_all_alignment(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Check if all symbols/timeframes have bars."""
        result = {}
        
        for symbol in self.symbols:
            symbol_bars = {}
            for timeframe in self.timeframes:
                buffer = self.bar_buffers[symbol][timeframe]
                if not buffer:
                    return None  # Missing data
                symbol_bars[timeframe] = buffer[0]  # Get oldest bar
            result[symbol] = symbol_bars
        
        return result
    
    def _check_any_alignment(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """Check if any symbol has all timeframes available."""
        for symbol in self.symbols:
            symbol_bars = {}
            all_present = True
            
            for timeframe in self.timeframes:
                buffer = self.bar_buffers[symbol][timeframe]
                if not buffer:
                    all_present = False
                    break
                symbol_bars[timeframe] = buffer[0]
            
            if all_present:
                return {symbol: symbol_bars}
        
        return None
    
    def _process_synchronized_bars(self, aligned_bars: Dict[str, Dict[str, Any]]) -> None:
        """Process synchronized bars."""
        # Extract bars from buffers
        processed_bars = {}
        features = {}
        
        for symbol, timeframe_bars in aligned_bars.items():
            processed_bars[symbol] = {}
            for timeframe, bar_data in timeframe_bars.items():
                # Remove from buffer
                self.bar_buffers[symbol][timeframe].pop(0)
                self.pending_count -= 1
                processed_bars[symbol][timeframe] = bar_data
                
                # Extract features if present
                if 'features' in bar_data:
                    if symbol not in features:
                        features[symbol] = {}
                    features[symbol].update(bar_data['features'])
        
        # Update last processed time
        # Find the latest bar timestamp
        latest_time = None
        for symbol_bars in processed_bars.values():
            for bar_data in symbol_bars.values():
                bar = bar_data.get('bar')
                if bar and hasattr(bar, 'timestamp'):
                    if latest_time is None or bar.timestamp > latest_time:
                        latest_time = bar.timestamp
        
        if latest_time:
            self.last_processed_time = latest_time
        
        # Notify via callback if set
        if self.on_synchronized_bars:
            self.on_synchronized_bars({
                'bars': processed_bars,
                'features': features,
                'timestamp': self.last_processed_time
            })
        
        # Emit synchronized event
        sync_event = Event(
            event_type='SYNCHRONIZED_BARS',  # Custom event type
            payload={
                'bars': processed_bars,
                'features': features,
                'timestamp': self.last_processed_time,
                'symbols': list(aligned_bars.keys()),
                'alignment_mode': self.alignment_mode
            },
            source_id=self.container.container_id if self.container else 'time_alignment_buffer'
        )
        
        # Publish to container's event bus
        if self.container:
            self.container.event_bus.publish(sync_event)
        
        logger.debug(f"Processed synchronized bars for {list(aligned_bars.keys())} "
                    f"at {self.last_processed_time}")
    
    def get_buffer_status(self) -> Dict[str, int]:
        """Get current buffer depths."""
        status = {}
        for symbol, timeframe_buffers in self.bar_buffers.items():
            for timeframe, buffer in timeframe_buffers.items():
                status[f"{symbol}_{timeframe}"] = len(buffer)
        return status
    
    def clear_buffers(self) -> None:
        """Clear all buffers."""
        for symbol in self.bar_buffers:
            for timeframe in self.bar_buffers[symbol]:
                self.bar_buffers[symbol][timeframe].clear()
        self.pending_count = 0
        logger.info("Cleared all TimeAlignmentBuffer buffers")
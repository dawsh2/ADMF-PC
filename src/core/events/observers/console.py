"""
Console Event Observer for Development and Debugging

Provides real-time console output of events with structured formatting.
Integrates with the existing tracing infrastructure.
"""

from typing import Dict, Any, Optional, List, Set, Callable
from datetime import datetime
import logging

from ..protocols import EventObserverProtocol
from ..types import Event

logger = logging.getLogger(__name__)


class ConsoleEventObserver:
    """
    Observer that prints events to console in a structured format.
    
    Designed for development and debugging - provides immediate visual feedback
    of event flow through the system with correlation ID tracking.
    """
    
    def __init__(self, 
                 correlation_id: Optional[str] = None,
                 event_filter: Optional[List[str]] = None,
                 show_payload: bool = True,
                 show_timing: bool = True,
                 max_payload_length: int = 100):
        """
        Initialize console observer.
        
        Args:
            correlation_id: Correlation ID for this observer session
            event_filter: List of event types to show (None = show all)
            show_payload: Whether to show event payload details
            show_timing: Whether to show timing information
            max_payload_length: Maximum characters to show in payload
        """
        self.correlation_id = correlation_id or f"console_{datetime.now().strftime('%H%M%S')}"
        self.event_filter = set(event_filter) if event_filter else None
        self.show_payload = show_payload
        self.show_timing = show_timing
        self.max_payload_length = max_payload_length
        
        # Statistics
        self.events_shown = 0
        self.events_filtered = 0
        self.start_time = datetime.now()
        
        logger.info(f"Console observer initialized - correlation_id: {self.correlation_id}")
        if self.event_filter:
            logger.info(f"Console filter: {sorted(self.event_filter)}")
    
    def on_publish(self, event: Event) -> None:
        """Print event to console when published."""
        if self._should_show_event(event):
            self._print_event(event, "PUBLISH")
            self.events_shown += 1
        else:
            self.events_filtered += 1
    
    def on_delivered(self, event: Event, handler: Callable) -> None:
        """Print delivery confirmation if enabled."""
        if self._should_show_event(event) and self.show_timing:
            handler_name = getattr(handler, '__name__', str(handler))
            if len(handler_name) > 20:
                handler_name = handler_name[:17] + "..."
            print(f"    ✓ → {handler_name}")
    
    def on_error(self, event: Event, handler: Callable, error: Exception) -> None:
        """Print error information."""
        if self._should_show_event(event):
            handler_name = getattr(handler, '__name__', str(handler))
            print(f"    ❌ → {handler_name}: {str(error)[:50]}")
    
    def _should_show_event(self, event: Event) -> bool:
        """Check if event should be shown based on filter."""
        if self.event_filter is None:
            return True
        return event.event_type in self.event_filter
    
    def _print_event(self, event: Event, stage: str) -> None:
        """Print event in structured format."""
        # Format timestamp
        timestamp = event.timestamp or datetime.now()
        time_str = timestamp.strftime("%H:%M:%S.%f")[:-3]
        
        # Format container and source
        container = event.container_id or "unknown"
        if len(container) > 15:
            container = container[:12] + "..."
        
        source = event.source_id or "unknown"
        if len(source) > 15:
            source = source[:12] + "..."
        
        # Format correlation ID
        correlation = event.correlation_id or "none"
        if len(correlation) > 12:
            correlation = correlation[:9] + "..."
        
        # Main event line
        print(f"🔍 {time_str} | {event.event_type:12} | {container:15} | {source:15} | {correlation:12}")
        
        # Payload details if enabled
        if self.show_payload and event.payload:
            payload_str = self._format_payload(event.payload)
            if payload_str:
                print(f"    📦 {payload_str}")
        
        # Metadata if interesting
        if event.metadata:
            metadata_str = self._format_metadata(event.metadata)
            if metadata_str:
                print(f"    📄 {metadata_str}")
    
    def _format_payload(self, payload: Any) -> str:
        """Format payload for display."""
        try:
            if isinstance(payload, dict):
                # Show key fields
                key_fields = []
                
                # Common important fields
                for key in ['symbol', 'price', 'signal', 'signal_value', 'quantity', 'direction', 'pnl']:
                    if key in payload:
                        value = payload[key]
                        if isinstance(value, float):
                            key_fields.append(f"{key}={value:.3f}")
                        else:
                            key_fields.append(f"{key}={value}")
                
                # Add other fields up to length limit
                remaining_fields = [f"{k}={v}" for k, v in payload.items() 
                                  if k not in ['symbol', 'price', 'signal', 'signal_value', 'quantity', 'direction', 'pnl']]
                
                all_fields = key_fields + remaining_fields[:3]  # Limit to prevent clutter
                result = " | ".join(all_fields)
                
                if len(result) > self.max_payload_length:
                    result = result[:self.max_payload_length-3] + "..."
                
                return result
            else:
                # Simple string representation
                result = str(payload)
                if len(result) > self.max_payload_length:
                    result = result[:self.max_payload_length-3] + "..."
                return result
                
        except Exception as e:
            return f"[payload format error: {e}]"
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format interesting metadata for display."""
        try:
            interesting_fields = []
            
            # Show event ID if present
            if 'event_id' in metadata:
                interesting_fields.append(f"id={metadata['event_id']}")
            
            # Show sequence if present
            if 'sequence_number' in metadata:
                interesting_fields.append(f"seq={metadata['sequence_number']}")
            
            # Show timing info if enabled and present
            if self.show_timing and 'timing' in metadata:
                timing = metadata['timing']
                if 'trace_enhanced' in timing:
                    interesting_fields.append("traced")
            
            if interesting_fields:
                return " | ".join(interesting_fields)
            return ""
            
        except Exception as e:
            return f"[metadata format error: {e}]"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get console observer summary."""
        duration = (datetime.now() - self.start_time).total_seconds()
        return {
            'correlation_id': self.correlation_id,
            'events_shown': self.events_shown,
            'events_filtered': self.events_filtered,
            'total_events': self.events_shown + self.events_filtered,
            'filter_active': self.event_filter is not None,
            'filtered_types': list(self.event_filter) if self.event_filter else None,
            'duration_seconds': duration,
            'events_per_second': self.events_shown / max(1, duration)
        }
    
    def print_summary(self) -> None:
        """Print summary statistics to console."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print(f"📊 Console Observer Summary - {summary['correlation_id']}")
        print("="*60)
        print(f"Events shown:    {summary['events_shown']}")
        print(f"Events filtered: {summary['events_filtered']}")
        print(f"Total events:    {summary['total_events']}")
        print(f"Duration:        {summary['duration_seconds']:.2f}s")
        print(f"Events/sec:      {summary['events_per_second']:.1f}")
        if summary['filter_active']:
            print(f"Active filter:   {summary['filtered_types']}")
        print("="*60)


def create_console_observer_from_config(config: Dict[str, Any]) -> ConsoleEventObserver:
    """
    Create console observer from configuration.
    
    Args:
        config: Configuration dict with:
            - correlation_id: Observer correlation ID
            - console_filter: List of event types to show
            - show_payload: Whether to show payload details
            - show_timing: Whether to show timing information
            - max_payload_length: Maximum payload display length
    
    Returns:
        Configured ConsoleEventObserver instance
    """
    return ConsoleEventObserver(
        correlation_id=config.get('correlation_id'),
        event_filter=config.get('console_filter'),
        show_payload=config.get('show_payload', True),
        show_timing=config.get('show_timing', True),
        max_payload_length=config.get('max_payload_length', 100)
    )
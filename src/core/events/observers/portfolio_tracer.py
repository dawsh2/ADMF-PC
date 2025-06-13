"""
Portfolio-specific event tracer.

This tracer is attached to portfolio containers to capture and store
events relevant to that specific portfolio, even when using a shared event bus.
"""

from typing import Optional, Dict, Any, List, Set
import logging
from datetime import datetime

from ..types import Event, EventType
from ..protocols import EventObserverProtocol
from ..storage.hierarchical import HierarchicalEventStorage, HierarchicalStorageConfig
from ..storage.sparse_signal_storage import SparseSignalStorage

logger = logging.getLogger(__name__)


class PortfolioTracer(EventObserverProtocol):
    """
    Event tracer specific to portfolio containers.
    
    This allows portfolio containers to maintain their own event storage
    even when sharing a root event bus with other containers.
    """
    
    def __init__(self, 
                 container_id: str,
                 workflow_id: str,
                 managed_strategies: List[str],
                 storage_config: Optional[Dict[str, Any]] = None):
        """
        Initialize portfolio tracer.
        
        Args:
            container_id: Portfolio container ID
            workflow_id: Workflow ID for storage organization
            managed_strategies: List of strategy names this portfolio manages
            storage_config: Storage configuration
        """
        self.container_id = container_id
        self.workflow_id = workflow_id
        self.managed_strategies = set(managed_strategies)
        
        # Event types to trace for portfolios
        self.traced_event_types = {
            EventType.SIGNAL.value,
            EventType.ORDER.value,
            EventType.FILL.value,
            EventType.POSITION_OPEN.value,
            EventType.POSITION_CLOSE.value,
            EventType.PORTFOLIO_UPDATE.value
        }
        
        # Initialize storage
        config = storage_config or {}
        storage_backend = config.get('storage_backend', 'hierarchical')
        
        if storage_backend == 'hierarchical':
            self.storage = HierarchicalEventStorage(
                HierarchicalStorageConfig(
                    base_dir=config.get('base_dir', 'workspaces'),
                    batch_size=config.get('batch_size', 1000),
                    format=config.get('format', 'jsonl'),
                    enable_container_isolation=True
                )
            )
            # Set context for this portfolio
            self.storage.set_context(
                workflow_id=workflow_id,
                container_id=container_id
            )
        else:
            self.storage = None
            
        self._event_count = 0
        logger.info(f"PortfolioTracer initialized for {container_id} managing strategies: {managed_strategies}")
    
    def on_event(self, event: Event) -> None:
        """
        Process event and store if relevant to this portfolio.
        
        Args:
            event: Event to process
        """
        # Only trace specific event types
        if event.event_type not in self.traced_event_types:
            return
            
        # For SIGNAL events, filter by strategy
        if event.event_type == EventType.SIGNAL.value:
            strategy_id = event.payload.get('strategy_id', '')
            
            # Check if this signal is from a managed strategy
            is_managed = False
            for strategy_name in self.managed_strategies:
                if strategy_name in strategy_id:
                    is_managed = True
                    break
                    
            if not is_managed:
                return
        
        # For ORDER/FILL events, could filter by portfolio_id if available
        # For now, trace all orders/fills since they're likely relevant
        
        # Store the event
        if self.storage:
            # Add portfolio context to event
            if not event.container_id:
                event.container_id = self.container_id
                
            self.storage.store(event)
            self._event_count += 1
            
            logger.debug(f"Portfolio {self.container_id} traced {event.event_type} event "
                        f"(total: {self._event_count})")
    
    def flush(self) -> None:
        """Flush any pending events to storage."""
        if self.storage and hasattr(self.storage, 'flush_all'):
            self.storage.flush_all()
            logger.info(f"Flushed {self._event_count} events for portfolio {self.container_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracer statistics."""
        stats = {
            'container_id': self.container_id,
            'managed_strategies': list(self.managed_strategies),
            'events_traced': self._event_count,
            'traced_types': list(self.traced_event_types)
        }
        
        if self.storage and hasattr(self.storage, 'get_statistics'):
            stats['storage_stats'] = self.storage.get_statistics()
            
        return stats
    
    # Implement required EventObserverProtocol methods
    def on_publish(self, event: Event) -> None:
        """Called when event is published - handled by on_event."""
        self.on_event(event)
    
    def on_delivered(self, event: Event, handler_count: int) -> None:
        """Called after event delivery - not used for tracing."""
        pass
    
    def on_error(self, event: Event, error: Exception) -> None:
        """Called on event error - log but continue."""
        logger.error(f"Portfolio tracer error for event {event.event_type}: {error}")
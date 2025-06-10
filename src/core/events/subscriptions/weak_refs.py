"""Advanced subscription management with weak references."""

from typing import Optional, Callable, Dict, Any, List
import weakref
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class WeakSubscriptionManager:
    """
    Subscription manager using weak references to prevent memory leaks.
    
    Critical for long-running systems where handlers may be destroyed.
    """
    
    def __init__(self):
        self._subscriptions: Dict[str, List[weakref.ref]] = defaultdict(list)
        
    def subscribe(self, event_type: str, handler: Callable, 
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Subscribe with weak reference to handler."""
        # Use weak reference to allow handler garbage collection
        weak_handler = weakref.ref(handler, 
            lambda ref: self._cleanup_dead_ref(event_type, ref))
        self._subscriptions[event_type].append(weak_handler)
        
    def _cleanup_dead_ref(self, event_type: str, dead_ref: weakref.ref) -> None:
        """Automatically clean up dead references."""
        self._subscriptions[event_type] = [
            ref for ref in self._subscriptions[event_type] 
            if ref is not dead_ref
        ]
        
    def get_handlers(self, event_type: str) -> List[Callable]:
        """Get live handlers for event type."""
        handlers = []
        dead_refs = []
        
        for weak_ref in self._subscriptions[event_type]:
            handler = weak_ref()
            if handler is not None:
                handlers.append(handler)
            else:
                dead_refs.append(weak_ref)
                
        # Clean up dead references
        for dead_ref in dead_refs:
            self._subscriptions[event_type].remove(dead_ref)
            
        return handlers
    
    def unsubscribe(self, event_type: str, handler: Callable) -> bool:
        """Unsubscribe handler from event type."""
        removed = False
        refs_to_remove = []
        
        for weak_ref in self._subscriptions[event_type]:
            if weak_ref() == handler:
                refs_to_remove.append(weak_ref)
                removed = True
                
        for ref in refs_to_remove:
            self._subscriptions[event_type].remove(ref)
            
        return removed
    
    def clear(self) -> None:
        """Clear all subscriptions."""
        self._subscriptions.clear()
        
    def get_subscription_count(self) -> Dict[str, int]:
        """Get count of live subscriptions by event type."""
        counts = {}
        for event_type, refs in self._subscriptions.items():
            # Count only live references
            live_count = sum(1 for ref in refs if ref() is not None)
            if live_count > 0:
                counts[event_type] = live_count
        return counts
"""
EventScopeDetector - Communication Pattern Detection Component for Logging System v3
Composable component for detecting and classifying event communication scopes
"""

from enum import Enum
from typing import Dict, Any, Optional, Set
from .protocols import EventScopeClassifier, PerformanceOptimized


class EventScope(Enum):
    """Event communication scope classification."""
    INTERNAL_BUS = "internal_bus"
    EXTERNAL_FAST = "external_fast_tier"
    EXTERNAL_STANDARD = "external_standard_tier"
    EXTERNAL_RELIABLE = "external_reliable_tier"
    COMPONENT_INTERNAL = "component_internal"
    LIFECYCLE_MANAGEMENT = "lifecycle_management"


class EventScopeDetector:
    """
    Event scope detection - composable component.
    
    This component automatically classifies events by their communication scope
    to enable proper logging categorization and performance optimization.
    
    Features:
    - Communication pattern detection
    - Event scope classification
    - Performance optimization for frequent detection
    - Configurable scope mapping
    """
    
    def __init__(self):
        """Initialize event scope detector."""
        # Pre-compiled scope detection patterns for performance
        self.scope_patterns = {
            'internal_scope': EventScope.INTERNAL_BUS.value,
            'lifecycle_operation': EventScope.LIFECYCLE_MANAGEMENT.value,
        }
        
        self.tier_mapping = {
            'fast': EventScope.EXTERNAL_FAST.value,
            'standard': EventScope.EXTERNAL_STANDARD.value,
            'reliable': EventScope.EXTERNAL_RELIABLE.value,
        }
        
        # Performance optimization
        self._detection_cache = {}  # Cache common patterns
        self._cache_hits = 0
        self._cache_misses = 0
    
    def detect_scope(self, context: Dict[str, Any]) -> str:
        """
        Detect the communication scope of an event.
        
        Args:
            context: Event context dictionary
            
        Returns:
            Event scope string (e.g., "internal_bus", "external_standard_tier")
        """
        # Create cache key from relevant context
        cache_key = self._create_cache_key(context)
        
        # Check cache first for performance
        if cache_key in self._detection_cache:
            self._cache_hits += 1
            return self._detection_cache[cache_key]
        
        # Perform detection
        scope = self._detect_scope_internal(context)
        
        # Cache result for future use
        self._detection_cache[cache_key] = scope
        self._cache_misses += 1
        
        # Prevent cache from growing too large
        if len(self._detection_cache) > 1000:
            self._trim_cache()
        
        return scope
    
    def _detect_scope_internal(self, context: Dict[str, Any]) -> str:
        """Internal scope detection logic."""
        # Fast path for common patterns
        for key, scope in self.scope_patterns.items():
            if key in context:
                return scope
        
        # Handle tier-based scopes (cross-container communication)
        if 'publish_tier' in context:
            tier = context['publish_tier']
            return self.tier_mapping.get(tier, EventScope.EXTERNAL_STANDARD.value)
        
        # Handle explicit event flow classification
        if 'event_flow' in context:
            return context['event_flow']
        
        # Handle container boundary detection
        if 'source_container' in context and 'target_container' in context:
            source = context['source_container']
            target = context['target_container']
            
            if source == target:
                return EventScope.INTERNAL_BUS.value
            else:
                # Cross-container - determine tier based on event type
                event_type = context.get('event_type', '')
                return self._classify_cross_container_scope(event_type)
        
        # Default to component internal
        return EventScope.COMPONENT_INTERNAL.value
    
    def _classify_cross_container_scope(self, event_type: str) -> str:
        """Classify cross-container events by type."""
        # High-frequency data events use fast tier
        if event_type in ['BAR', 'TICK', 'QUOTE', 'BOOK_UPDATE']:
            return EventScope.EXTERNAL_FAST.value
        
        # Critical execution events use reliable tier
        elif event_type in ['ORDER', 'FILL', 'SYSTEM', 'ERROR']:
            return EventScope.EXTERNAL_RELIABLE.value
        
        # Standard business logic uses standard tier
        else:
            return EventScope.EXTERNAL_STANDARD.value
    
    def _create_cache_key(self, context: Dict[str, Any]) -> str:
        """Create cache key from relevant context fields."""
        relevant_fields = [
            'internal_scope', 'lifecycle_operation', 'publish_tier',
            'event_flow', 'source_container', 'target_container', 'event_type'
        ]
        
        key_parts = []
        for field in relevant_fields:
            if field in context:
                key_parts.append(f"{field}={context[field]}")
        
        return "|".join(key_parts) if key_parts else "default"
    
    def _trim_cache(self):
        """Trim cache to prevent memory growth."""
        # Keep most recently used half of cache
        # This is a simple LRU approximation
        items = list(self._detection_cache.items())
        self._detection_cache = dict(items[len(items)//2:])
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scope detector statistics."""
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate_percent': cache_hit_rate,
            'cache_size': len(self._detection_cache),
            'total_requests': total_requests
        }
    
    def clear_cache(self):
        """Clear the detection cache."""
        self._detection_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


class OptimizedEventScopeDetector(EventScopeDetector):
    """
    Memory-optimized event scope detector for high-performance environments.
    
    This version uses more aggressive optimization techniques for scenarios
    with very high event volumes.
    """
    
    def __init__(self):
        """Initialize optimized detector."""
        super().__init__()
        
        # Pre-computed scope mappings for common patterns
        self.fast_lookup = {
            ('internal_scope', True): EventScope.INTERNAL_BUS.value,
            ('lifecycle_operation', True): EventScope.LIFECYCLE_MANAGEMENT.value,
            ('publish_tier', 'fast'): EventScope.EXTERNAL_FAST.value,
            ('publish_tier', 'standard'): EventScope.EXTERNAL_STANDARD.value,
            ('publish_tier', 'reliable'): EventScope.EXTERNAL_RELIABLE.value,
        }
        
        # Event type to scope mapping for performance
        self.event_type_scopes = {
            'BAR': EventScope.EXTERNAL_FAST.value,
            'TICK': EventScope.EXTERNAL_FAST.value,
            'QUOTE': EventScope.EXTERNAL_FAST.value,
            'BOOK_UPDATE': EventScope.EXTERNAL_FAST.value,
            'ORDER': EventScope.EXTERNAL_RELIABLE.value,
            'FILL': EventScope.EXTERNAL_RELIABLE.value,
            'SYSTEM': EventScope.EXTERNAL_RELIABLE.value,
            'ERROR': EventScope.EXTERNAL_RELIABLE.value,
            'SIGNAL': EventScope.EXTERNAL_STANDARD.value,
            'INDICATOR': EventScope.EXTERNAL_STANDARD.value,
            'PORTFOLIO_UPDATE': EventScope.EXTERNAL_STANDARD.value,
        }
    
    def detect_scope(self, context: Dict[str, Any]) -> str:
        """Optimized scope detection with minimal allocations."""
        # Ultra-fast lookup for common patterns
        for (key, value), scope in self.fast_lookup.items():
            if key in context and (value is True or context[key] == value):
                return scope
        
        # Fast event type lookup
        event_type = context.get('event_type')
        if event_type in self.event_type_scopes:
            return self.event_type_scopes[event_type]
        
        # Fallback to parent implementation
        return super().detect_scope(context)


class ConfigurableEventScopeDetector(EventScopeDetector):
    """
    Configurable event scope detector with runtime rule modification.
    
    This version allows dynamic configuration of scope detection rules
    for flexible deployment scenarios.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize configurable detector.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration for scope detection."""
        # Update tier mapping from config
        if 'tier_mapping' in self.config:
            self.tier_mapping.update(self.config['tier_mapping'])
        
        # Update scope patterns from config
        if 'scope_patterns' in self.config:
            self.scope_patterns.update(self.config['scope_patterns'])
        
        # Load custom event type classifications
        if 'event_type_classifications' in self.config:
            self.event_type_classifications = self.config['event_type_classifications']
        else:
            self.event_type_classifications = {}
    
    def update_configuration(self, config: Dict[str, Any]):
        """
        Update detector configuration at runtime.
        
        Args:
            config: New configuration dictionary
        """
        self.config.update(config)
        self._load_configuration()
        
        # Clear cache since rules have changed
        self.clear_cache()
    
    def _classify_cross_container_scope(self, event_type: str) -> str:
        """Enhanced classification with custom rules."""
        # Check custom classifications first
        if event_type in self.event_type_classifications:
            classification = self.event_type_classifications[event_type]
            return f"external_{classification}_tier"
        
        # Fallback to parent implementation
        return super()._classify_cross_container_scope(event_type)
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current detector configuration."""
        return {
            'tier_mapping': self.tier_mapping.copy(),
            'scope_patterns': self.scope_patterns.copy(),
            'event_type_classifications': getattr(self, 'event_type_classifications', {}),
            'cache_size': len(self._detection_cache),
            'statistics': self.get_statistics()
        }
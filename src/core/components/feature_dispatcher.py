"""
Feature Dispatcher for granular feature routing.

This component solves the problem of broadcasting ALL features to every strategy.
Instead, it receives the full FEATURES event and routes only the needed features
to each strategy based on their requirements.
"""

import logging
from typing import Dict, List, Set, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict

from ..events import Event, EventType
from ..types.trading import Bar

logger = logging.getLogger(__name__)


@dataclass
class StrategyFeatureRequirements:
    """Defines which features a strategy needs."""
    strategy_id: str
    strategy_type: str
    required_features: Set[str] = field(default_factory=set)
    feature_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def add_requirement(self, feature_name: str, params: Optional[Dict[str, Any]] = None):
        """Add a feature requirement."""
        self.required_features.add(feature_name)
        if params:
            self.feature_params[feature_name] = params


class FeatureDispatcher:
    """
    Routes features to strategies based on their requirements.
    
    This dispatcher:
    1. Receives FEATURES events containing all computed indicators
    2. Maintains a registry of what each strategy needs
    3. Filters and routes only needed features to each strategy
    4. Publishes targeted STRATEGY_FEATURES events
    """
    
    def __init__(self, root_event_bus=None):
        """
        Initialize the Feature Dispatcher.
        
        Args:
            root_event_bus: Event bus for publishing filtered features
        """
        self.root_event_bus = root_event_bus
        self.strategy_requirements: Dict[str, StrategyFeatureRequirements] = {}
        self.feature_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._initialized = False
        
        logger.info("Feature Dispatcher initialized")
    
    def register_strategy(
        self, 
        strategy_id: str, 
        strategy_type: str,
        required_features: List[str],
        handler: Optional[Callable] = None
    ):
        """
        Register a strategy's feature requirements.
        
        Args:
            strategy_id: Unique identifier for the strategy instance
            strategy_type: Type of strategy (momentum, mean_reversion, etc.)
            required_features: List of feature names this strategy needs
            handler: Optional direct handler for features (for stateless services)
        """
        requirements = StrategyFeatureRequirements(
            strategy_id=strategy_id,
            strategy_type=strategy_type
        )
        
        # Parse feature requirements
        for feature_spec in required_features:
            if isinstance(feature_spec, str):
                # Simple feature name
                requirements.add_requirement(feature_spec)
            elif isinstance(feature_spec, dict):
                # Feature with parameters
                feature_name = feature_spec.get('name')
                params = feature_spec.get('params', {})
                requirements.add_requirement(feature_name, params)
        
        self.strategy_requirements[strategy_id] = requirements
        
        # Register handler if provided
        if handler:
            self.feature_handlers[strategy_id].append(handler)
        
        logger.debug(
            f"Registered strategy {strategy_id} ({strategy_type}) "
            f"requiring features: {requirements.required_features}"
        )
    
    def handle_features(self, event: Event):
        """
        Handle incoming FEATURES event and dispatch to strategies.
        
        Args:
            event: FEATURES event containing all computed indicators
        """
        if event.event_type != EventType.FEATURES:
            logger.warning(f"Feature Dispatcher received non-FEATURES event: {event.event_type}")
            return
        
        # Extract feature data
        all_features = event.payload.get('features', {})
        symbol = event.payload.get('symbol')
        timeframe = event.payload.get('timeframe')
        bar = event.payload.get('bar')  # Include bar data for strategies
        timestamp = event.timestamp
        
        # Track which strategies received features
        dispatched_count = 0
        
        # Route to each registered strategy
        for strategy_id, requirements in self.strategy_requirements.items():
            # Filter features for this strategy
            filtered_features = self._filter_features(all_features, requirements)
            
            if not filtered_features:
                # Skip if no relevant features
                continue
            
            # Create targeted event
            strategy_event = Event(
                event_type=EventType.FEATURES,  # Keep same type for compatibility
                timestamp=timestamp,
                payload={
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'features': filtered_features,
                    'bar': bar,  # Include bar data
                    'strategy_id': strategy_id,  # Tag with target strategy
                    'filtered': True  # Mark as filtered
                },
                metadata={
                    'source': 'feature_dispatcher',
                    'original_event_id': event.metadata.get('event_id'),
                    'feature_count': len(filtered_features),
                    'target_strategy': strategy_id
                }
            )
            
            # Route to strategy
            if strategy_id in self.feature_handlers:
                # Direct handler routing (for stateless services)
                for handler in self.feature_handlers[strategy_id]:
                    try:
                        handler(strategy_event)
                        dispatched_count += 1
                    except Exception as e:
                        logger.error(f"Error dispatching to handler for {strategy_id}: {e}")
            
            elif self.root_event_bus:
                # Event bus routing (with strategy targeting)
                strategy_event.metadata['target_strategy'] = strategy_id
                self.root_event_bus.publish(strategy_event)
                dispatched_count += 1
        
        logger.debug(
            f"Dispatched features to {dispatched_count} strategies "
            f"(original had {len(all_features)} features)"
        )
    
    def _filter_features(
        self, 
        all_features: Dict[str, Any], 
        requirements: StrategyFeatureRequirements
    ) -> Dict[str, Any]:
        """
        Filter features based on strategy requirements.
        
        Args:
            all_features: All computed features
            requirements: Strategy's feature requirements
            
        Returns:
            Filtered features containing only what the strategy needs
        """
        filtered = {}
        
        for feature_name in requirements.required_features:
            if feature_name in all_features:
                filtered[feature_name] = all_features[feature_name]
            else:
                # Check for parameterized features (e.g., 'rsi_14' when requirement is 'rsi')
                for computed_name, value in all_features.items():
                    if computed_name.startswith(f"{feature_name}_"):
                        filtered[computed_name] = value
        
        return filtered
    
    def get_strategy_requirements(self, strategy_id: str) -> Optional[StrategyFeatureRequirements]:
        """Get requirements for a specific strategy."""
        return self.strategy_requirements.get(strategy_id)
    
    def get_all_required_features(self) -> Set[str]:
        """Get union of all features required by any strategy."""
        all_features = set()
        for requirements in self.strategy_requirements.values():
            all_features.update(requirements.required_features)
        return all_features
    
    def get_feature_consumers(self, feature_name: str) -> List[str]:
        """Get list of strategies that consume a specific feature."""
        consumers = []
        for strategy_id, requirements in self.strategy_requirements.items():
            if feature_name in requirements.required_features:
                consumers.append(strategy_id)
        return consumers
    
    def clear_registrations(self):
        """Clear all strategy registrations."""
        self.strategy_requirements.clear()
        self.feature_handlers.clear()
        logger.info("Cleared all feature dispatcher registrations")
    
    def get_dispatch_stats(self) -> Dict[str, Any]:
        """Get statistics about feature dispatching."""
        total_strategies = len(self.strategy_requirements)
        total_features = len(self.get_all_required_features())
        
        # Calculate feature usage
        feature_usage = defaultdict(int)
        for requirements in self.strategy_requirements.values():
            for feature in requirements.required_features:
                feature_usage[feature] += 1
        
        return {
            'registered_strategies': total_strategies,
            'unique_features': total_features,
            'feature_usage': dict(feature_usage),
            'avg_features_per_strategy': sum(
                len(r.required_features) for r in self.strategy_requirements.values()
            ) / max(total_strategies, 1)
        }
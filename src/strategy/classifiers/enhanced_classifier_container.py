"""
Enhanced Classifier Container following BACKTEST.MD architecture.

Implements the nested hierarchy:
Classifier Container -> Risk & Portfolio Containers -> Strategies
"""
from typing import Dict, List, Optional, Any, Tuple, Type
from datetime import datetime
import uuid

from ...core.containers import UniversalScopedContainer
from ...core.components import ComponentSpec
from ...core.events import Event, EventType
from ...risk.protocols import RiskManager, PortfolioManager
from ..protocols import Strategy, Signal
from .classifier import BaseClassifier
from .regime_types import MarketRegime, RegimeState


class EnhancedClassifierContainer(UniversalScopedContainer):
    """
    Classifier container that manages Risk & Portfolio sub-containers.
    
    Following BACKTEST.MD architecture:
    - Classifier determines regime/context
    - Multiple Risk & Portfolio containers for different risk profiles
    - Each Risk & Portfolio container has its own strategies
    - Regime context flows down through the hierarchy
    """
    
    def __init__(
        self,
        container_id: Optional[str] = None,
        classifier_type: str = "hmm",
        classifier_class: Optional[Type[BaseClassifier]] = None,
        shared_services: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize classifier container.
        
        Args:
            container_id: Unique container ID
            classifier_type: Type of classifier (hmm, pattern, etc.)
            classifier_class: Classifier class to instantiate
            shared_services: Shared read-only services
        """
        # Generate structured container ID
        if container_id is None:
            container_id = f"classifier_{classifier_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
        super().__init__(
            container_id=container_id,
            container_type=f"classifier_{classifier_type}",
            shared_services=shared_services
        )
        
        self.classifier_type = classifier_type
        self.classifier: Optional[BaseClassifier] = None
        
        # Risk & Portfolio sub-containers by risk profile
        self.risk_portfolio_containers: Dict[str, UniversalScopedContainer] = {}
        
        # Current regime context
        self.current_regime: Optional[RegimeContext] = None
        
        # Create classifier
        if classifier_class:
            self._create_classifier(classifier_class)
            
    def _create_classifier(self, classifier_class: Type[BaseClassifier]) -> None:
        """Create the regime classifier."""
        spec = ComponentSpec(
            name=f"{self.classifier_type}_classifier",
            class_name=classifier_class.__name__,
            parameters={},
            capabilities=['event_publisher', 'event_subscriber']
        )
        
        self.classifier = self.create_component(spec)
        
        # Subscribe to market data for classification
        self.event_bus.subscribe(EventType.BAR, self._handle_market_data)
        self.event_bus.subscribe(EventType.INDICATOR, self._handle_indicator_data)
        
    def add_risk_portfolio_container(
        self,
        risk_profile: str,
        risk_parameters: Dict[str, Any],
        portfolio_parameters: Dict[str, Any],
        strategies: List[Dict[str, Any]]
    ) -> UniversalScopedContainer:
        """
        Add a Risk & Portfolio container with strategies.
        
        Args:
            risk_profile: Name of risk profile (conservative, balanced, aggressive)
            risk_parameters: Parameters for risk manager
            portfolio_parameters: Parameters for portfolio manager
            strategies: List of strategy configurations
            
        Returns:
            Created Risk & Portfolio container
        """
        # Create sub-container for Risk & Portfolio
        container_id = f"{self.container_id}_{risk_profile}"
        risk_portfolio_container = self.create_subcontainer(
            container_id=container_id,
            container_type=f"risk_portfolio_{risk_profile}"
        )
        
        # Create risk manager
        risk_spec = ComponentSpec(
            name=f"risk_manager_{risk_profile}",
            class_name=risk_parameters.get('class', 'DefaultRiskManager'),
            parameters=risk_parameters.get('parameters', {
                'max_position_size': 0.02,
                'max_total_exposure': 0.1,
                'risk_profile': risk_profile
            }),
            capabilities=['event_publisher', 'event_subscriber']
        )
        risk_portfolio_container.create_component(risk_spec)
        
        # Create portfolio manager
        portfolio_spec = ComponentSpec(
            name=f"portfolio_{risk_profile}",
            class_name=portfolio_parameters.get('class', 'Portfolio'),
            parameters=portfolio_parameters.get('parameters', {
                'initial_capital': 100000,
                'risk_profile': risk_profile
            }),
            capabilities=['event_publisher']
        )
        risk_portfolio_container.create_component(portfolio_spec)
        
        # Add strategies to this Risk & Portfolio container
        for strategy_config in strategies:
            strategy_spec = ComponentSpec(
                name=strategy_config.get('name', f"strategy_{len(strategies)}"),
                class_name=strategy_config.get('class'),
                parameters=strategy_config.get('parameters', {}),
                capabilities=['event_publisher']
            )
            strategy = risk_portfolio_container.create_component(strategy_spec)
            
            # Subscribe strategy to regime changes
            risk_portfolio_container.event_bus.subscribe(
                EventType.REGIME_CHANGE,
                strategy.on_regime_change if hasattr(strategy, 'on_regime_change') else lambda e: None
            )
            
        # Store the container
        self.risk_portfolio_containers[risk_profile] = risk_portfolio_container
        
        # Set up event forwarding
        self._setup_container_event_forwarding(risk_portfolio_container)
        
        return risk_portfolio_container
        
    def _setup_container_event_forwarding(self, container: UniversalScopedContainer) -> None:
        """Set up event forwarding between containers."""
        # Forward market data to Risk & Portfolio container
        def forward_market_data(event: Event):
            if self.current_regime:
                # Add regime context to market data
                enhanced_event = Event(
                    event_type=event.event_type,
                    payload={
                        **event.payload,
                        'regime_context': self.current_regime.to_dict()
                    },
                    source_id=event.source_id
                )
                container.event_bus.publish(enhanced_event)
            else:
                container.event_bus.publish(event)
                
        self.event_bus.subscribe(EventType.BAR, forward_market_data)
        
        # Forward signals from Risk & Portfolio container to parent
        def forward_signals(event: Event):
            if event.event_type == EventType.SIGNAL:
                # Add classifier context
                signal_data = event.payload
                signal_data['classifier'] = self.classifier_type
                signal_data['regime'] = self.current_regime.regime.value if self.current_regime else 'unknown'
                
                self.event_bus.publish(Event(
                    event_type=EventType.SIGNAL,
                    payload=signal_data,
                    source_id=container.container_id
                ))
                
        container.event_bus.subscribe(EventType.SIGNAL, forward_signals)
        
    def _handle_market_data(self, event: Event) -> None:
        """Handle market data for regime classification."""
        if not self.classifier:
            return
            
        market_data = event.payload.get('market_data', {})
        timestamp = event.payload.get('timestamp')
        
        # Classify regime
        regime_context = self.classifier.classify_regime(market_data, timestamp)
        
        # Check if regime changed
        if self._has_regime_changed(regime_context):
            self._update_regime(regime_context)
            
    def _handle_indicator_data(self, event: Event) -> None:
        """Handle indicator data for classifier."""
        if self.classifier and hasattr(self.classifier, 'process_indicator_event'):
            self.classifier.process_indicator_event(event)
            
    def _has_regime_changed(self, new_context: RegimeContext) -> bool:
        """Check if regime has changed."""
        if not self.current_regime:
            return True
            
        return (new_context.regime != self.current_regime.regime or
                abs(new_context.confidence - self.current_regime.confidence) > 0.2)
                
    def _update_regime(self, new_context: RegimeContext) -> None:
        """Update regime and notify sub-containers."""
        old_regime = self.current_regime
        self.current_regime = new_context
        
        # Emit regime change event
        regime_event = Event(
            event_type=EventType.REGIME_CHANGE,
            payload={
                'classifier': self.classifier_type,
                'old_regime': old_regime.to_dict() if old_regime else None,
                'new_regime': new_context.to_dict(),
                'timestamp': datetime.now()
            },
            source_id=self.container_id
        )
        
        # Publish to own event bus (for logging/monitoring)
        self.event_bus.publish(regime_event)
        
        # Forward to all Risk & Portfolio containers
        for container in self.risk_portfolio_containers.values():
            container.event_bus.publish(regime_event)
            
        self.logger.info(
            f"Regime updated",
            classifier=self.classifier_type,
            new_regime=new_context.regime.value,
            confidence=new_context.confidence
        )
        
    def get_active_strategies_by_profile(self) -> Dict[str, List[str]]:
        """Get active strategies organized by risk profile."""
        result = {}
        
        for profile, container in self.risk_portfolio_containers.items():
            strategies = []
            for name, component in container._components.items():
                if 'strategy' in name.lower():
                    strategies.append(name)
            result[profile] = strategies
            
        return result
        
    def get_regime_history(self) -> List[Tuple[datetime, RegimeState, float]]:
        """Get regime classification history."""
        if self.classifier and hasattr(self.classifier, 'get_state_history'):
            return self.classifier.get_state_history()
        return []
        
    async def initialize_hierarchy(self) -> None:
        """Initialize the entire container hierarchy."""
        # Initialize self
        await self.initialize()
        
        # Initialize all Risk & Portfolio containers
        for container in self.risk_portfolio_containers.values():
            await container.initialize()
            
    async def start_hierarchy(self) -> None:
        """Start the entire container hierarchy."""
        # Start self
        await self.start()
        
        # Start all Risk & Portfolio containers
        for container in self.risk_portfolio_containers.values():
            await container.start()
            
    async def stop_hierarchy(self) -> None:
        """Stop the entire container hierarchy."""
        # Stop all Risk & Portfolio containers
        for container in self.risk_portfolio_containers.values():
            await container.stop()
            
        # Stop self
        await self.stop()


def create_classifier_hierarchy(
    classifier_type: str,
    classifier_class: Type[RegimeClassifier],
    risk_profiles: Dict[str, Dict[str, Any]],
    shared_services: Optional[Dict[str, Any]] = None
) -> EnhancedClassifierContainer:
    """
    Create a complete classifier hierarchy.
    
    Args:
        classifier_type: Type of classifier (hmm, pattern)
        classifier_class: Classifier class to use
        risk_profiles: Configuration for each risk profile
        shared_services: Shared services
        
    Returns:
        Configured classifier container with full hierarchy
    """
    # Create classifier container
    classifier_container = EnhancedClassifierContainer(
        classifier_type=classifier_type,
        classifier_class=classifier_class,
        shared_services=shared_services
    )
    
    # Add Risk & Portfolio containers for each profile
    for profile_name, profile_config in risk_profiles.items():
        classifier_container.add_risk_portfolio_container(
            risk_profile=profile_name,
            risk_parameters=profile_config.get('risk_parameters', {}),
            portfolio_parameters=profile_config.get('portfolio_parameters', {}),
            strategies=profile_config.get('strategies', [])
        )
        
    return classifier_container
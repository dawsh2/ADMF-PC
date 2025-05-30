"""
Multi-symbol architecture with risk containers.

This module implements the vision where:
- Risk modules are portfolio-aware containers that contain strategies
- Risk containers sit inside Classifier containers
- Strategies can run on one or more symbols
- Multiple risk containers can coexist with different strategies/symbols
"""

from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass, field
import logging

from ..containers import UniversalScopedContainer
from ..events import Event, EventType

logger = logging.getLogger(__name__)


@dataclass
class SymbolAllocation:
    """Defines which symbols a strategy trades."""
    strategy_id: str
    symbols: Set[str]
    weights: Dict[str, float] = field(default_factory=dict)  # Optional position sizing


class RiskContainer(UniversalScopedContainer):
    """
    Portfolio-aware risk container that manages multiple strategies.
    
    The RiskContainer:
    - Receives classifier signals and indicator data
    - Manages multiple strategy instances
    - Handles portfolio-level risk management
    - Coordinates multi-symbol execution
    """
    
    def __init__(self, 
                 container_id: str,
                 risk_config: Dict[str, Any],
                 parent_classifier: Optional[UniversalScopedContainer] = None):
        """
        Initialize risk container.
        
        Args:
            container_id: Unique identifier for this risk container
            risk_config: Risk management configuration
            parent_classifier: Parent classifier container
        """
        super().__init__(
            container_id=container_id,
            container_type="risk",
            parent_container=parent_classifier
        )
        
        self.risk_config = risk_config
        self.strategy_containers: Dict[str, UniversalScopedContainer] = {}
        self.symbol_allocations: Dict[str, SymbolAllocation] = {}
        
        # Portfolio state
        self.portfolio_state = {
            'positions': {},      # symbol -> position
            'cash': risk_config.get('initial_capital', 100000),
            'equity_curve': [],
            'risk_metrics': {}
        }
        
        # Risk limits
        self.risk_limits = risk_config.get('risk_limits', {
            'max_position_size': 0.1,      # 10% per position
            'max_sector_exposure': 0.3,     # 30% per sector
            'max_correlation': 0.7,         # Max correlation between positions
            'max_drawdown': 0.2            # 20% max drawdown
        })
        
        # Subscribe to parent classifier events if available
        if parent_classifier:
            self._subscribe_to_classifier_events()
    
    def add_strategy(self, 
                     strategy_config: Dict[str, Any],
                     symbols: List[str],
                     weights: Optional[Dict[str, float]] = None) -> str:
        """
        Add a strategy to the risk container.
        
        Args:
            strategy_config: Strategy configuration
            symbols: List of symbols the strategy trades
            weights: Optional position sizing weights per symbol
            
        Returns:
            Strategy container ID
        """
        # Generate strategy container ID
        strategy_id = f"{self.container_id}_strategy_{len(self.strategy_containers)}"
        
        # Create strategy container
        strategy_container = UniversalScopedContainer(
            container_id=strategy_id,
            container_type="strategy",
            parent_container=self
        )
        
        # Create strategy component within container
        strategy = strategy_container.create_component(strategy_config)
        
        # Track allocation
        allocation = SymbolAllocation(
            strategy_id=strategy_id,
            symbols=set(symbols),
            weights=weights or {}
        )
        self.symbol_allocations[strategy_id] = allocation
        
        # Store container
        self.strategy_containers[strategy_id] = strategy_container
        
        # Set up event routing for multi-symbol execution
        self._setup_symbol_routing(strategy_id, symbols)
        
        logger.info(f"Added strategy {strategy_id} trading symbols: {symbols}")
        return strategy_id
    
    def process_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Process market data for a symbol.
        
        Routes data to all strategies that trade this symbol.
        """
        # Find strategies that trade this symbol
        for strategy_id, allocation in self.symbol_allocations.items():
            if symbol in allocation.symbols:
                # Get strategy container
                strategy_container = self.strategy_containers[strategy_id]
                
                # Create symbol-specific event
                event = Event(
                    event_type=EventType.BAR,
                    payload={
                        'symbol': symbol,
                        'data': data,
                        'weight': allocation.weights.get(symbol, 1.0)
                    },
                    source_id=self.container_id,
                    container_id=strategy_id
                )
                
                # Route to strategy
                strategy_container.event_bus.publish(event)
    
    def process_signal(self, strategy_id: str, signal: Dict[str, Any]) -> None:
        """
        Process trading signal from a strategy.
        
        Applies risk management before execution.
        """
        symbol = signal['symbol']
        
        # Apply position sizing
        sized_signal = self._apply_position_sizing(signal)
        
        # Check risk limits
        if not self._check_risk_limits(sized_signal):
            logger.warning(f"Signal rejected due to risk limits: {signal}")
            return
        
        # Apply portfolio-level adjustments
        adjusted_signal = self._apply_portfolio_constraints(sized_signal)
        
        # Execute signal
        self._execute_signal(adjusted_signal)
        
        # Update risk metrics
        self._update_risk_metrics()
    
    def _apply_position_sizing(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Apply position sizing based on risk configuration."""
        sized_signal = signal.copy()
        
        # Kelly criterion, fixed fractional, or other sizing methods
        sizing_method = self.risk_config.get('sizing_method', 'fixed')
        
        if sizing_method == 'fixed':
            size = self.risk_config.get('position_size', 0.02)  # 2% default
        elif sizing_method == 'kelly':
            size = self._calculate_kelly_size(signal)
        elif sizing_method == 'volatility':
            size = self._calculate_volatility_size(signal)
        else:
            size = 0.02
        
        sized_signal['position_size'] = size
        return sized_signal
    
    def _check_risk_limits(self, signal: Dict[str, Any]) -> bool:
        """Check if signal passes risk limits."""
        symbol = signal['symbol']
        size = signal['position_size']
        
        # Check position size limit
        if size > self.risk_limits['max_position_size']:
            return False
        
        # Check correlation with existing positions
        if not self._check_correlation_limit(symbol):
            return False
        
        # Check drawdown limit
        if self._current_drawdown() > self.risk_limits['max_drawdown']:
            return False
        
        return True
    
    def _setup_symbol_routing(self, strategy_id: str, symbols: List[str]) -> None:
        """Set up event routing for multi-symbol execution."""
        # This would integrate with the parent classifier's IndicatorHub
        # to ensure the strategy receives data for all its symbols
        pass
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state."""
        return {
            'positions': self.portfolio_state['positions'].copy(),
            'cash': self.portfolio_state['cash'],
            'total_value': self._calculate_portfolio_value(),
            'risk_metrics': self.portfolio_state['risk_metrics'].copy()
        }
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        positions_value = sum(
            pos['quantity'] * pos['current_price'] 
            for pos in self.portfolio_state['positions'].values()
        )
        return self.portfolio_state['cash'] + positions_value


class MultiSymbolCoordinator:
    """
    Coordinates multi-symbol execution with nested container hierarchy.
    
    Manages the creation and coordination of:
    - Classifier containers (1-n)
    - Risk containers within classifiers (1-n per classifier)
    - Strategy containers within risk containers (1-n per risk container)
    """
    
    def __init__(self, coordinator: 'Coordinator'):
        self.coordinator = coordinator
        self.classifier_containers: Dict[str, UniversalScopedContainer] = {}
        self.risk_containers: Dict[str, List[RiskContainer]] = {}  # classifier_id -> risk containers
        
    def create_classifier_environment(self,
                                    classifier_type: str,
                                    classifier_config: Dict[str, Any],
                                    risk_configs: List[Dict[str, Any]]) -> str:
        """
        Create a complete classifier environment with risk containers.
        
        Args:
            classifier_type: Type of classifier (HMM, Volatility, etc.)
            classifier_config: Classifier configuration
            risk_configs: List of risk container configurations
            
        Returns:
            Classifier container ID
        """
        # Create classifier container
        classifier_id = f"classifier_{classifier_type}_{len(self.classifier_containers)}"
        classifier_container = UniversalScopedContainer(
            container_id=classifier_id,
            container_type="classifier"
        )
        
        # Create IndicatorHub within classifier
        indicator_hub = classifier_container.create_component({
            'name': 'indicator_hub',
            'class': 'IndicatorHub',
            'params': classifier_config.get('indicators', {})
        })
        
        # Create classifier
        classifier = classifier_container.create_component({
            'name': 'classifier',
            'class': classifier_type,
            'params': classifier_config
        })
        
        # Create risk containers
        risk_containers = []
        for risk_config in risk_configs:
            risk_id = f"{classifier_id}_risk_{len(risk_containers)}"
            risk_container = RiskContainer(
                container_id=risk_id,
                risk_config=risk_config,
                parent_classifier=classifier_container
            )
            risk_containers.append(risk_container)
        
        # Store references
        self.classifier_containers[classifier_id] = classifier_container
        self.risk_containers[classifier_id] = risk_containers
        
        return classifier_id
    
    def add_strategy_to_risk_container(self,
                                     classifier_id: str,
                                     risk_container_index: int,
                                     strategy_config: Dict[str, Any],
                                     symbols: List[str]) -> str:
        """
        Add a strategy to a specific risk container.
        
        Args:
            classifier_id: Classifier container ID
            risk_container_index: Index of risk container within classifier
            strategy_config: Strategy configuration
            symbols: Symbols the strategy trades
            
        Returns:
            Strategy container ID
        """
        risk_containers = self.risk_containers.get(classifier_id, [])
        if risk_container_index >= len(risk_containers):
            raise ValueError(f"Risk container {risk_container_index} not found")
        
        risk_container = risk_containers[risk_container_index]
        return risk_container.add_strategy(strategy_config, symbols)
    
    def process_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """
        Process market data across all classifier environments.
        
        The data flows:
        1. To each classifier's IndicatorHub
        2. Through the classifier
        3. To risk containers
        4. To relevant strategies
        """
        for classifier_id, classifier_container in self.classifier_containers.items():
            # Process through IndicatorHub
            indicator_hub = classifier_container.resolve('indicator_hub')
            indicator_data = indicator_hub.process_bar(data)
            
            # Process through classifier
            classifier = classifier_container.resolve('classifier')
            classification = classifier.classify(indicator_data)
            
            # Route to risk containers with classification context
            risk_containers = self.risk_containers.get(classifier_id, [])
            for risk_container in risk_containers:
                # Add classification to data
                enriched_data = {
                    **data,
                    'classification': classification,
                    'indicators': indicator_data
                }
                risk_container.process_market_data(symbol, enriched_data)
    
    def recontainerize_strategy(self,
                              strategy_id: str,
                              from_risk_container: str,
                              to_risk_container: str) -> None:
        """
        Move a strategy between risk containers.
        
        This demonstrates the ability to recontainerize as design evolves.
        """
        # This would handle the complex task of:
        # 1. Stopping the strategy in current container
        # 2. Extracting its state
        # 3. Creating it in new container
        # 4. Restoring its state
        # 5. Updating all references
        pass


def integrate_multi_symbol_support(coordinator: 'Coordinator') -> None:
    """
    Integrate multi-symbol architecture into the Coordinator.
    
    This adds the ability to manage nested container hierarchies for
    multi-symbol, multi-strategy, multi-risk-container execution.
    """
    # Add multi-symbol coordinator
    coordinator.multi_symbol = MultiSymbolCoordinator(coordinator)
    
    # Add helper methods
    def create_multi_symbol_workflow(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a complete multi-symbol workflow."""
        results = {}
        
        # Create classifier environments
        for classifier_config in config['classifiers']:
            classifier_id = self.multi_symbol.create_classifier_environment(
                classifier_config['type'],
                classifier_config['config'],
                classifier_config['risk_containers']
            )
            
            # Add strategies to risk containers
            for risk_idx, risk_config in enumerate(classifier_config['risk_containers']):
                for strategy_config in risk_config['strategies']:
                    strategy_id = self.multi_symbol.add_strategy_to_risk_container(
                        classifier_id,
                        risk_idx,
                        strategy_config['config'],
                        strategy_config['symbols']
                    )
                    
                    results[strategy_id] = {
                        'classifier': classifier_id,
                        'risk_container': risk_idx,
                        'symbols': strategy_config['symbols']
                    }
        
        return results
    
    coordinator.create_multi_symbol_workflow = create_multi_symbol_workflow.__get__(coordinator)
    
    logger.info("Multi-symbol architecture integrated into Coordinator")
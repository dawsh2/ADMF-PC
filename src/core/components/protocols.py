"""
Core component protocols for the ADMF-PC system.

This module defines the fundamental protocols that components can implement
to participate in the framework. Components only implement the protocols
they need, following the "pay for what you use" principle.
"""

from __future__ import annotations
from typing import Protocol, Dict, Any, Optional, List, Union, runtime_checkable
from abc import abstractmethod
from datetime import datetime

from ..events import EventBusProtocol, EventType, EventHandler


@runtime_checkable
class Component(Protocol):
    """Base protocol that all components must implement."""
    
    @property
    def component_id(self) -> str:
        """Unique identifier for the component."""
        ...


@runtime_checkable
class Lifecycle(Protocol):
    """Protocol for components with lifecycle management."""
    
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize the component with context."""
        ...
    
    def start(self) -> None:
        """Start the component."""
        ...
    
    def stop(self) -> None:
        """Stop the component."""
        ...
    
    def reset(self) -> None:
        """Reset component state."""
        ...
    
    def teardown(self) -> None:
        """Clean up resources."""
        ...


@runtime_checkable
class EventCapable(Protocol):
    """Protocol for components that use the event system."""
    
    @property
    def event_bus(self) -> EventBusProtocol:
        """Access to the container's event bus."""
        ...
    
    def initialize_events(self) -> None:
        """Initialize event subscriptions."""
        ...
    
    def teardown_events(self) -> None:
        """Clean up event subscriptions."""
        ...


@runtime_checkable
class Configurable(Protocol):
    """Protocol for components that can be configured."""
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return JSON schema for configuration validation."""
        ...
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Apply configuration to the component."""
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Return current configuration."""
        ...


@runtime_checkable
class Optimizable(Protocol):
    """Protocol for components that can be optimized."""
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Return the parameter space for optimization."""
        ...
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set optimizable parameters."""
        ...
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        ...


@runtime_checkable
class SignalGenerator(Protocol):
    """Protocol for components that generate trading signals."""
    
    def generate_signal(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a trading signal from market data."""
        ...


@runtime_checkable
class DataProvider(Protocol):
    """Protocol for components that provide market data."""
    
    def get_data(self, symbol: str, start: datetime, end: datetime) -> Any:
        """Retrieve market data for a symbol and time range."""
        ...
    
    def subscribe_symbol(self, symbol: str, handler: EventHandler) -> None:
        """Subscribe to real-time updates for a symbol."""
        ...
    
    def unsubscribe_symbol(self, symbol: str, handler: EventHandler) -> None:
        """Unsubscribe from symbol updates."""
        ...


@runtime_checkable
class RiskManager(Protocol):
    """Protocol for risk management components."""
    
    def check_risk(self, position: Dict[str, Any], order: Dict[str, Any]) -> bool:
        """Check if an order passes risk constraints."""
        ...
    
    def calculate_position_size(self, signal: Dict[str, Any], portfolio: Dict[str, Any]) -> float:
        """Calculate appropriate position size."""
        ...


@runtime_checkable
class OrderExecutor(Protocol):
    """Protocol for order execution components."""
    
    def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Submit an order for execution."""
        ...
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        ...
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Get the status of an order."""
        ...


@runtime_checkable
class Portfolio(Protocol):
    """Protocol for portfolio tracking components."""
    
    def get_positions(self) -> Dict[str, PositionPayload]:
        """Get current positions."""
        ...
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        ...
    
    def update_position(self, fill: Dict[str, Any]) -> None:
        """Update position based on a fill."""
        ...


@runtime_checkable
class Indicator(Protocol):
    """Protocol for technical indicators."""
    
    def calculate(self, data: Any) -> Any:
        """Calculate indicator values."""
        ...
    
    @property
    def lookback_period(self) -> int:
        """Number of bars needed for calculation."""
        ...


@runtime_checkable
class Monitorable(Protocol):
    """Protocol for components that provide monitoring data."""
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get component metrics."""
        ...
    
    def get_health_status(self) -> Dict[str, Any]:  # TODO: Use HealthStatus protocol
        """Get component health status."""
        ...


@runtime_checkable
class Stateful(Protocol):
    """Protocol for components with saveable state."""
    
    def save_state(self) -> Dict[str, Any]:
        """Save component state."""
        ...
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load component state."""
        ...


# ============================================================================
# STATELESS COMPONENT PROTOCOLS
# These protocols define pure function interfaces for the unified architecture
# ============================================================================

@runtime_checkable
class StatelessStrategy(Protocol):
    """
    Protocol for stateless strategy components.
    
    Strategies are pure functions that generate signals based on features and market data.
    They maintain no internal state - all required data is passed as parameters.
    """
    
    def generate_signal(self, features: Dict[str, Any], bar: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trading signal from features and current bar.
        
        Args:
            features: Calculated indicators and features from FeatureHub
            bar: Current market bar (OHLCV data)
            params: Strategy parameters (lookback periods, thresholds, etc.)
            
        Returns:
            Signal dictionary with at least:
            - direction: 'long', 'short', or 'flat'
            - strength: float between 0 and 1
            - metadata: optional additional information
        """
        ...
    
    @property
    def required_features(self) -> List[str]:
        """List of feature names this strategy requires."""
        ...


@runtime_checkable
class StatelessClassifier(Protocol):
    """
    Protocol for stateless market regime classifier components.
    
    Classifiers are pure functions that detect market regimes based on features.
    They maintain no internal state - all required data is passed as parameters.
    """
    
    def classify_regime(self, features: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify the current market regime.
        
        Args:
            features: Calculated indicators and features from FeatureHub
            params: Classifier parameters (thresholds, model params, etc.)
            
        Returns:
            Regime dictionary with at least:
            - regime: string identifier (e.g., 'bull', 'bear', 'sideways')
            - confidence: float between 0 and 1
            - metadata: optional additional information
        """
        ...
    
    @property
    def required_features(self) -> List[str]:
        """List of feature names this classifier requires."""
        ...


@runtime_checkable
class StatelessRiskValidator(Protocol):
    """
    Protocol for stateless risk validation components.
    
    Risk validators are pure functions that validate orders against risk limits.
    They maintain no internal state - portfolio state is passed as a parameter.
    """
    
    def validate_order(
        self, 
        order: Dict[str, Any], 
        portfolio_state: Dict[str, Any], 
        risk_limits: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate an order against risk limits.
        
        Args:
            order: Order to validate with symbol, quantity, side, etc.
            portfolio_state: Current portfolio state (positions, cash, etc.)
            risk_limits: Risk parameters (max position, max drawdown, etc.)
            market_data: Current market prices and conditions
            
        Returns:
            Validation result with:
            - approved: bool indicating if order passes risk checks
            - adjusted_quantity: optional adjusted order size
            - reason: string explanation if rejected
            - risk_metrics: calculated risk metrics
        """
        ...
    
    def calculate_position_size(
        self,
        signal: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        risk_params: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> float:
        """
        Calculate appropriate position size for a signal.
        
        Args:
            signal: Trading signal with direction and strength
            portfolio_state: Current portfolio state
            risk_params: Risk parameters for sizing
            market_data: Current market prices
            
        Returns:
            Position size (number of shares/contracts)
        """
        ...


# Capability definitions
class Capability:
    """Enumeration of available component capabilities."""
    
    LIFECYCLE = "lifecycle"
    EVENTS = "events"
    CONFIGURABLE = "configurable"
    OPTIMIZABLE = "optimizable"
    MONITORABLE = "monitorable"
    STATEFUL = "stateful"
    
    # Trading-specific capabilities
    SIGNAL_GENERATOR = "signal_generator"
    DATA_PROVIDER = "data_provider"
    RISK_MANAGER = "risk_manager"
    ORDER_EXECUTOR = "order_executor"
    PORTFOLIO = "portfolio"
    INDICATOR = "indicator"


# Protocol mapping for capability detection
CAPABILITY_PROTOCOLS = {
    Capability.LIFECYCLE: Lifecycle,
    Capability.EVENTS: EventCapable,
    Capability.CONFIGURABLE: Configurable,
    Capability.OPTIMIZABLE: Optimizable,
    Capability.MONITORABLE: Monitorable,
    Capability.STATEFUL: Stateful,
    Capability.SIGNAL_GENERATOR: SignalGenerator,
    Capability.DATA_PROVIDER: DataProvider,
    Capability.RISK_MANAGER: RiskManager,
    Capability.ORDER_EXECUTOR: OrderExecutor,
    Capability.PORTFOLIO: Portfolio,
    Capability.INDICATOR: Indicator
}


def detect_capabilities(component: Any) -> List[str]:
    """
    Detect which capabilities a component implements.
    
    Args:
        component: The component to analyze
        
    Returns:
        List of capability names the component implements
    """
    capabilities = []
    
    for capability, protocol in CAPABILITY_PROTOCOLS.items():
        if isinstance(component, protocol):
            capabilities.append(capability)
    
    return capabilities


def has_capability(component: Any, capability: str) -> bool:
    """
    Check if a component has a specific capability.
    
    Args:
        component: The component to check
        capability: The capability name
        
    Returns:
        True if the component has the capability
    """
    protocol = CAPABILITY_PROTOCOLS.get(capability)
    if protocol:
        return isinstance(component, protocol)
    return False


def require_capability(component: Any, capability: str) -> None:
    """
    Ensure a component has a required capability.
    
    Args:
        component: The component to check
        capability: The required capability
        
    Raises:
        TypeError: If the component doesn't have the capability
    """
    if not has_capability(component, capability):
        raise TypeError(
            f"Component {component} does not implement required "
            f"capability '{capability}'"
        )
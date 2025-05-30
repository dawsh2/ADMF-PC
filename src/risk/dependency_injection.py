"""
Dependency injection infrastructure for the risk module.

This module provides dependency injection patterns that align with the core
system's DI container and factory patterns.
"""

from typing import Protocol, Type, Any, Dict, Optional, List, Callable
from abc import abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from ..core.dependencies.container import DependencyContainer
from ..core.components.factory import ComponentFactory, create_component
from .protocols import (
    PositionSizerProtocol,
    RiskLimitProtocol,
    SignalProcessorProtocol,
    PortfolioStateProtocol,
)


class RiskDependencyProvider(Protocol):
    """Protocol for providing risk management dependencies."""
    
    @abstractmethod
    def get_position_sizer(self, name: str) -> PositionSizerProtocol:
        """Get position sizer by name."""
        ...
    
    @abstractmethod
    def get_risk_limits(self) -> List[RiskLimitProtocol]:
        """Get all configured risk limits."""
        ...
    
    @abstractmethod
    def get_signal_processor(self) -> SignalProcessorProtocol:
        """Get signal processor."""
        ...
    
    @abstractmethod
    def get_portfolio_state(self) -> PortfolioStateProtocol:
        """Get portfolio state tracker."""
        ...


@dataclass
class RiskComponentSpec:
    """Specification for creating risk components."""
    component_type: str
    name: str
    class_name: str
    params: Dict[str, Any]
    metadata: Dict[str, Any]


class RiskComponentFactory:
    """Factory for creating risk management components with proper DI."""
    
    def __init__(self, dependency_container: DependencyContainer):
        """Initialize with dependency container."""
        self._container = dependency_container
        self._component_factory = ComponentFactory()
        
        # Register component types
        self._register_position_sizers()
        self._register_risk_limits()
        self._register_processors()
    
    def create_position_sizer(
        self,
        spec: RiskComponentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> PositionSizerProtocol:
        """Create position sizer from specification."""
        return self._create_component(spec, context)
    
    def create_risk_limit(
        self,
        spec: RiskComponentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> RiskLimitProtocol:
        """Create risk limit from specification."""
        return self._create_component(spec, context)
    
    def create_signal_processor(
        self,
        spec: RiskComponentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> SignalProcessorProtocol:
        """Create signal processor from specification."""
        return self._create_component(spec, context)
    
    def _create_component(
        self,
        spec: RiskComponentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Create component using factory with DI support."""
        # Get class from registry
        component_class = self._get_component_class(spec.class_name)
        
        # Merge context
        merged_context = {
            'container': self._container,
            'component_id': spec.name,
            **(context or {})
        }
        
        # Create with factory
        return self._component_factory.create(
            component_class,
            context=merged_context,
            **spec.params
        )
    
    def _get_component_class(self, class_name: str) -> Type[Any]:
        """Get component class by name."""
        # Import dynamically to avoid circular imports
        if class_name == 'FixedPositionSizer':
            from .position_sizing import FixedPositionSizer
            return FixedPositionSizer
        elif class_name == 'PercentagePositionSizer':
            from .position_sizing import PercentagePositionSizer
            return PercentagePositionSizer
        elif class_name == 'VolatilityBasedSizer':
            from .position_sizing import VolatilityBasedSizer
            return VolatilityBasedSizer
        elif class_name == 'KellyCriterionSizer':
            from .position_sizing import KellyCriterionSizer
            return KellyCriterionSizer
        elif class_name == 'ATRBasedSizer':
            from .position_sizing import ATRBasedSizer
            return ATRBasedSizer
        elif class_name == 'MaxPositionLimit':
            from .risk_limits import MaxPositionLimit
            return MaxPositionLimit
        elif class_name == 'MaxExposureLimit':
            from .risk_limits import MaxExposureLimit
            return MaxExposureLimit
        elif class_name == 'MaxDrawdownLimit':
            from .risk_limits import MaxDrawdownLimit
            return MaxDrawdownLimit
        elif class_name == 'VaRLimit':
            from .risk_limits import VaRLimit
            return VaRLimit
        elif class_name == 'ConcentrationLimit':
            from .risk_limits import ConcentrationLimit
            return ConcentrationLimit
        elif class_name == 'LeverageLimit':
            from .risk_limits import LeverageLimit
            return LeverageLimit
        elif class_name == 'DailyLossLimit':
            from .risk_limits import DailyLossLimit
            return DailyLossLimit
        elif class_name == 'SymbolRestrictionLimit':
            from .risk_limits import SymbolRestrictionLimit
            return SymbolRestrictionLimit
        elif class_name == 'SignalProcessor':
            from .signal_processing import SignalProcessor
            return SignalProcessor
        elif class_name == 'RiskAdjustedSignalProcessor':
            from .signal_advanced import RiskAdjustedSignalProcessor
            return RiskAdjustedSignalProcessor
        else:
            raise ValueError(f"Unknown component class: {class_name}")
    
    def _register_position_sizers(self) -> None:
        """Register position sizer types."""
        # Position sizers are created on-demand, not registered globally
        pass
    
    def _register_risk_limits(self) -> None:
        """Register risk limit types."""
        # Risk limits are created on-demand, not registered globally
        pass
    
    def _register_processors(self) -> None:
        """Register signal processor types."""
        # Processors are created on-demand, not registered globally
        pass


class RiskDependencyResolver:
    """Resolves dependencies for risk components."""
    
    def __init__(
        self,
        container: DependencyContainer,
        factory: RiskComponentFactory
    ):
        """Initialize with container and factory."""
        self._container = container
        self._factory = factory
        
        # Component registrations
        self._position_sizers: Dict[str, PositionSizerProtocol] = {}
        self._risk_limits: List[RiskLimitProtocol] = []
        self._signal_processor: Optional[SignalProcessorProtocol] = None
        self._portfolio_state: Optional[PortfolioStateProtocol] = None
    
    def register_position_sizer(
        self,
        name: str,
        spec: RiskComponentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> PositionSizerProtocol:
        """Register and create position sizer."""
        sizer = self._factory.create_position_sizer(spec, context)
        self._position_sizers[name] = sizer
        return sizer
    
    def register_risk_limit(
        self,
        spec: RiskComponentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> RiskLimitProtocol:
        """Register and create risk limit."""
        limit = self._factory.create_risk_limit(spec, context)
        self._risk_limits.append(limit)
        return limit
    
    def register_signal_processor(
        self,
        spec: RiskComponentSpec,
        context: Optional[Dict[str, Any]] = None
    ) -> SignalProcessorProtocol:
        """Register and create signal processor."""
        processor = self._factory.create_signal_processor(spec, context)
        self._signal_processor = processor
        return processor
    
    def register_portfolio_state(
        self,
        portfolio_state: PortfolioStateProtocol
    ) -> None:
        """Register portfolio state instance."""
        self._portfolio_state = portfolio_state
    
    def get_position_sizer(self, name: str) -> PositionSizerProtocol:
        """Get position sizer by name."""
        if name not in self._position_sizers:
            raise ValueError(f"Position sizer '{name}' not registered")
        return self._position_sizers[name]
    
    def get_risk_limits(self) -> List[RiskLimitProtocol]:
        """Get all risk limits."""
        return self._risk_limits.copy()
    
    def get_signal_processor(self) -> SignalProcessorProtocol:
        """Get signal processor."""
        if not self._signal_processor:
            raise ValueError("Signal processor not registered")
        return self._signal_processor
    
    def get_portfolio_state(self) -> PortfolioStateProtocol:
        """Get portfolio state."""
        if not self._portfolio_state:
            raise ValueError("Portfolio state not registered")
        return self._portfolio_state


def create_position_sizer_spec(
    sizer_type: str,
    name: str,
    **params
) -> RiskComponentSpec:
    """Create position sizer specification."""
    class_mapping = {
        'fixed': 'FixedPositionSizer',
        'percentage': 'PercentagePositionSizer',
        'volatility': 'VolatilityBasedSizer',
        'kelly': 'KellyCriterionSizer',
        'atr': 'ATRBasedSizer'
    }
    
    class_name = class_mapping.get(sizer_type)
    if not class_name:
        raise ValueError(f"Unknown position sizer type: {sizer_type}")
    
    return RiskComponentSpec(
        component_type='position_sizer',
        name=name,
        class_name=class_name,
        params=params,
        metadata={'type': sizer_type}
    )


def create_risk_limit_spec(
    limit_type: str,
    name: str,
    **params
) -> RiskComponentSpec:
    """Create risk limit specification."""
    class_mapping = {
        'position': 'MaxPositionLimit',
        'exposure': 'MaxExposureLimit',
        'drawdown': 'MaxDrawdownLimit',
        'var': 'VaRLimit',
        'concentration': 'ConcentrationLimit',
        'leverage': 'LeverageLimit',
        'daily_loss': 'DailyLossLimit',
        'symbol_restriction': 'SymbolRestrictionLimit'
    }
    
    class_name = class_mapping.get(limit_type)
    if not class_name:
        raise ValueError(f"Unknown risk limit type: {limit_type}")
    
    return RiskComponentSpec(
        component_type='risk_limit',
        name=name,
        class_name=class_name,
        params=params,
        metadata={'type': limit_type}
    )


def create_signal_processor_spec(
    processor_type: str = 'standard',
    name: str = 'signal_processor',
    **params
) -> RiskComponentSpec:
    """Create signal processor specification."""
    class_mapping = {
        'standard': 'SignalProcessor',
        'risk_adjusted': 'RiskAdjustedSignalProcessor'
    }
    
    class_name = class_mapping.get(processor_type)
    if not class_name:
        raise ValueError(f"Unknown signal processor type: {processor_type}")
    
    return RiskComponentSpec(
        component_type='signal_processor',
        name=name,
        class_name=class_name,
        params=params,
        metadata={'type': processor_type}
    )

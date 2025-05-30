"""
Backtest Container Factory following BACKTEST.MD architecture.

Creates the complete nested container hierarchy for backtesting.
"""
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
import uuid
from dataclasses import dataclass, field

from ..core.containers import UniversalScopedContainer, ComponentSpec
from ..core.events import EventType
from ..strategy.components import IndicatorHub, IndicatorConfig, IndicatorType
from ..strategy.classifiers import (
    HMMClassifier, 
    PatternClassifier,
    EnhancedClassifierContainer,
    create_classifier_hierarchy
)
from ..data.streamer import DataStreamer
from .backtest_engine import UnifiedBacktestEngine, BacktestConfig


@dataclass
class ClassifierConfig:
    """Configuration for a classifier with risk profiles."""
    type: str  # 'hmm', 'pattern', etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    risk_profiles: List['RiskProfileConfig'] = field(default_factory=list)


@dataclass
class RiskProfileConfig:
    """Configuration for a risk profile."""
    name: str  # 'conservative', 'balanced', 'aggressive'
    capital_allocation: float
    risk_parameters: Dict[str, Any] = field(default_factory=dict)
    portfolio_parameters: Dict[str, Any] = field(default_factory=dict)
    strategies: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BacktestContainerConfig:
    """Complete configuration for a backtest container."""
    container_id: Optional[str] = None
    data_config: Dict[str, Any] = field(default_factory=dict)
    indicator_configs: List[IndicatorConfig] = field(default_factory=list)
    classifiers: List[ClassifierConfig] = field(default_factory=list)
    execution_config: Dict[str, Any] = field(default_factory=dict)
    shared_services: Dict[str, Any] = field(default_factory=dict)


class BacktestContainerFactory:
    """
    Factory for creating backtest containers per BACKTEST.MD.
    
    Creates the standardized hierarchy:
    - BacktestContainer (top-level)
      - DataStreamer
      - IndicatorHub (shared computation)
      - Classifier Containers
        - Classifier Component
        - Risk & Portfolio Containers
          - Risk Manager
          - Portfolio
          - Strategies
      - BacktestEngine
    """
    
    @staticmethod
    def create_instance(config: BacktestContainerConfig) -> UniversalScopedContainer:
        """
        Create a backtest container with full hierarchy.
        
        Args:
            config: Complete backtest configuration
            
        Returns:
            Configured backtest container
        """
        # Generate container ID if not provided
        if config.container_id is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            config.container_id = f"backtest_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Create top-level backtest container
        backtest_container = UniversalScopedContainer(
            container_id=config.container_id,
            container_type="backtest",
            shared_services=config.shared_services
        )
        
        # 1. Create Data Layer
        BacktestContainerFactory._create_data_layer(backtest_container, config)
        
        # 2. Create Indicator Hub (Shared Computation)
        indicator_hub = BacktestContainerFactory._create_indicator_hub(
            backtest_container, config
        )
        
        # 3. Create Classifier Containers with Risk & Portfolio sub-containers
        classifier_containers = BacktestContainerFactory._create_classifier_hierarchy(
            backtest_container, config
        )
        
        # 4. Create Execution Layer
        backtest_engine = BacktestContainerFactory._create_execution_layer(
            backtest_container, config
        )
        
        # 5. Wire Event Flows
        BacktestContainerFactory._wire_event_flows(
            backtest_container, 
            indicator_hub,
            classifier_containers,
            backtest_engine
        )
        
        return backtest_container
    
    @staticmethod
    def _create_data_layer(
        container: UniversalScopedContainer,
        config: BacktestContainerConfig
    ) -> None:
        """Create data streaming components."""
        # Create data streamer
        data_spec = ComponentSpec(
            name="data_streamer",
            class_name=config.data_config.get('streamer_class', 'HistoricalDataStreamer'),
            parameters=config.data_config.get('parameters', {}),
            capabilities=['event_publisher']
        )
        container.create_component(data_spec)
    
    @staticmethod
    def _create_indicator_hub(
        container: UniversalScopedContainer,
        config: BacktestContainerConfig
    ) -> IndicatorHub:
        """Create centralized indicator hub."""
        # Create indicator hub with configured indicators
        indicator_hub = IndicatorHub(
            indicators=config.indicator_configs,
            cache_size=1000,
            event_bus=container.event_bus
        )
        
        # Register as a component
        container.register_component("indicator_hub", indicator_hub)
        
        # Subscribe to market data
        container.event_bus.subscribe(
            EventType.BAR,
            indicator_hub.process_market_data
        )
        
        return indicator_hub
    
    @staticmethod
    def _create_classifier_hierarchy(
        container: UniversalScopedContainer,
        config: BacktestContainerConfig
    ) -> List[EnhancedClassifierContainer]:
        """Create classifier containers with risk & portfolio sub-containers."""
        classifier_containers = []
        
        for classifier_config in config.classifiers:
            # Map classifier type to class
            classifier_class = BacktestContainerFactory._get_classifier_class(
                classifier_config.type
            )
            
            # Create classifier container
            classifier_container = EnhancedClassifierContainer(
                container_id=f"{container.container_id}_{classifier_config.type}",
                classifier_type=classifier_config.type,
                classifier_class=classifier_class,
                shared_services=container.shared_services
            )
            
            # Add as sub-container
            container.add_subcontainer(classifier_container)
            
            # Add risk & portfolio containers
            for risk_profile in classifier_config.risk_profiles:
                classifier_container.add_risk_portfolio_container(
                    risk_profile=risk_profile.name,
                    risk_parameters=risk_profile.risk_parameters,
                    portfolio_parameters={
                        **risk_profile.portfolio_parameters,
                        'initial_capital': risk_profile.capital_allocation
                    },
                    strategies=risk_profile.strategies
                )
            
            classifier_containers.append(classifier_container)
        
        return classifier_containers
    
    @staticmethod
    def _create_execution_layer(
        container: UniversalScopedContainer,
        config: BacktestContainerConfig
    ) -> UnifiedBacktestEngine:
        """Create backtest execution engine."""
        # Create backtest engine configuration
        engine_config = BacktestConfig(
            start_date=config.data_config.get('start_date'),
            end_date=config.data_config.get('end_date'),
            initial_capital=sum(
                risk_profile.capital_allocation 
                for classifier in config.classifiers
                for risk_profile in classifier.risk_profiles
            ),
            slippage_model=config.execution_config.get('slippage_model'),
            commission_model=config.execution_config.get('commission_model')
        )
        
        # Create engine
        engine = UnifiedBacktestEngine(
            config=engine_config,
            event_bus=container.event_bus
        )
        
        # Register as component
        container.register_component("backtest_engine", engine)
        
        return engine
    
    @staticmethod
    def _wire_event_flows(
        container: UniversalScopedContainer,
        indicator_hub: IndicatorHub,
        classifier_containers: List[EnhancedClassifierContainer],
        backtest_engine: UnifiedBacktestEngine
    ) -> None:
        """Wire up event flows per BACKTEST.MD."""
        
        # 1. Indicator events to classifiers
        for classifier_container in classifier_containers:
            # Forward indicator events to classifier
            container.event_bus.subscribe(
                EventType.INDICATOR,
                lambda event: classifier_container.event_bus.publish(event)
            )
        
        # 2. Order events from risk containers to backtest engine
        for classifier_container in classifier_containers:
            for risk_container in classifier_container.risk_portfolio_containers.values():
                # Subscribe to orders from risk container
                risk_container.event_bus.subscribe(
                    EventType.ORDER,
                    lambda event: container.event_bus.publish(event)
                )
        
        # Subscribe backtest engine to orders
        container.event_bus.subscribe(
            EventType.ORDER,
            backtest_engine.process_event
        )
        
        # 3. Fill events from backtest engine to risk containers
        def distribute_fill(event):
            """Distribute fill events to appropriate risk containers."""
            for classifier_container in classifier_containers:
                for risk_container in classifier_container.risk_portfolio_containers.values():
                    risk_container.event_bus.publish(event)
        
        container.event_bus.subscribe(
            EventType.FILL,
            distribute_fill
        )
    
    @staticmethod
    def _get_classifier_class(classifier_type: str) -> Type:
        """Get classifier class from type string."""
        mapping = {
            'hmm': HMMClassifier,
            'pattern': PatternClassifier,
            # Add more classifiers as implemented
        }
        
        if classifier_type not in mapping:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        return mapping[classifier_type]
    
    @staticmethod
    def create_from_workflow_config(
        workflow_config: Dict[str, Any]
    ) -> UniversalScopedContainer:
        """
        Create backtest container from workflow configuration.
        
        Args:
            workflow_config: Workflow configuration dict
            
        Returns:
            Configured backtest container
        """
        # Extract and transform configuration
        indicator_configs = []
        for ind_config in workflow_config.get('indicators', []):
            indicator_configs.append(IndicatorConfig(
                name=ind_config['name'],
                indicator_type=IndicatorType(ind_config.get('type', 'CUSTOM').lower()),
                parameters=ind_config.get('parameters', {}),
                enabled=True
            ))
        
        # Transform classifier configs
        classifiers = []
        for class_config in workflow_config.get('classifiers', []):
            risk_profiles = []
            for risk_config in class_config.get('risk_profiles', []):
                risk_profiles.append(RiskProfileConfig(
                    name=risk_config['name'],
                    capital_allocation=risk_config.get('capital_allocation', 100000),
                    risk_parameters=risk_config.get('risk_parameters', {}),
                    portfolio_parameters=risk_config.get('portfolio_parameters', {}),
                    strategies=risk_config.get('strategies', [])
                ))
            
            classifiers.append(ClassifierConfig(
                type=class_config['type'],
                parameters=class_config.get('parameters', {}),
                risk_profiles=risk_profiles
            ))
        
        # Create backtest config
        backtest_config = BacktestContainerConfig(
            data_config=workflow_config.get('data', {}),
            indicator_configs=indicator_configs,
            classifiers=classifiers,
            execution_config=workflow_config.get('execution', {}),
            shared_services=workflow_config.get('shared_services', {})
        )
        
        return BacktestContainerFactory.create_instance(backtest_config)


def create_example_backtest_container() -> UniversalScopedContainer:
    """Create an example backtest container for testing."""
    config = BacktestContainerConfig(
        data_config={
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'symbols': ['AAPL', 'GOOGL', 'MSFT']
        },
        indicator_configs=[
            IndicatorConfig(
                name='SMA_20',
                indicator_type=IndicatorType.TREND,
                parameters={'period': 20}
            ),
            IndicatorConfig(
                name='RSI_14',
                indicator_type=IndicatorType.MOMENTUM,
                parameters={'period': 14}
            )
        ],
        classifiers=[
            ClassifierConfig(
                type='hmm',
                parameters={'n_states': 3},
                risk_profiles=[
                    RiskProfileConfig(
                        name='conservative',
                        capital_allocation=300000,
                        risk_parameters={
                            'max_position_size': 0.02,
                            'max_total_exposure': 0.1
                        },
                        strategies=[
                            {
                                'name': 'momentum',
                                'class': 'MomentumStrategy',
                                'parameters': {
                                    'fast_period': 10,
                                    'slow_period': 30
                                }
                            }
                        ]
                    ),
                    RiskProfileConfig(
                        name='aggressive',
                        capital_allocation=500000,
                        risk_parameters={
                            'max_position_size': 0.05,
                            'max_total_exposure': 0.3
                        },
                        strategies=[
                            {
                                'name': 'breakout',
                                'class': 'BreakoutStrategy',
                                'parameters': {
                                    'breakout_period': 20
                                }
                            }
                        ]
                    )
                ]
            )
        ],
        execution_config={
            'slippage_model': 'fixed',
            'commission_model': 'percentage'
        }
    )
    
    return BacktestContainerFactory.create_instance(config)
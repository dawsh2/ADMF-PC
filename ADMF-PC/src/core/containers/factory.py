"""
Container factory for creating specialized containers.

This module provides factory methods for creating different types
of containers with appropriate component configurations.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Type
import logging
from datetime import datetime

from .universal import UniversalScopedContainer, ContainerType
from .lifecycle import ContainerLifecycleManager
from .naming import (
    ContainerNamingStrategy,
    ContainerType as NamingContainerType,
    Phase,
    ClassifierType,
    RiskProfile,
    create_backtest_container_id,
    create_optimization_container_id
)


logger = logging.getLogger(__name__)


class ContainerFactory:
    """
    Factory for creating specialized containers.
    
    This factory knows how to create and configure containers
    for different use cases (backtest, optimization, etc.).
    """
    
    def __init__(self, lifecycle_manager: Optional[ContainerLifecycleManager] = None):
        """
        Initialize the container factory.
        
        Args:
            lifecycle_manager: Lifecycle manager to use
        """
        self.lifecycle_manager = lifecycle_manager or ContainerLifecycleManager()
        
        # Container type configurations
        self._container_configs: Dict[str, Dict[str, Any]] = {
            ContainerType.BACKTEST.value: {
                'components': [
                    {
                        'name': 'Portfolio',
                        'class_name': 'Portfolio',
                        'params': {'initial_cash': 100000},
                        'capabilities': ['lifecycle', 'events', 'reset']
                    },
                    {
                        'name': 'RiskManager',
                        'class_name': 'RiskManager',
                        'dependencies': ['Portfolio'],
                        'capabilities': ['lifecycle', 'events']
                    },
                    {
                        'name': 'ExecutionSimulator',
                        'class_name': 'ExecutionSimulator',
                        'dependencies': ['Portfolio'],
                        'capabilities': ['lifecycle', 'events']
                    }
                ]
            },
            ContainerType.OPTIMIZATION.value: {
                'components': [
                    {
                        'name': 'Portfolio',
                        'class_name': 'Portfolio',
                        'params': {'initial_cash': 100000},
                        'capabilities': ['lifecycle', 'events', 'reset']
                    },
                    {
                        'name': 'MetricsCollector',
                        'class_name': 'MetricsCollector',
                        'dependencies': ['Portfolio'],
                        'capabilities': ['lifecycle', 'events']
                    }
                ]
            },
            ContainerType.LIVE_TRADING.value: {
                'components': [
                    {
                        'name': 'Portfolio',
                        'class_name': 'LivePortfolio',
                        'capabilities': ['lifecycle', 'events', 'stateful']
                    },
                    {
                        'name': 'RiskManager',
                        'class_name': 'LiveRiskManager',
                        'dependencies': ['Portfolio'],
                        'capabilities': ['lifecycle', 'events', 'monitorable']
                    },
                    {
                        'name': 'OrderManager',
                        'class_name': 'OrderManager',
                        'dependencies': ['Portfolio', 'RiskManager'],
                        'capabilities': ['lifecycle', 'events']
                    },
                    {
                        'name': 'BrokerConnector',
                        'class_name': 'BrokerConnector',
                        'dependencies': ['OrderManager'],
                        'capabilities': ['lifecycle', 'events']
                    }
                ]
            },
            ContainerType.INDICATOR.value: {
                'components': []  # Indicators added dynamically
            },
            ContainerType.DATA.value: {
                'components': [
                    {
                        'name': 'DataStore',
                        'class_name': 'DataStore',
                        'capabilities': ['lifecycle']
                    }
                ]
            }
        }
    
    def create_backtest_container(
        self,
        strategy_spec: Dict[str, Any],
        shared_services: Optional[Dict[str, Any]] = None,
        container_id: Optional[str] = None,
        additional_components: Optional[List[Dict[str, Any]]] = None,
        phase: Optional[Phase] = None,
        classifier: Optional[ClassifierType] = None,
        risk_profile: Optional[RiskProfile] = None
    ) -> str:
        """
        Create a container for backtesting.
        
        Args:
            strategy_spec: Strategy specification
            shared_services: Shared services
            container_id: Optional container ID
            additional_components: Extra components to add
            phase: Workflow phase
            classifier: Classifier type
            risk_profile: Risk profile
            
        Returns:
            Container ID
        """
        # Generate structured container ID if not provided
        if container_id is None:
            container_id = create_backtest_container_id(
                phase=phase or Phase.COMPUTATION,
                classifier=classifier or ClassifierType.NONE,
                risk_profile=risk_profile or RiskProfile.BALANCED,
                metadata={'strategy': strategy_spec.get('name', 'unknown')}
            )
        
        # Get base components
        components = self._container_configs[ContainerType.BACKTEST.value]['components'].copy()
        
        # Add strategy
        components.append({
            'name': 'Strategy',
            'class_name': strategy_spec['class'],
            'params': strategy_spec.get('parameters', {}),
            'dependencies': ['Portfolio', 'RiskManager'],
            'capabilities': strategy_spec.get('capabilities', ['lifecycle', 'events']),
            'config': strategy_spec.get('config', {})
        })
        
        # Add any additional components
        if additional_components:
            components.extend(additional_components)
        
        # Create container
        return self.lifecycle_manager.create_container(
            container_type=ContainerType.BACKTEST.value,
            container_id=container_id,
            shared_services=shared_services,
            specs=components,
            initialize=True,
            start=False
        )
    
    def create_optimization_container(
        self,
        strategy_spec: Dict[str, Any],
        trial_id: str,
        shared_services: Optional[Dict[str, Any]] = None,
        phase: Optional[Phase] = None,
        classifier: Optional[ClassifierType] = None
    ) -> str:
        """
        Create a container for optimization trial.
        
        Args:
            strategy_spec: Strategy specification
            trial_id: Optimization trial ID
            shared_services: Shared services
            phase: Workflow phase
            classifier: Classifier type
            
        Returns:
            Container ID
        """
        # Generate structured container ID
        container_id = create_optimization_container_id(
            phase=phase or Phase.PHASE1_GRID_SEARCH,
            trial_number=int(trial_id.split('_')[-1]) if '_' in trial_id else 0,
            classifier=classifier or ClassifierType.NONE,
            metadata={'strategy': strategy_spec.get('name', 'unknown')}
        )
        
        # Get base components
        components = self._container_configs[ContainerType.OPTIMIZATION.value]['components'].copy()
        
        # Add strategy with optimization capability
        strategy_spec = strategy_spec.copy()
        caps = strategy_spec.get('capabilities', ['lifecycle', 'events'])
        if 'optimization' not in caps:
            caps.append('optimization')
        
        components.append({
            'name': 'Strategy',
            'class_name': strategy_spec['class'],
            'params': strategy_spec.get('parameters', {}),
            'dependencies': ['Portfolio'],
            'capabilities': caps,
            'config': strategy_spec.get('config', {})
        })
        
        # Create container
        return self.lifecycle_manager.create_container(
            container_type=ContainerType.OPTIMIZATION.value,
            container_id=container_id,
            shared_services=shared_services,
            specs=components,
            initialize=True,
            start=False
        )
    
    def create_live_trading_container(
        self,
        strategy_spec: Dict[str, Any],
        broker_config: Dict[str, Any],
        shared_services: Optional[Dict[str, Any]] = None,
        container_id: Optional[str] = None
    ) -> str:
        """
        Create a container for live trading.
        
        Args:
            strategy_spec: Strategy specification
            broker_config: Broker configuration
            shared_services: Shared services
            container_id: Optional container ID
            
        Returns:
            Container ID
        """
        # Get base components
        components = self._container_configs[ContainerType.LIVE_TRADING.value]['components'].copy()
        
        # Update broker configuration
        for comp in components:
            if comp['name'] == 'BrokerConnector':
                comp['config'] = broker_config
                break
        
        # Add strategy
        components.append({
            'name': 'Strategy',
            'class_name': strategy_spec['class'],
            'params': strategy_spec.get('parameters', {}),
            'dependencies': ['Portfolio', 'RiskManager'],
            'capabilities': ['lifecycle', 'events', 'monitorable'],
            'config': strategy_spec.get('config', {})
        })
        
        # Create container
        return self.lifecycle_manager.create_container(
            container_type=ContainerType.LIVE_TRADING.value,
            container_id=container_id,
            shared_services=shared_services,
            specs=components,
            initialize=True,
            start=False
        )
    
    def create_indicator_container(
        self,
        indicators: List[Dict[str, Any]],
        shared_services: Optional[Dict[str, Any]] = None,
        container_id: Optional[str] = None
    ) -> str:
        """
        Create a container for shared indicators.
        
        Args:
            indicators: List of indicator specifications
            shared_services: Shared services
            container_id: Optional container ID
            
        Returns:
            Container ID
        """
        components = []
        
        # Create indicator components
        for i, indicator_spec in enumerate(indicators):
            components.append({
                'name': f"{indicator_spec['type']}_{i}",
                'class_name': indicator_spec['type'],
                'params': indicator_spec.get('params', {}),
                'capabilities': ['events'],
                'config': indicator_spec.get('config', {})
            })
        
        # Create indicator hub to manage them
        components.append({
            'name': 'IndicatorHub',
            'class_name': 'IndicatorHub',
            'dependencies': [comp['name'] for comp in components],
            'capabilities': ['lifecycle', 'events']
        })
        
        # Create container
        return self.lifecycle_manager.create_container(
            container_type=ContainerType.INDICATOR.value,
            container_id=container_id,
            shared_services=shared_services,
            specs=components,
            initialize=True,
            start=False
        )
    
    def create_custom_container(
        self,
        container_type: str,
        components: List[Dict[str, Any]],
        shared_services: Optional[Dict[str, Any]] = None,
        container_id: Optional[str] = None,
        initialize: bool = True,
        start: bool = False
    ) -> str:
        """
        Create a custom container with specified components.
        
        Args:
            container_type: Type of container
            components: Component specifications
            shared_services: Shared services
            container_id: Optional container ID
            initialize: Whether to initialize
            start: Whether to start
            
        Returns:
            Container ID
        """
        return self.lifecycle_manager.create_container(
            container_type=container_type,
            container_id=container_id,
            shared_services=shared_services,
            specs=components,
            initialize=initialize,
            start=start
        )
    
    def register_container_config(
        self,
        container_type: str,
        config: Dict[str, Any]
    ) -> None:
        """
        Register a configuration for a container type.
        
        Args:
            container_type: Type name
            config: Container configuration
        """
        self._container_configs[container_type] = config
        logger.info(f"Registered container configuration for type: {container_type}")
    
    def get_container(self, container_id: str) -> UniversalScopedContainer:
        """Get a container by ID."""
        return self.lifecycle_manager.get_container(container_id)
    
    def dispose_container(self, container_id: str) -> None:
        """Dispose of a container."""
        self.lifecycle_manager.dispose_container(container_id)
    
    def list_containers(
        self,
        container_type: Optional[str] = None
    ) -> List[str]:
        """List active containers."""
        return self.lifecycle_manager.list_containers(container_type=container_type)
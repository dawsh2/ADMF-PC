"""
Enhanced Backtest Container Factory with Event Tracing

This module extends the standard backtest container factory to integrate
event tracing capabilities for debugging and pattern discovery.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from ..core.events.tracing import TracedEventBus, EventTracer
from ..core.containers import UniversalScopedContainer
from .backtest_container_factory import (
    BacktestContainerFactory,
    BacktestContainerConfig
)
import logging

logger = logging.getLogger(__name__)


class TracedBacktestContainerFactory(BacktestContainerFactory):
    """
    Extended factory that creates backtest containers with event tracing.
    
    This factory:
    1. Creates containers with TracedEventBus instead of regular EventBus
    2. Initializes EventTracer with proper correlation IDs
    3. Enables event lineage tracking throughout the backtest
    """
    
    @staticmethod
    def create_instance(config: BacktestContainerConfig) -> UniversalScopedContainer:
        """
        Create a backtest container with full event tracing.
        
        Args:
            config: Complete backtest configuration
            
        Returns:
            Configured backtest container with tracing enabled
        """
        # Generate container ID if not provided
        if config.container_id is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            config.container_id = f"backtest_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Create event tracer for this backtest
        correlation_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        event_tracer = EventTracer(correlation_id=correlation_id)
        
        # Create traced event bus
        traced_bus = TracedEventBus(name="backtest_main")
        traced_bus.set_tracer(event_tracer)
        
        # Create top-level backtest container with traced bus
        # Note: We need to modify UniversalScopedContainer to accept custom event bus
        # For now, we'll create a standard container and replace its event bus
        backtest_container = UniversalScopedContainer(
            container_id=config.container_id,
            container_type="backtest",
            shared_services={
                **config.shared_services,
                'event_tracer': event_tracer,  # Make tracer available to all containers
                'correlation_id': correlation_id
            }
        )
        
        # Replace the event bus with traced version
        # This is a workaround - ideally UniversalScopedContainer would accept event_bus param
        backtest_container._event_bus = traced_bus
        
        logger.info(f"Backtest starting with correlation_id: {correlation_id}")
        
        # Continue with standard factory setup
        # 1. Create Data Layer
        TracedBacktestContainerFactory._create_data_layer(backtest_container, config)
        
        # 2. Create Indicator Hub (Shared Computation)
        indicator_hub = TracedBacktestContainerFactory._create_indicator_hub(
            backtest_container, config
        )
        
        # 3. Create Classifier Containers with Risk & Portfolio sub-containers
        classifier_containers = TracedBacktestContainerFactory._create_classifier_hierarchy(
            backtest_container, config
        )
        
        # 4. Create Execution Layer
        backtest_engine = TracedBacktestContainerFactory._create_execution_layer(
            backtest_container, config
        )
        
        # 5. Wire Event Flows
        TracedBacktestContainerFactory._wire_event_flows(
            backtest_container, 
            indicator_hub,
            classifier_containers,
            backtest_engine
        )
        
        # 6. Add event tracer to container for access
        backtest_container.event_tracer = event_tracer
        
        return backtest_container


def create_traced_backtest_container(workflow_config: Dict[str, Any]) -> UniversalScopedContainer:
    """
    Convenience function to create a traced backtest container from workflow config.
    
    Args:
        workflow_config: Workflow configuration dictionary
        
    Returns:
        Backtest container with event tracing enabled
    """
    # Transform workflow config to backtest config
    backtest_config = BacktestContainerConfig()
    
    # Copy data config
    backtest_config.data_config = workflow_config.get('data', {})
    
    # Transform indicators
    for ind_config in workflow_config.get('indicators', []):
        from ..strategy.components import IndicatorConfig, IndicatorType
        backtest_config.indicator_configs.append(
            IndicatorConfig(
                name=ind_config['name'],
                indicator_type=IndicatorType(ind_config.get('type', 'CUSTOM').lower()),
                parameters=ind_config.get('parameters', {}),
                enabled=True
            )
        )
    
    # Transform classifiers
    from .backtest_container_factory import ClassifierConfig, RiskProfileConfig
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
        
        backtest_config.classifiers.append(ClassifierConfig(
            type=class_config['type'],
            parameters=class_config.get('parameters', {}),
            risk_profiles=risk_profiles
        ))
    
    # Copy execution config
    backtest_config.execution_config = workflow_config.get('execution', {})
    backtest_config.shared_services = workflow_config.get('shared_services', {})
    
    # Use traced factory
    return TracedBacktestContainerFactory.create_instance(backtest_config)
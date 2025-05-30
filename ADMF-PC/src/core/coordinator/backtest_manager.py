"""
Backtest workflow manager that creates proper container hierarchy.

This manager uses the BacktestContainerFactory to create containers
that follow the BACKTEST.MD architecture.
"""
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from ..containers import UniversalScopedContainer
from ..events import Event, EventType
from ...execution.backtest_container_factory import BacktestContainerFactory
from .types import WorkflowConfig, WorkflowResult, ExecutionContext
from .protocols import WorkflowManager


logger = logging.getLogger(__name__)


class BacktestWorkflowManager(WorkflowManager):
    """
    Manages backtest workflow execution with proper container hierarchy.
    
    Creates containers following BACKTEST.MD architecture:
    - BacktestContainer with DataStreamer and IndicatorHub
    - Classifier containers with Risk & Portfolio sub-containers
    - Strategies organized under Risk & Portfolio containers
    - Unified BacktestEngine for execution
    """
    
    def __init__(self, container_id: str, shared_services: Optional[Dict[str, Any]] = None):
        """Initialize backtest workflow manager."""
        self.container_id = container_id
        self.shared_services = shared_services or {}
        self.backtest_container: Optional[UniversalScopedContainer] = None
        
    async def validate_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Validate backtest configuration."""
        errors = []
        warnings = []
        
        # Check required sections
        if not config.data_config:
            errors.append("Missing data configuration")
            
        if not config.backtest_config:
            warnings.append("Missing backtest configuration, using defaults")
            
        # Check for classifiers and strategies
        classifiers = config.parameters.get('classifiers', [])
        if not classifiers:
            warnings.append("No classifiers defined, using default configuration")
            
        # Validate each classifier has strategies
        for classifier in classifiers:
            risk_profiles = classifier.get('risk_profiles', [])
            if not risk_profiles:
                errors.append(f"Classifier {classifier.get('type')} has no risk profiles")
            else:
                for profile in risk_profiles:
                    if not profile.get('strategies'):
                        errors.append(f"Risk profile {profile.get('name')} has no strategies")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    async def execute(
        self,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute backtest workflow with proper container hierarchy."""
        logger.info(
            f"Starting backtest workflow execution",
            workflow_id=context.workflow_id
        )
        
        try:
            # Merge all configuration sources
            workflow_config = self._build_workflow_config(config)
            
            # Create backtest container using factory
            self.backtest_container = BacktestContainerFactory.create_from_workflow_config(
                workflow_config
            )
            
            # Initialize container hierarchy
            await self._initialize_container_hierarchy()
            
            # Start container hierarchy
            await self._start_container_hierarchy()
            
            # Run the backtest
            results = await self._run_backtest(workflow_config)
            
            # Create workflow result
            workflow_result = WorkflowResult(
                workflow_id=context.workflow_id,
                workflow_type=config.workflow_type,
                success=True,
                results=results
            )
            
            workflow_result.finalize()
            return workflow_result
            
        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            
            workflow_result = WorkflowResult(
                workflow_id=context.workflow_id,
                workflow_type=config.workflow_type,
                success=False,
                errors=[str(e)]
            )
            
            workflow_result.finalize()
            return workflow_result
            
        finally:
            # Clean up
            if self.backtest_container:
                await self._stop_container_hierarchy()
    
    def _build_workflow_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build complete workflow configuration."""
        # Start with base config
        workflow_config = {
            'data': config.data_config or {},
            'execution': config.backtest_config.get('execution', {}),
            'shared_services': self.shared_services
        }
        
        # Add indicators
        workflow_config['indicators'] = config.parameters.get('indicators', [
            {'name': 'SMA_20', 'type': 'SMA', 'parameters': {'period': 20}},
            {'name': 'RSI_14', 'type': 'RSI', 'parameters': {'period': 14}}
        ])
        
        # Add classifiers with default if none provided
        workflow_config['classifiers'] = config.parameters.get('classifiers', [
            {
                'type': 'hmm',
                'parameters': {'n_states': 3},
                'risk_profiles': [{
                    'name': 'default',
                    'capital_allocation': config.backtest_config.get('initial_capital', 100000),
                    'risk_parameters': {
                        'max_position_size': 0.02,
                        'max_total_exposure': 0.1
                    },
                    'strategies': config.parameters.get('strategies', [])
                }]
            }
        ])
        
        return workflow_config
    
    async def _initialize_container_hierarchy(self) -> None:
        """Initialize the entire container hierarchy."""
        if not self.backtest_container:
            return
            
        # Initialize main container
        await self.backtest_container.initialize()
        
        # Initialize all sub-containers (classifiers)
        for sub_container in self.backtest_container._subcontainers.values():
            if hasattr(sub_container, 'initialize_hierarchy'):
                await sub_container.initialize_hierarchy()
            else:
                await sub_container.initialize()
    
    async def _start_container_hierarchy(self) -> None:
        """Start the entire container hierarchy."""
        if not self.backtest_container:
            return
            
        # Start main container
        await self.backtest_container.start()
        
        # Start all sub-containers
        for sub_container in self.backtest_container._subcontainers.values():
            if hasattr(sub_container, 'start_hierarchy'):
                await sub_container.start_hierarchy()
            else:
                await sub_container.start()
    
    async def _stop_container_hierarchy(self) -> None:
        """Stop the entire container hierarchy."""
        if not self.backtest_container:
            return
            
        # Stop all sub-containers first
        for sub_container in self.backtest_container._subcontainers.values():
            if hasattr(sub_container, 'stop_hierarchy'):
                await sub_container.stop_hierarchy()
            else:
                await sub_container.stop()
        
        # Stop main container
        await self.backtest_container.stop()
    
    async def _run_backtest(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run the actual backtest."""
        if not self.backtest_container:
            raise RuntimeError("Backtest container not initialized")
            
        # Get the backtest engine
        engine = self.backtest_container.get_component("backtest_engine")
        if not engine:
            raise RuntimeError("Backtest engine not found in container")
            
        # Get data streamer
        data_streamer = self.backtest_container.get_component("data_streamer")
        if not data_streamer:
            raise RuntimeError("Data streamer not found in container")
            
        # Configure data streamer
        await data_streamer.configure({
            'symbols': workflow_config['data'].get('symbols', []),
            'start_date': workflow_config['data'].get('start_date'),
            'end_date': workflow_config['data'].get('end_date')
        })
        
        # Track progress
        total_bars = 0
        processed_bars = 0
        
        def track_progress(event: Event):
            nonlocal processed_bars
            if event.event_type == EventType.BAR:
                processed_bars += 1
                if processed_bars % 100 == 0:
                    logger.info(f"Processed {processed_bars} bars")
        
        self.backtest_container.event_bus.subscribe(EventType.BAR, track_progress)
        
        # Run backtest by streaming data
        logger.info("Starting data stream for backtest")
        
        async for timestamp, market_data in data_streamer.stream():
            # Publish market data event
            event = Event(
                event_type=EventType.BAR,
                payload={
                    'timestamp': timestamp,
                    'market_data': market_data
                },
                source_id="data_streamer"
            )
            
            self.backtest_container.event_bus.publish(event)
            total_bars += 1
        
        logger.info(f"Backtest complete. Processed {total_bars} bars")
        
        # Get results from engine
        results = engine.get_results()
        
        # Add container structure info
        results['container_structure'] = self._get_container_structure()
        
        return results
    
    def _get_container_structure(self) -> Dict[str, Any]:
        """Get the container hierarchy structure for debugging."""
        if not self.backtest_container:
            return {}
            
        structure = {
            'main_container': self.backtest_container.container_id,
            'components': list(self.backtest_container._components.keys()),
            'classifiers': {}
        }
        
        # Add classifier containers
        for name, sub_container in self.backtest_container._subcontainers.items():
            if hasattr(sub_container, 'classifier_type'):
                structure['classifiers'][name] = {
                    'type': sub_container.classifier_type,
                    'risk_profiles': list(sub_container.risk_portfolio_containers.keys())
                }
        
        return structure
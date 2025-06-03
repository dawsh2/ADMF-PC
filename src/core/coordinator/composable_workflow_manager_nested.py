"""
Composable Workflow Manager with Proper Container Nesting.

This version implements the correct container hierarchy:
Risk > Portfolio > Strategy
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..logging.container_logger import get_container_logger
from .protocols import WorkflowManagerProtocol
from .types import WorkflowConfig, WorkflowResult, WorkflowStatus, ExecutionMode
from ..containers.composition_engine import CompositionEngine, get_global_registry
from ..containers.composable import ComposableContainerProtocol
from ..events.types import Event, EventType

logger = logging.getLogger(__name__)


class NestedWorkflowManager(WorkflowManagerProtocol):
    """Workflow manager that creates properly nested container hierarchies."""
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.container = None
        self.completion_event = asyncio.Event()
    
    async def execute_workflow(self, config: WorkflowConfig) -> WorkflowResult:
        """Execute workflow with nested container structure."""
        try:
            # Build composition config
            composition_config = self._build_composition_config(config)
            
            # Create nested container structure
            engine = CompositionEngine()
            
            # Build the nested structure: Risk > Portfolio > Strategy
            logger.info("Creating nested container structure: Risk > Portfolio > Strategy")
            
            # Create custom nested structure
            nested_structure = {
                "root": {
                    "role": "backtest",
                    "children": {
                        "data": {"role": "data"},
                        "indicators": {"role": "indicator"},
                        "risk": {
                            "role": "risk",
                            "children": {
                                "portfolio": {
                                    "role": "portfolio",
                                    "children": {
                                        "strategy": {"role": "strategy"}
                                    }
                                }
                            }
                        },
                        "execution": {"role": "execution"}
                    }
                }
            }
            
            self.container = engine.compose_custom_pattern(nested_structure, composition_config)
            logger.info("Created nested container hierarchy")
            
            # Log the structure
            self._log_container_structure(self.container)
            
            # Get all containers for pipeline setup
            all_containers = self._collect_all_containers(self.container)
            
            # Setup pipeline communication with proper ordering
            if hasattr(self.coordinator, 'communication_layer') and self.coordinator.communication_layer:
                # Configure pipeline order for nested structure
                pipeline_order = self._determine_nested_pipeline_order(all_containers)
                
                # Find the pipeline adapter and setup
                for adapter in self.coordinator.communication_layer.adapters:
                    if hasattr(adapter, 'setup_pipeline'):
                        # Pass ALL containers to adapter
                        adapter.setup_pipeline(all_containers)
                        logger.info(f"Updated pipeline adapter with {len(all_containers)} containers")
                        
                        # Set up pipeline connections for nested structure
                        adapter.connections.clear()
                        
                        # Special handling for nested structure
                        # Data -> Indicator -> Risk (which contains Portfolio/Strategy)
                        data_container = next(c for c in all_containers if c.metadata.role.value == 'data')
                        indicator_container = next(c for c in all_containers if c.metadata.role.value == 'indicator')
                        risk_container = next(c for c in all_containers if c.metadata.role.value == 'risk')
                        execution_container = next(c for c in all_containers if c.metadata.role.value == 'execution')
                        
                        # Main pipeline connections
                        adapter.connections.append((data_container, indicator_container))
                        adapter.connections.append((indicator_container, risk_container))
                        
                        # For nested containers, we need to ensure signals flow from Strategy to Execution
                        # This requires setting up the internal routing within Risk container
                        strategy_container = self._find_nested_container(risk_container, 'strategy')
                        if strategy_container:
                            adapter.connections.append((strategy_container, execution_container))
                        
                        logger.info(f"Set up {len(adapter.connections)} pipeline connections for nested structure")
                        
                        # Start the adapter
                        if hasattr(adapter, 'start'):
                            adapter.start()
                            logger.info("Started pipeline adapter with nested structure")
                        break
            
            # Setup monitoring
            self._setup_completion_monitoring()
            
            # Initialize containers
            await self.container.initialize()
            
            # Start execution
            await self.container.start()
            
            # Wait for completion
            await self._wait_for_completion()
            
            # Extract results
            result = await self._extract_results()
            
            return result
            
        finally:
            # Cleanup
            if self.container:
                await self.container.stop()
    
    def _log_container_structure(self, container: ComposableContainerProtocol, level: int = 0):
        """Log the container hierarchy structure."""
        indent = "  " * level
        logger.info(f"{indent}Container: {container.metadata.name} (role: {container.metadata.role.value})")
        for child in container.child_containers:
            self._log_container_structure(child, level + 1)
    
    def _find_nested_container(self, parent: ComposableContainerProtocol, role: str) -> Optional[ComposableContainerProtocol]:
        """Find a container with specific role in the hierarchy."""
        if parent.metadata.role.value == role:
            return parent
        
        for child in parent.child_containers:
            result = self._find_nested_container(child, role)
            if result:
                return result
        
        return None
    
    def _determine_nested_pipeline_order(self, containers: List[ComposableContainerProtocol]) -> List[ComposableContainerProtocol]:
        """Determine pipeline order for nested structure."""
        # For nested structure, we need a different approach
        # The order should respect the nesting but still allow proper event flow
        
        role_map = {}
        for container in containers:
            role = container.metadata.role.value
            role_map[role] = container
        
        # Order for nested structure
        # Note: Strategy is nested but needs to connect to Execution
        pipeline_order = []
        
        if 'data' in role_map:
            pipeline_order.append(role_map['data'])
        if 'indicator' in role_map:
            pipeline_order.append(role_map['indicator'])
        if 'risk' in role_map:
            pipeline_order.append(role_map['risk'])
        if 'strategy' in role_map:
            pipeline_order.append(role_map['strategy'])
        if 'execution' in role_map:
            pipeline_order.append(role_map['execution'])
        
        logger.info(f"Nested pipeline order: {[c.metadata.name for c in pipeline_order]}")
        return pipeline_order
    
    def _collect_all_containers(self, root: ComposableContainerProtocol) -> List[ComposableContainerProtocol]:
        """Collect all containers in the hierarchy."""
        containers = [root]
        for child in root.child_containers:
            containers.extend(self._collect_all_containers(child))
        return containers
    
    def _build_composition_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build configuration for container composition."""
        composition_config = {
            'data': config.data_config,
            'backtest': config.backtest_config,
        }
        
        # Add configurations from parameters
        if 'risk' in config.parameters:
            composition_config['risk'] = config.parameters['risk']
            
        if 'portfolio' in config.parameters:
            composition_config['portfolio'] = config.parameters['portfolio']
            
        if 'indicator' in config.parameters:
            composition_config['indicator'] = config.parameters['indicator']
            
        if 'execution' in config.parameters:
            composition_config['execution'] = config.parameters['execution']
        
        # Extract strategies if present
        if 'strategies' in config.parameters:
            composition_config['strategies'] = config.parameters['strategies']
            
            # Also set strategy config directly
            composition_config['strategy'] = {
                'strategies': config.parameters['strategies']
            }
        
        return composition_config
    
    def _setup_completion_monitoring(self):
        """Setup monitoring for workflow completion."""
        # Subscribe to system events
        if self.container:
            # Find data container for END_OF_DATA events
            data_containers = [c for c in self._collect_all_containers(self.container) 
                             if c.metadata.role.value == 'data']
            
            for data_container in data_containers:
                data_container.event_bus.subscribe(
                    EventType.SYSTEM,
                    self._handle_system_event
                )
    
    def _handle_system_event(self, event: Event):
        """Handle system events for completion detection."""
        if event.payload.get('message') == 'END_OF_DATA':
            logger.info("Received END_OF_DATA signal")
            self.completion_event.set()
    
    async def _wait_for_completion(self):
        """Wait for workflow completion."""
        # Use a timeout to prevent hanging
        timeout = 300  # 5 minutes
        
        try:
            await asyncio.wait_for(self.completion_event.wait(), timeout)
            logger.info("Workflow completed normally")
        except asyncio.TimeoutError:
            logger.warning(f"Workflow timeout after {timeout} seconds")
            
        # Give containers time to process final events
        await asyncio.sleep(2)
    
    async def _extract_results(self) -> Dict[str, Any]:
        """Extract results from completed workflow."""
        result = {
            'success': True,
            'container_id': self.container.metadata.container_id if self.container else None,
            'completion_time': datetime.now(),
        }
        
        # Extract portfolio performance from nested structure
        try:
            # Find portfolio container in nested structure
            risk_container = self._find_nested_container(self.container, 'risk')
            if risk_container:
                portfolio_container = self._find_nested_container(risk_container, 'portfolio')
                if portfolio_container:
                    # Get portfolio state
                    if hasattr(portfolio_container, 'portfolio_state'):
                        portfolio = portfolio_container.portfolio_state
                        if portfolio:
                            result['portfolio'] = {
                                'final_value': float(portfolio.get_total_value()),
                                'cash_balance': float(portfolio.get_cash_balance()),
                                'positions': len(portfolio.get_all_positions()),
                                'position_details': {
                                    symbol: {
                                        'quantity': float(pos.quantity),
                                        'avg_price': float(pos.average_price)
                                    } for symbol, pos in portfolio.get_all_positions().items()
                                }
                            }
                            logger.info(f"Extracted portfolio results from nested structure: {result['portfolio']}")
                        else:
                            logger.warning("Portfolio state is None")
                    else:
                        logger.warning("Portfolio container has no portfolio_state attribute")
                else:
                    logger.warning("Could not find portfolio container in nested structure")
            else:
                logger.warning("Could not find risk container")
                
        except Exception as e:
            logger.error(f"Error extracting portfolio results from nested structure: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result
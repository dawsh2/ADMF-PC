"""
Nested execution strategy for hierarchical container structures.

This execution strategy creates hierarchical container structures like:
Risk > Portfolio > Strategy, with parent-child event routing.
"""

import asyncio
import logging
from typing import Dict, Any

from . import ExecutionStrategy
from ....types.workflow import WorkflowConfig, ExecutionContext, WorkflowResult

logger = logging.getLogger(__name__)


class NestedExecutor(ExecutionStrategy):
    """Nested container execution strategy for hierarchical structures."""
    
    async def execute_single_pattern(
        self,
        pattern_info: Dict[str, Any],
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute pattern using nested container structure."""
        
        pattern_name = pattern_info['name']
        pattern_config = pattern_info['config']
        
        logger.info(f"Executing nested pattern: {pattern_name}")
        
        container_key = f"{context.workflow_id}_{pattern_name}_nested"
        
        try:
            # Create custom nested structure
            if pattern_name == 'simple_backtest':
                nested_structure = self._create_simple_backtest_nested_structure()
            else:
                # Use pattern-specific nested structure
                nested_structure = self._create_pattern_nested_structure(pattern_name)
            
            # Create container using custom structure
            root_container = self.factory.compose_custom_pattern(nested_structure, pattern_config)
            
            # Store container for cleanup
            self.workflow_manager.active_containers[container_key] = root_container
            
            # Log the nested structure
            self._log_container_hierarchy(root_container)
            
            # Setup hierarchical communication
            await self._setup_hierarchical_communication(root_container, context)
            
            # Execute container
            result = await self._execute_container(root_container, config, context)
            result.metadata['container_pattern'] = f"{pattern_name}_nested"
            result.metadata['execution_mode'] = 'nested'
            
            return result
            
        except Exception as e:
            logger.error(f"Nested pattern execution failed: {e}")
            return WorkflowResult(
                workflow_id=context.workflow_id,
                workflow_type=config.workflow_type,
                success=False,
                errors=[str(e)],
                metadata={'execution_mode': 'nested', 'pattern': pattern_name}
            )
        finally:
            # Clean up
            if container_key in self.workflow_manager.active_containers:
                await self.workflow_manager.active_containers[container_key].dispose()
                del self.workflow_manager.active_containers[container_key]
    
    def _create_simple_backtest_nested_structure(self) -> Dict[str, Any]:
        """Create nested structure for simple backtest: Risk > Portfolio > Strategy."""
        
        return {
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
    
    def _create_pattern_nested_structure(self, pattern_name: str) -> Dict[str, Any]:
        """Create nested structure for other patterns."""
        
        if pattern_name == 'full_backtest':
            return {
                "root": {
                    "role": "backtest",
                    "children": {
                        "data": {"role": "data"},
                        "indicators": {"role": "indicator"},
                        "classifier": {"role": "classifier"},
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
        else:
            # Default to simple backtest structure
            return self._create_simple_backtest_nested_structure()
    
    async def _setup_hierarchical_communication(self, root_container, context: ExecutionContext) -> None:
        """Setup hierarchical parent-child communication."""
        try:
            # Get all containers in hierarchy
            all_containers = self._collect_all_containers(root_container)
            container_map = {c.metadata.name: c for c in all_containers}
            
            # Create hierarchical adapter configuration
            hierarchical_config = {
                'adapters': [{
                    'name': 'hierarchical_adapter',
                    'type': 'hierarchical',
                    'parent': root_container.metadata.name,
                    'children': [
                        {'name': c.metadata.name, 'role': c.metadata.role.value} 
                        for c in all_containers if c != root_container
                    ],
                    'log_level': 'INFO'
                }]
            }
            
            # Create adapters
            adapters = self.adapter_factory.create_adapters_from_config(
                hierarchical_config['adapters'],
                container_map
            )
            
            # Store adapters for cleanup
            self.workflow_manager.active_adapters.extend(adapters)
            
            # Start adapters
            self.adapter_factory.start_all()
            
            logger.info(f"Hierarchical communication setup complete with {len(adapters)} adapters")
            
        except Exception as e:
            logger.error(f"Failed to setup hierarchical communication: {e}")
            # Don't raise - allow workflow to continue without communication
    
    async def _execute_container(self, root_container, config: WorkflowConfig, context: ExecutionContext) -> WorkflowResult:
        """Execute container with hierarchical coordination."""
        
        try:
            # Initialize container hierarchy
            await root_container.initialize()
            
            # Start execution (cascades to children)
            await root_container.start()
            
            # Wait for completion with hierarchical monitoring
            await self._wait_for_hierarchical_completion(root_container, config)
            
            # Collect results from hierarchy
            container_status = root_container.get_status()
            
            result = WorkflowResult(
                workflow_id=context.workflow_id,
                workflow_type=config.workflow_type,
                success=root_container.state.value in ['stopped', 'initialized', 'running'],
                final_results={
                    'container_status': container_status,
                    'container_structure': self._get_container_structure(root_container),
                    'metrics': container_status.get('metrics', {}),
                    'final_state': root_container.state.value,
                    'hierarchy_depth': self._calculate_hierarchy_depth(root_container)
                }
            )
            
            # Stop container hierarchy
            await root_container.stop()
            
            return result
            
        except Exception as e:
            logger.error(f"Hierarchical container execution failed: {e}")
            raise e
    
    async def _wait_for_hierarchical_completion(self, root_container, config: WorkflowConfig) -> None:
        """Wait for completion considering hierarchical structure."""
        
        # Wait for data streaming completion at root
        await asyncio.sleep(2.0)
        
        # Monitor hierarchy for completion
        max_wait = 10.0  # Longer wait for hierarchical structures
        check_interval = 0.5
        
        start_time = asyncio.get_event_loop().time()
        
        while True:
            current_time = asyncio.get_event_loop().time()
            
            if current_time - start_time > max_wait:
                logger.info("Hierarchical completion detection finished after max wait time")
                break
            
            # Check if all containers in hierarchy are stable
            all_containers = self._collect_all_containers(root_container)
            all_stable = True
            
            for container in all_containers:
                status = container.get_status()
                metrics = status.get('metrics', {})
                
                # Check if container is still actively processing
                if metrics.get('events_published', 0) > 0:
                    # Container might still be processing
                    all_stable = False
                    break
            
            if all_stable:
                logger.info("All containers in hierarchy appear stable")
                break
            
            await asyncio.sleep(check_interval)
    
    def _log_container_hierarchy(self, container, level: int = 0) -> None:
        """Log the container hierarchy structure."""
        indent = "  " * level
        logger.info(f"{indent}Container: {container.metadata.name} (role: {container.metadata.role.value})")
        for child in container.child_containers:
            self._log_container_hierarchy(child, level + 1)
    
    def _get_container_structure(self, container) -> Dict[str, Any]:
        """Get hierarchical structure of container."""
        structure = {
            'id': container.metadata.container_id,
            'role': container.metadata.role.value,
            'name': container.metadata.name,
            'state': container.state.value,
            'children': []
        }
        
        for child in container.child_containers:
            structure['children'].append(self._get_container_structure(child))
        
        return structure
    
    def _collect_all_containers(self, root) -> list:
        """Collect all containers in the hierarchy."""
        containers = [root]
        for child in root.child_containers:
            containers.extend(self._collect_all_containers(child))
        return containers
    
    def _calculate_hierarchy_depth(self, container, current_depth: int = 0) -> int:
        """Calculate maximum depth of container hierarchy."""
        if not container.child_containers:
            return current_depth
        
        max_child_depth = 0
        for child in container.child_containers:
            child_depth = self._calculate_hierarchy_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
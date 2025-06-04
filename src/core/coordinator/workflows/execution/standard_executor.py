"""
Standard execution strategy for container patterns.

This is the basic execution strategy that:
1. Creates containers using factory patterns
2. Sets up basic communication
3. Executes containers sequentially
4. Collects results using event tracing
"""

import asyncio
import logging
from typing import Dict, Any

from . import ExecutionStrategy
from ....types.workflow import WorkflowConfig, ExecutionContext, WorkflowResult
from ....containers.protocols import ComposableContainer

logger = logging.getLogger(__name__)


class StandardExecutor(ExecutionStrategy):
    """Standard container execution strategy."""
    
    async def execute_single_pattern(
        self,
        pattern_info: Dict[str, Any],
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute pattern using standard approach."""
        
        pattern_name = pattern_info['name']
        pattern_config = pattern_info['config']
        
        logger.info(f"Executing standard pattern: {pattern_name}")
        
        container_key = f"{context.workflow_id}_{pattern_name}"
        
        try:
            # 1. Create containers using factory
            root_container = self.factory.compose_pattern(
                pattern_name=pattern_name,
                config_overrides=pattern_config
            )
            
            # Store container for cleanup
            self.workflow_manager.active_containers[container_key] = root_container
            
            # 2. Setup basic communication
            await self._setup_basic_communication(pattern_name, root_container, context)
            
            # 3. Execute container
            result = await self._execute_container(root_container, config, context)
            result.metadata['container_pattern'] = pattern_name
            result.metadata['execution_mode'] = 'standard'
            
            return result
            
        except Exception as e:
            logger.error(f"Standard pattern execution failed: {e}")
            return WorkflowResult(
                workflow_id=context.workflow_id,
                workflow_type=config.workflow_type,
                success=False,
                errors=[str(e)],
                metadata={'execution_mode': 'standard', 'pattern': pattern_name}
            )
        finally:
            # Clean up
            if container_key in self.workflow_manager.active_containers:
                await self.workflow_manager.active_containers[container_key].dispose()
                del self.workflow_manager.active_containers[container_key]
    
    async def _setup_basic_communication(
        self,
        pattern_name: str,
        root_container: ComposableContainer,
        context: ExecutionContext
    ) -> None:
        """Setup basic communication for the pattern."""
        try:
            # Get all containers in hierarchy
            all_containers = self._collect_all_containers(root_container)
            container_map = {c.metadata.name: c for c in all_containers}
            
            # Get communication config for this pattern
            from ..patterns.communication_patterns import get_communication_config
            communication_config = get_communication_config(pattern_name, all_containers)
            
            # Create adapters if configuration exists
            if communication_config.get('adapters'):
                adapters = self.adapter_factory.create_adapters_from_config(
                    communication_config['adapters'],
                    container_map
                )
                
                # Store adapters for cleanup
                self.workflow_manager.active_adapters.extend(adapters)
                
                # Start adapters
                self.adapter_factory.start_all()
                
                logger.info(f"Basic communication setup complete for {pattern_name} with {len(adapters)} adapters")
            else:
                logger.info(f"No communication configuration for pattern: {pattern_name}")
                
        except Exception as e:
            logger.error(f"Failed to setup basic communication: {e}")
            # Don't raise - allow workflow to continue without communication
    
    async def _execute_container(
        self,
        root_container: ComposableContainer,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute a single container and return results."""
        
        try:
            # Initialize container
            await root_container.initialize()
            
            # Start execution
            await root_container.start()
            
            # Wait for completion based on workflow type
            await self._wait_for_completion(root_container, config)
            
            # Collect results using event tracing
            container_status = root_container.get_status()
            
            # The event tracing system will handle detailed result extraction
            # We just collect basic container information here
            result = WorkflowResult(
                workflow_id=context.workflow_id,
                workflow_type=config.workflow_type,
                success=root_container.state.value in ['stopped', 'initialized', 'running'],
                final_results={
                    'container_status': container_status,
                    'container_structure': self._get_container_structure(root_container),
                    'metrics': container_status.get('metrics', {}),
                    'final_state': root_container.state.value
                }
            )
            
            # Stop container
            await root_container.stop()
            
            return result
            
        except Exception as e:
            logger.error(f"Container execution failed: {e}")
            raise e
    
    async def _wait_for_completion(
        self,
        root_container: ComposableContainer,
        config: WorkflowConfig
    ) -> None:
        """Wait for container execution to complete."""
        
        # For backtest workflows, wait for data streaming completion
        if config.workflow_type.value == 'backtest':
            await self._wait_for_data_streaming_completion(root_container)
        else:
            # For other workflows, use shorter default wait
            await asyncio.sleep(1.0)
    
    async def _wait_for_data_streaming_completion(
        self,
        root_container: ComposableContainer
    ) -> None:
        """Wait for data streaming to complete by monitoring container state."""
        
        # Wait for initial data processing
        await asyncio.sleep(2.0)
        
        # Find the data container
        data_container = root_container
        if data_container.metadata.role.value != 'data':
            logger.warning("Root container is not DataContainer, using default completion detection")
            await asyncio.sleep(1.0)
            return
        
        # Monitor for completion
        max_additional_wait = 5.0  # Maximum additional wait time
        check_interval = 0.2  # Check every 200ms
        idle_threshold = 1.0  # Consider complete if idle for 1 second
        
        start_time = asyncio.get_event_loop().time()
        last_events_published = 0
        stable_count = 0
        
        while True:
            current_time = asyncio.get_event_loop().time()
            
            # Check timeout
            if current_time - start_time > max_additional_wait:
                logger.info("Container completion detection finished after max wait time")
                break
            
            # Get current event count
            status = data_container.get_status()
            metrics = status.get('metrics', {})
            events_published = metrics.get('events_published', 0)
            
            # Check if event publishing has stopped
            if events_published == last_events_published:
                stable_count += 1
                # If stable for multiple checks, consider complete
                if stable_count >= (idle_threshold / check_interval):
                    logger.info(f"Data streaming complete (no new events for {stable_count * check_interval:.1f}s)")
                    break
            else:
                # Reset stability counter
                stable_count = 0
                last_events_published = events_published
            
            await asyncio.sleep(check_interval)
    
    def _get_container_structure(self, container: ComposableContainer) -> Dict[str, Any]:
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
    
    def _collect_all_containers(self, root: ComposableContainer) -> list[ComposableContainer]:
        """Collect all containers in the hierarchy."""
        containers = [root]
        for child in root.child_containers:
            containers.extend(self._collect_all_containers(child))
        return containers
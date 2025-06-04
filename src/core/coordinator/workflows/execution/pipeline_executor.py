"""
Pipeline execution strategy for container communication workflows.

This execution strategy creates containers with pipeline communication adapters
for orchestrated data flow between containers.
"""

import asyncio
import logging
from typing import Dict, Any, List

from . import ExecutionStrategy
from ....types.workflow import WorkflowConfig, ExecutionContext, WorkflowResult

logger = logging.getLogger(__name__)


class PipelineExecutor(ExecutionStrategy):
    """Pipeline container execution strategy for orchestrated communication."""
    
    async def execute_single_pattern(
        self,
        pattern_info: Dict[str, Any],
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute pattern using pipeline communication."""
        
        pattern_name = pattern_info['name']
        pattern_config = pattern_info['config']
        
        logger.info(f"Executing pipeline pattern: {pattern_name}")
        
        container_key = f"{context.workflow_id}_{pattern_name}_pipeline"
        
        try:
            # Create container using standard factory
            root_container = self.factory.compose_pattern(
                pattern_name=pattern_name,
                config_overrides=pattern_config
            )
            
            # Store container for cleanup
            self.workflow_manager.active_containers[container_key] = root_container
            
            # Setup pipeline communication
            await self._setup_pipeline_communication(root_container, context)
            
            # Execute container
            result = await self._execute_container(root_container, config, context)
            result.metadata['container_pattern'] = f"{pattern_name}_pipeline"
            result.metadata['execution_mode'] = 'pipeline'
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline pattern execution failed: {e}")
            return WorkflowResult(
                workflow_id=context.workflow_id,
                workflow_type=config.workflow_type,
                success=False,
                errors=[str(e)],
                metadata={'execution_mode': 'pipeline', 'pattern': pattern_name}
            )
        finally:
            # Clean up
            if container_key in self.workflow_manager.active_containers:
                await self.workflow_manager.active_containers[container_key].dispose()
                del self.workflow_manager.active_containers[container_key]
    
    async def execute_multi_pattern(
        self,
        patterns: List[Dict[str, Any]],
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute multiple patterns with pipeline communication between them."""
        
        logger.info(f"Executing multi-pattern pipeline with {len(patterns)} patterns")
        
        try:
            # Create all containers
            containers = []
            container_map = {}
            
            for i, pattern_info in enumerate(patterns):
                pattern_name = pattern_info['name']
                pattern_config = pattern_info['config']
                
                container = self.factory.compose_pattern(
                    pattern_name=pattern_name,
                    config_overrides=pattern_config
                )
                
                container_key = f"{context.workflow_id}_{pattern_name}_{i}"
                containers.append(container)
                container_map[container_key] = container
                self.workflow_manager.active_containers[container_key] = container
            
            # Setup inter-pattern communication
            await self._setup_multi_pattern_communication(containers, context)
            
            # Execute all containers in coordination
            results = await self._execute_coordinated_containers(containers, config, context)
            
            return WorkflowResult(
                workflow_id=context.workflow_id,
                workflow_type=config.workflow_type,
                success=all(r['success'] for r in results),
                final_results={
                    'pattern_results': results,
                    'execution_mode': 'multi_pattern_pipeline'
                },
                metadata={
                    'execution_mode': 'pipeline',
                    'patterns_executed': [p['name'] for p in patterns],
                    'containers_created': len(containers)
                }
            )
            
        except Exception as e:
            logger.error(f"Multi-pattern pipeline execution failed: {e}")
            return WorkflowResult(
                workflow_id=context.workflow_id,
                workflow_type=config.workflow_type,
                success=False,
                errors=[str(e)],
                metadata={'execution_mode': 'pipeline', 'pattern_count': len(patterns)}
            )
        finally:
            # Clean up all containers
            for container_key in list(self.workflow_manager.active_containers.keys()):
                if container_key.startswith(context.workflow_id):
                    await self.workflow_manager.active_containers[container_key].dispose()
                    del self.workflow_manager.active_containers[container_key]
    
    async def _setup_pipeline_communication(self, root_container, context: ExecutionContext) -> None:
        """Setup pipeline communication adapters."""
        try:
            # Get all containers that need pipeline communication
            all_containers = self._collect_all_containers(root_container)
            container_map = {c.metadata.name: c for c in all_containers}
            
            # Create pipeline adapter configuration
            pipeline_config = {
                'adapters': [{
                    'name': 'pipeline_adapter',
                    'type': 'pipeline',
                    'containers': [c.metadata.name for c in all_containers],
                    'flow_order': self._determine_pipeline_flow_order(all_containers),
                    'log_level': 'INFO'
                }]
            }
            
            # Create adapters
            adapters = self.adapter_factory.create_adapters_from_config(
                pipeline_config['adapters'],
                container_map
            )
            
            # Store adapters for cleanup
            self.workflow_manager.active_adapters.extend(adapters)
            
            # Start adapters
            self.adapter_factory.start_all()
            
            logger.info(f"Pipeline communication setup complete with {len(adapters)} adapters")
            
        except Exception as e:
            logger.error(f"Failed to setup pipeline communication: {e}")
            # Don't raise - allow workflow to continue without communication
    
    async def _setup_multi_pattern_communication(self, containers: List, context: ExecutionContext) -> None:
        """Setup communication between multiple pattern containers."""
        try:
            container_map = {f"container_{i}": container for i, container in enumerate(containers)}
            
            # Create broadcast adapter for inter-pattern communication
            broadcast_config = {
                'adapters': [{
                    'name': 'multi_pattern_broadcast',
                    'type': 'broadcast',
                    'containers': list(container_map.keys()),
                    'log_level': 'INFO'
                }]
            }
            
            # Create adapters
            adapters = self.adapter_factory.create_adapters_from_config(
                broadcast_config['adapters'],
                container_map
            )
            
            # Store adapters for cleanup
            self.workflow_manager.active_adapters.extend(adapters)
            
            # Start adapters
            self.adapter_factory.start_all()
            
            logger.info(f"Multi-pattern communication setup complete with {len(adapters)} adapters")
            
        except Exception as e:
            logger.error(f"Failed to setup multi-pattern communication: {e}")
    
    async def _execute_container(self, root_container, config: WorkflowConfig, context: ExecutionContext) -> WorkflowResult:
        """Execute container with pipeline coordination."""
        
        try:
            # Initialize container
            await root_container.initialize()
            
            # Start execution
            await root_container.start()
            
            # Wait for completion with pipeline monitoring
            await self._wait_for_pipeline_completion(root_container, config)
            
            # Collect results
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
                    'pipeline_depth': self._calculate_pipeline_depth(root_container)
                }
            )
            
            # Stop container
            await root_container.stop()
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline container execution failed: {e}")
            raise e
    
    async def _execute_coordinated_containers(self, containers: List, config: WorkflowConfig, context: ExecutionContext) -> List[Dict[str, Any]]:
        """Execute multiple containers in coordinated fashion."""
        
        results = []
        
        # Initialize all containers
        for container in containers:
            await container.initialize()
        
        # Start all containers
        for container in containers:
            await container.start()
        
        # Wait for all to complete
        await self._wait_for_multi_container_completion(containers, config)
        
        # Collect results from all containers
        for i, container in enumerate(containers):
            try:
                container_status = container.get_status()
                results.append({
                    'container_index': i,
                    'success': True,
                    'container_status': container_status,
                    'final_state': container.state.value,
                    'metrics': container_status.get('metrics', {})
                })
                
                # Stop container
                await container.stop()
                
            except Exception as e:
                logger.error(f"Container {i} execution failed: {e}")
                results.append({
                    'container_index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    async def _wait_for_pipeline_completion(self, root_container, config: WorkflowConfig) -> None:
        """Wait for completion considering pipeline flow."""
        
        # Initial wait for pipeline setup
        await asyncio.sleep(2.0)
        
        # Monitor pipeline for completion
        max_wait = 8.0  # Pipeline may take longer
        check_interval = 0.5
        
        start_time = asyncio.get_event_loop().time()
        
        while True:
            current_time = asyncio.get_event_loop().time()
            
            if current_time - start_time > max_wait:
                logger.info("Pipeline completion detection finished after max wait time")
                break
            
            # Check if pipeline is stable
            all_containers = self._collect_all_containers(root_container)
            pipeline_stable = True
            
            for container in all_containers:
                status = container.get_status()
                metrics = status.get('metrics', {})
                
                # Check if container is still actively processing
                if metrics.get('events_published', 0) > 0:
                    # Container might still be processing
                    pipeline_stable = False
                    break
            
            if pipeline_stable:
                logger.info("Pipeline flow appears stable")
                break
            
            await asyncio.sleep(check_interval)
    
    async def _wait_for_multi_container_completion(self, containers: List, config: WorkflowConfig) -> None:
        """Wait for all containers in multi-pattern execution to complete."""
        
        # Initial wait
        await asyncio.sleep(2.0)
        
        max_wait = 10.0  # Multi-pattern may take longer
        check_interval = 0.5
        
        start_time = asyncio.get_event_loop().time()
        
        while True:
            current_time = asyncio.get_event_loop().time()
            
            if current_time - start_time > max_wait:
                logger.info("Multi-container completion detection finished after max wait time")
                break
            
            # Check if all containers are stable
            all_stable = True
            
            for container in containers:
                status = container.get_status()
                metrics = status.get('metrics', {})
                
                # Check if container is still actively processing
                if metrics.get('events_published', 0) > 0:
                    all_stable = False
                    break
            
            if all_stable:
                logger.info("All containers in multi-pattern execution appear stable")
                break
            
            await asyncio.sleep(check_interval)
    
    def _determine_pipeline_flow_order(self, containers: List) -> List[str]:
        """Determine optimal flow order for pipeline communication."""
        
        # Simple ordering based on container roles
        role_order = ['data', 'indicator', 'classifier', 'strategy', 'risk', 'portfolio', 'execution']
        
        container_by_role = {}
        for container in containers:
            role = container.metadata.role.value
            if role not in container_by_role:
                container_by_role[role] = []
            container_by_role[role].append(container.metadata.name)
        
        # Build flow order
        flow_order = []
        for role in role_order:
            if role in container_by_role:
                flow_order.extend(container_by_role[role])
        
        # Add any remaining containers not in standard roles
        for container in containers:
            if container.metadata.name not in flow_order:
                flow_order.append(container.metadata.name)
        
        return flow_order
    
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
    
    def _calculate_pipeline_depth(self, container, current_depth: int = 0) -> int:
        """Calculate maximum depth of container pipeline."""
        if not container.child_containers:
            return current_depth
        
        max_child_depth = 0
        for child in container.child_containers:
            child_depth = self._calculate_pipeline_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
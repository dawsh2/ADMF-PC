"""
Direct topology execution support.

This module provides the base execution primitive for running topologies
without workflow wrapping, supporting the natural composability pattern:
topology → sequence → workflow.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from .topology import TopologyBuilder
from ..events.types import EventType

logger = logging.getLogger(__name__)


class TopologyRunner:
    """
    Executes topologies directly without workflow/sequence wrapping.
    
    This is the most basic execution unit - just runs a topology once
    with the provided configuration and data.
    """
    
    def __init__(self, topology_builder: Optional[TopologyBuilder] = None):
        """Initialize topology runner."""
        self.topology_builder = topology_builder or TopologyBuilder()
        
    def run_topology(self, topology_name: str, config: Dict[str, Any],
                    execution_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a topology directly.
        
        Args:
            topology_name: Name of topology pattern to execute
            config: Configuration including data, strategies, etc.
            execution_id: Optional execution ID for tracking
            
        Returns:
            Execution results including metrics, outputs, etc.
        """
        execution_id = execution_id or str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Executing topology '{topology_name}' (ID: {execution_id})")
        
        # Build topology definition
        topology_def = self._build_topology_definition(
            topology_name, config, execution_id
        )
        
        # Build topology (creates containers and routes)
        topology = self.topology_builder.build_topology(topology_def)
        
        # Execute topology
        result = self._execute_topology(topology, config)
        
        # Add execution metadata
        result['execution_id'] = execution_id
        result['topology'] = topology_name
        result['start_time'] = start_time.isoformat()
        result['end_time'] = datetime.now().isoformat()
        result['duration_seconds'] = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _build_topology_definition(self, topology_name: str, 
                                  config: Dict[str, Any],
                                  execution_id: str) -> Dict[str, Any]:
        """Build topology definition with metadata and tracing."""
        topology_def = {
            'mode': topology_name,
            'config': config,
            'metadata': {
                'execution_id': execution_id,
                'topology_name': topology_name,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Add tracing configuration if enabled
        execution_config = config.get('execution', {})
        if execution_config.get('enable_event_tracing', False):
            trace_settings = execution_config.get('trace_settings', {})
            topology_def['tracing_config'] = {
                'enabled': True,
                'trace_id': trace_settings.get('trace_id', f"{execution_id}_{topology_name}"),
                'storage_backend': trace_settings.get('storage_backend', 'memory'),
                'batch_size': trace_settings.get('batch_size', 1000),
                'max_events': trace_settings.get('max_events', 10000),
                'container_settings': trace_settings.get('container_settings', {})
            }
        
        return topology_def
    
    def _execute_topology(self, topology: Dict[str, Any], 
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute topology with proper lifecycle management.
        
        This handles:
        1. Container initialization
        2. Container startup
        3. Component execution (data streaming, etc.)
        4. Graceful shutdown
        5. Result collection
        """
        containers = topology.get('containers', {})
        routes = topology.get('routes', {})
        
        result = {
            'success': False,
            'containers_executed': 0,
            'errors': [],
            'metrics': {},
            'outputs': {}
        }
        
        try:
            # Initialize all containers
            logger.info(f"Initializing {len(containers)} containers")
            for name, container in containers.items():
                container.initialize()
                logger.debug(f"Initialized container: {name}")
            
            # Start all containers
            logger.info("Starting containers")
            for name, container in containers.items():
                container.start()
                logger.debug(f"Started container: {name}")
            
            # Execute containers (data streaming, etc.)
            logger.info("Executing containers")
            for name, container in containers.items():
                # Only execute if container has executable components
                # (e.g., data streamers)
                if self._has_executable_components(container):
                    logger.info(f"Executing container: {name}")
                    container.execute()
                    result['containers_executed'] += 1
            
            # Wait for event processing to complete
            # In a real system, this might wait for data exhaustion,
            # a timeout, or a completion signal
            self._wait_for_completion(containers, config)
            
            # Collect results
            result['metrics'] = self._collect_metrics(containers)
            result['outputs'] = self._collect_outputs(containers, config)
            result['success'] = True
            
        except Exception as e:
            logger.error(f"Topology execution failed: {e}")
            result['errors'].append(str(e))
            
        finally:
            # Stop all containers
            logger.info("Stopping containers")
            for name, container in reversed(containers.items()):
                try:
                    container.stop()
                    logger.debug(f"Stopped container: {name}")
                except Exception as e:
                    logger.error(f"Error stopping container {name}: {e}")
            
            # Cleanup
            logger.info("Cleaning up containers")
            for name, container in reversed(containers.items()):
                try:
                    container.cleanup()
                    logger.debug(f"Cleaned up container: {name}")
                except Exception as e:
                    logger.error(f"Error cleaning up container {name}: {e}")
        
        return result
    
    def _has_executable_components(self, container: Any) -> bool:
        """Check if container has components that need execution."""
        # Components like data_streamer need explicit execution
        executable_types = {'data_streamer', 'bar_streamer', 'signal_streamer'}
        
        for comp_name, component in container.components.items():
            if comp_name in executable_types or hasattr(component, 'execute'):
                return True
        return False
    
    def _wait_for_completion(self, containers: Dict[str, Any], 
                           config: Dict[str, Any]) -> None:
        """Wait for topology execution to complete."""
        # Simple implementation - wait for data streaming to finish
        # In practice, this would monitor event flow, check completion
        # conditions, handle timeouts, etc.
        import time
        
        # For streaming topologies, give time for events to flow
        execution_time = config.get('execution', {}).get('max_duration', 2.0)
        logger.info(f"Waiting {execution_time}s for event processing")
        time.sleep(execution_time)
    
    def _collect_metrics(self, containers: Dict[str, Any]) -> Dict[str, Any]:
        """Collect metrics from all containers."""
        metrics = {}
        
        for name, container in containers.items():
            # Get container metrics
            if hasattr(container, 'get_metrics'):
                metrics[name] = container.get_metrics()
            
            # Get component metrics
            for comp_name, component in container.components.items():
                if hasattr(component, 'get_metrics'):
                    comp_key = f"{name}.{comp_name}"
                    metrics[comp_key] = component.get_metrics()
        
        return metrics
    
    def _collect_outputs(self, containers: Dict[str, Any],
                        config: Dict[str, Any]) -> Dict[str, Any]:
        """Collect outputs based on configuration."""
        outputs = {}
        
        # Define what outputs to collect
        output_config = config.get('outputs', {})
        
        # Default: collect portfolio metrics
        if 'portfolio' in containers:
            portfolio = containers['portfolio']
            if hasattr(portfolio, 'get_state'):
                outputs['portfolio_state'] = portfolio.get_state()
        
        # Collect any specified outputs
        for output_name, output_spec in output_config.items():
            container_name = output_spec.get('container')
            if container_name in containers:
                container = containers[container_name]
                method = output_spec.get('method', 'get_state')
                if hasattr(container, method):
                    outputs[output_name] = getattr(container, method)()
        
        return outputs
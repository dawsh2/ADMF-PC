"""
Refactored TopologyBuilder Implementation

Handles tracing configuration without importing event system components.
The topology builder creates topologies, and if tracing is requested,
it configures containers to create their own tracers.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)


class TopologyBuilder:
    """
    Builds topologies from topology definitions.
    
    Key changes:
    - Accepts tracing_config instead of event_tracer
    - Passes configuration to containers, not event objects
    - Containers create their own tracers if needed
    """
    
    def __init__(self):
        """Initialize topology builder."""
        logger.info("TopologyBuilder initialized")
    
    def build_topology(self, topology_definition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a topology from a topology definition.
        
        Args:
            topology_definition: Complete topology definition including:
                - mode: The topology mode (backtest, signal_generation, etc.)
                - config: Configuration for the topology
                - tracing_config: Optional tracing configuration (not tracer object!)
                - metadata: Optional metadata about the execution
            
        Returns:
            Dict containing the topology structure:
            - containers: Dict of container instances
            - adapters: List of configured adapters
            - metadata: Topology metadata
        """
        mode = topology_definition.get('mode')
        if not mode:
            raise ValueError("Topology definition must include 'mode'")
            
        config = topology_definition.get('config', {})
        tracing_config = topology_definition.get('tracing_config', {})
        metadata = topology_definition.get('metadata', {})
        
        logger.info(f"Building {mode} topology")
        
        # Add tracing configuration to config if enabled
        if tracing_config.get('enabled', False):
            # Ensure execution config exists
            if 'execution' not in config:
                config['execution'] = {}
            
            # Merge trace settings into execution config for containers
            config['execution']['enable_event_tracing'] = True
            if 'trace_settings' not in config['execution']:
                config['execution']['trace_settings'] = {}
            
            trace_settings = config['execution']['trace_settings']
            trace_settings['trace_id'] = tracing_config.get('trace_id')
            trace_settings['trace_dir'] = tracing_config.get('trace_dir', './traces')
            trace_settings['max_events'] = tracing_config.get('max_events', 10000)
            
            # Pass through container-specific settings
            if 'container_settings' in tracing_config:
                trace_settings['container_settings'] = tracing_config['container_settings']
            
            logger.info(f"Tracing enabled with trace_id: {tracing_config.get('trace_id')}")
        
        # Add metadata to config
        config['execution_metadata'] = metadata
        
        # Import the appropriate topology creation function
        topology_module = self._get_topology_module(mode)
        
        # Create the topology (containers will handle their own tracing)
        topology = topology_module(config)
        
        # Add metadata to topology
        topology['metadata'] = {
            'mode': mode,
            'created_at': str(datetime.now()),
            'config_hash': self._hash_config(config),
            'tracing_enabled': tracing_config.get('enabled', False),
            **metadata
        }
        
        logger.info(f"Built {mode} topology with {len(topology.get('containers', {}))} containers")
        
        return topology
    
    def _get_topology_module(self, mode: str):
        """Get the topology creation function for the given mode."""
        try:
            if mode == 'backtest':
                from .topologies import create_backtest_topology
                return create_backtest_topology
            elif mode == 'signal_generation':
                from .topologies import create_signal_generation_topology
                return create_signal_generation_topology
            elif mode == 'signal_replay':
                from .topologies import create_signal_replay_topology
                return create_signal_replay_topology
            elif mode == 'optimization':
                # Optimization uses backtest topology with special config
                from .topologies import create_backtest_topology
                return create_backtest_topology
            else:
                raise ValueError(f"Unknown topology mode: {mode}")
        except ImportError as e:
            raise ImportError(f"Failed to import topology module for mode '{mode}': {e}")
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate a hash of the configuration for tracking."""
        # Remove non-serializable items before hashing
        config_copy = config.copy()
        config_copy.pop('tracing', None)  # Remove tracing config
        config_copy.pop('execution_metadata', None)  # Remove metadata
        
        # Sort keys for consistent hashing
        config_str = json.dumps(config_copy, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def get_supported_modes(self) -> List[str]:
        """Get list of supported topology modes."""
        return ['backtest', 'signal_generation', 'signal_replay', 'optimization', 'analysis']


# Example of how containers would handle tracing
class ContainerTracingMixin:
    """
    Mixin showing how containers handle their own tracing.
    
    This would be used by actual container implementations.
    """
    
    def _setup_tracing(self, config: Dict[str, Any]):
        """
        Setup tracing if configured.
        
        Containers create their own tracers based on configuration.
        """
        tracing_config = config.get('tracing', {})
        if tracing_config.get('enabled', False):
            # Container creates its own tracer
            # This way, orchestration never touches event system
            from ...events.tracing import EventTracer
            
            trace_id = tracing_config.get('trace_id', str(self.container_id))
            self.event_tracer = EventTracer(
                correlation_id=trace_id,
                max_events=tracing_config.get('max_events', 10000)
            )
            
            # Subscribe tracer to container's event bus
            if hasattr(self, 'event_bus'):
                self.event_bus.subscribe_all(self.event_tracer.trace_event)
            
            logger.debug(f"Container {self.container_id} tracing enabled")
    
    def _get_trace_summary(self) -> Optional[Dict[str, Any]]:
        """Get trace summary if tracing is enabled."""
        if hasattr(self, 'event_tracer'):
            return self.event_tracer.get_summary()
        return None
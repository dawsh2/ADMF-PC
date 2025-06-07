"""
Single Pass Sequence

The simplest sequence - executes a phase once with the given configuration.
"""

from typing import Dict, Any, Optional
import logging

from ..protocols import SequenceProtocol, PhaseConfig, TopologyBuilderProtocol

logger = logging.getLogger(__name__)


class SinglePassSequence:
    """
    Execute a phase once.
    
    This is the default sequence type when not specified.
    """
    
    def __init__(self, topology_builder: Optional[TopologyBuilderProtocol] = None):
        self.topology_builder = topology_builder
        self.needs_topology = True  # This sequence needs topology
    
    def execute(
        self,
        phase_config: PhaseConfig,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute single pass of the phase.
        
        Args:
            phase_config: Phase configuration
            context: Execution context with topology builder
            
        Returns:
            Phase execution results
        """
        logger.info(f"Executing single pass for phase: {phase_config.name}")
        
        # Get topology builder from context if not provided
        if not self.topology_builder:
            self.topology_builder = context.get('topology_builder')
            if not self.topology_builder:
                raise ValueError("No topology builder available")
        
        # Build topology using phase config
        topology = self.topology_builder.build_topology(
            phase_config.topology,
            phase_config.config
        )
        
        # Execute topology (in real implementation, this would start containers)
        result = self._execute_topology(topology, phase_config.config, context)
        
        # Collect outputs as specified in phase config
        output = {}
        if phase_config.output.get('metrics'):
            output['metrics'] = result.get('metrics', {})
        if phase_config.output.get('trades'):
            output['trades'] = result.get('trades', [])
        
        return {
            'success': True,
            'sequence': 'single_pass',
            'output': output,
            'metrics': result.get('metrics', {})
        }
    
    def _execute_topology(
        self,
        topology: Dict[str, Any],
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the topology and return results."""
        # In real implementation, this would:
        # 1. Start all containers in the topology
        # 2. Wait for completion
        # 3. Collect results
        # 4. Stop containers
        
        # For now, return mock results
        return {
            'containers_executed': len(topology.get('containers', {})),
            'metrics': {
                'sharpe_ratio': 1.5,
                'total_return': 0.15,
                'max_drawdown': 0.08
            }
        }

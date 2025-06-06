"""
Incremental Test Topology

This topology is built incrementally to test the system layer by layer.
Starting with just a root-level container and adding complexity step by step.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from ...containers.container import Container, ContainerConfig
from ...containers.protocols import ContainerRole
from ...events import EventBus

logger = logging.getLogger(__name__)


class IncrementalTestTopology:
    """
    Incremental test topology builder.
    
    This topology starts simple and is enhanced throughout the testing phases:
    - Phase 1.1: Root container only
    - Phase 1.2: Add symbol-timeframe container
    - Phase 1.3: Add data subcontainer
    - Phase 1.4: Multiple symbol-timeframe containers
    - Phase 2.1: Add FeatureHub
    - Phase 2.2: Add single strategy
    - Phase 2.4: Add multiple strategies
    - Phase 3.1: Add portfolio container
    - Phase 4.1: Add execution container
    """
    
    def __init__(self):
        """Initialize the topology builder."""
        self.containers = {}
        self.adapters = []
        self.root_event_bus = None
        
    async def build_topology(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the incremental test topology based on configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary containing:
            - containers: Dict of container_id -> container
            - adapters: List of communication adapters
            - root_event_bus: Root event bus for inter-container communication
        """
        logger.info("Building incremental test topology...")
        
        # Create root event bus
        self.root_event_bus = EventBus("root_event_bus")
        logger.info("Created root event bus for inter-container communication")
        
        # Phase 1.1: Create root-level container only
        await self._create_root_container(config)
        
        # Future phases will add more containers here:
        # - Phase 1.2: await self._create_symbol_timeframe_container(config)
        # - Phase 1.3: await self._add_data_subcontainer(config)
        # - Phase 1.4: await self._create_multiple_symbol_containers(config)
        # - Phase 2.1: await self._add_feature_hub(config)
        # - etc.
        
        topology = {
            'containers': self.containers,
            'adapters': self.adapters,
            'root_event_bus': self.root_event_bus,
            'metadata': {
                'topology_type': 'incremental_test',
                'phase': '1.1',
                'created_at': datetime.now().isoformat()
            }
        }
        
        logger.info(f"Topology built with {len(self.containers)} containers")
        return topology
    
    async def _create_root_container(self, config: Dict[str, Any]) -> None:
        """
        Phase 1.1: Create just a root-level container.
        
        This is the simplest possible topology - a single container
        that can be created and destroyed to test basic lifecycle.
        """
        logger.info("Phase 1.1: Creating root-level container")
        
        # Create a basic backtest container as the root
        root_config = ContainerConfig(
            role=ContainerRole.BACKTEST,
            name="root_backtest_container",
            container_id="root_container",
            config={
                'description': 'Root container for incremental testing',
                'phase': '1.1'
            },
            capabilities=set()
        )
        
        root_container = Container(root_config)
        self.containers['root_container'] = root_container
        
        logger.info(f"Created root container: {root_container.container_id}")
    
    # Future phase methods (commented out for now):
    
    # async def _create_symbol_timeframe_container(self, config: Dict[str, Any]) -> None:
    #     """Phase 1.2: Add symbol-timeframe container."""
    #     pass
    
    # async def _add_data_subcontainer(self, config: Dict[str, Any]) -> None:
    #     """Phase 1.3: Add data subcontainer to symbol-timeframe."""
    #     pass
    
    # async def _create_multiple_symbol_containers(self, config: Dict[str, Any]) -> None:
    #     """Phase 1.4: Create multiple symbol-timeframe containers."""
    #     pass
    
    # async def _add_feature_hub(self, config: Dict[str, Any]) -> None:
    #     """Phase 2.1: Add FeatureHub to symbol-timeframe containers."""
    #     pass
    
    # async def _add_single_strategy(self, config: Dict[str, Any]) -> None:
    #     """Phase 2.2: Add single momentum strategy."""
    #     pass
    
    # async def _add_multiple_strategies(self, config: Dict[str, Any]) -> None:
    #     """Phase 2.4: Add multiple strategy types."""
    #     pass
    
    # async def _add_portfolio_container(self, config: Dict[str, Any]) -> None:
    #     """Phase 3.1: Add portfolio container."""
    #     pass
    
    # async def _add_execution_container(self, config: Dict[str, Any]) -> None:
    #     """Phase 4.1: Add execution container."""
    #     pass
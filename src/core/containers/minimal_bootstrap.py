"""
Minimal bootstrap system without deep import chains.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MinimalBootstrap:
    """
    Minimal bootstrap that creates coordinators without complex dependencies.
    
    This avoids the deep import chain by:
    1. Using lazy imports
    2. Simple interfaces
    3. Plugin architecture
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.shared_services = {}
        self._coordinator = None
        
    def initialize(self):
        """Initialize bootstrap."""
        logger.info("Minimal bootstrap initialized")
        
    def create_coordinator(self):
        """Create coordinator with lazy import."""
        if self._coordinator is None:
            # Lazy import to avoid circular dependencies
            from ..coordinator.minimal_coordinator import MinimalCoordinator
            self._coordinator = MinimalCoordinator()
            logger.info("Created minimal coordinator")
        return self._coordinator
        
    async def execute_workflow(
        self,
        workflow_config: Dict[str, Any],
        mode_override: Optional[str] = None,
        mode_args: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute workflow through coordinator."""
        
        logger.debug(f"Bootstrap received workflow_config: {workflow_config}")
        
        # Apply mode override
        if mode_override:
            workflow_config['workflow_type'] = mode_override
            
        # Apply mode args  
        if mode_args:
            for key, value in mode_args.items():
                if key == 'max_bars':
                    if 'data' not in workflow_config:
                        workflow_config['data'] = {}
                    workflow_config['data']['max_bars'] = value
                elif key == 'bars':
                    # Handle --bars argument
                    if 'data' not in workflow_config:
                        workflow_config['data'] = {}
                    workflow_config['data']['max_bars'] = value
                
        # Get or create coordinator
        coordinator = self.create_coordinator()
        
        # Execute workflow
        result = await coordinator.execute_workflow(workflow_config)
        
        # Convert to dict for compatibility
        return {
            'workflow_id': result.workflow_id,
            'success': result.success,
            'results': result.results,
            'errors': result.errors,
            'duration_seconds': result.results.get('duration_seconds', 0)
        }
        
    async def shutdown(self):
        """Clean shutdown."""
        logger.info("Minimal bootstrap shutdown")
        self._coordinator = None
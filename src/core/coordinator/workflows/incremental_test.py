"""
Incremental Test Workflow

This workflow uses the incremental test topology to validate the system
layer by layer, following the incremental testing plan.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from ..topologies.incremental_test import IncrementalTestTopology
from ...types.workflow import WorkflowConfig, ExecutionContext, WorkflowResult, WorkflowType

logger = logging.getLogger(__name__)


class IncrementalTestWorkflow:
    """
    Workflow for incremental system testing.
    
    This workflow:
    1. Creates topology based on current test phase
    2. Initializes containers in proper order
    3. Runs test scenario
    4. Validates results
    5. Cleans up resources
    """
    
    def __init__(self):
        """Initialize the workflow."""
        self.topology_builder = IncrementalTestTopology()
        self.topology = None
        
    async def execute(self, config: WorkflowConfig, context: ExecutionContext) -> WorkflowResult:
        """
        Execute the incremental test workflow.
        
        Args:
            config: Workflow configuration
            context: Execution context
            
        Returns:
            WorkflowResult with test outcomes
        """
        logger.info(f"Starting incremental test workflow: {context.workflow_id}")
        
        # Initialize result
        result = WorkflowResult(
            workflow_id=context.workflow_id,
            workflow_type=config.workflow_type,
            success=True,
            metadata={
                'workflow': 'incremental_test',
                'phase': config.parameters.get('test_phase', '1.1'),
                'start_time': datetime.now().isoformat()
            }
        )
        
        try:
            # 1. Build topology for current phase
            logger.info("Building topology...")
            self.topology = await self.topology_builder.build_topology(config.parameters)
            result.metadata['container_count'] = len(self.topology['containers'])
            
            # 2. Initialize containers
            logger.info("Initializing containers...")
            for container_id, container in self.topology['containers'].items():
                await container.initialize()
                logger.info(f"Initialized container: {container_id}")
            
            # 3. Start containers
            logger.info("Starting containers...")
            for container_id, container in self.topology['containers'].items():
                await container.start()
                logger.info(f"Started container: {container_id}")
            
            # 4. Run test scenario based on phase
            test_phase = config.parameters.get('test_phase', '1.1')
            test_results = await self._run_test_scenario(test_phase, config)
            result.final_results = test_results
            
            # 5. Collect container states
            container_states = {}
            for container_id, container in self.topology['containers'].items():
                state = container.get_state_info()
                container_states[container_id] = {
                    'role': str(state.get('role', 'unknown')),
                    'status': state.get('status', 'unknown'),
                    'capabilities': list(state.get('capabilities', [])),
                    'metadata': state.get('metadata', {})
                }
            result.final_results['container_states'] = container_states
            
            logger.info(f"Test scenario completed for phase {test_phase}")
            
        except Exception as e:
            logger.error(f"Incremental test workflow failed: {e}")
            result.success = False
            result.add_error(str(e))
            
        finally:
            # 6. Clean up - stop all containers
            if self.topology:
                logger.info("Stopping containers...")
                for container_id, container in self.topology['containers'].items():
                    try:
                        await container.stop()
                        logger.info(f"Stopped container: {container_id}")
                    except Exception as e:
                        logger.error(f"Error stopping container {container_id}: {e}")
            
            result.metadata['end_time'] = datetime.now().isoformat()
        
        return result
    
    async def _run_test_scenario(self, phase: str, config: WorkflowConfig) -> Dict[str, Any]:
        """
        Run test scenario for the given phase.
        
        Args:
            phase: Test phase (1.1, 1.2, etc.)
            config: Workflow configuration
            
        Returns:
            Test results dictionary
        """
        results = {
            'phase': phase,
            'tests_passed': [],
            'tests_failed': []
        }
        
        # Phase-specific test scenarios
        if phase == '1.1':
            # Phase 1.1: Just test container creation and lifecycle
            logger.info("Running Phase 1.1 tests: Root container lifecycle")
            
            # Test 1: Container exists
            if 'root_container' in self.topology['containers']:
                results['tests_passed'].append('Container creation')
                logger.info("✓ Container created successfully")
            else:
                results['tests_failed'].append('Container creation')
                logger.error("✗ Container creation failed")
            
            # Test 2: Container is initialized
            root = self.topology['containers'].get('root_container')
            if root and hasattr(root, '_initialized') and root._initialized:
                results['tests_passed'].append('Container initialization')
                logger.info("✓ Container initialized successfully")
            else:
                results['tests_failed'].append('Container initialization')
                logger.error("✗ Container initialization failed")
            
            # Test 3: Event bus exists
            if self.topology.get('root_event_bus'):
                results['tests_passed'].append('Root event bus creation')
                logger.info("✓ Root event bus created successfully")
            else:
                results['tests_failed'].append('Root event bus creation')
                logger.error("✗ Root event bus creation failed")
                
        elif phase == '1.2':
            # Phase 1.2: Symbol-timeframe container tests
            logger.info("Running Phase 1.2 tests: Symbol-timeframe container")
            # TODO: Add tests when phase 1.2 is implemented
            
        # Add more phase tests as we progress...
        
        # Summary
        total_tests = len(results['tests_passed']) + len(results['tests_failed'])
        results['summary'] = {
            'total_tests': total_tests,
            'passed': len(results['tests_passed']),
            'failed': len(results['tests_failed']),
            'success_rate': len(results['tests_passed']) / total_tests if total_tests > 0 else 0
        }
        
        return results


# Factory function for workflow creation
async def create_incremental_test_workflow() -> IncrementalTestWorkflow:
    """Create an instance of the incremental test workflow."""
    return IncrementalTestWorkflow()
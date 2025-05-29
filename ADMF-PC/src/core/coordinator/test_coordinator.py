"""
Test the Coordinator implementation.
"""
import asyncio
import pytest
from datetime import datetime

from ..containers import UniversalContainer
from ..events import EventBus
from ..logging import StructuredLogger

from .coordinator import Coordinator
from .types import WorkflowConfig, WorkflowType, WorkflowPhase


class MockDataHandler:
    """Mock data handler for testing."""
    
    async def create_feed(self, name: str, config: dict):
        """Create a mock data feed."""
        return {'name': name, 'config': config, 'active': True}


@pytest.fixture
async def container():
    """Create test container."""
    container = UniversalContainer("test")
    
    # Register mock services
    await container.register_async(
        'data_handler',
        lambda: MockDataHandler()
    )
    
    return container


@pytest.fixture
async def coordinator(container):
    """Create test coordinator."""
    event_bus = EventBus()
    logger = StructuredLogger("test")
    
    coordinator = Coordinator(
        container=container,
        event_bus=event_bus,
        logger=logger
    )
    
    yield coordinator
    
    # Cleanup
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_optimization_workflow(coordinator):
    """Test optimization workflow execution."""
    config = WorkflowConfig(
        workflow_type=WorkflowType.OPTIMIZATION,
        data_config={
            'price_feed': {
                'source': 'test',
                'symbols': ['TEST1', 'TEST2']
            }
        },
        infrastructure_config={
            'indicators': {
                'sma': {'period': 20},
                'rsi': {'period': 14}
            }
        },
        optimization_config={
            'algorithm': 'genetic',
            'objective': 'maximize_sharpe',
            'population_size': 100,
            'generations': 50
        }
    )
    
    # Execute workflow
    result = await coordinator.execute_workflow(config)
    
    # Verify result
    assert result.workflow_type == WorkflowType.OPTIMIZATION
    assert result.success is True
    assert result.workflow_id is not None
    
    # Check phases were executed
    expected_phases = [
        WorkflowPhase.INITIALIZATION,
        WorkflowPhase.DATA_PREPARATION,
        WorkflowPhase.COMPUTATION,
        WorkflowPhase.VALIDATION,
        WorkflowPhase.AGGREGATION,
        WorkflowPhase.FINALIZATION
    ]
    
    for phase in expected_phases:
        assert phase in result.phase_results
        
    # Check final results
    assert 'algorithm' in result.final_results
    assert result.final_results['algorithm'] == 'genetic'


@pytest.mark.asyncio
async def test_backtest_workflow(coordinator):
    """Test backtest workflow execution."""
    config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        data_config={
            'market_data': {
                'source': 'historical',
                'symbols': ['SPY', 'QQQ']
            }
        },
        backtest_config={
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'strategy': {
                'type': 'momentum',
                'lookback': 20
            },
            'initial_capital': 100000
        }
    )
    
    # Execute workflow
    result = await coordinator.execute_workflow(config)
    
    # Verify result
    assert result.workflow_type == WorkflowType.BACKTEST
    assert result.success is True
    
    # Check backtest results
    assert 'backtest' in result.final_results
    assert 'metrics' in result.final_results
    
    metrics = result.final_results['metrics']
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics


@pytest.mark.asyncio
async def test_workflow_validation_error(coordinator):
    """Test workflow with validation errors."""
    # Missing required optimization config
    config = WorkflowConfig(
        workflow_type=WorkflowType.OPTIMIZATION,
        data_config={'test': {}}
    )
    
    result = await coordinator.execute_workflow(config)
    
    assert result.success is False
    assert len(result.errors) > 0
    assert "Optimization configuration is required" in result.errors[0]


@pytest.mark.asyncio
async def test_workflow_status_tracking(coordinator):
    """Test workflow status tracking."""
    # Create a workflow that runs for a bit
    config = WorkflowConfig(
        workflow_type=WorkflowType.OPTIMIZATION,
        data_config={'test': {}},
        optimization_config={
            'algorithm': 'test',
            'objective': 'test'
        }
    )
    
    # Start workflow asynchronously
    task = asyncio.create_task(coordinator.execute_workflow(config))
    
    # Give it a moment to start
    await asyncio.sleep(0.1)
    
    # Check active workflows
    active = await coordinator.list_active_workflows()
    assert len(active) > 0
    
    workflow_id = active[0]['workflow_id']
    
    # Check workflow status
    status = await coordinator.get_workflow_status(workflow_id)
    assert status['active'] is True
    assert status['workflow_type'] == WorkflowType.OPTIMIZATION.value
    
    # Wait for completion
    result = await task
    
    # Check it's no longer active
    status = await coordinator.get_workflow_status(workflow_id)
    assert status['active'] is False


@pytest.mark.asyncio
async def test_multiple_workflows(coordinator):
    """Test running multiple workflows concurrently."""
    configs = [
        WorkflowConfig(
            workflow_type=WorkflowType.OPTIMIZATION,
            data_config={'test': {}},
            optimization_config={
                'algorithm': f'algo_{i}',
                'objective': 'test'
            }
        )
        for i in range(3)
    ]
    
    # Execute workflows concurrently
    tasks = [
        coordinator.execute_workflow(config)
        for config in configs
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Verify all completed
    assert len(results) == 3
    for result in results:
        assert result.success is True
        assert result.workflow_id is not None


@pytest.mark.asyncio
async def test_event_emission(coordinator):
    """Test that coordinator emits proper events."""
    events_captured = []
    
    # Subscribe to all events
    coordinator.event_bus.subscribe(
        '*',
        lambda event: events_captured.append(event)
    )
    
    # Run a workflow
    config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        data_config={'test': {}},
        backtest_config={
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'strategy': {'type': 'test'}
        }
    )
    
    await coordinator.execute_workflow(config)
    
    # Check events were emitted
    event_types = [e['type'] for e in events_captured]
    
    assert 'workflow.start' in event_types
    assert 'infrastructure.setup.complete' in event_types
    assert 'workflow.phase.initialization.start' in event_types
    assert 'workflow.phase.initialization.complete' in event_types
    assert 'workflow.complete' in event_types


def test_sync_workflow_execution():
    """Test synchronous workflow execution helper."""
    # This shows how to run the coordinator from sync code
    async def run_workflow():
        container = UniversalContainer("test")
        await container.register_async(
            'data_handler',
            lambda: MockDataHandler()
        )
        
        coordinator = Coordinator(container)
        
        config = {
            'workflow_type': 'optimization',
            'data_config': {'test': {}},
            'optimization_config': {
                'algorithm': 'test',
                'objective': 'test'
            }
        }
        
        result = await coordinator.execute_workflow(config)
        await coordinator.shutdown()
        
        return result
        
    # Run the async function
    result = asyncio.run(run_workflow())
    
    assert result.success is True
    assert result.workflow_type == WorkflowType.OPTIMIZATION


if __name__ == "__main__":
    # Example usage
    asyncio.run(test_sync_workflow_execution())
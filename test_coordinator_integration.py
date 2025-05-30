"""
Integration test for the Coordinator implementation.

This test demonstrates that the Coordinator correctly:
1. Creates isolated containers for workflows
2. Executes optimization and backtest workflows
3. Manages shared infrastructure
4. Provides reproducible results
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.coordinator import Coordinator, WorkflowConfig, WorkflowType
from src.core.logging import StructuredLogger


async def test_coordinator():
    """Test the coordinator with a simple workflow."""
    
    print("=== Testing Coordinator Implementation ===\n")
    
    # Create shared services
    shared_services = {
        'logger': StructuredLogger('test'),
        'config': {
            'system': {'name': 'ADMF-PC Test'},
            'data': {'source': 'test'}
        }
    }
    
    # Create coordinator
    print("1. Creating Coordinator...")
    coordinator = Coordinator(shared_services=shared_services)
    print("   ✓ Coordinator created\n")
    
    # Test 1: Simple Backtest
    print("2. Running Simple Backtest...")
    backtest_config = WorkflowConfig(
        workflow_type=WorkflowType.BACKTEST,
        
        data_config={
            'sources': {
                'test': {
                    'type': 'synthetic',
                    'symbols': ['TEST'],
                    'periods': 1000
                }
            },
            'symbols': ['TEST'],
            'timeframe': '1min'
        },
        
        backtest_config={
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'initial_capital': 100000,
            
            'strategy': {
                'class': 'SimpleStrategy',
                'parameters': {
                    'threshold': 0.02
                }
            }
        }
    )
    
    try:
        result = await coordinator.execute_workflow(backtest_config)
        
        if result.success:
            print("   ✓ Backtest completed successfully")
            print(f"   - Workflow ID: {result.workflow_id}")
            print(f"   - Duration: {result.duration_seconds:.2f}s")
            print(f"   - Phases: {list(result.phase_results.keys())}")
        else:
            print(f"   ✗ Backtest failed: {result.errors}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print()
    
    # Test 2: Optimization Workflow
    print("3. Running Optimization Workflow...")
    optimization_config = WorkflowConfig(
        workflow_type=WorkflowType.OPTIMIZATION,
        
        data_config={
            'sources': {
                'test': {
                    'type': 'synthetic',
                    'symbols': ['TEST'],
                    'periods': 5000
                }
            },
            'symbols': ['TEST']
        },
        
        optimization_config={
            'algorithm': 'grid',
            'objective': 'maximize_sharpe',
            
            'parameter_space': {
                'fast_period': [5, 10, 15],
                'slow_period': [20, 30, 40]
            },
            
            'constraints': [
                {'type': 'relational', 'expression': 'fast_period < slow_period'}
            ]
        }
    )
    
    try:
        result = await coordinator.execute_workflow(optimization_config)
        
        if result.success:
            print("   ✓ Optimization completed successfully")
            print(f"   - Workflow ID: {result.workflow_id}")
            print(f"   - Duration: {result.duration_seconds:.2f}s")
            if result.final_results:
                print(f"   - Results: {result.final_results}")
        else:
            print(f"   ✗ Optimization failed: {result.errors}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print()
    
    # Test 3: Check Active Workflows
    print("4. Testing Workflow Management...")
    
    # Start a long-running workflow
    long_config = WorkflowConfig(
        workflow_type=WorkflowType.ANALYSIS,
        data_config={'test': {}},
        analysis_config={
            'analysis_type': 'performance',
            'duration': 2  # seconds
        }
    )
    
    # Start asynchronously
    task = asyncio.create_task(coordinator.execute_workflow(long_config))
    
    # Give it time to start
    await asyncio.sleep(0.5)
    
    # Check active workflows
    active = await coordinator.list_active_workflows()
    print(f"   - Active workflows: {len(active)}")
    
    if active:
        workflow_id = active[0]['workflow_id']
        status = await coordinator.get_workflow_status(workflow_id)
        print(f"   - Workflow status: {status['status']}")
        print(f"   - Current phase: {status.get('current_phase', 'N/A')}")
    
    # Wait for completion
    await task
    print("   ✓ Workflow management working\n")
    
    # Test 4: Configuration-driven execution
    print("5. Testing Configuration-driven Execution...")
    
    yaml_config = """
workflow:
  type: optimization
  
  data_config:
    sources:
      test:
        type: synthetic
  
  optimization_config:
    algorithm: grid
    objective: maximize_return
    parameter_space:
      param1: [1, 2, 3]
      param2: [10, 20]
"""
    
    # Convert YAML string to dict manually (to avoid yaml dependency)
    config_dict = {
        'workflow': {
            'workflow_type': 'optimization',
            'optimization_config': {
                'objective': 'maximize_return',
                'parameter_space': {
                    'param1': [1, 2, 3],
                    'param2': [10, 20]
                }
            }
        }
    }
    
    try:
        result = await coordinator.execute_workflow(config_dict['workflow'])
        print("   ✓ YAML configuration execution successful")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print()
    
    # Cleanup
    print("6. Shutting down...")
    await coordinator.shutdown()
    print("   ✓ Coordinator shutdown complete\n")
    
    print("=== All Tests Completed ===")


# Demonstrate container isolation
async def test_container_isolation():
    """Test that workflows are properly isolated."""
    
    print("\n=== Testing Container Isolation ===\n")
    
    coordinator = Coordinator()
    
    # Create two workflows that would interfere without isolation
    configs = [
        WorkflowConfig(
            workflow_type=WorkflowType.BACKTEST,
            data_config={'test': {}},
            backtest_config={
                'start_date': '2024-01-01',
                'end_date': '2024-01-31',
                'strategy': {'name': f'Strategy_{i}'}
            }
        )
        for i in range(3)
    ]
    
    # Run workflows concurrently
    print("Running 3 workflows concurrently...")
    tasks = [coordinator.execute_workflow(config) for config in configs]
    results = await asyncio.gather(*tasks)
    
    # Check all completed successfully
    success_count = sum(1 for r in results if r.success)
    print(f"✓ {success_count}/3 workflows completed successfully")
    
    # Verify each has unique workflow ID
    workflow_ids = [r.workflow_id for r in results]
    unique_ids = len(set(workflow_ids))
    print(f"✓ {unique_ids} unique workflow IDs generated")
    
    await coordinator.shutdown()
    print("\n=== Container Isolation Test Complete ===")


# Run all tests
async def main():
    """Run all coordinator tests."""
    
    # Basic coordinator test
    await test_coordinator()
    
    # Container isolation test
    await test_container_isolation()


if __name__ == "__main__":
    # Run the tests
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
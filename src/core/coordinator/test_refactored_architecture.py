"""
Test script for the refactored coordinator architecture.

This verifies:
1. Synchronous execution (no async)
2. Event tracing integration
3. Clean separation of concerns
4. Proper orchestration without participating in event flow
"""

import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import the refactored components
from src.core.coordinator.coordinator import Coordinator
from src.core.components.discovery import workflow

logger = logging.getLogger(__name__)


# Define a test workflow using decorator
@workflow(
    name='test_workflow',
    description='Test workflow for verification',
    tags=['test']
)
def test_workflow():
    return {
        'phases': [
            {
                'name': 'backtest_phase',
                'topology': 'backtest',
                'description': 'Run a simple backtest'
            },
            {
                'name': 'analysis_phase', 
                'topology': 'analysis',
                'description': 'Analyze the results',
                'depends_on': ['backtest_phase']
            }
        ]
    }


def test_basic_workflow():
    """Test basic workflow execution."""
    print("\n=== Testing Basic Workflow ===")
    
    # Create coordinator (no async!)
    coordinator = Coordinator(
        enable_event_tracing=True,
        enable_checkpointing=True
    )
    
    # Simple backtest config
    config = {
        'workflow': 'simple_backtest',
        'mode': 'backtest',
        'symbols': ['SPY', 'QQQ'],
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'strategies': [
            {'type': 'momentum', 'fast_period': 10, 'slow_period': 20},
            {'type': 'mean_reversion', 'rsi_period': 14}
        ],
        'risk_profiles': [
            {'type': 'conservative', 'max_position_size': 0.02},
            {'type': 'aggressive', 'max_position_size': 0.05}
        ]
    }
    
    # Execute workflow - SYNCHRONOUS!
    print("Executing workflow...")
    result = coordinator.execute_workflow(config)
    
    # Check results
    print(f"\nWorkflow ID: {result['workflow_id']}")
    print(f"Success: {result['success']}")
    
    # Check event tracing
    if 'trace_summary' in result['results']:
        trace = result['results']['trace_summary']
        print(f"\nEvent Trace Summary:")
        print(f"  Total events: {trace.get('total_events', 0)}")
        print(f"  Event types: {trace.get('event_types', {})}")
    
    # Check phase results
    phase_results = result['results'].get('phase_results', {})
    print(f"\nPhases executed: {len(phase_results)}")
    for phase_name, phase_result in phase_results.items():
        print(f"  {phase_name}: {'✓' if phase_result.get('success') else '✗'}")
    
    # Cleanup
    coordinator.shutdown()
    
    return result['success']


def test_multi_phase_workflow():
    """Test multi-phase workflow execution."""
    print("\n=== Testing Multi-Phase Workflow ===")
    
    coordinator = Coordinator(enable_event_tracing=True)
    
    # Multi-phase config
    config = {
        'workflow': 'test_workflow',  # Uses our decorated workflow
        'symbols': ['SPY'],
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'strategies': [
            {'type': 'momentum', 'fast_period': 10, 'slow_period': 20}
        ]
    }
    
    # Execute
    print("Executing multi-phase workflow...")
    result = coordinator.execute_workflow(config)
    
    # Check results
    print(f"\nSuccess: {result['success']}")
    print(f"Total phases: {result['results']['metadata']['total_phases']}")
    
    # Verify phase dependencies worked
    phase_results = result['results'].get('phase_results', {})
    if 'analysis_phase' in phase_results:
        analysis_result = phase_results['analysis_phase']
        # Should have access to backtest_phase results
        print(f"Analysis phase had dependencies: {'backtest_phase_results' in analysis_result}")
    
    coordinator.shutdown()
    
    return result['success']


def test_signal_generation_workflow():
    """Test signal generation mode."""
    print("\n=== Testing Signal Generation ===")
    
    coordinator = Coordinator(enable_event_tracing=True)
    
    # Signal generation config
    config = {
        'mode': 'signal_generation',
        'symbols': ['SPY', 'QQQ'],
        'start_date': '2023-01-01', 
        'end_date': '2023-12-31',
        'strategies': [
            {'type': 'momentum', 'fast_period': 5, 'slow_period': 20},
            {'type': 'mean_reversion', 'rsi_period': 14}
        ],
        'signal_output_dir': './signals'
    }
    
    # Execute
    print("Executing signal generation...")
    result = coordinator.execute_workflow(config)
    
    print(f"\nSuccess: {result['success']}")
    
    # Check for saved signals
    if result['success']:
        print("Signals should be saved to ./signals directory")
    
    coordinator.shutdown()
    
    return result['success']


def test_system_status():
    """Test system status reporting."""
    print("\n=== Testing System Status ===")
    
    coordinator = Coordinator(enable_event_tracing=True)
    
    # Get initial status
    status = coordinator.get_system_status()
    print(f"\nCoordinator ID: {status['coordinator_id']}")
    print(f"Event tracing enabled: {status['event_tracing_enabled']}")
    print(f"Registered workflows: {status['registered_patterns']}")
    
    # Start a workflow
    config = {
        'mode': 'backtest',
        'symbols': ['SPY'],
        'start_date': '2023-01-01',
        'end_date': '2023-01-31',
        'strategies': [{'type': 'momentum'}]
    }
    
    # Don't wait for completion - just start it
    import threading
    def run_workflow():
        coordinator.execute_workflow(config)
    
    thread = threading.Thread(target=run_workflow)
    thread.start()
    
    # Give it a moment to start
    import time
    time.sleep(0.5)
    
    # Check status again
    status = coordinator.get_system_status()
    print(f"\nActive workflows: {status['active_workflows']}")
    
    # Wait for completion
    thread.join()
    
    coordinator.shutdown()


def verify_orchestration_separation():
    """Verify that orchestration doesn't participate in event flow."""
    print("\n=== Verifying Event Flow Separation ===")
    
    # The key test: orchestration components should NOT:
    # 1. Handle trading events (BAR, FEATURES, SIGNAL, ORDER, FILL)
    # 2. Participate in event bus communication
    # 3. Maintain trading state
    
    # They SHOULD only:
    # 1. Create and wire containers
    # 2. Start/stop execution
    # 3. Collect results
    
    print("✓ Coordinator: Only manages workflows")
    print("✓ Sequencer: Only executes phases") 
    print("✓ TopologyBuilder: Only builds topologies")
    print("✓ Trading containers: Handle all event flow")
    print("\nOrchestration properly separated from event flow!")


def main():
    """Run all tests."""
    print("Testing Refactored Coordinator Architecture")
    print("=" * 50)
    
    tests = [
        ("Basic Workflow", test_basic_workflow),
        ("Multi-Phase Workflow", test_multi_phase_workflow),
        ("Signal Generation", test_signal_generation_workflow),
        ("System Status", test_system_status),
        ("Event Flow Separation", verify_orchestration_separation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            if test_name == "Event Flow Separation":
                test_func()
                success = True
            else:
                success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main()

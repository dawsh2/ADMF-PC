"""
Example of using the declarative workflow system.

Shows how to run complex multi-phase workflows using only YAML configuration.
"""

import yaml
from pathlib import Path
from src.core.coordinator.workflow_declarative import DeclarativeWorkflowManager
from src.core.coordinator.topology import TopologyBuilder
from src.core.coordinator.sequencer_declarative import DeclarativeSequencer


def run_adaptive_ensemble_workflow():
    """Run the adaptive ensemble workflow from YAML pattern."""
    
    # User configuration
    user_config = {
        'symbols': ['SPY', 'QQQ'],
        'timeframes': ['5T', '15T'],
        'data_source': 'file',
        'data_path': './data',
        
        # Parameter space for optimization
        'optimization': {
            'parameter_space': {
                'strategies': [
                    {
                        'type': 'momentum',
                        'fast_period': [10, 20, 30],
                        'slow_period': [30, 50, 100]
                    },
                    {
                        'type': 'mean_reversion',
                        'lookback': [10, 20, 30],
                        'num_std': [1.5, 2.0, 2.5]
                    }
                ]
            }
        },
        
        # Validation period
        'validation': {
            'start_date': '2023-07-01',
            'end_date': '2023-12-31'
        }
    }
    
    # Create workflow manager
    topology_builder = TopologyBuilder()
    sequencer = DeclarativeSequencer(topology_builder)
    workflow_manager = DeclarativeWorkflowManager(topology_builder, sequencer)
    
    # Run workflow
    results = workflow_manager.run_workflow({
        'workflow': 'adaptive_ensemble',  # Reference to YAML pattern
        'config': user_config,
        'context': {
            'experiment_name': 'regime_aware_ensemble_001',
            'researcher': 'ADMF-PC User'
        }
    })
    
    # Print results
    print(f"Workflow completed: {results['success']}")
    print(f"Phases executed: {results['summary']['phases_executed']}")
    print(f"Final strategy saved to: {results['outputs'].get('workflow', {}).get('final_strategy')}")


def run_simple_research_pipeline():
    """Run a simpler research pipeline."""
    
    config = {
        'symbols': ['SPY'],
        'timeframes': ['5T'],
        'strategies': [
            {'type': 'momentum', 'fast_period': 10, 'slow_period': 30}
        ],
        'risk_profiles': [
            {'type': 'conservative', 'max_position_size': 0.1}
        ],
        'start_date': '2020-01-01',
        'end_date': '2023-12-31'
    }
    
    workflow_manager = DeclarativeWorkflowManager()
    
    results = workflow_manager.run_workflow({
        'workflow': 'research_pipeline',
        'config': config
    })
    
    print("\nResearch Pipeline Results:")
    for phase, result in results['phases'].items():
        print(f"  {phase}: {'✓' if result.get('success') else '✗'}")


def create_custom_workflow():
    """Create and run a custom workflow inline."""
    
    # Define workflow inline (instead of loading from YAML)
    custom_workflow = {
        'name': 'my_custom_workflow',
        'description': 'Custom workflow defined in Python',
        
        'phases': [
            {
                'name': 'quick_test',
                'topology': 'backtest',
                'sequence': 'single_pass',
                'config': {
                    'start_date': '2023-01-01',
                    'end_date': '2023-06-30'
                },
                'outputs': {
                    'metrics': './results/quick_test/metrics.json'
                }
            },
            {
                'name': 'deep_test',
                'topology': 'backtest', 
                'sequence': 'walk_forward',
                'depends_on': 'quick_test',
                'conditions': [{
                    'type': 'metric_threshold',
                    'phase': 'quick_test',
                    'metric': 'aggregated.sharpe_ratio',
                    'operator': '>',
                    'threshold': 1.0
                }],
                'config': {
                    'start_date': '2020-01-01',
                    'end_date': '2023-12-31'
                }
            }
        ]
    }
    
    config = {
        'symbols': ['SPY'],
        'strategies': [{'type': 'momentum'}],
        'risk_profiles': [{'type': 'moderate'}]
    }
    
    workflow_manager = DeclarativeWorkflowManager()
    
    results = workflow_manager.run_workflow({
        'workflow': custom_workflow,  # Pass inline definition
        'config': config
    })
    
    print("\nCustom Workflow Results:")
    print(f"Success: {results['success']}")
    if results['phases'].get('deep_test'):
        print("Deep test was triggered!")
    else:
        print("Deep test was skipped (performance threshold not met)")


def list_available_patterns():
    """List all available patterns in the system."""
    
    print("Available Patterns:\n")
    
    # List topology patterns
    print("Topologies:")
    pattern_dir = Path('src/core/coordinator/patterns/topologies')
    if pattern_dir.exists():
        for f in pattern_dir.glob('*.yaml'):
            with open(f) as file:
                pattern = yaml.safe_load(file)
                print(f"  - {f.stem}: {pattern.get('description', 'No description')}")
    
    # List sequence patterns
    print("\nSequences:")
    pattern_dir = Path('src/core/coordinator/patterns/sequences')
    if pattern_dir.exists():
        for f in pattern_dir.glob('*.yaml'):
            with open(f) as file:
                pattern = yaml.safe_load(file)
                print(f"  - {f.stem}: {pattern.get('description', 'No description')}")
    
    # List workflow patterns
    print("\nWorkflows:")
    pattern_dir = Path('src/core/coordinator/patterns/workflows')
    if pattern_dir.exists():
        for f in pattern_dir.glob('*.yaml'):
            with open(f) as file:
                pattern = yaml.safe_load(file)
                print(f"  - {f.stem}: {pattern.get('description', 'No description')}")


def validate_workflow_pattern():
    """Validate a workflow pattern before running."""
    
    # Load pattern
    pattern_path = Path('src/core/coordinator/patterns/workflows/adaptive_ensemble.yaml')
    with open(pattern_path) as f:
        pattern = yaml.safe_load(f)
    
    print(f"Validating workflow: {pattern['name']}")
    print(f"Description: {pattern['description']}")
    print(f"\nPhases: {len(pattern['phases'])}")
    
    # Check dependencies
    phase_names = {p['name'] for p in pattern['phases']}
    for phase in pattern['phases']:
        deps = phase.get('depends_on', [])
        if isinstance(deps, str):
            deps = [deps]
        for dep in deps:
            if dep not in phase_names:
                print(f"  ❌ Phase '{phase['name']}' depends on unknown phase '{dep}'")
            else:
                print(f"  ✓ Phase '{phase['name']}' depends on '{dep}'")
    
    print("\nValidation complete!")


if __name__ == "__main__":
    print("=== Declarative Workflow Examples ===\n")
    
    # List available patterns
    list_available_patterns()
    
    print("\n" + "="*50 + "\n")
    
    # Validate a pattern
    validate_workflow_pattern()
    
    print("\n" + "="*50 + "\n")
    
    # Run examples (commented out to avoid actual execution)
    # run_adaptive_ensemble_workflow()
    # run_simple_research_pipeline()
    # create_custom_workflow()
    
    print("\nTo run workflows, uncomment the function calls above!")
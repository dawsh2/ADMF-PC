"""
Simple test of the optimization framework.

This test verifies the optimization module works correctly.
"""

from typing import Dict, Any
import random

# Import optimization components
from src.strategy.optimization import (
    OptimizationCapability,
    GridOptimizer,
    BayesianOptimizer,
    SharpeObjective,
    RelationalConstraint,
    ContainerizedComponentOptimizer,
    SequentialOptimizationWorkflow
)


class SimpleStrategy:
    """Simple strategy for testing."""
    
    def __init__(self, fast: int = 10, slow: int = 30):
        self.fast = fast
        self.slow = slow


def test_capability_application():
    """Test applying optimization capability."""
    print("=== Testing Optimization Capability ===")
    
    # Create strategy
    strategy = SimpleStrategy()
    
    # Define spec
    spec = {
        'parameter_space': {
            'fast': [5, 10, 15, 20],
            'slow': [20, 30, 40, 50]
        },
        'default_params': {'fast': 10, 'slow': 30}
    }
    
    # Apply capability
    capability = OptimizationCapability()
    optimizable = capability.apply(strategy, spec)
    
    # Test methods were added
    assert hasattr(optimizable, 'get_parameter_space')
    assert hasattr(optimizable, 'set_parameters')
    assert hasattr(optimizable, 'get_parameters')
    assert hasattr(optimizable, 'validate_parameters')
    
    print("✓ Optimization methods added")
    
    # Test parameter operations
    params = optimizable.get_parameters()
    print(f"✓ Current parameters: {params}")
    
    new_params = {'fast': 15, 'slow': 40}
    optimizable.set_parameters(new_params)
    assert optimizable.get_parameters() == new_params
    print(f"✓ Parameters updated: {new_params}")
    
    # Test validation
    valid, msg = optimizable.validate_parameters({'fast': 10, 'slow': 30})
    assert valid
    print("✓ Parameter validation working")
    
    return optimizable


def test_grid_optimization():
    """Test grid search optimization."""
    print("\n=== Testing Grid Optimization ===")
    
    optimizer = GridOptimizer()
    objective = SharpeObjective()
    
    # Define evaluation function
    def evaluate(params):
        # Mock evaluation
        sharpe = 1.0 + (params['slow'] - params['fast']) * 0.01
        results = {'sharpe_ratio': sharpe}
        return objective.calculate(results)
    
    # Run optimization
    param_space = {
        'fast': [5, 10, 15],
        'slow': [30, 40, 50]
    }
    
    best = optimizer.optimize(evaluate, parameter_space=param_space)
    
    print(f"✓ Best parameters: {best}")
    print(f"✓ Best score: {optimizer.get_best_score():.4f}")
    print(f"✓ Trials run: {len(optimizer.get_optimization_history())}")


def test_constraints():
    """Test parameter constraints."""
    print("\n=== Testing Constraints ===")
    
    # Relational constraint
    constraint = RelationalConstraint('fast', '<', 'slow')
    
    valid_params = {'fast': 10, 'slow': 30}
    invalid_params = {'fast': 40, 'slow': 30}
    
    assert constraint.is_satisfied(valid_params)
    assert not constraint.is_satisfied(invalid_params)
    print("✓ Relational constraint working")
    
    # Test adjustment
    adjusted = constraint.validate_and_adjust(invalid_params)
    assert constraint.is_satisfied(adjusted)
    print(f"✓ Constraint adjustment: {invalid_params} -> {adjusted}")


def test_sequential_workflow():
    """Test sequential optimization workflow."""
    print("\n=== Testing Sequential Workflow ===")
    
    # Define stages
    stages = [
        {
            'name': 'coarse_search',
            'optimizer': {'type': 'grid'},
            'objective': {'type': 'sharpe'},
            'component': {
                'class': 'SimpleStrategy',
                'parameter_space': {
                    'fast': [5, 15],
                    'slow': [30, 50]
                }
            },
            'n_trials': 4
        },
        {
            'name': 'fine_tuning',
            'optimizer': {'type': 'bayesian'},
            'objective': {'type': 'sharpe'},
            'component': {
                'class': 'SimpleStrategy',
                'parameter_space': {
                    'fast': (5, 20),
                    'slow': (25, 55)
                }
            },
            'n_trials': 10,
            'use_previous_best': True
        }
    ]
    
    # Create workflow
    workflow = SequentialOptimizationWorkflow(stages)
    
    # Run
    results = workflow.run()
    
    print("✓ Workflow completed")
    for stage_name in ['coarse_search', 'fine_tuning']:
        if stage_name in results and 'best_parameters' in results[stage_name]:
            print(f"  {stage_name}: {results[stage_name]['best_parameters']}")


def main():
    """Run all tests."""
    print("=== Testing Optimization Framework ===\n")
    
    # Test 1: Capability application
    optimizable = test_capability_application()
    
    # Test 2: Grid optimization
    test_grid_optimization()
    
    # Test 3: Constraints
    test_constraints()
    
    # Test 4: Sequential workflow
    test_sequential_workflow()
    
    print("\n=== All Tests Passed! ===")


if __name__ == "__main__":
    main()
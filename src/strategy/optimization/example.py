"""
Example usage of the optimization framework.

This demonstrates how to make any component optimizable and run
optimization workflows.
"""

from typing import Dict, Any
import random

# Import optimization components
from .capabilities import OptimizationCapability
from .optimizers import GridOptimizer, BayesianOptimizer
from .objectives import SharpeObjective, CompositeObjective, MinDrawdownObjective
from .workflows import SequentialOptimizationWorkflow
from .constraints import RelationalConstraint


# Example 1: Simple Strategy Component
class SimpleMovingAverageStrategy:
    """Example strategy that can be optimized."""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30, 
                 threshold: float = 0.02):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.threshold = threshold
    
    def execute(self, data):
        """Strategy execution logic."""
        # This would contain actual strategy logic
        pass


def example_basic_optimization():
    """Example of basic grid search optimization."""
    print("\n=== Basic Grid Search Example ===")
    
    # Create strategy instance
    strategy = SimpleMovingAverageStrategy()
    
    # Apply optimization capability
    capability = OptimizationCapability()
    strategy_spec = {
        'parameter_space': {
            'fast_period': [5, 10, 15, 20],
            'slow_period': [20, 30, 40, 50],
            'threshold': [0.01, 0.02, 0.03]
        },
        'default_params': {
            'fast_period': 10,
            'slow_period': 30,
            'threshold': 0.02
        }
    }
    
    # Make strategy optimizable
    optimizable_strategy = capability.apply(strategy, strategy_spec)
    
    # Create optimizer and objective
    optimizer = GridOptimizer()
    objective = SharpeObjective()
    
    # Define evaluation function (mock backtest)
    def evaluate(params):
        # Apply parameters
        optimizable_strategy.set_parameters(params)
        
        # Run backtest (mock)
        results = {
            'sharpe_ratio': random.uniform(0.5, 2.0),
            'total_return': random.uniform(-0.1, 0.5),
            'max_drawdown': random.uniform(0.05, 0.3)
        }
        
        return objective.calculate(results)
    
    # Run optimization
    best_params = optimizer.optimize(
        evaluate,
        parameter_space=optimizable_strategy.get_parameter_space()
    )
    
    print(f"Best parameters: {best_params}")
    print(f"Best score: {optimizer.get_best_score():.4f}")
    
    # Apply best parameters
    optimizable_strategy.set_parameters(best_params)
    print(f"Strategy updated with best parameters")


def example_bayesian_with_constraints():
    """Example of Bayesian optimization with constraints."""
    print("\n=== Bayesian Optimization with Constraints ===")
    
    # Create optimizer with composite objective
    optimizer = BayesianOptimizer(acquisition_function='expected_improvement')
    
    # Composite objective: 70% Sharpe, 30% Drawdown
    objective = CompositeObjective([
        (SharpeObjective(), 0.7),
        (MinDrawdownObjective(), 0.3)
    ])
    
    # Define constraint: fast_period < slow_period
    constraint = RelationalConstraint('fast_period', '<', 'slow_period')
    
    # Parameter space with continuous ranges
    param_space = {
        'fast_period': (5, 25),    # Continuous range
        'slow_period': (20, 60),   # Continuous range
        'threshold': [0.01, 0.02, 0.03]  # Discrete values
    }
    
    # Evaluation function with constraint checking
    def evaluate(params):
        # Check constraint
        if not constraint.is_satisfied(params):
            return float('-inf')  # Invalid
        
        # Mock backtest
        results = {
            'sharpe_ratio': 1.5 + 0.1 * (params['slow_period'] - params['fast_period']) / 10,
            'total_return': 0.2,
            'max_drawdown': 0.1 + 0.01 * params['threshold']
        }
        
        return objective.calculate(results)
    
    # Run optimization
    best_params = optimizer.optimize(
        evaluate,
        n_trials=50,
        parameter_space=param_space
    )
    
    print(f"Best parameters: {best_params}")
    print(f"Best score: {optimizer.get_best_score():.4f}")
    print(f"Constraint satisfied: {constraint.is_satisfied(best_params)}")


def example_sequential_workflow():
    """Example of multi-stage sequential optimization."""
    print("\n=== Sequential Workflow Example ===")
    
    # Define workflow stages
    stages = [
        {
            'name': 'coarse_search',
            'optimizer': {'type': 'grid'},
            'objective': {'type': 'sharpe'},
            'component': {
                'class': 'SimpleMovingAverageStrategy',
                'parameter_space': {
                    'fast_period': [5, 15, 25],
                    'slow_period': [30, 45, 60]
                }
            },
            'n_trials': 9  # 3x3 grid
        },
        {
            'name': 'fine_tuning',
            'optimizer': {
                'type': 'bayesian',
                'acquisition': 'expected_improvement'
            },
            'objective': {
                'type': 'composite',
                'components': [
                    {'type': 'sharpe', 'weight': 0.6},
                    {'type': 'drawdown', 'weight': 0.4}
                ]
            },
            'component': {
                'class': 'SimpleMovingAverageStrategy',
                'parameter_space': {
                    'threshold': (0.005, 0.05)
                }
            },
            'n_trials': 30,
            'use_previous_best': True  # Use best params from stage 1
        }
    ]
    
    # Create and run workflow
    workflow = SequentialOptimizationWorkflow(stages)
    
    print("Starting sequential optimization...")
    results = workflow.run()
    
    # Display results
    print("\nWorkflow completed!")
    for stage_name, stage_results in results.items():
        if 'best_parameters' in stage_results:
            print(f"\n{stage_name}:")
            print(f"  Parameters: {stage_results['best_parameters']}")
            print(f"  Score: {stage_results['best_score']:.4f}")


def main():
    """Run all examples."""
    print("=== ADMF-PC Optimization Framework Examples ===")
    
    # Basic grid search
    example_basic_optimization()
    
    # Bayesian with constraints
    example_bayesian_with_constraints()
    
    # Sequential workflow
    example_sequential_workflow()
    
    print("\n=== Examples Complete ===")


if __name__ == "__main__":
    main()
"""
Final test of optimization framework functionality.

This test demonstrates all key features of the optimization framework.
"""

from typing import Dict, Any
import random

# Import optimization components
from src.strategy.optimization import (
    OptimizationCapability,
    GridOptimizer,
    BayesianOptimizer,
    SharpeObjective,
    MinDrawdownObjective,
    CompositeObjective,
    RelationalConstraint,
    RangeConstraint
)


def test_1_capability():
    """Test 1: Optimization Capability"""
    print("=== Test 1: Optimization Capability ===")
    
    # Create a simple strategy class
    class TrendFollower:
        def __init__(self, lookback=20, threshold=0.02):
            self.lookback = lookback
            self.threshold = threshold
    
    # Create instance
    strategy = TrendFollower()
    
    # Define optimization spec
    spec = {
        'parameter_space': {
            'lookback': [10, 20, 30, 40],
            'threshold': [0.01, 0.02, 0.03]
        },
        'default_params': {
            'lookback': 20,
            'threshold': 0.02
        }
    }
    
    # Apply optimization capability
    capability = OptimizationCapability()
    strategy = capability.apply(strategy, spec)
    
    # Verify methods were added
    assert hasattr(strategy, 'get_parameter_space')
    assert hasattr(strategy, 'set_parameters')
    assert hasattr(strategy, 'get_parameters')
    assert hasattr(strategy, 'validate_parameters')
    print("✓ Optimization methods added successfully")
    
    # Test parameter operations
    current = strategy.get_parameters()
    print(f"✓ Current parameters: {current}")
    
    # Update parameters
    new_params = {'lookback': 30, 'threshold': 0.01}
    strategy.set_parameters(new_params)
    assert strategy.get_parameters() == new_params
    print(f"✓ Parameters updated: {new_params}")
    
    # Test validation
    valid, msg = strategy.validate_parameters({'lookback': 10, 'threshold': 0.02})
    assert valid
    print("✓ Parameter validation working")
    
    # Test invalid parameters
    valid, msg = strategy.validate_parameters({'lookback': 50})  # Missing threshold
    assert not valid
    print(f"✓ Invalid parameters detected: {msg}")
    
    return strategy


def test_2_grid_optimization():
    """Test 2: Grid Search Optimization"""
    print("\n=== Test 2: Grid Search Optimization ===")
    
    # Create optimizer
    optimizer = GridOptimizer()
    objective = SharpeObjective()
    
    # Define parameter space
    param_space = {
        'fast_ma': [5, 10, 15, 20],
        'slow_ma': [20, 30, 40, 50],
        'threshold': [0.01, 0.02]
    }
    
    # Define evaluation function
    def evaluate(params):
        # Simulate backtest results
        # Better results when slow > fast and reasonable threshold
        fast = params['fast_ma']
        slow = params['slow_ma']
        threshold = params['threshold']
        
        if slow <= fast:
            sharpe = 0.5  # Poor performance
        else:
            sharpe = 1.0 + (slow - fast) * 0.02 + (0.02 - threshold) * 5
        
        results = {
            'sharpe_ratio': sharpe,
            'total_return': sharpe * 0.15,
            'max_drawdown': 0.2 / sharpe
        }
        
        return objective.calculate(results)
    
    # Run optimization
    best_params = optimizer.optimize(evaluate, parameter_space=param_space)
    
    print(f"✓ Best parameters found: {best_params}")
    print(f"✓ Best Sharpe ratio: {optimizer.get_best_score():.4f}")
    
    # Verify all combinations tested
    history = optimizer.get_optimization_history()
    expected_trials = len(param_space['fast_ma']) * len(param_space['slow_ma']) * len(param_space['threshold'])
    assert len(history) == expected_trials
    print(f"✓ All {expected_trials} combinations tested")
    
    return best_params


def test_3_bayesian_optimization():
    """Test 3: Bayesian Optimization"""
    print("\n=== Test 3: Bayesian Optimization ===")
    
    # Create Bayesian optimizer
    optimizer = BayesianOptimizer(acquisition_function='expected_improvement')
    
    # Create composite objective (70% Sharpe, 30% Drawdown)
    objective = CompositeObjective([
        (SharpeObjective(), 0.7),
        (MinDrawdownObjective(), 0.3)
    ])
    
    # Define continuous parameter space
    param_space = {
        'lookback': (5, 50),         # Continuous range
        'threshold': (0.005, 0.05),  # Continuous range
        'risk_level': [1, 2, 3]      # Discrete choices
    }
    
    # Evaluation function
    def evaluate(params):
        lookback = params['lookback']
        threshold = params['threshold']
        risk = params['risk_level']
        
        # Simulate performance
        sharpe = 1.5 - abs(lookback - 20) * 0.02 - abs(threshold - 0.02) * 10
        sharpe *= (1 + risk * 0.1)  # Higher risk can mean higher return
        
        drawdown = 0.1 + risk * 0.05 + abs(threshold - 0.02) * 2
        
        results = {
            'sharpe_ratio': max(0.1, sharpe),
            'max_drawdown': min(0.5, drawdown),
            'total_return': sharpe * 0.2
        }
        
        return objective.calculate(results)
    
    # Run optimization with fewer trials
    best_params = optimizer.optimize(
        evaluate, 
        n_trials=30,
        parameter_space=param_space
    )
    
    print(f"✓ Best parameters: {best_params}")
    print(f"✓ Best composite score: {optimizer.get_best_score():.4f}")
    print(f"✓ Total Bayesian trials: {len(optimizer.get_optimization_history())}")
    
    return best_params


def test_4_constraints():
    """Test 4: Parameter Constraints"""
    print("\n=== Test 4: Parameter Constraints ===")
    
    # Test relational constraint
    print("Testing relational constraints...")
    rel_constraint = RelationalConstraint('fast_ma', '<', 'slow_ma')
    
    valid = {'fast_ma': 10, 'slow_ma': 30}
    invalid = {'fast_ma': 40, 'slow_ma': 30}
    
    assert rel_constraint.is_satisfied(valid)
    assert not rel_constraint.is_satisfied(invalid)
    print("✓ Relational constraint working")
    
    # Test adjustment
    adjusted = rel_constraint.validate_and_adjust(invalid)
    assert rel_constraint.is_satisfied(adjusted)
    print(f"✓ Auto-adjusted: {invalid} -> {adjusted}")
    
    # Test range constraint
    print("\nTesting range constraints...")
    range_constraint = RangeConstraint('risk_factor', min_value=0.0, max_value=1.0)
    
    assert range_constraint.is_satisfied({'risk_factor': 0.5})
    assert not range_constraint.is_satisfied({'risk_factor': 1.5})
    
    adjusted = range_constraint.validate_and_adjust({'risk_factor': 1.5})
    assert adjusted['risk_factor'] == 1.0
    print("✓ Range constraint working")
    
    # Test with optimization
    print("\nTesting constrained optimization...")
    optimizer = GridOptimizer()
    objective = SharpeObjective()
    
    param_space = {
        'fast': [5, 10, 15, 20, 25, 30],
        'slow': [10, 20, 30, 40],
        'threshold': [0.02]
    }
    
    valid_count = 0
    
    def evaluate_constrained(params):
        nonlocal valid_count
        if not rel_constraint.is_satisfied(params):
            return float('-inf')
        
        valid_count += 1
        # Simple evaluation
        return 1.0 + (params['slow'] - params['fast']) * 0.01
    
    best = optimizer.optimize(evaluate_constrained, parameter_space=param_space)
    
    print(f"✓ Valid combinations: {valid_count} out of {6*4*1}")
    print(f"✓ Best constrained params: {best}")
    assert rel_constraint.is_satisfied(best)
    print("✓ Best params satisfy constraint")


def test_5_optimization_history():
    """Test 5: Optimization History and Analysis"""
    print("\n=== Test 5: Optimization History and Analysis ===")
    
    # Apply capability to track history
    class Strategy:
        def __init__(self, param1=1.0, param2=2.0):
            self.param1 = param1
            self.param2 = param2
    
    strategy = Strategy()
    spec = {
        'parameter_space': {
            'param1': [0.5, 1.0, 1.5, 2.0],
            'param2': [1.0, 2.0, 3.0, 4.0]
        }
    }
    
    capability = OptimizationCapability()
    strategy = capability.apply(strategy, spec)
    
    # Simulate optimization trials
    print("Simulating optimization trials...")
    scores = []
    
    for p1 in spec['parameter_space']['param1']:
        for p2 in spec['parameter_space']['param2']:
            params = {'param1': p1, 'param2': p2}
            strategy.set_parameters(params)
            
            # Mock score
            score = p1 * p2 + random.uniform(-0.1, 0.1)
            scores.append(score)
            
            # Update best if better
            strategy.update_best_parameters(params, score)
    
    # Check optimization stats
    stats = strategy.get_optimization_stats()
    print(f"✓ Total trials: {stats['total_trials']}")
    print(f"✓ Best score: {stats['best_score']:.4f}")
    print(f"✓ Best parameters: {stats['best_parameters']}")
    
    # Verify history
    history = strategy.get_parameter_history()
    assert len(history) == 16  # 4x4 grid
    print(f"✓ Parameter history tracked: {len(history)} entries")


def main():
    """Run all tests."""
    print("=== ADMF-PC Optimization Framework Tests ===\n")
    
    # Test 1: Basic capability
    strategy = test_1_capability()
    
    # Test 2: Grid search
    grid_best = test_2_grid_optimization()
    
    # Test 3: Bayesian optimization
    bayes_best = test_3_bayesian_optimization()
    
    # Test 4: Constraints
    test_4_constraints()
    
    # Test 5: History tracking
    test_5_optimization_history()
    
    print("\n=== All Tests Passed! ===")
    print("\nOptimization framework features verified:")
    print("✓ Dynamic capability application")
    print("✓ Grid search optimization")
    print("✓ Bayesian optimization")
    print("✓ Composite objectives")
    print("✓ Parameter constraints")
    print("✓ Optimization history tracking")
    print("\nThe optimization framework is ready for use!")


if __name__ == "__main__":
    main()
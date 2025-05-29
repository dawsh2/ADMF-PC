"""
Test optimization workflow implementation.

This test follows the approach outlined in TEST_WORKFLOW.MD to verify
the optimization framework works correctly.
"""

import asyncio
from datetime import datetime
import random
from typing import Dict, Any, List

# Import our modules
from src.core.components import ComponentFactory
from src.strategy.optimization import (
    OptimizationCapability,
    GridOptimizer,
    BayesianOptimizer,
    SharpeObjective,
    CompositeObjective,
    MinDrawdownObjective,
    RelationalConstraint,
    SequentialOptimizationWorkflow,
    RegimeBasedOptimizationWorkflow,
    OptimizationContainer
)


# Test fixtures
class TestFixtures:
    """Provides test data and mock components"""
    
    @staticmethod
    def create_mock_strategy_spec() -> Dict[str, Any]:
        """Create a mock strategy specification"""
        return {
            'name': 'test_strategy',
            'class': 'MockStrategy',
            'capabilities': ['optimization'],
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
    
    @staticmethod
    def create_mock_backtest_runner():
        """Create a mock backtest runner"""
        def backtest(component) -> Dict[str, Any]:
            # Simulate results based on parameters
            params = component.get_parameters() if hasattr(component, 'get_parameters') else {}
            
            # Simulate that faster periods work better in trends
            fast = params.get('fast_period', 10)
            slow = params.get('slow_period', 30)
            threshold = params.get('threshold', 0.02)
            
            # Mock calculation
            base_sharpe = 1.0
            if fast < slow:
                base_sharpe += 0.2
            if slow - fast > 15:
                base_sharpe += 0.1
            if threshold < 0.02:
                base_sharpe += 0.15
            
            # Add some randomness
            sharpe = base_sharpe + random.uniform(-0.2, 0.2)
            
            return {
                'sharpe_ratio': sharpe,
                'total_return': sharpe * 0.15,  # Simplified
                'max_drawdown': 0.15 / sharpe,  # Inverse relationship
                'num_trades': int(100 * threshold / 0.02),
                'returns': [random.gauss(0.001, 0.01) for _ in range(252)]
            }
        
        return backtest


# Mock strategy class
class MockStrategy:
    """Mock strategy for testing"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30, 
                 threshold: float = 0.02):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.threshold = threshold
        self._parameters = {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'threshold': threshold
        }


# Phase 1: Test Grid Search Optimization
def test_phase1_grid_search():
    """Test basic grid search optimization"""
    print("\n=== Phase 1: Grid Search Optimization ===")
    
    # Create optimizer and objective
    optimizer = GridOptimizer()
    objective = SharpeObjective()
    
    # Create mock backtest
    backtest = TestFixtures.create_mock_backtest_runner()
    
    # Define parameter space
    param_space = {
        'fast_period': [5, 10, 15],
        'slow_period': [20, 30, 40],
        'threshold': [0.01, 0.02, 0.03]
    }
    
    # Run optimization
    print("Running grid search optimization...")
    
    def evaluate(params):
        # Create mock component with params
        component = MockStrategy(**params)
        component.get_parameters = lambda: params
        results = backtest(component)
        return objective.calculate(results)
    
    best_params = optimizer.optimize(
        evaluate,
        parameter_space=param_space
    )
    
    print(f"Best parameters found: {best_params}")
    print(f"Best score: {optimizer.get_best_score():.4f}")
    print(f"Total trials: {len(optimizer.get_optimization_history())}")
    
    # Verify all combinations tested
    expected_trials = len(param_space['fast_period']) * \
                     len(param_space['slow_period']) * \
                     len(param_space['threshold'])
    assert len(optimizer.get_optimization_history()) == expected_trials
    print("✓ All parameter combinations tested")
    
    return optimizer.get_optimization_history()


# Phase 2: Test Bayesian Optimization
def test_phase2_bayesian_optimization():
    """Test Bayesian optimization"""
    print("\n=== Phase 2: Bayesian Optimization ===")
    
    # Create optimizer with composite objective
    optimizer = BayesianOptimizer()
    
    # Composite objective: 70% Sharpe, 30% drawdown
    objective = CompositeObjective([
        (SharpeObjective(), 0.7),
        (MinDrawdownObjective(), 0.3)
    ])
    
    # Create mock backtest
    backtest = TestFixtures.create_mock_backtest_runner()
    
    # Define continuous parameter space
    param_space = {
        'fast_period': (5, 20),      # Continuous range
        'slow_period': (20, 50),     # Continuous range  
        'threshold': [0.01, 0.02, 0.03]  # Discrete
    }
    
    # Run optimization
    print("Running Bayesian optimization...")
    
    def evaluate(params):
        component = MockStrategy(**params)
        component.get_parameters = lambda: params
        results = backtest(component)
        return objective.calculate(results)
    
    best_params = optimizer.optimize(
        evaluate,
        n_trials=30,
        parameter_space=param_space
    )
    
    print(f"Best parameters found: {best_params}")
    print(f"Best score: {optimizer.get_best_score():.4f}")
    print(f"Total trials: {len(optimizer.get_optimization_history())}")
    print("✓ Bayesian optimization completed")


# Phase 3: Test Optimization with Constraints
def test_phase3_constrained_optimization():
    """Test optimization with parameter constraints"""
    print("\n=== Phase 3: Constrained Optimization ===")
    
    # Create constrained parameter space
    constraint = RelationalConstraint('fast_period', '<', 'slow_period')
    
    # Grid optimizer with constraints
    optimizer = GridOptimizer()
    objective = SharpeObjective()
    
    backtest = TestFixtures.create_mock_backtest_runner()
    
    param_space = {
        'fast_period': [5, 10, 15, 20, 25],
        'slow_period': [10, 20, 30, 40],
        'threshold': [0.02]
    }
    
    print("Running constrained optimization...")
    valid_count = 0
    
    def evaluate(params):
        nonlocal valid_count
        # Check constraint
        if not constraint.is_satisfied(params):
            return float('-inf')  # Invalid combination
        
        valid_count += 1
        component = MockStrategy(**params)
        component.get_parameters = lambda: params
        results = backtest(component)
        return objective.calculate(results)
    
    best_params = optimizer.optimize(
        evaluate,
        parameter_space=param_space
    )
    
    print(f"Best parameters found: {best_params}")
    print(f"Valid combinations tested: {valid_count}")
    
    # Verify constraint is satisfied
    assert constraint.is_satisfied(best_params)
    print("✓ Best parameters satisfy constraint")


# Phase 4: Test Sequential Workflow
def test_phase4_sequential_workflow():
    """Test multi-stage sequential optimization"""
    print("\n=== Phase 4: Sequential Optimization Workflow ===")
    
    # Create two-stage workflow
    stages = [
        {
            'name': 'rough_search',
            'optimizer': {'type': 'grid'},
            'objective': {'type': 'sharpe'},
            'component': {
                'class': 'MockStrategy',
                'parameter_space': {
                    'fast_period': [5, 10, 15, 20],
                    'slow_period': [20, 30, 40, 50]
                }
            },
            'n_trials': 16
        },
        {
            'name': 'fine_tuning',
            'optimizer': {'type': 'bayesian'},
            'objective': {
                'type': 'composite',
                'components': [
                    {'type': 'sharpe', 'weight': 0.7},
                    {'type': 'drawdown', 'weight': 0.3}
                ]
            },
            'component': {
                'class': 'MockStrategy',
                'parameter_space': {
                    'threshold': (0.005, 0.05)  # Fine-tune threshold
                }
            },
            'n_trials': 20,
            'use_previous_best': True  # Use best params from stage 1
        }
    ]
    
    # Override the backtest runner creation
    original_create_backtest = SequentialOptimizationWorkflow._create_backtest_runner
    SequentialOptimizationWorkflow._create_backtest_runner = lambda self, config: TestFixtures.create_mock_backtest_runner()
    
    workflow = SequentialOptimizationWorkflow(stages)
    
    print("Running sequential workflow...")
    results = workflow.run()
    
    # Restore original
    SequentialOptimizationWorkflow._create_backtest_runner = original_create_backtest
    
    print("\nWorkflow Results:")
    for stage_name, stage_results in results.items():
        if 'best_parameters' in stage_results:
            print(f"  {stage_name}:")
            print(f"    Best params: {stage_results['best_parameters']}")
            print(f"    Best score: {stage_results['best_score']:.4f}")
    
    print("✓ Sequential workflow completed")


# Phase 5: Test Container Isolation
async def test_phase5_container_isolation():
    """Test that optimization trials are properly isolated"""
    print("\n=== Phase 5: Container Isolation Test ===")
    
    # Create optimization container
    base_config = TestFixtures.create_mock_strategy_spec()
    container = OptimizationContainer("test_opt", base_config)
    
    # Initialize container
    container.initialize_scope()
    
    print("Testing parameter isolation...")
    
    # Run trials with different parameters
    params1 = {'fast_period': 10, 'slow_period': 30, 'threshold': 0.01}
    params2 = {'fast_period': 15, 'slow_period': 40, 'threshold': 0.02}
    
    backtest = TestFixtures.create_mock_backtest_runner()
    
    # Run trials
    result1 = container.run_trial(params1, backtest)
    result2 = container.run_trial(params2, backtest)
    
    # Verify isolation
    assert result1['parameters'] == params1
    assert result2['parameters'] == params2
    assert result1['trial_id'] != result2['trial_id']
    
    print(f"Trial 1: {result1['trial_id']} -> Sharpe: {result1['sharpe_ratio']:.4f}")
    print(f"Trial 2: {result2['trial_id']} -> Sharpe: {result2['sharpe_ratio']:.4f}")
    print("✓ Container isolation verified")
    
    # Clean up
    container.dispose()


# Phase 6: Test Regime-Based Optimization
def test_phase6_regime_based_optimization():
    """Test regime-specific optimization"""
    print("\n=== Phase 6: Regime-Based Optimization ===")
    
    # Configure regime-based workflow
    regime_config = {
        'volatility_window': 20,
        'trend_window': 50
    }
    
    component_config = {
        'class': 'MockStrategy',
        'parameter_space': {
            'fast_period': [5, 10, 15],
            'slow_period': [20, 30, 40],
            'threshold': [0.01, 0.02, 0.03]
        }
    }
    
    optimizer_config = {
        'type': 'grid',
        'objective': 'sharpe',
        'n_trials_per_regime': 10
    }
    
    workflow = RegimeBasedOptimizationWorkflow(
        regime_config,
        component_config,
        optimizer_config
    )
    
    print("Running regime-based optimization...")
    results = workflow.run()
    
    print("\nRegime-Specific Results:")
    for regime in results.get('detected_regimes', []):
        if regime in results:
            regime_result = results[regime]
            if 'best_parameters' in regime_result:
                print(f"  {regime}:")
                print(f"    Best params: {regime_result['best_parameters']}")
                print(f"    Best score: {regime_result['best_score']:.4f}")
    
    # Verify adaptive config created
    assert 'adaptive_config' in results
    assert 'regime_parameters' in results['adaptive_config']
    print("✓ Regime-based optimization completed")


# Main test runner
async def main():
    """Run all optimization workflow tests"""
    print("=== Testing Optimization Workflow Implementation ===")
    
    try:
        # Phase 1: Basic grid search
        test_phase1_grid_search()
        
        # Phase 2: Bayesian optimization
        test_phase2_bayesian_optimization()
        
        # Phase 3: Constrained optimization
        test_phase3_constrained_optimization()
        
        # Phase 4: Sequential workflow
        test_phase4_sequential_workflow()
        
        # Phase 5: Container isolation
        await test_phase5_container_isolation()
        
        # Phase 6: Regime-based optimization
        test_phase6_regime_based_optimization()
        
        print("\n=== All Tests Passed! ===")
        print("\nOptimization framework is working correctly:")
        print("✓ Grid search optimization")
        print("✓ Bayesian optimization")
        print("✓ Constraint handling")
        print("✓ Sequential workflows")
        print("✓ Container isolation")
        print("✓ Regime-based optimization")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
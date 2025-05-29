"""
Integration test for the optimization framework.

This test verifies that the optimization module integrates properly
with the rest of the ADMF-PC system.
"""

import asyncio
from typing import Dict, Any

# Import core components
from src.core.components import ComponentFactory
from src.core.containers import UniversalScopedContainer

# Import optimization components
from src.strategy.optimization import (
    OptimizationCapability,
    GridOptimizer,
    SharpeObjective,
    RelationalConstraint,
    ContainerizedComponentOptimizer
)


class TestStrategy:
    """Test strategy for integration testing."""
    
    def __init__(self, lookback: int = 20, threshold: float = 0.02):
        self.lookback = lookback
        self.threshold = threshold
        self.trades = []
    
    def execute(self, data):
        """Execute strategy logic."""
        # Mock execution
        if len(data) >= self.lookback:
            signal = data[-1] > data[-self.lookback:].mean()
            if abs(signal - 0.5) > self.threshold:
                self.trades.append({'signal': signal, 'price': data[-1]})


async def test_optimization_integration():
    """Test optimization framework integration."""
    print("=== Testing Optimization Framework Integration ===\n")
    
    # Step 1: Create a container
    print("1. Creating container...")
    container = UniversalScopedContainer(
        container_id="test_integration",
        container_type="test"
    )
    container.initialize_scope()
    container.start()
    
    # Step 2: Register optimization capability
    print("2. Registering optimization capability...")
    factory = ComponentFactory()
    factory.add_enhancer('optimization', OptimizationCapability())
    
    # Step 3: Create optimizable component
    print("3. Creating optimizable strategy component...")
    strategy_spec = {
        'name': 'test_strategy',
        'class_name': TestStrategy,
        'capabilities': ['optimization'],
        'parameter_space': {
            'lookback': [10, 20, 30, 40],
            'threshold': [0.01, 0.02, 0.03, 0.04]
        },
        'default_params': {
            'lookback': 20,
            'threshold': 0.02
        }
    }
    
    strategy = container.create_component(strategy_spec, initialize=True)
    
    # Verify optimization capability was applied
    assert hasattr(strategy, 'get_parameter_space'), "Strategy should have optimization methods"
    assert hasattr(strategy, 'set_parameters'), "Strategy should have parameter setter"
    print("✓ Strategy has optimization capability")
    
    # Step 4: Test parameter management
    print("\n4. Testing parameter management...")
    
    # Get parameter space
    param_space = strategy.get_parameter_space()
    print(f"Parameter space: {param_space}")
    
    # Set parameters
    new_params = {'lookback': 30, 'threshold': 0.03}
    strategy.set_parameters(new_params)
    
    # Verify parameters were set
    current_params = strategy.get_parameters()
    assert current_params == new_params, "Parameters should be updated"
    print(f"✓ Parameters updated: {current_params}")
    
    # Step 5: Test constraint validation
    print("\n5. Testing constraints...")
    constraint = RelationalConstraint('threshold', '<', 0.05)
    
    valid_params = {'lookback': 20, 'threshold': 0.02}
    invalid_params = {'lookback': 20, 'threshold': 0.1}
    
    assert constraint.is_satisfied(valid_params), "Valid params should satisfy constraint"
    assert not constraint.is_satisfied(invalid_params), "Invalid params should not satisfy"
    print("✓ Constraints working correctly")
    
    # Step 6: Test containerized optimization
    print("\n6. Testing containerized optimization...")
    
    # Create optimizer
    optimizer = GridOptimizer()
    objective = SharpeObjective()
    
    # Mock backtest function
    def mock_backtest(component):
        # Simulate backtest based on parameters
        params = component.get_parameters() if hasattr(component, 'get_parameters') else {}
        lookback = params.get('lookback', 20)
        threshold = params.get('threshold', 0.02)
        
        # Mock results
        sharpe = 1.0 + (40 - lookback) * 0.01 + (0.02 - threshold) * 10
        
        return {
            'sharpe_ratio': sharpe,
            'total_return': sharpe * 0.15,
            'max_drawdown': 0.1 / sharpe,
            'num_trades': int(100 / threshold)
        }
    
    # Run optimization
    containerized_optimizer = ContainerizedComponentOptimizer(
        optimizer, objective, use_containers=True
    )
    
    results = containerized_optimizer.optimize_component(
        strategy_spec,
        mock_backtest,
        n_trials=16  # 4x4 grid
    )
    
    print(f"Best parameters: {results['best_parameters']}")
    print(f"Best score: {results['best_score']:.4f}")
    print(f"Total trials: {len(results['all_results'])}")
    print("✓ Containerized optimization completed")
    
    # Step 7: Verify isolation
    print("\n7. Verifying isolation...")
    
    # Original strategy should not be affected
    original_params = strategy.get_parameters()
    print(f"Original strategy params: {original_params}")
    assert original_params == new_params, "Original strategy should not be modified"
    print("✓ Component isolation maintained")
    
    # Clean up
    container.stop()
    container.dispose()
    
    print("\n=== Integration Test Passed! ===")
    print("\nKey features verified:")
    print("✓ Optimization capability registration")
    print("✓ Dynamic capability application")
    print("✓ Parameter management")
    print("✓ Constraint validation")
    print("✓ Containerized optimization")
    print("✓ Component isolation")


if __name__ == "__main__":
    asyncio.run(test_optimization_integration())
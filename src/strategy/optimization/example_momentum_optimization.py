"""
Example of using MomentumStrategy with OptimizationCapability.

This shows how optimization methods are added through capabilities
rather than being built into the strategy class.
"""

from typing import Dict, Any

# Example 1: Creating strategy with optimization capability
def create_optimizable_momentum_strategy():
    """Create momentum strategy with optimization support."""
    from src.core.components import ComponentFactory
    
    # Define parameter space for optimization
    parameter_space = {
        'lookback_period': {
            'type': 'int',
            'min': 10,
            'max': 30,
            'default': 20
        },
        'momentum_threshold': {
            'type': 'float',
            'min': 0.01,
            'max': 0.05,
            'default': 0.02
        },
        'rsi_period': {
            'type': 'int',
            'min': 10,
            'max': 20,
            'default': 14
        },
        'rsi_oversold': {
            'type': 'float',
            'min': 20,
            'max': 40,
            'default': 30
        },
        'rsi_overbought': {
            'type': 'float',
            'min': 60,
            'max': 80,
            'default': 70
        }
    }
    
    # Create strategy with optimization capability
    strategy = ComponentFactory().create_component({
        'class': 'MomentumStrategy',
        'params': {
            'lookback_period': 20,
            'momentum_threshold': 0.02
        },
        'capabilities': [
            {
                'type': 'optimization',
                'parameter_space': parameter_space,
                'constraints': [
                    {
                        'type': 'parameter_dependency',
                        'rule': lambda p: p['rsi_oversold'] < p['rsi_overbought']
                    }
                ]
            }
        ]
    })
    
    return strategy


# Example 2: Using the optimization methods
def optimize_strategy_example():
    """Example of using optimization methods added by capability."""
    strategy = create_optimizable_momentum_strategy()
    
    # These methods are added by OptimizationCapability:
    
    # Get parameter space
    param_space = strategy.get_parameter_space()
    print(f"Parameter space: {param_space}")
    
    # Get current parameters
    current_params = strategy.get_parameters()
    print(f"Current parameters: {current_params}")
    
    # Validate new parameters
    new_params = {
        'lookback_period': 25,
        'momentum_threshold': 0.03,
        'rsi_period': 16,
        'rsi_oversold': 25,
        'rsi_overbought': 75
    }
    
    valid, error = strategy.validate_parameters(new_params)
    if valid:
        # Apply new parameters
        strategy.set_parameters(new_params)
        print("Parameters updated successfully")
    else:
        print(f"Invalid parameters: {error}")
    
    # Update best parameters after optimization run
    strategy.update_best_parameters(new_params, score=0.85)
    
    # Get optimization statistics
    stats = strategy.get_optimization_stats()
    print(f"Optimization stats: {stats}")
    
    # Reset to default parameters
    strategy.reset_to_defaults()


# Example 3: Integration with optimizer
def full_optimization_example():
    """Complete example with optimizer integration."""
    from src.strategy.optimization import (
        GridSearchOptimizer,
        SharpeRatioObjective,
        OptimizationContainer
    )
    
    # Create optimizable strategy
    strategy = create_optimizable_momentum_strategy()
    
    # Create optimizer
    optimizer = GridSearchOptimizer(n_jobs=4)
    
    # Create objective
    objective = SharpeRatioObjective(risk_free_rate=0.02)
    
    # Create optimization container
    container = OptimizationContainer(
        optimizer=optimizer,
        objective=objective,
        component=strategy
    )
    
    # Define sample data for backtesting
    historical_data = [
        {'timestamp': '2024-01-01', 'close': 100},
        {'timestamp': '2024-01-02', 'close': 101},
        # ... more data ...
    ]
    
    # Run optimization
    result = container.optimize(
        data=historical_data,
        n_trials=100
    )
    
    print(f"Best parameters: {result['best_params']}")
    print(f"Best score: {result['best_score']}")
    
    # Apply best parameters to strategy
    strategy.set_parameters(result['best_params'])


# Example 4: Without optimization capability
def plain_strategy_example():
    """Example of strategy without optimization capability."""
    from src.strategy.strategies import MomentumStrategy
    
    # Create plain strategy (no optimization methods)
    strategy = MomentumStrategy(
        lookback_period=20,
        momentum_threshold=0.02
    )
    
    # These methods are NOT available on plain strategy:
    # strategy.get_parameter_space()  # AttributeError
    # strategy.set_parameters()       # AttributeError
    # strategy.validate_parameters()  # AttributeError
    
    # Only the core strategy methods are available:
    market_data = {'close': 100, 'symbol': 'BTC-USD'}
    signal = strategy.generate_signal(market_data)
    
    # To add optimization, create through ComponentFactory
    # with optimization capability as shown above


if __name__ == "__main__":
    # Run examples
    print("=== Example 1: Create optimizable strategy ===")
    strategy = create_optimizable_momentum_strategy()
    print(f"Strategy has optimization methods: {hasattr(strategy, 'get_parameter_space')}")
    
    print("\n=== Example 2: Use optimization methods ===")
    optimize_strategy_example()
    
    print("\n=== Example 4: Plain strategy ===")
    plain_strategy_example()
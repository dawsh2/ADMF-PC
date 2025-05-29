"""
Demonstration of Protocol + Composition Architecture in Strategy Module.

This example shows how to use strategies with capabilities, without
any inheritance. Everything is composed at runtime.
"""

from typing import Dict, Any
import asyncio

# Assuming these imports based on the architecture
# from src.core.components import ComponentFactory
# from src.core.containers import UniversalScopedContainer
# from src.strategy import MomentumStrategy, MeanReversionStrategy
# from src.strategy.optimization import OptimizationContainer, GridOptimizer


def demonstrate_pure_strategy():
    """Show that strategies are just plain classes."""
    print("=== Pure Strategy (No Capabilities) ===")
    
    # Direct instantiation - just a plain class
    strategy = MomentumStrategy(
        lookback_period=20,
        momentum_threshold=0.02
    )
    
    # Has basic strategy methods
    print(f"Strategy name: {strategy.name}")
    
    # Generate signal
    market_data = {
        'symbol': 'AAPL',
        'close': 150.00,
        'timestamp': None
    }
    signal = strategy.generate_signal(market_data)
    print(f"Signal: {signal}")
    
    # Does NOT have optimization methods
    print(f"Has optimization methods: {hasattr(strategy, 'get_parameter_space')}")
    print()


def demonstrate_strategy_with_capabilities():
    """Show how capabilities add functionality through composition."""
    print("=== Strategy with Capabilities ===")
    
    # This is how you would use ComponentFactory
    # strategy = ComponentFactory().create_component({
    #     'class': 'MomentumStrategy',
    #     'params': {
    #         'lookback_period': 20,
    #         'momentum_threshold': 0.02
    #     },
    #     'capabilities': [
    #         'strategy',        # Adds signal tracking, event handling
    #         'optimization',    # Adds parameter optimization
    #         'indicators',      # Adds indicator management
    #         'regime_adaptive'  # Adds regime adaptation
    #     ],
    #     'parameter_space': {
    #         'lookback_period': [10, 20, 30],
    #         'momentum_threshold': [0.01, 0.02, 0.03]
    #     }
    # })
    
    # With capabilities, the strategy now has:
    # - Signal tracking (from StrategyCapability)
    # - get_signal_count(), get_last_signal(), etc.
    # - Optimization methods (from OptimizationCapability)
    # - get_parameter_space(), set_parameters(), validate_parameters()
    # - Indicator management (from IndicatorCapability)
    # - register_indicator(), update_indicators()
    # - Regime adaptation (from RegimeAdaptiveCapability)
    # - on_regime_change(), get_regime_parameters()
    
    print("Strategy now has optimization methods through capability")
    print("Strategy now has indicator management through capability")
    print("Strategy now has regime adaptation through capability")
    print()


def demonstrate_container_isolation():
    """Show how strategies run in isolated containers."""
    print("=== Container Isolation for Strategies ===")
    
    # Each strategy instance runs in its own container
    # container = UniversalScopedContainer(
    #     container_id="strategy_001",
    #     container_type="strategy"
    # )
    
    # Create strategy in container
    # strategy_spec = {
    #     'name': 'momentum_strategy',
    #     'class': 'MomentumStrategy',
    #     'params': {'lookback_period': 20},
    #     'capabilities': ['strategy', 'optimization', 'events']
    # }
    # container.create_component(strategy_spec)
    
    # Container provides:
    # - Complete isolation between strategy instances
    # - Event bus scoping
    # - Resource management
    # - Lifecycle management
    
    print("Strategies run in isolated containers")
    print("Each container has its own event bus and resources")
    print()


def demonstrate_optimization_workflow():
    """Show how optimization works with containers."""
    print("=== Optimization with Container Isolation ===")
    
    # OptimizationContainer manages isolated trials
    # opt_container = OptimizationContainer(
    #     container_id="opt_001",
    #     base_config={
    #         'class': 'MomentumStrategy',
    #         'capabilities': ['strategy', 'optimization']
    #     }
    # )
    
    # Each parameter trial runs in complete isolation
    # results = opt_container.run_trial(
    #     parameters={'lookback_period': 30, 'momentum_threshold': 0.01},
    #     evaluator=backtest_evaluator
    # )
    
    print("Each optimization trial runs in an isolated container")
    print("No shared state between trials")
    print("Clean separation ensures reproducible results")
    print()


def demonstrate_composition_patterns():
    """Show various composition patterns."""
    print("=== Composition Patterns ===")
    
    # 1. Minimal strategy (no capabilities)
    minimal = MomentumStrategy()
    print(f"Minimal strategy methods: {[m for m in dir(minimal) if not m.startswith('_')][:5]}...")
    
    # 2. Strategy with specific capabilities
    # with_indicators = ComponentFactory().create_component({
    #     'class': 'MomentumStrategy',
    #     'capabilities': ['indicators']  # Only indicator management
    # })
    
    # 3. Full-featured strategy
    # full_featured = ComponentFactory().create_component({
    #     'class': 'MomentumStrategy',
    #     'capabilities': [
    #         'strategy',
    #         'optimization',
    #         'indicators',
    #         'rules',
    #         'regime_adaptive',
    #         'lifecycle',
    #         'events'
    #     ]
    # })
    
    # 4. Ensemble of strategies
    # ensemble_container = UniversalScopedContainer("ensemble_001")
    # strategies = [
    #     ensemble_container.create_component({
    #         'name': f'momentum_{i}',
    #         'class': 'MomentumStrategy',
    #         'params': {'lookback_period': period},
    #         'capabilities': ['strategy']
    #     })
    #     for i, period in enumerate([10, 20, 30])
    # ]
    
    print("Strategies can be composed with exactly the capabilities needed")
    print("No unnecessary functionality, no forced inheritance")
    print()


def demonstrate_protocol_compliance():
    """Show that all strategies satisfy the Strategy protocol."""
    print("=== Protocol Compliance ===")
    
    from ..protocols import Strategy
    
    # All our strategies satisfy the protocol
    strategies = [
        MomentumStrategy(),
        MeanReversionStrategy(),
        # TrendFollowingStrategy(),
        # ArbitrageStrategy(),
        # MarketMakingStrategy()
    ]
    
    for strategy in strategies:
        # Check protocol compliance
        is_strategy = isinstance(strategy, Strategy)
        print(f"{strategy.__class__.__name__}: implements Strategy = {is_strategy}")
    
    print("\nAll strategies implement the Strategy protocol without inheritance!")
    print()


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("Protocol + Composition Architecture Demonstration")
    print("=" * 60 + "\n")
    
    demonstrate_pure_strategy()
    demonstrate_strategy_with_capabilities()
    demonstrate_container_isolation()
    demonstrate_optimization_workflow()
    demonstrate_composition_patterns()
    demonstrate_protocol_compliance()
    
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("- No inheritance anywhere in the strategy module")
    print("- Strategies are simple classes that implement protocols")
    print("- Capabilities add functionality through composition")
    print("- Containers provide isolation for parallel execution")
    print("- Pay for what you use - only add needed capabilities")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
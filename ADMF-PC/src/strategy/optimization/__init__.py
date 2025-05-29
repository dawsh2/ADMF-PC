"""
Optimization framework for ADMF-PC.

This module provides protocol-based optimization capabilities that can be
added to any component without inheritance. It includes optimizers, objectives,
constraints, and workflow management.

Example Usage:
    ```python
    from src.core.optimization import (
        OptimizationCapability,
        GridOptimizer,
        SharpeObjective
    )
    
    # Make any component optimizable
    strategy = create_component({
        'class': 'MyStrategy',
        'capabilities': ['optimization'],
        'parameter_space': {
            'fast_period': [5, 10, 15, 20],
            'slow_period': [20, 30, 40, 50]
        }
    })
    
    # Optimize it
    optimizer = GridOptimizer()
    objective = SharpeObjective()
    
    best_params = optimizer.optimize(
        lambda params: backtest_with_params(strategy, params),
        parameter_space=strategy.get_parameter_space()
    )
    ```
"""

from .protocols import (
    Optimizable,
    Optimizer,
    Objective,
    Constraint,
    OptimizationWorkflow
)

from .capabilities import (
    OptimizationCapability
)

from .optimizers import (
    GridOptimizer,
    BayesianOptimizer,
    GeneticOptimizer
)

from .objectives import (
    SharpeObjective,
    MaxReturnObjective,
    MinDrawdownObjective,
    CompositeObjective
)

from .constraints import (
    ParameterConstraint,
    RelationalConstraint,
    RangeConstraint
)

from .workflows import (
    SequentialOptimizationWorkflow,
    RegimeBasedOptimizationWorkflow,
    ContainerizedComponentOptimizer
)

from .containers import (
    OptimizationContainer,
    OptimizationResultsCollector
)


__all__ = [
    # Protocols
    "Optimizable",
    "Optimizer", 
    "Objective",
    "Constraint",
    "OptimizationWorkflow",
    
    # Capabilities
    "OptimizationCapability",
    
    # Optimizers
    "GridOptimizer",
    "BayesianOptimizer",
    "GeneticOptimizer",
    
    # Objectives
    "SharpeObjective",
    "MaxReturnObjective",
    "MinDrawdownObjective",
    "CompositeObjective",
    
    # Constraints
    "ParameterConstraint",
    "RelationalConstraint",
    "RangeConstraint",
    
    # Workflows
    "SequentialOptimizationWorkflow",
    "RegimeBasedOptimizationWorkflow",
    "ContainerizedComponentOptimizer",
    
    # Containers
    "OptimizationContainer",
    "OptimizationResultsCollector"
]
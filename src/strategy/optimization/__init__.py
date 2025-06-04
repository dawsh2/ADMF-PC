"""
Optimization framework for ADMF-PC strategies.

This module provides protocol-based optimization components that
work seamlessly with the container isolation architecture.

Example Usage:
    ```python
    from src.strategy.optimization import (
        OptimizationContainer,
        GridOptimizer,
        SharpeObjective
    )
    
    # Create optimization container
    container = OptimizationContainer(
        container_id="opt_001",
        base_config={
            'class': 'MomentumStrategy',
            'capabilities': ['strategy', 'optimization']
        }
    )
    
    # Run optimization
    optimizer = GridOptimizer()
    objective = SharpeObjective()
    
    best_params = optimizer.optimize(
        evaluate_func=lambda params: container.run_trial(params, evaluator),
        parameter_space={
            'lookback_period': [10, 20, 30],
            'momentum_threshold': [0.01, 0.02, 0.03]
        }
    )
    ```
"""

# Protocols
from .protocols import (
    Optimizer,
    Objective,
    Constraint,
    ParameterSpace,
    OptimizationWorkflow,
    RegimeAnalyzer,
    OptimizationContainer as OptimizationContainerProtocol,
    ParameterSampler
)

# Capabilities
# from .capabilities import OptimizationCapability  # Removed - capabilities.py doesn't exist

# Temporary stub for OptimizationCapability
OptimizationCapability = None

# Containers
from .containers import (
    OptimizationContainer,
    OptimizationResultsCollector,
    RegimeTracker,
    RegimeAwareOptimizationContainer
)

# Workflows - commented out to break circular import
# These can be imported directly when needed:
# from src.strategy.optimization.workflows import ...
#
# from .workflows import (
#     ContainerizedComponentOptimizer,
#     SequentialOptimizationWorkflow,
#     RegimeBasedOptimizationWorkflow,
#     PhaseAwareOptimizationWorkflow,
#     create_phase_aware_workflow
# )

# Optimizers
from .optimizers import (
    GridOptimizer,
    RandomOptimizer
)

# Objectives
from .objectives import (
    SharpeObjective,
    MaxReturnObjective,
    MinDrawdownObjective,
    CompositeObjective,
    CalmarObjective,
    SortinoObjective
)

# Constraints
from .constraints import (
    RelationalConstraint,
    RangeConstraint,
    DiscreteConstraint,
    FunctionalConstraint,
    CompositeConstraint
)

# Walk-forward validation
from .walk_forward import (
    WalkForwardPeriod,
    WalkForwardValidator,
    WalkForwardAnalyzer,
    ContainerizedWalkForward
)


__all__ = [
    # Protocols
    "Optimizer",
    "Objective",
    "Constraint",
    "ParameterSpace",
    "OptimizationWorkflow",
    "RegimeAnalyzer",
    "OptimizationContainerProtocol",
    "ParameterSampler",
    
    # Capabilities
    "OptimizationCapability",
    
    # Containers
    "OptimizationContainer",
    "OptimizationResultsCollector",
    "RegimeTracker",
    "RegimeAwareOptimizationContainer",
    
    # Workflows - commented out to break circular import
    # "ContainerizedComponentOptimizer",
    # "SequentialOptimizationWorkflow",
    # "RegimeBasedOptimizationWorkflow",
    # "PhaseAwareOptimizationWorkflow",
    # "create_phase_aware_workflow",
    
    # Optimizers
    "GridOptimizer",
    "RandomOptimizer",
    
    # Objectives
    "SharpeObjective",
    "MaxReturnObjective",
    "MinDrawdownObjective",
    "CompositeObjective",
    "CalmarObjective",
    "SortinoObjective",
    
    # Constraints
    "RelationalConstraint",
    "RangeConstraint",
    "DiscreteConstraint",
    "FunctionalConstraint",
    "CompositeConstraint",
    
    # Walk-forward validation
    "WalkForwardPeriod",
    "WalkForwardValidator",
    "WalkForwardAnalyzer",
    "ContainerizedWalkForward"
]
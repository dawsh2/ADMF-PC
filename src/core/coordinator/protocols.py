"""
Protocol definitions for the refactored coordinator system.

This module defines the protocols (interfaces) used throughout
the system to ensure clean separation of concerns.
"""

from typing import Dict, Any, List, Optional, Protocol, runtime_checkable, Tuple
from dataclasses import dataclass, field


@runtime_checkable
class TopologyBuilderProtocol(Protocol):
    """Protocol for topology builders - ONLY builds topologies."""
    
    def build_topology(self, mode: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a topology for the given mode and configuration.
        
        Args:
            mode: The topology mode (backtest, signal_generation, etc.)
            config: Configuration for the topology
            
        Returns:
            Dict containing the topology structure
        """
        ...


@runtime_checkable
class SequencerProtocol(Protocol):
    """Protocol for phase sequencers."""
    
    def execute_phases(
        self, 
        pattern: Dict[str, Any], 
        config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute all phases in a workflow pattern.
        
        Args:
            pattern: Workflow pattern with phases
            config: Configuration for execution
            context: Execution context
            
        Returns:
            Dict with execution results
        """
        ...


@runtime_checkable
class ResultStreamerProtocol(Protocol):
    """Protocol for result streaming."""
    
    def write_result(self, result: Dict[str, Any]) -> None:
        """Write a result to the stream."""
        ...
    
    def flush(self) -> None:
        """Flush any buffered results."""
        ...
    
    def get_aggregated_results(self) -> Dict[str, Any]:
        """Get aggregated results."""
        ...
    
    def close(self) -> None:
        """Close the streamer and clean up resources."""
        ...


@runtime_checkable
class DataManagerProtocol(Protocol):
    """Protocol for inter-phase data management."""
    
    def store_phase_output(
        self,
        workflow_id: str,
        phase_name: str,
        output: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store output from a phase."""
        ...
    
    def get_phase_output(
        self,
        workflow_id: str,
        phase_name: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve output from a phase."""
        ...


@runtime_checkable
class CheckpointManagerProtocol(Protocol):
    """Protocol for checkpoint management."""
    
    def save_state(self, workflow_id: str, state: Dict[str, Any]) -> None:
        """Save workflow state."""
        ...
    
    def restore_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Restore workflow state."""
        ...
    
    def delete_checkpoint(self, workflow_id: str) -> None:
        """Delete checkpoint."""
        ...


# ============= NEW WORKFLOW AND SEQUENCE PROTOCOLS =============

@dataclass
class PhaseConfig:
    """Configuration for a single phase in a workflow."""
    name: str
    sequence: str  # Which sequence pattern to use (single_pass, train_test, etc.)
    topology: str  # Which topology to use (backtest, signal_generation, etc.)
    description: str
    config: Dict[str, Any]  # Phase-specific configuration
    input: Dict[str, str] = field(default_factory=dict)  # Inter-phase data references
    output: Dict[str, Any] = field(default_factory=dict)  # What to collect/store
    depends_on: List[str] = field(default_factory=list)  # Phase dependencies


@dataclass
class WorkflowBranch:
    """Defines a conditional branch in workflow execution."""
    condition: callable  # Function that evaluates result -> bool
    workflow: str  # Name of workflow to execute if condition is true
    config_modifier: Optional[callable] = None  # Function to modify config for branch


@runtime_checkable
class WorkflowProtocol(Protocol):
    """
    Protocol for high-level workflows.
    
    Workflows define WHAT to execute - the business process.
    They compose sequences and topologies to achieve business goals.
    
    Workflows can optionally be composable, supporting:
    - Iterative execution until conditions are met
    - Conditional branching based on results
    - Dynamic configuration modification
    """
    
    defaults: Dict[str, Any]  # Default configuration values
    
    def get_phases(self, config: Dict[str, Any]) -> Dict[str, PhaseConfig]:
        """
        Convert user configuration into phase definitions.
        
        Args:
            config: User-provided configuration
            
        Returns:
            Dict mapping phase names to PhaseConfig objects
        """
        ...
    
    # Optional composable methods
    
    def should_continue(self, result: Dict[str, Any], iteration: int) -> bool:
        """
        Check if workflow should continue iterating.
        
        Optional method for composable workflows.
        
        Args:
            result: Result from previous iteration
            iteration: Current iteration number
            
        Returns:
            True if workflow should continue, False to stop
        """
        return False  # Default: don't iterate
    
    def get_branches(self, result: Dict[str, Any]) -> Optional[List[WorkflowBranch]]:
        """
        Get conditional branches based on results.
        
        Optional method for composable workflows.
        
        Args:
            result: Result from current workflow execution
            
        Returns:
            List of possible branches, or None
        """
        return None  # Default: no branching
    
    def modify_config_for_next(self, config: Dict[str, Any], 
                              result: Dict[str, Any], 
                              iteration: int) -> Dict[str, Any]:
        """
        Modify configuration for next iteration.
        
        Optional method for composable workflows.
        
        Args:
            config: Current configuration
            result: Result from previous iteration
            iteration: Current iteration number
            
        Returns:
            Modified configuration for next iteration
        """
        return config  # Default: no modification


@runtime_checkable
class SequenceProtocol(Protocol):
    """
    Protocol for execution sequences.
    
    Sequences define HOW to execute phases - the orchestration pattern.
    Examples: single_pass, train_test, walk_forward, monte_carlo
    """
    
    def execute(
        self, 
        phase_config: PhaseConfig,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a phase according to the sequence pattern.
        
        Args:
            phase_config: Configuration for this phase
            context: Execution context with inter-phase data
            
        Returns:
            Dict with phase results
        """
        ...


@runtime_checkable
class PhaseEnhancerProtocol(Protocol):
    """
    Protocol for composable phase enhancers.
    
    These components can modify phase configurations to add
    capabilities like logging, monitoring, caching, etc.
    """
    
    def enhance(self, phases: Dict[str, PhaseConfig]) -> Dict[str, PhaseConfig]:
        """
        Enhance phase configurations.
        
        Args:
            phases: Original phase configurations
            
        Returns:
            Enhanced phase configurations
        """
        ...


@runtime_checkable
class ResultAggregatorProtocol(Protocol):
    """
    Protocol for result aggregation across phases.
    
    These components aggregate results from multiple phases
    into summary metrics, reports, etc.
    """
    
    def aggregate(self, phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate results from multiple phases.
        
        Args:
            phase_results: Results from each phase
            
        Returns:
            Aggregated results
        """
        ...


@runtime_checkable
class OptimizerProtocol(Protocol):
    """
    Protocol for parameter optimizers.
    
    Used by sequences like train_test to optimize parameters.
    """
    
    def optimize(
        self,
        parameter_space: Dict[str, Any],
        objective_function: callable,
        method: str = 'grid'
    ) -> Dict[str, Any]:
        """
        Optimize parameters.
        
        Args:
            parameter_space: Space of parameters to search
            objective_function: Function to optimize
            method: Optimization method (grid, random, bayesian, etc.)
            
        Returns:
            Dict with optimal parameters and performance
        """
        ...
    
    def select_parameters(
        self,
        results: Dict[str, Any],
        method: str = 'top_n',
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Select parameters based on optimization results.
        
        Args:
            results: Optimization results with parameter performance
            method: Selection method (top_n, threshold, clustering, etc.)
            **kwargs: Method-specific arguments (n for top_n, etc.)
            
        Returns:
            List of selected parameter sets
        """
        ...


@runtime_checkable
class Optimizable(Protocol):
    """
    Protocol for components that can be optimized.
    
    Any component implementing this protocol can participate
    in optimization workflows. This supports the recursive
    parameter extraction for ensemble strategies.
    """
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dict mapping parameter names to:
                - List[Any]: discrete values
                - Tuple[float, float]: continuous range (min, max)
                - Dict with 'type', 'min', 'max', 'step', 'default', etc.
        """
        ...
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Apply parameter values."""
        ...
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values."""
        ...
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate parameter values.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        ...


@runtime_checkable
class StrategyCompilerProtocol(Protocol):
    """Protocol for compiling compositional strategy configurations into executable strategies."""
    
    def compile_strategies(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Compile strategy configuration into list of executable strategy specs.
        
        For grid search, this expands to many strategies.
        For single strategy, returns list of one.
        
        Args:
            config: Strategy configuration (can be atomic, composite, conditional, etc.)
        
        Returns:
            List of dicts with:
                - 'id': Unique strategy identifier
                - 'function': Compiled strategy function
                - 'features': List[FeatureSpec] needed
                - 'metadata': Additional info (params, composition structure, etc.)
        """
        ...
    
    def extract_features(self, config: Dict[str, Any]) -> List['FeatureSpec']:
        """
        Recursively extract all features needed by the compositional strategy.
        
        Args:
            config: Strategy configuration
            
        Returns:
            List of FeatureSpec objects for all required features
        """
        ...

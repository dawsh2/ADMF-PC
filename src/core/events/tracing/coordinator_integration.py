"""
Integration of event tracing with the Coordinator and workflow management.
Ensures tracing works across complex multi-phase workflows while respecting
container isolation and route communication patterns.
"""
import logging
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

from src.core.events.tracing.event_store import EventStore
from src.core.events.tracing.storage_backends import StorageBackend
from src.core.events.tracing.route_integration import (
    TracingConfig, ContainerIsolationTracer, create_tracing_route
)
from src.core.coordinator.protocols import Coordinator, WorkflowPhase
from src.core.containers.protocols import Container


@dataclass
class WorkflowTracingContext:
    """Context for tracing a complete workflow execution."""
    workflow_id: str
    workflow_type: str  # "backtest", "optimization", "walk_forward", etc.
    correlation_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    phases_completed: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Container isolation tracking
    isolation_groups: Dict[str, str] = field(default_factory=dict)  # container -> group
    cross_boundary_events: List[str] = field(default_factory=list)  # event_ids
    
    def add_phase_completion(self, phase_name: str) -> None:
        """Record phase completion."""
        self.phases_completed.append(phase_name)
        
    def finalize(self) -> None:
        """Mark workflow as complete."""
        self.end_time = datetime.now()
        self.metadata['duration_seconds'] = (
            self.end_time - self.start_time
        ).total_seconds()


class TracingCoordinatorMixin:
    """
    Mixin for Coordinators to add workflow-aware event tracing.
    Handles complex scenarios like optimization runs with multiple isolated containers.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_store: Optional[EventStore] = None
        self.storage_backend: Optional[StorageBackend] = None
        self.isolation_tracer: Optional[ContainerIsolationTracer] = None
        self.workflow_contexts: Dict[str, WorkflowTracingContext] = {}
        self.logger = logging.getLogger(f"{__name__}.TracingCoordinator")
        
        # Tracing configuration
        self.tracing_enabled = self.config.get('tracing', {}).get('enabled', True)
        self.trace_optimization_details = self.config.get('tracing', {}).get(
            'trace_optimization_details', True
        )
        
    def setup_tracing(
        self, 
        storage_backend: StorageBackend,
        event_store: Optional[EventStore] = None
    ) -> None:
        """Initialize tracing for the coordinator."""
        self.storage_backend = storage_backend
        self.event_store = event_store or EventStore(storage_backend)
        self.isolation_tracer = ContainerIsolationTracer(self.event_store)
        
        self.logger.info("Tracing enabled for coordinator")
        
    @contextmanager
    def trace_workflow(
        self,
        workflow_type: str,
        correlation_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracing a complete workflow."""
        if not self.tracing_enabled or not self.event_store:
            yield None
            return
            
        # Create workflow context
        workflow_id = f"{workflow_type}_{correlation_id}"
        context = WorkflowTracingContext(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            correlation_id=correlation_id,
            metadata=metadata or {}
        )
        
        self.workflow_contexts[workflow_id] = context
        
        try:
            self.logger.info(f"Starting traced workflow: {workflow_id}")
            yield context
        finally:
            # Finalize and store workflow context
            context.finalize()
            self._store_workflow_context(context)
            
            # Clean up
            del self.workflow_contexts[workflow_id]
            self.logger.info(
                f"Completed traced workflow: {workflow_id} "
                f"(duration: {context.metadata.get('duration_seconds', 0):.2f}s)"
            )
            
    def _store_workflow_context(self, context: WorkflowTracingContext) -> None:
        """Store workflow context as a special event."""
        workflow_event = {
            'event_type': 'WorkflowComplete',
            'workflow_id': context.workflow_id,
            'workflow_type': context.workflow_type,
            'correlation_id': context.correlation_id,
            'start_time': context.start_time.isoformat(),
            'end_time': context.end_time.isoformat() if context.end_time else None,
            'duration_seconds': context.metadata.get('duration_seconds'),
            'phases_completed': context.phases_completed,
            'metadata': context.metadata,
            'isolation_summary': {
                'groups': dict(context.isolation_groups),
                'cross_boundary_event_count': len(context.cross_boundary_events)
            }
        }
        
        # Store as a traced event
        from src.core.events.tracing.traced_event import TracedEvent
        traced_event = TracedEvent(
            event_id=f"workflow_{context.workflow_id}",
            correlation_id=context.correlation_id,
            causation_id=None,
            source_container="coordinator",
            target_container=None,
            event_type="WorkflowComplete",
            timestamp=datetime.now(),
            data=workflow_event,
            metadata={'is_workflow_summary': True}
        )
        
        self.event_store.store_single(traced_event)
        
    def trace_phase_transition(
        self,
        from_phase: Optional[str],
        to_phase: str,
        workflow_id: str
    ) -> None:
        """Trace phase transitions in multi-phase workflows."""
        if not self.tracing_enabled or workflow_id not in self.workflow_contexts:
            return
            
        context = self.workflow_contexts[workflow_id]
        context.add_phase_completion(from_phase or "init")
        
        # Store phase transition event
        transition_event = {
            'event_type': 'PhaseTransition',
            'workflow_id': workflow_id,
            'from_phase': from_phase,
            'to_phase': to_phase,
            'timestamp': datetime.now().isoformat()
        }
        
        from src.core.events.tracing.traced_event import TracedEvent
        traced_event = TracedEvent(
            event_id=f"phase_{workflow_id}_{to_phase}",
            correlation_id=context.correlation_id,
            causation_id=None,
            source_container="coordinator",
            target_container=None,
            event_type="PhaseTransition",
            timestamp=datetime.now(),
            data=transition_event
        )
        
        self.event_store.store_single(traced_event)
        
    def setup_container_isolation_tracing(
        self,
        container: Container,
        isolation_group: str,
        workflow_id: str,
        allowed_targets: Optional[List[str]] = None
    ) -> None:
        """
        Setup isolation tracking for a container.
        Critical for optimization runs where each parameter set gets isolated containers.
        """
        if not self.isolation_tracer:
            return
            
        # Register isolation context
        isolation_context = {
            'isolation_group': isolation_group,
            'workflow_id': workflow_id,
            'allowed_targets': allowed_targets or [],
            'created_at': datetime.now().isoformat()
        }
        
        self.isolation_tracer.register_isolation_context(
            container.name,
            isolation_context
        )
        
        # Track in workflow context
        if workflow_id in self.workflow_contexts:
            context = self.workflow_contexts[workflow_id]
            context.isolation_groups[container.name] = isolation_group
            
        self.logger.debug(
            f"Setup isolation tracing for {container.name} "
            f"in group {isolation_group}"
        )


class OptimizationTracingExtension:
    """
    Specialized tracing for optimization workflows.
    Handles parameter sweeps, objective tracking, and result correlation.
    """
    
    def __init__(self, coordinator: TracingCoordinatorMixin):
        self.coordinator = coordinator
        self.event_store = coordinator.event_store
        self.logger = logging.getLogger(f"{__name__}.OptimizationTracing")
        
        # Track optimization runs
        self.optimization_runs: Dict[str, Dict[str, Any]] = {}
        
    def trace_optimization_start(
        self,
        optimization_id: str,
        strategy_type: str,
        parameter_space: Dict[str, Any],
        objective_function: str
    ) -> None:
        """Trace the start of an optimization run."""
        run_info = {
            'optimization_id': optimization_id,
            'strategy_type': strategy_type,
            'parameter_space': parameter_space,
            'objective_function': objective_function,
            'start_time': datetime.now(),
            'parameter_sets_evaluated': 0,
            'best_result': None
        }
        
        self.optimization_runs[optimization_id] = run_info
        
        # Store as event
        self._store_optimization_event(
            optimization_id,
            'OptimizationStart',
            run_info
        )
        
    def trace_parameter_evaluation(
        self,
        optimization_id: str,
        parameter_set: Dict[str, Any],
        correlation_id: str,
        isolation_group: str
    ) -> None:
        """Trace evaluation of a specific parameter set."""
        if optimization_id not in self.optimization_runs:
            return
            
        # Update run info
        run_info = self.optimization_runs[optimization_id]
        run_info['parameter_sets_evaluated'] += 1
        
        # Store parameter evaluation event
        eval_event = {
            'optimization_id': optimization_id,
            'parameter_set': parameter_set,
            'correlation_id': correlation_id,
            'isolation_group': isolation_group,
            'evaluation_number': run_info['parameter_sets_evaluated']
        }
        
        self._store_optimization_event(
            optimization_id,
            'ParameterEvaluation',
            eval_event,
            correlation_id=correlation_id
        )
        
    def trace_evaluation_result(
        self,
        optimization_id: str,
        correlation_id: str,
        result: Dict[str, Any]
    ) -> None:
        """Trace the result of a parameter evaluation."""
        if optimization_id not in self.optimization_runs:
            return
            
        # Check if this is the best result
        run_info = self.optimization_runs[optimization_id]
        if (run_info['best_result'] is None or 
            result.get('objective_value', float('-inf')) > 
            run_info['best_result'].get('objective_value', float('-inf'))):
            run_info['best_result'] = result
            
        # Store result event
        result_event = {
            'optimization_id': optimization_id,
            'correlation_id': correlation_id,
            'objective_value': result.get('objective_value'),
            'metrics': result.get('metrics', {}),
            'is_best_so_far': result == run_info['best_result']
        }
        
        self._store_optimization_event(
            optimization_id,
            'EvaluationResult',
            result_event,
            correlation_id=correlation_id
        )
        
    def trace_optimization_complete(
        self,
        optimization_id: str,
        best_parameters: Dict[str, Any],
        best_correlation_id: str
    ) -> None:
        """Trace optimization completion."""
        if optimization_id not in self.optimization_runs:
            return
            
        run_info = self.optimization_runs[optimization_id]
        
        # Store completion event
        complete_event = {
            'optimization_id': optimization_id,
            'total_evaluations': run_info['parameter_sets_evaluated'],
            'best_parameters': best_parameters,
            'best_correlation_id': best_correlation_id,
            'best_objective_value': run_info['best_result'].get('objective_value')
            if run_info['best_result'] else None,
            'duration_seconds': (
                datetime.now() - run_info['start_time']
            ).total_seconds()
        }
        
        self._store_optimization_event(
            optimization_id,
            'OptimizationComplete',
            complete_event
        )
        
        # Clean up
        del self.optimization_runs[optimization_id]
        
    def _store_optimization_event(
        self,
        optimization_id: str,
        event_type: str,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> None:
        """Store an optimization-related event."""
        from src.core.events.tracing.traced_event import TracedEvent
        
        traced_event = TracedEvent(
            event_id=f"opt_{optimization_id}_{event_type}_{datetime.now().timestamp()}",
            correlation_id=correlation_id or optimization_id,
            causation_id=None,
            source_container="optimization_coordinator",
            target_container=None,
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            metadata={'optimization_id': optimization_id}
        )
        
        self.event_store.store_single(traced_event)
        
    def get_optimization_trace(
        self,
        optimization_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get complete trace for an optimization run.
        Includes all parameter evaluations and results.
        """
        events = list(self.event_store.query_events(
            metadata={'optimization_id': optimization_id}
        ))
        
        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        # Convert to dictionaries for easy analysis
        return [
            {
                'timestamp': e.timestamp,
                'event_type': e.event_type,
                'correlation_id': e.correlation_id,
                'data': e.data
            }
            for e in events
        ]


class WalkForwardTracingExtension:
    """
    Specialized tracing for walk-forward analysis.
    Tracks training/testing windows and out-of-sample performance.
    """
    
    def __init__(self, coordinator: TracingCoordinatorMixin):
        self.coordinator = coordinator
        self.event_store = coordinator.event_store
        self.logger = logging.getLogger(f"{__name__}.WalkForwardTracing")
        
    def trace_window(
        self,
        window_id: str,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
        correlation_id: str
    ) -> None:
        """Trace a walk-forward window."""
        window_event = {
            'window_id': window_id,
            'train_period': {
                'start': train_start.isoformat(),
                'end': train_end.isoformat(),
                'days': (train_end - train_start).days
            },
            'test_period': {
                'start': test_start.isoformat(),
                'end': test_end.isoformat(),
                'days': (test_end - test_start).days
            }
        }
        
        from src.core.events.tracing.traced_event import TracedEvent
        traced_event = TracedEvent(
            event_id=f"wf_window_{window_id}",
            correlation_id=correlation_id,
            causation_id=None,
            source_container="walk_forward_coordinator",
            target_container=None,
            event_type="WalkForwardWindow",
            timestamp=datetime.now(),
            data=window_event
        )
        
        self.event_store.store_single(traced_event)
        
    def trace_window_result(
        self,
        window_id: str,
        correlation_id: str,
        in_sample_metrics: Dict[str, float],
        out_of_sample_metrics: Dict[str, float]
    ) -> None:
        """Trace results for a walk-forward window."""
        result_event = {
            'window_id': window_id,
            'in_sample_metrics': in_sample_metrics,
            'out_of_sample_metrics': out_of_sample_metrics,
            'degradation': {
                metric: (
                    (in_sample_metrics.get(metric, 0) - out_of_sample_metrics.get(metric, 0)) /
                    in_sample_metrics.get(metric, 1) * 100
                    if in_sample_metrics.get(metric, 0) != 0 else 0
                )
                for metric in in_sample_metrics
                if metric in out_of_sample_metrics
            }
        }
        
        from src.core.events.tracing.traced_event import TracedEvent
        traced_event = TracedEvent(
            event_id=f"wf_result_{window_id}",
            correlation_id=correlation_id,
            causation_id=None,
            source_container="walk_forward_coordinator",
            target_container=None,
            event_type="WalkForwardResult",
            timestamp=datetime.now(),
            data=result_event
        )
        
        self.event_store.store_single(traced_event)
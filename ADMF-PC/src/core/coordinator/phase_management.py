"""
Phase management components for the Coordinator.

This module implements the critical architectural decisions from TEST_WORKFLOW.MD:
1. Event flow between phases with clear data transitions
2. Container naming & tracking strategy
3. Result storage & streaming aggregation
4. Cross-regime strategy identity tracking
5. Coordinator state management with checkpointing
6. Shared service versioning
"""

from typing import Dict, Any, List, Optional, Callable, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import asyncio
from collections import deque
import logging

from ..containers import UniversalScopedContainer
from ..events import Event, EventType
from .types import WorkflowPhase
from .protocols import ResultStreamer, CheckpointManager

logger = logging.getLogger(__name__)


@dataclass
class PhaseTransition:
    """
    Manages data flow between phases.
    
    Critical Decision #1: Clear pattern for phase data flow.
    """
    phase_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __init__(self):
        self.phase1_outputs = {
            'signals_by_regime': {},      # For weight optimization
            'parameter_performance': {},   # For analysis
            'regime_transitions': []       # For robustness
        }
        self.phase2_outputs = {
            'regime_best_params': {},      # Best params per regime
            'regime_performance': {},      # Performance per regime
            'classifier_comparison': {}    # Cross-classifier analysis
        }
        self.phase3_outputs = {
            'optimal_weights': {},         # Signal weights
            'ensemble_performance': {},    # Combined performance
            'weight_stability': {}         # Weight consistency
        }
        
    def record_phase_output(self, phase: str, key: str, value: Any) -> None:
        """Record output from a phase for use in subsequent phases."""
        phase_key = f"phase{phase}_outputs"
        if hasattr(self, phase_key):
            outputs = getattr(self, phase_key)
            outputs[key] = value
            
    def get_phase_input(self, phase: str, key: str) -> Optional[Any]:
        """Get input data for a phase from previous phase outputs."""
        # Phase 2 gets data from Phase 1
        if phase == "2" and hasattr(self, "phase1_outputs"):
            return self.phase1_outputs.get(key)
        # Phase 3 gets data from Phase 1 and 2
        elif phase == "3":
            # Check Phase 2 first
            if hasattr(self, "phase2_outputs") and key in self.phase2_outputs:
                return self.phase2_outputs[key]
            # Fall back to Phase 1
            if hasattr(self, "phase1_outputs") and key in self.phase1_outputs:
                return self.phase1_outputs[key]
        return None


class ContainerNamingStrategy:
    """
    Container naming & tracking strategy.
    
    Critical Decision #2: Consistent naming scheme for debugging.
    """
    
    @staticmethod
    def generate_container_id(
        phase: str,
        regime: str,
        strategy: str,
        params: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Generate consistent container ID with all context.
        
        Format: {phase}_{regime}_{strategy}_{params_hash}_{timestamp}
        Example: phase1_hmm_bull_ma520_hash123_20240115
        """
        # Generate params hash
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        
        # Format timestamp
        ts = timestamp or datetime.now()
        ts_str = ts.strftime("%Y%m%d_%H%M%S")
        
        # Clean regime and strategy names
        regime_clean = regime.lower().replace(" ", "_")
        strategy_clean = strategy.lower().replace(" ", "_")
        
        # Build container ID
        container_id = f"{phase}_{regime_clean}_{strategy_clean}_{params_hash}_{ts_str}"
        
        return container_id
    
    @staticmethod
    def parse_container_id(container_id: str) -> Dict[str, str]:
        """Parse container ID to extract context."""
        parts = container_id.split("_")
        
        if len(parts) >= 6:
            return {
                'phase': parts[0],
                'regime': parts[1],
                'strategy': parts[2],
                'params_hash': parts[3],
                'date': parts[4],
                'time': parts[5]
            }
        return {}


class StreamingResultWriter:
    """Streams results to disk immediately to avoid memory issues."""
    
    def __init__(self, output_dir: str, buffer_size: int = 100):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self._buffers: Dict[str, deque] = {}
        self._file_handles: Dict[str, Any] = {}
        
    def write(self, container_id: str, result: Dict[str, Any]) -> None:
        """Stream result to disk immediately."""
        # Get or create buffer for this container
        if container_id not in self._buffers:
            self._buffers[container_id] = deque(maxlen=self.buffer_size)
            
        # Add to buffer
        self._buffers[container_id].append({
            'timestamp': datetime.now().isoformat(),
            'result': result
        })
        
        # Write if buffer is full
        if len(self._buffers[container_id]) >= self.buffer_size:
            self._flush_buffer(container_id)
    
    def _flush_buffer(self, container_id: str) -> None:
        """Flush buffer to disk."""
        if container_id not in self._buffers:
            return
            
        buffer = self._buffers[container_id]
        if not buffer:
            return
            
        # Create file path
        file_path = self.output_dir / f"{container_id}_results.jsonl"
        
        # Write buffer to file (append mode)
        with open(file_path, 'a') as f:
            while buffer:
                record = buffer.popleft()
                f.write(json.dumps(record) + '\n')
    
    def flush_all(self) -> None:
        """Flush all buffers."""
        for container_id in list(self._buffers.keys()):
            self._flush_buffer(container_id)
    
    def close(self) -> None:
        """Close writer and flush remaining data."""
        self.flush_all()
        self._buffers.clear()


class ResultAggregator:
    """
    Result storage & aggregation pattern.
    
    Critical Decision #3: Stream to disk, cache top performers only.
    """
    
    def __init__(self, output_dir: str, cache_size: int = 1000):
        self.streaming_writer = StreamingResultWriter(output_dir)
        self.in_memory_cache: Dict[str, Any] = {}
        self.cache_size = cache_size
        self.top_performers: List[Tuple[str, float]] = []  # (container_id, score)
        
    def handle_container_result(self, container_id: str, result: Dict[str, Any]) -> None:
        """Handle result from container - stream to disk and cache if top performer."""
        # Stream to disk immediately
        self.streaming_writer.write(container_id, result)
        
        # Cache top performers only
        if self._is_top_performer(result):
            score = self._extract_score(result)
            
            # Add to top performers
            self.top_performers.append((container_id, score))
            self.top_performers.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only top N
            if len(self.top_performers) > self.cache_size:
                self.top_performers = self.top_performers[:self.cache_size]
            
            # Update cache
            top_ids = {cid for cid, _ in self.top_performers}
            
            # Remove non-top performers from cache
            for cid in list(self.in_memory_cache.keys()):
                if cid not in top_ids:
                    del self.in_memory_cache[cid]
            
            # Add to cache if in top performers
            if container_id in top_ids:
                self.in_memory_cache[container_id] = result
    
    def _is_top_performer(self, result: Dict[str, Any]) -> bool:
        """Check if result qualifies as top performer."""
        score = self._extract_score(result)
        
        # If cache not full, it's a top performer
        if len(self.top_performers) < self.cache_size:
            return True
            
        # Otherwise, check if better than worst cached
        if self.top_performers:
            worst_score = self.top_performers[-1][1]
            return score > worst_score
            
        return True
    
    def _extract_score(self, result: Dict[str, Any]) -> float:
        """Extract performance score from result."""
        # Try different score fields
        for field in ['sharpe_ratio', 'total_return', 'score', 'performance']:
            if field in result:
                return float(result[field])
        return 0.0
    
    def get_top_results(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N results from cache."""
        top_n = []
        for container_id, _ in self.top_performers[:n]:
            if container_id in self.in_memory_cache:
                top_n.append(self.in_memory_cache[container_id])
        return top_n
    
    def close(self) -> None:
        """Close aggregator and flush data."""
        self.streaming_writer.close()


@dataclass
class StrategyIdentity:
    """
    Cross-regime strategy identity tracking.
    
    Critical Decision #4: Track same strategy across regime environments.
    """
    canonical_id: str
    base_class: str
    base_params: Dict[str, Any]
    regime_instances: Dict[str, str] = field(default_factory=dict)  # regime -> container_id
    
    def __init__(self, base_class: str, base_params: Dict[str, Any]):
        self.base_class = base_class
        self.base_params = base_params
        self.canonical_id = self._generate_canonical_id(base_class, base_params)
        self.regime_instances = {}
        
    def _generate_canonical_id(self, base_class: str, base_params: Dict[str, Any]) -> str:
        """Generate consistent ID for strategy regardless of regime."""
        # Sort params for consistency
        params_str = json.dumps(base_params, sort_keys=True)
        
        # Create hash
        id_str = f"{base_class}_{params_str}"
        return hashlib.sha256(id_str.encode()).hexdigest()[:16]
    
    def add_regime_instance(self, regime: str, container_id: str) -> None:
        """Track strategy instance in specific regime."""
        self.regime_instances[regime] = container_id
    
    def get_regime_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance across all regimes for this strategy."""
        performance = {}
        
        for regime, container_id in self.regime_instances.items():
            if container_id in results:
                performance[regime] = results[container_id]
                
        return performance


class WorkflowState:
    """Tracks workflow state for checkpointing."""
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.current_phase: Optional[str] = None
        self.completed_phases: Set[str] = set()
        self.phase_results: Dict[str, Any] = {}
        self.active_containers: Set[str] = set()
        self.created_at = datetime.now()
        self.last_checkpoint = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'workflow_id': self.workflow_id,
            'current_phase': self.current_phase,
            'completed_phases': list(self.completed_phases),
            'phase_results': self.phase_results,
            'active_containers': list(self.active_containers),
            'created_at': self.created_at.isoformat(),
            'last_checkpoint': self.last_checkpoint.isoformat() if self.last_checkpoint else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        """Create from dictionary."""
        state = cls(data['workflow_id'])
        state.current_phase = data.get('current_phase')
        state.completed_phases = set(data.get('completed_phases', []))
        state.phase_results = data.get('phase_results', {})
        state.active_containers = set(data.get('active_containers', []))
        state.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('last_checkpoint'):
            state.last_checkpoint = datetime.fromisoformat(data['last_checkpoint'])
        return state


class CheckpointManager:
    """
    Manages workflow checkpoints for resumability.
    
    Critical Decision #5: Large optimizations need resumability.
    """
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_state(self, state: WorkflowState) -> str:
        """Save workflow state to checkpoint."""
        state.last_checkpoint = datetime.now()
        
        # Create checkpoint file
        checkpoint_file = self.checkpoint_dir / f"{state.workflow_id}_checkpoint.json"
        
        # Write state
        with open(checkpoint_file, 'w') as f:
            json.dump(state.to_dict(), f, indent=2)
            
        logger.info(f"Saved checkpoint for workflow {state.workflow_id}")
        return str(checkpoint_file)
    
    def restore_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Restore workflow state from checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{workflow_id}_checkpoint.json"
        
        if not checkpoint_file.exists():
            return None
            
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            state = WorkflowState.from_dict(data)
            logger.info(f"Restored checkpoint for workflow {workflow_id}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return None
    
    def delete_checkpoint(self, workflow_id: str) -> None:
        """Delete checkpoint after successful completion."""
        checkpoint_file = self.checkpoint_dir / f"{workflow_id}_checkpoint.json"
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info(f"Deleted checkpoint for workflow {workflow_id}")


class WorkflowCoordinator:
    """
    Enhanced coordinator with phase management.
    
    Critical Decision #5: Track complex state across phases.
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.workflow_state = WorkflowState("")  # Will be set per workflow
        self.checkpointing = CheckpointManager(checkpoint_dir)
        self.phase_transitions = PhaseTransition()
        self.result_aggregator: Optional[ResultAggregator] = None
        
    def execute_phase(self, phase_name: str, phase_func: Callable) -> Any:
        """Execute phase with checkpointing and error recovery."""
        # Update state
        self.workflow_state.current_phase = phase_name
        
        # Checkpoint before phase
        self.checkpointing.save_state(self.workflow_state)
        
        try:
            # Execute phase
            result = phase_func()
            
            # Record completion
            self.workflow_state.completed_phases.add(phase_name)
            self.workflow_state.phase_results[phase_name] = result
            
            # Record phase output for transitions
            if hasattr(result, 'items'):
                for key, value in result.items():
                    self.phase_transitions.record_phase_output(
                        phase_name.replace("phase", ""), 
                        key, 
                        value
                    )
            
            return result
            
        except Exception as e:
            # Can resume from checkpoint
            logger.error(f"Phase {phase_name} failed: {e}")
            
            # Try to restore and retry
            restored_state = self.checkpointing.restore_state(
                self.workflow_state.workflow_id
            )
            if restored_state:
                self.workflow_state = restored_state
                logger.info(f"Restored state from checkpoint")
            
            raise


class SharedServiceRegistry:
    """
    Shared service versioning for compatibility.
    
    Critical Decision #6: Prevent breaking changes.
    """
    
    def __init__(self):
        self.services: Dict[str, Any] = {}
        self.service_versions: Dict[str, str] = {}
        
    def register_service(self, name: str, service: Any, version: str = "1.0") -> None:
        """Version shared services for compatibility."""
        versioned_name = f"{name}_v{version}"
        
        # Register versioned service
        self.services[versioned_name] = service
        
        # Update latest pointer
        self.services[name] = service
        
        # Track version
        self.service_versions[name] = version
        
        logger.info(f"Registered service {name} version {version}")
    
    def get_service(self, name: str, version: Optional[str] = None) -> Any:
        """Get service by name and optional version."""
        if version:
            versioned_name = f"{name}_v{version}"
            if versioned_name in self.services:
                return self.services[versioned_name]
            else:
                raise ValueError(f"Service {name} version {version} not found")
        else:
            # Return latest version
            if name in self.services:
                return self.services[name]
            else:
                raise ValueError(f"Service {name} not found")
    
    def get_service_version(self, name: str) -> Optional[str]:
        """Get current version of a service."""
        return self.service_versions.get(name)


# Walk-forward validation support
class WalkForwardValidator:
    """
    Ensures identical execution paths for walk-forward validation.
    
    Critical for multi-period optimization validation.
    """
    
    def __init__(self, coordinator: 'Coordinator'):
        self.coordinator = coordinator
        self.execution_traces: List[Dict[str, Any]] = []
        
    def record_execution(self, phase: str, container_id: str, params: Dict[str, Any]) -> None:
        """Record execution for reproducibility."""
        trace = {
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'container_id': container_id,
            'params': params.copy()
        }
        self.execution_traces.append(trace)
    
    def validate_execution_path(self, expected_traces: List[Dict[str, Any]]) -> bool:
        """Validate that execution follows expected path."""
        if len(self.execution_traces) != len(expected_traces):
            return False
            
        for actual, expected in zip(self.execution_traces, expected_traces):
            # Compare phase and params (ignore timestamps and container IDs)
            if actual['phase'] != expected['phase']:
                return False
            if actual['params'] != expected['params']:
                return False
                
        return True
    
    def create_forward_period(
        self, 
        optimization_results: Dict[str, Any],
        test_start: datetime,
        test_end: datetime
    ) -> Dict[str, Any]:
        """Create configuration for walk-forward test period."""
        return {
            'period': {
                'start': test_start,
                'end': test_end
            },
            'optimal_params': optimization_results['best_params'],
            'regime_params': optimization_results.get('regime_params', {}),
            'expected_regimes': optimization_results.get('detected_regimes', [])
        }


def integrate_phase_management(coordinator: 'Coordinator') -> None:
    """
    Integrate phase management capabilities into existing Coordinator.
    
    This function adds the critical architectural features to the Coordinator.
    """
    # Add phase management attributes
    coordinator.phase_transitions = PhaseTransition()
    coordinator.container_naming = ContainerNamingStrategy()
    coordinator.checkpointing = CheckpointManager("./checkpoints")
    coordinator.service_registry = SharedServiceRegistry()
    coordinator.walk_forward_validator = WalkForwardValidator(coordinator)
    
    # Add strategy identity tracking
    coordinator.strategy_identities: Dict[str, StrategyIdentity] = {}
    
    # Override execute_phase method
    original_execute = coordinator.execute_workflow
    
    async def enhanced_execute_workflow(config):
        """Enhanced workflow execution with phase management."""
        workflow_id = config.get('workflow_id', str(uuid.uuid4()))
        
        # Initialize workflow state
        coordinator.workflow_state = WorkflowState(workflow_id)
        
        # Check for existing checkpoint
        restored_state = coordinator.checkpointing.restore_state(workflow_id)
        if restored_state:
            coordinator.workflow_state = restored_state
            logger.info(f"Resuming workflow {workflow_id} from checkpoint")
        
        # Initialize result aggregator
        output_dir = config.get('output_dir', f"./results/{workflow_id}")
        coordinator.result_aggregator = ResultAggregator(output_dir)
        
        try:
            # Execute with enhanced tracking
            result = await original_execute(config)
            
            # Clean up checkpoint on success
            coordinator.checkpointing.delete_checkpoint(workflow_id)
            
            return result
            
        finally:
            # Close result aggregator
            if coordinator.result_aggregator:
                coordinator.result_aggregator.close()
    
    coordinator.execute_workflow = enhanced_execute_workflow
    
    # Add helper methods
    def track_strategy_identity(self, strategy_class: str, params: Dict[str, Any]) -> StrategyIdentity:
        """Track strategy identity across regimes."""
        identity = StrategyIdentity(strategy_class, params)
        self.strategy_identities[identity.canonical_id] = identity
        return identity
    
    coordinator.track_strategy_identity = track_strategy_identity.__get__(coordinator)
    
    logger.info("Phase management integrated into Coordinator")
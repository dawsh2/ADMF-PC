# Unified Migration Plan: Clean Architecture with Event-Based Result Streaming

## Overview

This migration plan incorporates key architectural insights:
1. **Result Streaming via Event Tracing** - Events are the single source of truth for all results
2. **Clean Architectural Boundaries** - Orchestration layer stays OUTSIDE event system
3. **Synchronous Execution** - No async needed for ADMF-PC's current architecture
4. **Phase Isolation** - Each phase gets fresh event system, torn down after execution

## Core Architectural Principles

### 1. Layer Separation

```
Orchestration Layer (NO EVENTS):
├── Coordinator      # Manages workflows
├── Sequencer       # Manages phases  
└── TopologyBuilder # Builds topologies

Execution Layer (EVENTS):
├── Containers      # Isolated execution units
├── EventBus        # Phase-scoped events
└── Components      # Event producers/consumers
```

### 2. Phase Isolation

Each phase:
- Gets a FRESH event system
- Runs in complete isolation
- Has event system torn down after execution
- Cannot contaminate other phases

## Phase 1: Event Tracing Enhancement with Result Extraction

### 1.1 Create Result Extraction Framework

```python
# src/core/events/result_extraction.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pickle
import zlib
import base64

class ResultExtractor(ABC):
    """Base class for extracting specific results from event stream."""
    
    @abstractmethod
    def can_extract(self, event: Event) -> bool:
        """Check if this extractor can process this event."""
        pass
    
    @abstractmethod
    def extract(self, event: Event) -> Optional[Dict[str, Any]]:
        """Extract result data from event."""
        pass

class PortfolioMetricsExtractor(ResultExtractor):
    """Extract portfolio performance metrics."""
    
    def can_extract(self, event: Event) -> bool:
        return event.type == EventType.PORTFOLIO_UPDATE
    
    def extract(self, event: Event) -> Optional[Dict[str, Any]]:
        if not self.can_extract(event):
            return None
            
        return {
            'timestamp': event.timestamp,
            'container_id': event.source,
            'metrics': {
                'total_value': event.data.get('total_value'),
                'pnl': event.data.get('pnl'),
                'sharpe_ratio': event.data.get('metrics', {}).get('sharpe_ratio'),
                'max_drawdown': event.data.get('metrics', {}).get('max_drawdown')
            }
        }
```

### 1.2 Create Extractor Registry

```python
# src/core/events/extractor_registry.py
class ExtractorRegistry:
    """Central registry for all result extractors."""
    
    def __init__(self):
        self.extractors = {}
        self.sql_mappings = {}  # Extractor -> SQL table
    
    def register(self, name: str, extractor: ResultExtractor, table: str):
        self.extractors[name] = extractor
        self.sql_mappings[name] = table
    
    def get_extractors_for_workflow(self, workflow_type: str) -> List[ResultExtractor]:
        """Get relevant extractors based on workflow type."""
        base_extractors = []
        
        # Always include core extractors
        if 'portfolio_metrics' in self.extractors:
            base_extractors.append(self.extractors['portfolio_metrics'])
        
        # Add workflow-specific extractors
        if workflow_type == 'optimization':
            if 'pattern_discovery' in self.extractors:
                base_extractors.append(self.extractors['pattern_discovery'])
        
        return base_extractors
```

### 1.3 Enhance EventTracer with Result Processing

```python
# src/core/events/enhanced_tracer.py
class EnhancedEventTracer:
    """Event tracer with integrated result extraction."""
    
    def __init__(
        self, 
        trace_id: str,
        trace_file_path: Optional[str] = None,
        result_extractors: Optional[List[ResultExtractor]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.trace_id = trace_id
        self.trace_file_path = trace_file_path
        self.events = []
        self.result_extractors = result_extractors or []
        self.extracted_results = defaultdict(list)
        self.config = config or {}
        
    def trace_event(self, event: Event):
        """Trace event and optionally extract results."""
        # Store event
        self.events.append(event)
        
        # Write to trace file if configured
        if self.trace_file_path:
            self._write_to_file(event)
        
        # Extract results if extractors configured
        for extractor in self.result_extractors:
            if extractor.can_extract(event):
                result = extractor.extract(event)
                if result:
                    extractor_name = extractor.__class__.__name__
                    self.extracted_results[extractor_name].append(result)
    
    def get_extracted_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all extracted results before teardown."""
        return dict(self.extracted_results)
```

## Phase 2: Clean TopologyBuilder (Building Only)

### 2.1 Create Clean TopologyBuilder

```python
# src/core/coordinator/topology_clean.py
class TopologyBuilder:
    """
    Builds topologies based on mode and configuration.
    
    That's it. No workflow logic, no execution, no phase management.
    Just a factory for creating topologies.
    """
    
    def __init__(self):
        """Initialize topology builder."""
        self.topology_creators = {
            'backtest': self._create_backtest_topology,
            'signal_generation': self._create_signal_generation_topology,
            'signal_replay': self._create_signal_replay_topology,
            'analysis': self._create_analysis_topology
        }
        
    def build_topology(self, mode: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a topology for the given mode and configuration.
        
        This is the ONLY public method.
        """
        creator = self.topology_creators.get(mode)
        if not creator:
            raise ValueError(f"Unknown topology mode: {mode}")
        
        topology = creator(config)
        
        # Add metadata
        topology['metadata'] = {
            'mode': mode,
            'created_at': str(datetime.now()),
            'config_hash': self._hash_config(config)
        }
        
        return topology
```

## Phase 3: Sequencer with Clean Event Isolation

### 3.1 Create Phase Data Manager (NOT Event-Based)

```python
# src/core/orchestration/phase_data.py
class PhaseDataManager:
    """
    Simple phase data storage - NOT event-based.
    Manages inter-phase data flow at orchestration level.
    """
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.storage_path = Path(f"./workflows/{workflow_id}/phases")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._memory_cache = {}
        
        # Thresholds for storage strategy
        self.MEMORY_THRESHOLD = 1_000_000      # 1MB
        self.COMPRESS_THRESHOLD = 10_000_000   # 10MB
    
    def store_phase_output(self, phase_name: str, output: Any):
        """Store phase output with size-based strategy."""
        output_size = len(pickle.dumps(output))
        
        if output_size < self.MEMORY_THRESHOLD:
            # Small: Keep in memory
            self._memory_cache[phase_name] = output
            
        elif output_size < self.COMPRESS_THRESHOLD:
            # Medium: Store compressed
            path = self.storage_path / f"{phase_name}_output.pkl.gz"
            compressed = zlib.compress(pickle.dumps(output))
            with open(path, 'wb') as f:
                f.write(compressed)
                
        else:
            # Large: Store uncompressed
            path = self.storage_path / f"{phase_name}_output.pkl"
            with open(path, 'wb') as f:
                pickle.dump(output, f)
    
    def get_phase_output(self, phase_name: str) -> Any:
        """Retrieve phase output."""
        # Check memory first
        if phase_name in self._memory_cache:
            return self._memory_cache[phase_name]
        
        # Check compressed file
        compressed_path = self.storage_path / f"{phase_name}_output.pkl.gz"
        if compressed_path.exists():
            with open(compressed_path, 'rb') as f:
                return pickle.loads(zlib.decompress(f.read()))
        
        # Check uncompressed file
        path = self.storage_path / f"{phase_name}_output.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                return pickle.load(f)
        
        return None
```

### 3.2 Create Clean Sequencer

```python
# src/core/coordinator/sequencer_clean.py
class Sequencer:
    """
    Executes workflow phases in sequence.
    
    Key principles:
    - Orchestration layer - NO event access
    - Each phase gets fresh event system
    - Extract results BEFORE teardown
    - Archive traces for analysis
    - SYNCHRONOUS execution
    """
    
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        trace_dir: Optional[str] = "./traces",
        extractor_registry: Optional[ExtractorRegistry] = None
    ):
        self.checkpoint_dir = checkpoint_dir
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.extractor_registry = extractor_registry or ExtractorRegistry()
        
        # Phase tracking
        self.completed_phases: List[str] = []
        
    def execute_phases(
        self,
        pattern: Dict[str, Any],
        config: Dict[str, Any],
        context: Dict[str, Any],
        topology_builder: Any = None
    ) -> Dict[str, Any]:
        """
        Execute all phases in the workflow pattern.
        SYNCHRONOUS - no async/await!
        """
        workflow_id = context['workflow_id']
        phases = pattern.get('phases', [])
        
        # Create phase data manager
        phase_data_mgr = PhaseDataManager(workflow_id)
        
        # Initialize result
        result = {
            'workflow_id': workflow_id,
            'success': True,
            'phase_results': {},
            'extracted_results': {}
        }
        
        # Execute each phase synchronously
        for phase_index, phase_config in enumerate(phases):
            phase_name = phase_config.get('name', f'phase_{phase_index + 1}')
            
            # Get dependencies from phase data manager
            dependencies = self._load_dependencies(
                phase_config, 
                phase_data_mgr, 
                self.completed_phases
            )
            
            # Merge configs
            phase_config_merged = {**config, **dependencies}
            if 'config_override' in phase_config:
                phase_config_merged.update(phase_config['config_override'])
            
            # Execute phase with isolated event system
            phase_result = self._execute_phase(
                phase_config,
                phase_config_merged,
                context,
                phase_index,
                topology_builder
            )
            
            # Store results
            result['phase_results'][phase_name] = phase_result
            
            # Store phase output for dependencies
            if phase_result.get('output'):
                phase_data_mgr.store_phase_output(phase_name, phase_result['output'])
            
            # Store extracted results
            if phase_result.get('extracted_results'):
                result['extracted_results'][phase_name] = phase_result['extracted_results']
            
            # Mark phase completed
            self.completed_phases.append(phase_name)
            
            # Check for failure
            if not phase_result.get('success', True):
                result['success'] = False
                break
        
        return result
    
    def _execute_phase(
        self,
        phase_config: Dict[str, Any],
        merged_config: Dict[str, Any],
        context: Dict[str, Any],
        phase_index: int,
        topology_builder: Any
    ) -> Dict[str, Any]:
        """Execute a single phase with isolated event system."""
        phase_name = phase_config.get('name', f'phase_{phase_index + 1}')
        topology_mode = phase_config.get('topology', 'backtest')
        workflow_id = context['workflow_id']
        workflow_type = context.get('workflow_type', 'backtest')
        
        # Create FRESH event bus for THIS phase only
        event_bus = EventBus(f"{workflow_id}_{phase_name}")
        
        # Get extractors for this workflow type
        extractors = self.extractor_registry.get_extractors_for_workflow(workflow_type)
        
        # Create tracer with result extraction
        trace_file = self.trace_dir / workflow_id / f"{phase_name}_trace.jsonl"
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        
        tracer = EnhancedEventTracer(
            trace_id=f"{workflow_id}_{phase_name}",
            trace_file_path=str(trace_file),
            result_extractors=extractors,
            config={'enabled': merged_config.get('tracing', {}).get('enabled', True)}
        )
        
        # Subscribe tracer to event bus
        event_bus.subscribe_all(tracer.trace_event)
        
        try:
            # Build topology
            if topology_builder:
                topology = topology_builder.build_topology(topology_mode, merged_config)
            else:
                topology = {'containers': {}, 'adapters': []}
            
            # Add event bus to topology (for containers to use)
            topology['event_bus'] = event_bus
            
            # Execute the topology SYNCHRONOUSLY
            execution_result = self._run_topology(topology, event_bus)
            
            # Extract results BEFORE teardown
            extracted_results = tracer.get_extracted_results()
            
            # Build phase result
            phase_result = {
                'phase_name': phase_name,
                'topology_mode': topology_mode,
                'success': True,
                'execution_result': execution_result,
                'extracted_results': extracted_results,
                'trace_file': str(trace_file),
                'output': execution_result.get('final_state', {})
            }
            
            return phase_result
            
        except Exception as e:
            logger.error(f"Phase {phase_name} execution failed: {e}")
            return {
                'phase_name': phase_name,
                'success': False,
                'error': str(e)
            }
        finally:
            # ALWAYS clean up - ensures no event contamination
            event_bus.shutdown()
            logger.info(f"Cleaned up phase {phase_name} event system")
    
    def _run_topology(self, topology: Dict[str, Any], event_bus: EventBus) -> Dict[str, Any]:
        """
        Execute a topology synchronously.
        This is where actual container execution happens.
        """
        containers = topology.get('containers', {})
        
        # Initialize containers
        for container_name, container in containers.items():
            if hasattr(container, 'initialize'):
                container.initialize()
        
        # TODO: Actual execution would:
        # 1. Stream data through containers
        # 2. Containers emit events to event bus
        # 3. Events flow between containers
        # 4. Final state is collected
        
        # For now, return mock results
        return {
            'containers_executed': len(containers),
            'final_state': {'mock': 'results'}
        }
```

## Phase 4: Clean Coordinator

```python
# src/core/coordinator/coordinator_clean.py
class Coordinator:
    """
    The workflow manager that orchestrates everything.
    
    Uses composition, not inheritance:
    - Owns a TopologyBuilder (for creating topologies) 
    - Owns a Sequencer (for phase execution)
    - Owns workflow patterns (for workflow definitions)
    
    NO event access at this layer!
    """
    
    def __init__(
        self,
        shared_services: Optional[Dict[str, Any]] = None,
        enable_checkpointing: bool = True
    ):
        """Initialize coordinator with pluggable components."""
        self.shared_services = shared_services or {}
        self.enable_checkpointing = enable_checkpointing
        
        # Initialize extractor registry
        self.extractor_registry = ExtractorRegistry()
        self._register_default_extractors()
        
        # Composed components (not inherited!)
        self.topology_builder = TopologyBuilder()
        self.sequencer = Sequencer(
            checkpoint_dir="./checkpoints" if enable_checkpointing else None,
            extractor_registry=self.extractor_registry
        )
        
        # Workflow patterns
        self.workflow_patterns = self._initialize_patterns()
        
        # Active workflows tracking
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        
    def execute_workflow(
        self,
        config: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute workflow by delegating to sequencer.
        ALL workflows go through the sequencer.
        """
        workflow_id = workflow_id or str(uuid.uuid4())
        workflow_type = config.get('workflow', 'simple_backtest')
        
        # Create execution context
        context = {
            'workflow_id': workflow_id,
            'workflow_type': workflow_type,
            'start_time': datetime.now()
        }
        
        # Get workflow pattern
        pattern = self.workflow_patterns.get(workflow_type, self._default_pattern())
        
        # Execute through sequencer
        result = self.sequencer.execute_phases(
            pattern,
            config,
            context,
            topology_builder=self.topology_builder
        )
        
        # Post-process extracted results if needed
        if result.get('extracted_results'):
            self._process_extracted_results(workflow_id, result['extracted_results'])
        
        return result
```

## Phase 5: Post-Execution Analysis

### 5.1 Create Trace Analyzer

```python
# src/core/analysis/trace_analyzer.py
class TraceAnalyzer:
    """Analyze completed traces for patterns and insights."""
    
    def __init__(self, trace_dir: Path):
        self.trace_dir = trace_dir
        self.extractors = self._load_extractors()
    
    def analyze_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Analyze all phases of a workflow."""
        workflow_dir = self.trace_dir / workflow_id
        
        if not workflow_dir.exists():
            raise ValueError(f"No traces found for workflow {workflow_id}")
        
        results = {
            'workflow_id': workflow_id,
            'phase_analysis': {},
            'cross_phase_patterns': [],
            'aggregate_metrics': {}
        }
        
        # Analyze each phase
        for trace_file in workflow_dir.glob("*_trace.jsonl"):
            phase_name = trace_file.stem.replace('_trace', '')
            phase_results = self._analyze_phase_trace(trace_file)
            results['phase_analysis'][phase_name] = phase_results
        
        # Cross-phase analysis
        results['cross_phase_patterns'] = self._analyze_cross_phase_patterns(
            results['phase_analysis']
        )
        
        # Aggregate metrics
        results['aggregate_metrics'] = self._compute_aggregate_metrics(
            results['phase_analysis']
        )
        
        return results
```

## Benefits of This Unified Approach

1. **Single Source of Truth**: Events contain everything
2. **Clean Architecture**: Clear separation between orchestration and execution
3. **Perfect Reproducibility**: Each phase starts fresh
4. **Rich Analysis**: Traces contain all data for mining
5. **Simple Phase Data**: No complex event-based data flow at orchestration level
6. **Synchronous Execution**: Aligned with ADMF-PC patterns

## Migration Steps

1. **Phase 1**: Implement result extraction framework
2. **Phase 2**: Create clean TopologyBuilder (137 lines vs 2,296!)
3. **Phase 3**: Implement Sequencer with phase isolation
4. **Phase 4**: Update Coordinator to use new components
5. **Phase 5**: Add post-execution analysis tools

## Key Principles

1. **Events for Execution, Not Orchestration**
2. **Fresh Event System per Phase**
3. **Extract Results Before Teardown**
4. **Simple Phase Data Management**
5. **Rich Traces for Analysis**
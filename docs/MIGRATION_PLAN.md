# Coordinator Architecture Migration Plan

## Overview

This document outlines the migration from the current mixed-responsibility coordinator architecture to the clean Protocol + Composition design in the `coordinator_refactor/` directory, including the integration of result streaming with event tracing based on the data mining architecture.

## Current Problems

1. **TopologyBuilder does too much**: Currently builds AND executes workflows
2. **Mixed execution paths in Coordinator**: Conditional logic for single vs multi-phase
3. **Circular dependencies**: Components depend on each other in circular ways
4. **Single Responsibility Principle violations**: Each component has multiple responsibilities
5. **Duplicate result capture**: Results captured separately from events

## Target Architecture

Clean separation of concerns:
- **Coordinator**: Workflow management and orchestration
- **Sequencer**: ALL workflow execution (no special cases)
- **TopologyBuilder**: ONLY builds topologies
- **EventTracer**: Captures all events with result extraction


## Architecture Principles

### 1. Clean Architectural Boundaries
- **Orchestration Layer** (Coordinator, Sequencer, TopologyBuilder): NO events, manages workflow execution
- **Execution Layer** (Containers, Components): Events flow here, isolated per phase
- Each phase gets a fresh event system that is torn down after execution

### 2. Events as Single Source of Truth (Within Phases)
- All execution activity within a phase flows through events
- Results are extracted from events before phase teardown
- Event traces are immutable and archived per phase
- Perfect reproducibility: same phase config → same event sequence

### 3. Two-Layer Analysis Architecture
- **Layer 1**: SQL database for high-level metrics and discovery
- **Layer 2**: Event stream storage (per phase) for deep behavioral analysis

### 4. Protocol + Composition
- No inheritance, only protocols
- Components composed together
- Clean dependency flow: Coordinator → Sequencer → TopologyBuilder
- No async unless absolutely necessary (keep it simple)

## Migration Steps

### Phase 1: Preparation (✓ Completed)
1. ✓ Create comprehensive backup of current coordinator module
2. ✓ Add workflow discovery to existing discovery.py system
3. ✓ Move discovery.py from containers/ to components/
4. Document all breaking changes for users

### Phase 2: Phase Data Management
1. **Create PhaseDataStore**:
   ```python
   class PhaseDataStore:
       """Simple phase data storage at orchestration layer."""
       
       def __init__(self, workflow_id: str):
           self.workflow_id = workflow_id
           self.outputs = {}
           self.storage_path = Path(f"./workflows/{workflow_id}/phases")
           self.storage_path.mkdir(parents=True, exist_ok=True)
           
       def store_output(self, phase_name: str, output: Any):
           """Store phase output with size-based strategy."""
           if self._is_small(output):  # < 1MB
               self.outputs[phase_name] = output
           else:
               # Store to disk for larger outputs
               path = self.storage_path / f"{phase_name}_output.pkl"
               with open(path, 'wb') as f:
                   pickle.dump(output, f)
               self.outputs[phase_name] = {'_file': str(path)}
               
       def get_output(self, phase_name: str) -> Any:
           """Retrieve phase output."""
           if phase_name not in self.outputs:
               return None
               
           output = self.outputs[phase_name]
           if isinstance(output, dict) and '_file' in output:
               with open(output['_file'], 'rb') as f:
                   return pickle.load(f)
           return output
   ```

2. **Archive Phase Traces**:
   ```python
   class PhaseTraceArchiver:
       """Archive event traces per phase for analysis."""
       
       def archive_trace(self, tracer: EventTracer, workflow_id: str, phase_name: str):
           """Save phase trace for later analysis."""
           archive_path = Path(f"./traces/{workflow_id}/{phase_name}")
           archive_path.mkdir(parents=True, exist_ok=True)
           
           # Save events
           trace_file = archive_path / "events.parquet"
           events_df = self._events_to_dataframe(tracer.events)
           events_df.to_parquet(trace_file)
           
           # Save metadata
           metadata = {
               'workflow_id': workflow_id,
               'phase_name': phase_name,
               'event_count': len(tracer.events),
               'start_time': tracer.start_time,
               'end_time': tracer.end_time
           }
           with open(archive_path / "metadata.json", 'w') as f:
               json.dump(metadata, f)
   ```

### Phase 3: Event Tracing Enhancement (Within Phases)
1. **Phase-Scoped Event Tracer**:
   ```python
   class PhaseEventTracer(EventTracer):
       """Event tracer scoped to a single phase execution."""
       
       def __init__(self, workflow_id: str, phase_name: str):
           trace_id = f"{workflow_id}_{phase_name}"
           super().__init__(trace_id)
           self.workflow_id = workflow_id
           self.phase_name = phase_name
           self.result_extractors = []
           
       def configure_extractors(self, extractors: List[ResultExtractor]):
           """Configure result extractors for this phase."""
           self.result_extractors = extractors
           
       def extract_results(self) -> Dict[str, Any]:
           """Extract results before phase teardown."""
           results = {}
           for extractor in self.result_extractors:
               extractor_results = []
               for event in self.events:
                   if extractor.can_extract(event):
                       result = extractor.extract(event)
                       if result:
                           extractor_results.append(result)
               
               if extractor_results:
                   results[extractor.__class__.__name__] = extractor_results
                   
           return results
   ```

2. **Enhanced Event Structure** (for use within phases):
   ```python
   @dataclass
   class TracedEvent:
       # Identity
       event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
       event_type: EventType
       timestamp: datetime = field(default_factory=datetime.now)
       
       # Phase context (NOT for orchestration communication)
       phase_id: str  # workflow_id + phase_name
       source_container: str
       
       # Payload
       data: Dict[str, Any]
       
       # Metadata for analysis
       metadata: Dict[str, Any] = field(default_factory=dict)
       
       def mark_as_result(self, category: str):
           """Mark for result extraction within the phase."""
           self.metadata['contains_result'] = True
           self.metadata['result_category'] = category
   ```

3. **Result Extractors** (operate within phase boundary):
   ```python
   # In src/core/events/extractors/base.py
   class ResultExtractor(Protocol):
       """Extract results from events within a phase."""
       
       def can_extract(self, event: TracedEvent) -> bool:
           """Check if this extractor handles this event."""
           ...
           
       def extract(self, event: TracedEvent) -> Optional[Dict[str, Any]]:
           """Extract result data from event."""
           ...
   
   # Example extractors
   class PortfolioMetricsExtractor(ResultExtractor):
       def can_extract(self, event: TracedEvent) -> bool:
           return event.event_type == EventType.PORTFOLIO_UPDATE
           
       def extract(self, event: TracedEvent) -> Optional[Dict[str, Any]]:
           return {
               'timestamp': event.timestamp,
               'total_value': event.data.get('total_value'),
               'pnl': event.data.get('pnl'),
               'sharpe_ratio': event.data.get('metrics', {}).get('sharpe_ratio')
           }
   ```

### Phase 4: Core Migration
1. **Move refactored components**:
   - `coordinator_refactor/coordinator.py` → `coordinator/coordinator.py`
   - `coordinator_refactor/sequencer.py` → `coordinator/sequencer.py`
   - `coordinator_refactor/topology_builder.py` → `coordinator/topology.py`
   - `coordinator_refactor/protocols.py` → `coordinator/protocols.py`

2. **Remove execution logic from TopologyBuilder**:
   - Remove `execute()`, `execute_pattern()`, `_execute_phase()` methods
   - Keep only topology building methods
   - Remove workflow management responsibilities

3. **Update Coordinator for clean boundaries**:
   - Remove ALL event system usage from orchestration layer
   - Coordinator does NOT create or use EventBus
   - Sequencer manages phase execution with isolated event systems:
   ```python
   def execute_workflow(self, config: WorkflowConfig) -> WorkflowResult:
       # Orchestration layer - NO events here
       workflow_id = str(uuid.uuid4())
       workflow_pattern = self.get_workflow_pattern(config['type'])
       
       # Simple phase data storage
       phase_data_store = PhaseDataStore(workflow_id)
       
       # Delegate to sequencer (synchronous, no async)
       result = self.sequencer.execute_phases(
           pattern=workflow_pattern,
           config=config,
           context={
               'workflow_id': workflow_id,
               'phase_data_store': phase_data_store
           }
       )
       
       return result
   ```

4. **Update Sequencer for phase isolation**:
   ```python
   def execute_phase(self, phase_config, context):
       """Execute phase with isolated event system."""
       phase_name = phase_config['name']
       workflow_id = context['workflow_id']
       
       # Get dependencies from phase data store (NOT events)
       dependencies = self._load_phase_dependencies(
           phase_config, 
           context['phase_data_store']
       )
       
       # Create FRESH event system for THIS phase only
       event_bus = EventBus()
       tracer = EventTracer(f"{workflow_id}_{phase_name}")
       
       try:
           # Build topology
           topology = self.topology_builder.build_topology(
               phase_config['topology'],
               {**phase_config, 'dependencies': dependencies}
           )
           
           # Execute with isolated events
           result = self._run_topology(topology, event_bus, tracer)
           
           # Extract results BEFORE teardown
           if self.result_extractors:
               extracted_results = self._extract_results(tracer)
               
           # Archive trace for later analysis
           self._archive_phase_trace(tracer, workflow_id, phase_name)
           
           # Store output for next phases (NOT via events)
           context['phase_data_store'].store_output(phase_name, result)
           
           return result
           
       finally:
           # ALWAYS clean up - ensures no event contamination
           event_bus.shutdown()
           tracer.close()
   ```

### Phase 5: Data Mining Integration
1. **Create SQL Schema**:
   ```sql
   -- Core tables matching data-mining-architecture.md
   CREATE TABLE optimization_runs (...);
   CREATE TABLE trades (...);
   CREATE TABLE discovered_patterns (...);
   CREATE TABLE signal_patterns (...);
   CREATE TABLE pattern_interactions (...);
   CREATE TABLE pattern_regime_performance (...);
   CREATE TABLE pattern_performance_history (...);
   ```

2. **Implement Enhanced ETL Pipeline**:
   ```python
   # In src/core/analysis/etl.py
   class OptimizationETL:
       def __init__(self):
           self.registry = ExtractorRegistry()
           self.sql_db = PostgresConnection()
           
       def nightly_etl(self, date: datetime):
           """Nightly ETL using ResultExtractors."""
           # Load events for the day
           daily_events = self.load_events_for_date(date)
           
           # Get all registered extractors
           for name, extractor in self.registry.extractors.items():
               results = []
               for event in daily_events:
                   if extractor.can_extract(event):
                       result = extractor.extract(event)
                       if result:
                           results.append(result)
               
               # Write to appropriate SQL table
               if results:
                   df = pd.DataFrame(results)
                   table_name = self.registry.sql_mappings[name]
                   df.to_sql(table_name, self.sql_db, if_exists='append')
   ```

3. **Pattern Discovery Pipeline**:
   ```python
   # In src/core/analysis/pattern_discovery.py
   class PatternDiscoveryPipeline:
       """Unified pattern discovery using extractors."""
       
       def __init__(self):
           self.extractors = [
               SignalPatternExtractor(),
               RegimeTransitionExtractor(),
               RiskDecisionExtractor()
           ]
           
       def mine_patterns(self, events: List[Event]):
           # Use extractors to prepare data for pattern mining
           pattern_data = defaultdict(list)
           
           for event in events:
               for extractor in self.extractors:
                   if extractor.can_extract(event):
                       result = extractor.extract(event)
                       if result:
                           pattern_data[extractor.category].append(result)
           
           # Feed to pattern discovery algorithms
           return self.discover_patterns(pattern_data)
   ```

### Phase 6: Storage Architecture
1. **Implement Tiered Storage**:
   ```yaml
   storage_config:
     hot:
       # Real-time analysis
       location: "./results/hot"
       format: "jsonl"
       retention: "30d"
       
     warm:
       # SQL + Parquet
       location: "./results/warm"
       format: "parquet"
       retention: "1y"
       
     cold:
       # Compressed archive
       location: "./results/cold"
       format: "compressed_parquet"
       retention: "indefinite"
   ```

2. **Intelligent Sampling**:
   ```python
   # Implement IntelligentEventSampler
   - Always keep critical events (SIGNAL, ORDER, FILL, RISK_BREACH)
   - Keep context window around critical events
   - Adaptive sampling based on activity levels
   ```

### Phase 7: Testing & Validation
1. **Test Event Flow**:
   - Verify all events have correlation_id
   - Test result extraction accuracy
   - Validate SQL metrics match event-derived metrics

2. **Test Pattern Discovery**:
   - Verify patterns are detected correctly
   - Test pattern performance tracking
   - Validate pattern decay monitoring

3. **Performance Testing**:
   - Measure extraction overhead
   - Test streaming performance
   - Validate storage efficiency

### Phase 8: Advanced Features
1. **Live Pattern Matching**:
   ```python
   class LivePatternMatcher:
       def check_live_patterns(self, event_stream):
           # Real-time pattern detection
           # Alert on anti-patterns
           # Track pattern performance
   ```

2. **Pattern Lifecycle Management**:
   ```python
   class PatternLifecycleManager:
       def evaluate_pattern_health(self, pattern_id):
           # Monitor pattern degradation
           # Adaptive parameter adjustment
           # Automated retirement
   ```

3. **Scientific Method Workflow**:
   ```python
   class TradingScientificMethod:
       def run_experiment(self, hypothesis):
           # SQL Discovery → Event Analysis → Validation → Production
   ```

## Implementation Priorities

### Critical Path (Must Have)
1. **TracedEvent Structure**: Enhanced event with full metadata
2. **ExtractorRegistry**: Central registry with SQL mappings
3. **UnifiedEventTracer**: New tracer inheriting from EventTracer
4. **Core Extractors**: Portfolio, Trade, Signal extractors
5. **ETL Pipeline**: Nightly batch processing

### High Priority (Should Have)
1. **PatternDiscoveryPipeline**: Structured pattern mining
2. **Workflow-Specific Extraction**: Registry.get_extractors_for_workflow()
3. **SQL Schema**: Full schema from data-mining-architecture.md
4. **IntelligentEventSampler**: Smart sampling for storage efficiency

### Medium Priority (Nice to Have)
1. **LivePatternMatcher**: Real-time pattern detection
2. **PatternLifecycleManager**: Pattern health monitoring
3. **Event Compression**: Semantic compression for long-term storage
4. **TradingScientificMethod**: Hypothesis testing framework

## Implementation Timeline

### Week 1: Foundation & Architecture
- Day 1: Create PhaseDataStore for orchestration layer
- Day 2: Implement PhaseTraceArchiver
- Day 3: Create PhaseEventTracer (phase-scoped)
- Day 4: Define clean architectural boundaries in code
- Day 5: Remove all async, ensure synchronous execution

### Week 2: Core Migration
- Day 1-2: Move refactored coordinator components
- Day 3: Update Coordinator (no events, no async)
- Day 4: Update Sequencer with phase isolation
- Day 5: Test phase isolation and reproducibility

### Week 3: Event System Enhancement
- Day 1: Enhanced TracedEvent structure
- Day 2: Create base ResultExtractor protocol
- Day 3: Implement core extractors (Portfolio, Signal, Trade)
- Day 4: Integrate extractors with PhaseEventTracer
- Day 5: Test result extraction within phases

### Week 4: Data Mining Integration
- Day 1: SQL schema creation
- Day 2: ETL pipeline for archived traces
- Day 3: Pattern discovery from phase traces
- Day 4: Cross-phase analysis tools
- Day 5: Integration testing

### Week 5: Production Readiness
- Day 1: Performance optimization
- Day 2: Storage tiering implementation
- Day 3: Documentation update
- Day 4: Migration guide for users
- Day 5: Final validation

## Key Implementation Files

### New Files to Create
```
src/core/orchestration/
├── phase_data.py              # PhaseDataStore
└── trace_archiver.py          # PhaseTraceArchiver

src/core/events/
├── extractors/
│   ├── __init__.py
│   ├── base.py                # ResultExtractor protocol
│   ├── portfolio.py           # Portfolio metrics extraction
│   ├── signals.py             # Signal extraction
│   ├── trades.py              # Trade extraction
│   └── patterns.py            # Pattern candidate extraction
├── tracing/
│   ├── __init__.py
│   └── phase_tracer.py        # PhaseEventTracer (phase-scoped)
└── analysis/
    ├── __init__.py
    ├── phase_analyzer.py      # Analyze archived phase traces
    ├── cross_phase.py         # Cross-phase analysis
    └── etl.py                 # ETL from phase traces to SQL

src/core/analysis/
├── sql/
│   ├── schema.sql             # Database schema
│   ├── views.sql              # Materialized views
│   └── indexes.sql            # Performance indexes
└── patterns/
    ├── __init__.py
    ├── discovery.py           # Pattern discovery from traces
    └── library.py             # Pattern library
```

### Files to Modify
```
src/core/coordinator/coordinator.py   # Remove events, remove async
src/core/coordinator/sequencer.py     # Add phase isolation, PhaseDataStore
src/core/coordinator/topology.py      # Remove execution methods
src/core/events/types.py              # Add phase-scoped TracedEvent
src/core/components/discovery.py      # Already updated with @workflow
```

### Files to Remove/Deprecate
```
src/core/coordinator_refactor/data_management.py  # Replaced by PhaseDataStore
Any async coordinator implementations               # Convert to sync
```

## Configuration Example

```yaml
workflow:
  type: "optimization"
  
  # Phase configuration (no async)
  phases:
    - name: "data_validation"
      topology: "validation"
      
    - name: "parameter_sweep"
      topology: "backtest"
      depends_on: ["data_validation"]
      config:
        parameters:
          fast_ma: [10, 20, 30]
          slow_ma: [50, 100, 200]
          
    - name: "analysis"
      topology: "analysis"
      depends_on: ["parameter_sweep"]
      
  # Phase data management (orchestration layer)
  phase_data:
    storage_path: "./workflows/{workflow_id}/phases"
    compression_threshold: 1048576  # 1MB
    
  # Event tracing (per phase)
  tracing:
    enabled: true
    archive_path: "./traces/{workflow_id}/{phase_name}"
    
  # Result extraction (within phases)
  result_extraction:
    enabled: true
    extractors:
      - portfolio_metrics
      - signals
      - trades
    output_format: "parquet"
    
  # Post-execution analysis
  analysis:
    etl_enabled: true
    pattern_discovery: true
    sql_output: true
```

## Breaking Changes

### For Users
1. No async execution - all workflows run synchronously
2. Phase outputs stored separately from events
3. Event traces archived per phase, not globally
4. New configuration structure for phases

### For Developers
1. Orchestration layer (Coordinator/Sequencer) has NO event access
2. Each phase gets fresh EventBus that's destroyed after execution
3. Phase data passed via PhaseDataStore, not events
4. Result extraction happens within phase boundaries
5. No async/await in orchestration code

## Success Criteria

1. ✓ Clean architectural boundaries enforced
2. ✓ Perfect phase isolation - each phase gets fresh event system
3. ✓ No events in orchestration layer
4. ✓ Synchronous execution throughout
5. ✓ Phase data flows through PhaseDataStore
6. ✓ Event traces archived per phase
7. ✓ Result extraction works within phases
8. ✓ Post-execution analysis from archived traces
9. ✓ Perfect reproducibility maintained
10. ✓ Complete documentation

## Rollback Plan

If issues arise:
1. Event tracing can be disabled via config
2. Result extraction can be turned off
3. Revert to separate result capture (temporary)
4. Use backup from Phase 1

## Unified Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Unified Data Flow Architecture               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Container → Event → UnifiedEventTracer (with ExtractorRegistry) │
│                           ↓                                      │
│                    Event Storage (Parquet)                       │
│                           ↓                                      │
│                 ┌─────────┴─────────┐                           │
│                 ↓                   ↓                           │
│          Real-time Results     Batch ETL Process                │
│          (Streaming)           (Nightly)                         │
│                 ↓                   ↓                           │
│          Live Monitoring      Analytics DB (SQL)                │
│                 ↓                   ↓                           │
│          Pattern Matching     Data Mining Queries               │
│                 ↓                   ↓                           │
│          Alerts/Actions       Pattern Discovery                 │
│                                     ↓                           │
│                              Pattern Library                     │
│                                     ↓                           │
│                           Continuous Improvement                 │
└─────────────────────────────────────────────────────────────────┘
```

## Benefits of This Architecture

1. **Clean Boundaries**: Orchestration and execution layers completely separated
2. **Perfect Reproducibility**: Each phase starts fresh, no contamination
3. **True Parallelization**: Phases can run in parallel without interference
4. **Simple Phase Data**: PhaseDataStore handles inter-phase data simply
5. **Events Scoped Correctly**: Events flow within phases, not between them
6. **Rich Analysis**: Phase traces archived for deep analysis
7. **Pattern Discovery**: Mine patterns from phase execution traces
8. **No Async Complexity**: Synchronous execution is simpler and more debuggable
9. **Result Extraction**: Works within phase boundaries, before teardown
10. **Testability**: Each phase can be tested in complete isolation

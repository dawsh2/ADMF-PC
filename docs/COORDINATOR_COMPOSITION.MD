# ADMF-PC Core Concepts: Coordinator and Containers

This document outlines the core concepts of the ADMF-PC (Adaptive Dynamic Market Framework - Protocol Components) system, focusing on its coordinator, composable containers, and workflow composition, as detailed in the `src/core/coordinator` and `src/core/containers` modules.

## I. Coordinator

The Coordinator is the central brain of the ADMF-PC system, responsible for orchestrating all high-level operations and ensuring consistency and reproducibility.

### A. Overview and Role
* **Central Orchestration**: Acts as the single entry point for all workflows, including optimization, backtesting, live trading, analysis, and validation.
* **Reproducibility & Consistency**: Enforces a standardized execution path, consistent initialization, controlled random seeds, and complete configuration capture for reproducible results.
* **Workflow Management**: Reads and validates workflow configurations, sets up shared infrastructure, creates isolated containers for workflows/trials, delegates execution, aggregates results, and ensures resource cleanup.

### B. Key Features
* **Configuration-Driven**: Complex workflows are defined declaratively (e.g., in YAML) rather than hard-coded.
* **Container Isolation**: Each workflow or trial runs in its own isolated container with a dedicated event bus, independent component instances, and separate state management.
* **Extensibility**: Supports custom workflow managers by extending `BaseWorkflowManager`.

### C. Architecture & Execution (`src/core/coordinator/coordinator.py`)
* **Lazy Loading**: Employs lazy loading for complex dependencies (e.g., composition engine, container registry, workflow managers) to maintain clean imports and separation of concerns.
* **Execution Modes**:
    * `AUTO`: Coordinator determines the best execution mode.
    * `TRADITIONAL`: Uses traditional workflow managers.
    * `COMPOSABLE`: Leverages composable container patterns. This mode is chosen if the config explicitly requests a container pattern, involves complex multi-pattern workflows, or if the coordinator determines the workflow would benefit (e.g., multi-classifier scenarios, signal generation/replay).
    * `HYBRID`: Mixes traditional and composable approaches, potentially using composable patterns for specific phases within a larger, traditionally orchestrated workflow.
* **Workflow Lifecycle**: Manages configuration validation, container creation, infrastructure setup, phase execution, result aggregation, and resource cleanup.
* **Advanced Features**: Supports parallel execution of multiple workflows and provides methods for workflow monitoring (listing active workflows, getting status, cancellation).

## II. Composable Containers

The system employs a sophisticated container model designed for flexibility, isolation, and reusability, primarily defined in `src/core/containers/`.

### A. Core Philosophy and Protocols (`src/core/containers/composable.py`)
* **`ComposableContainerProtocol`**: Defines the contract for all containers. Key aspects include:
    * `metadata`: For identification (ID, role, name, parent ID).
    * `state`: Lifecycle state (e.g., Uninitialized, Running, Stopped).
    * `event_bus`: A container-scoped message bus.
    * `parent_container` & `child_containers`: For establishing hierarchies.
    * Lifecycle methods: `initialize`, `start`, `stop`, `dispose`.
    * Composition methods: `add_child_container`, `remove_child_container`, `find_containers_by_role`.
    * Event handling: `process_event`, `publish_event`.
    * Configuration: `update_config`, `get_status`, `get_capabilities`.
* **`ContainerRole`**: Standardized roles such as DATA, INDICATOR, CLASSIFIER, RISK, STRATEGY, EXECUTION, etc.
* **`ContainerState`**: Defines lifecycle states like UNINITIALIZED, INITIALIZING, RUNNING, STOPPED, ERROR.
* **`BaseComposableContainer`**: A base class providing common functionality for containers, simplifying implementation.

### B. Container Composition Engine (`src/core/containers/composition_engine.py`)
* **`ContainerRegistry`**:
    * Manages registration of container types (via factory functions) and their associated capabilities.
    * Stores definitions of `ContainerPattern` (e.g., "full_backtest", "signal_generation", "signal_replay", "simple_backtest"). These patterns define a hierarchical structure of containers for specific use cases.
* **`ContainerCompositionEngine`**:
    * Instantiates containers using the factories in the registry.
    * Constructs complex container hierarchies based on registered `ContainerPattern` names or custom-defined structures.
    * Can infer dependencies, such as required indicators for a strategy, and configure relevant containers (e.g., Indicator Hub) accordingly.
    * Validates patterns for structural integrity and capability requirements.
* **Global Access**: Provides global instances (`get_global_registry`, `get_global_composition_engine`) for system-wide use.

### C. Foundational Container Implementations
* **`UniversalScopedContainer` (`src/core/containers/universal.py`)**:
    * Provides the fundamental building block for isolated execution environments.
    * Manages its own dependency container (`BaseScopedContainer`) and component factory.
    * Features an isolated event bus created via `get_isolation_manager()`.
    * Tracks component specifications, initialization order, and lifecycle states (Created, Initializing, Running, Stopped, Disposed, Failed).
* **`EnhancedContainer` (`src/core/containers/enhanced_container.py`)**:
    * Extends `UniversalScopedContainer` to explicitly support sub-container hierarchies.
    * Allows `create_subcontainer`, which establishes parent-child relationships.
    * Implements hierarchical component resolution (searches current then parent containers).
    * Uses a `ScopedEventBus` that can propagate events to a parent bus.

### D. Container Factory (`src/core/containers/factory.py`)
* The `ContainerFactory` is designed to create specialized containers (e.g., for Backtesting, Optimization, Live Trading) by assembling them with pre-defined sets of components.
* It uses a `ContainerLifecycleManager` to manage the containers it creates.
* It defines structured container IDs using `ContainerNamingStrategy` for better traceability (e.g., `create_backtest_container_id`).

### E. Container Lifecycle Management (`src/core/containers/lifecycle.py`)
* **`ContainerLifecycleManager`**:
    * Central point for creating, initializing, starting, stopping, resetting, and disposing of `UniversalScopedContainer` instances.
    * Manages active containers and can implement pooling for container reuse to improve performance.
    * Can enforce limits on the maximum number of containers and evict old ones if necessary.
    * Supports lifecycle hooks for custom actions at different stages (e.g., Created, Initialized, Started).

### F. Event System Integration
* Each container typically has its own isolated event bus, ensuring that events do not cross-contaminate between parallel workflow executions.
* The `EnhancedContainer`'s `ScopedEventBus` allows for controlled event propagation (e.g., bubbling to a parent).
* Components within a container subscribe to and publish events on the container's local event bus.

## III. Composable Workflows

The system allows complex, multi-phase workflows to be constructed by composing simpler, standardized workflow types. This is evident in the `Coordinator`'s ability to use different managers and container patterns.

### A. Concept and Rationale (`docs/WORKFLOW_COMPOSITION.MD`)
* **Building Blocks**: Simple workflows (Backtest, Optimization, Analysis, etc.) serve as fundamental building blocks.
* **Declarative Composition**: New, complex workflow patterns are defined through configuration (e.g., YAML or `CompositeWorkflowTemplate`) by sequencing these blocks, rather than writing new manager code for each variant.
* **Reusability**: Leverages existing, proven infrastructure (managers, container patterns) for each phase of a composite workflow.

### B. `ComposableWorkflowManager` (`src/core/coordinator/composable_workflow_manager.py`)
* Acts as a bridge, enabling the use of composable container patterns within the broader workflow management system orchestrated by the `Coordinator`.
* **Pattern Determination**: Analyzes the `WorkflowConfig` to decide which `ContainerPattern`(s) (e.g., "simple_backtest", "full_backtest", "signal_generation", "signal_replay") are appropriate for the given workflow type (Backtest, Optimization, Analysis).
* **Configuration Building**: Contains logic to construct specific configuration dictionaries tailored for each determined container pattern based on the overall `WorkflowConfig` (e.g., `_build_simple_backtest_config`, `_build_signal_generation_config`).
* **Execution Orchestration**:
    * For single-pattern workflows, it uses the `CompositionEngine` to compose and execute the appropriate container pattern.
    * For multi-pattern workflows (e.g., an optimization workflow that first generates signals and then replays them), it executes the sequence of patterns, potentially passing results from one pattern to the next (e.g., signal output path from a signal generation pattern becomes an input for a signal replay pattern).
* **Result Handling**: Collects and formats results from the executed container(s).

### C. Phase Management (`src/core/coordinator/phase_management.py`)
This module provides critical infrastructure for robust multi-phase workflows:
* **`PhaseTransition`**: Manages the data flow and dependencies between different phases of a complex workflow, ensuring outputs from one phase (e.g., `phase1_outputs`) are available as inputs to subsequent phases.
* **`ContainerNamingStrategy`**: Implements a consistent naming scheme for containers based on phase, regime, strategy, parameters, and timestamp, aiding in debugging and tracking.
* **`StreamingResultWriter` & `ResultAggregator`**: Address potential memory issues in large-scale optimizations by streaming detailed results from individual container runs to disk immediately. The aggregator can then maintain an in-memory cache of only the top-performing results.
* **`StrategyIdentity`**: Allows tracking of the same logical strategy (base class and parameters) across different regime-specific container instances, facilitating cross-regime performance analysis.
* **`WorkflowState` & `CheckpointManager`**: Provide resumability for long-running optimization workflows by saving and restoring the workflow's state (current phase, completed phases, intermediate results) at checkpoints.
* **`SharedServiceRegistry`**: Manages versions of shared services to prevent compatibility issues and breaking changes.
* The `integrate_phase_management` function shows how these capabilities can be added to an existing Coordinator.

### D. Data Structures and Protocols (`src/core/coordinator/types.py`, `src/core/coordinator/protocols.py`)
* Defines key data structures like `WorkflowConfig`, `ExecutionContext`, `WorkflowResult`, `PhaseResult`, and enums like `WorkflowType` and `WorkflowPhase`. These are used throughout the coordinator and workflow management modules, with Pydantic support for validation if available.
* Specifies protocols for `WorkflowManager`, `PhaseExecutor`, `ResultStreamer`, `CheckpointManager`, `ResourceManager`, and `ResultAggregator`, defining clear interfaces for these components.

## IV. Integration and Data Flow (Illustrative Examples)

As seen in `docs/BACKTEST_README.md` and `docs/MULTIPHASE_OPTIMIZATION.MD`:

* **Nested Container Hierarchy**: The system supports a hierarchical structure (e.g., Classifier Container > Risk Container > Portfolio Container > Strategy). This allows for logical grouping and efficient resource sharing (e.g., a shared Indicator Hub within a Classifier Container).
* **Event-Driven Flow**: Communication within and between containers (especially in backtesting scenarios) is often event-driven (e.g., SIGNAL -> ORDER -> FILL events).
* **Specialized Container Patterns for Workflows**:
    * **Full Backtest**: Standard pattern for complete simulation from market data to execution.
    * **Signal Replay**: Uses pre-generated signal logs as input, bypassing indicator and classifier computation, primarily for ensemble optimization. This is significantly faster.
    * **Signal Generation**: Focuses on producing signals without execution, used for analysis, MAE/MFE optimization, and classifier tuning.
* **File-Based Communication in Multi-Phase Optimization**: The Coordinator often sets up a workspace, and different phases (managed by Optimizer or Backtester components) read inputs from and write outputs to specified file paths within this workspace. This facilitates checkpointing and inspectability.

This architecture emphasizes modularity, clear separation of concerns, and declarative workflow definition, allowing for robust and flexible construction of complex trading system operations.
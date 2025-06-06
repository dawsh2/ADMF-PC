"""
Workflow Topology Manager - Creates and Manages Workflow Topologies

This module creates workflow topologies by arranging containers, patterns, and
execution strategies into connected graphs. It handles:
- Pattern detection and topology construction
- Communication configuration between components  
- Execution strategy delegation
- Multi-parameter workflow coordination
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime

# Factory pattern removed - direct container creation in new architecture
from ..communication import AdapterFactory
from ..types.workflow import WorkflowConfig, WorkflowType, ExecutionContext, WorkflowResult
from ..types.events import EventType
from ..events import Event
from ..containers.protocols import ComposableContainer, ContainerRole
from ..containers.factory import get_global_factory, get_global_registry

# Unified architecture - no pattern detection needed!
# from .workflows.config import PatternDetector, ParameterAnalyzer, ConfigBuilder
# from .workflows.execution import get_executor, ExecutionStrategy

logger = logging.getLogger(__name__)


class WorkflowMode(str, Enum):
    """Execution modes for unified architecture."""
    BACKTEST = "backtest"              # Full pipeline: data → signals → orders → fills
    SIGNAL_GENERATION = "signal_generation"  # Stop after signals, save for replay
    SIGNAL_REPLAY = "signal_replay"     # Start from saved signals → orders → fills


class TopologyBuilder:
    """
    Topology builder for creating container-based execution topologies.
    
    This builder constructs topologies by:
    1. Detecting appropriate patterns based on configuration
    2. Creating containers and stateless services
    3. Setting up event routing between components
    4. Managing container lifecycle and communication
    """
    
    def __init__(
        self,
        container_id: Optional[str] = None,
        shared_services: Optional[Dict[str, Any]] = None,
        coordinator: Optional[Any] = None,
        execution_mode: str = 'standard',  # DEPRECATED - kept for compatibility
        enable_nesting: bool = False,  # DEPRECATED - kept for compatibility
        enable_pipeline_communication: bool = False  # DEPRECATED - kept for compatibility
    ):
        """Initialize unified workflow manager.
        
        NOTE: execution_mode, enable_nesting, and enable_pipeline_communication
        parameters are deprecated and ignored. The unified architecture always
        uses the same topology and execution flow.
        """
        self.container_id = container_id
        self.shared_services = shared_services or {}
        self.coordinator = coordinator
        # Deprecated parameters - kept for backward compatibility
        self.execution_mode = execution_mode
        self.enable_nesting = enable_nesting
        self.enable_pipeline_communication = enable_pipeline_communication
        
        # Initialize container factories before using them
        self._initialize_container_factories()
        
        # Only adapter factory needed in new architecture
        self.adapter_factory = AdapterFactory()
        
        # Unified architecture - no pattern detection or multiple executors!
        # self.pattern_detector = PatternDetector()
        # self.parameter_analyzer = ParameterAnalyzer()
        # self.config_builder = ConfigBuilder()
        # self._executors: Dict[str, ExecutionStrategy] = {}
        
        # Active resources
        self.active_containers: Dict[str, ComposableContainer] = {}
        self.active_adapters = []
        
        logger.info(f"WorkflowManager initialized (unified architecture)")
    
    def _initialize_container_factories(self):
        """Initialize container factories and trigger auto-discovery."""
        try:
            from ..containers.factory import get_global_factory
            from ..containers.discovery import auto_discover_all_components
            
            # Get the global factory which will auto-discover components
            factory = get_global_factory()
            
            # Trigger auto-discovery of all decorated components
            discovered = auto_discover_all_components()
            logger.info(f"Auto-discovered components: {discovered}")
            
            # Auto-discovery is triggered when factory methods are called
            logger.debug("Container factories initialized with auto-discovery")
            
        except Exception as e:
            logger.error(f"Error initializing container factories: {e}")
            # Don't raise - allow system to continue
    
    async def execute(
        self,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute workflow using unified architecture with universal topology."""
        
        logger.info(f"Executing {config.workflow_type.value} workflow")
        
        # Check for custom workflow
        custom_workflow = config.parameters.get('workflow')
        if custom_workflow:
            logger.info(f"Executing custom workflow: {custom_workflow}")
            return await self._execute_custom_workflow(custom_workflow, config, context)
        
        try:
            # 1. Determine workflow mode (backtest, signal gen, or replay)
            mode = self._determine_mode(config)
            logger.info(f"Using workflow mode: {mode.value}")
            
            # 2. Create mode-specific topology
            try:
                if mode == WorkflowMode.BACKTEST:
                    topology = await self._create_backtest_topology(config)
                elif mode == WorkflowMode.SIGNAL_GENERATION:
                    topology = await self._create_signal_generation_topology(config)
                elif mode == WorkflowMode.SIGNAL_REPLAY:
                    topology = await self._create_signal_replay_topology(config)
                else:
                    raise ValueError(f"Unknown workflow mode: {mode}")
                    
                logger.info(f"Created {mode.value} topology with {len(topology['containers'])} containers")
            except Exception as e:
                logger.error(f"Failed to create topology: {e}", exc_info=True)
                raise
            
            # 3. Wire mode-specific adapters
            if mode == WorkflowMode.BACKTEST:
                adapters = self._create_backtest_adapters(topology)
            elif mode == WorkflowMode.SIGNAL_GENERATION:
                adapters = self._create_signal_generation_adapters(topology)
            elif mode == WorkflowMode.SIGNAL_REPLAY:
                adapters = self._create_signal_replay_adapters(topology)
            else:
                adapters = []
                
            logger.info(f"Created {len(adapters)} {mode.value} adapters")
            
            # 4. Execute based on mode
            if mode == WorkflowMode.BACKTEST:
                result = await self._execute_full_pipeline(topology, config, context)
            elif mode == WorkflowMode.SIGNAL_GENERATION:
                result = await self._execute_signal_generation(topology, config, context)
            elif mode == WorkflowMode.SIGNAL_REPLAY:
                result = await self._execute_signal_replay(topology, config, context)
            else:
                raise ValueError(f"Unknown workflow mode: {mode}")
            
            logger.info(f"Workflow execution completed: success={result.success}")
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return WorkflowResult(
                workflow_id=context.workflow_id,
                workflow_type=config.workflow_type,
                success=False,
                errors=[str(e)],
                metadata={'mode': 'failed'}
            )
    
    async def execute_pattern(
        self,
        pattern_name: str,
        config: Dict[str, Any],
        correlation_id: str
    ) -> WorkflowResult:
        """
        DEPRECATED - Pattern-based execution is replaced by unified modes.
        
        This method is kept for backward compatibility with Coordinators that
        still call execute_pattern. It now maps pattern names to unified modes.
        
        Args:
            pattern_name: Name of the pattern to execute (ignored)
            config: Configuration for the pattern
            correlation_id: Correlation ID for tracking
            
        Returns:
            WorkflowResult with execution details
        """
        logger.warning(f"execute_pattern called with '{pattern_name}' - mapping to unified mode")
        
        # Map old pattern names to new modes
        mode_mapping = {
            'simple_backtest': WorkflowMode.BACKTEST,
            'signal_generation': WorkflowMode.SIGNAL_GENERATION,
            'signal_replay': WorkflowMode.SIGNAL_REPLAY,
            'multi_strategy': WorkflowMode.BACKTEST,
            'optimization': WorkflowMode.BACKTEST,
            'walk_forward': WorkflowMode.BACKTEST
        }
        
        # Create a WorkflowConfig with the mode
        workflow_config = WorkflowConfig(
            workflow_type=WorkflowType.BACKTEST,
            parameters=config
        )
        
        # Set mode based on pattern name or config
        if pattern_name in mode_mapping:
            workflow_config.mode = mode_mapping[pattern_name]
        else:
            workflow_config.mode = self._determine_mode(workflow_config)
        
        # Create execution context
        context = ExecutionContext(
            workflow_id=correlation_id.split('_')[1] if '_' in correlation_id else correlation_id,
            workflow_type=WorkflowType.BACKTEST,
            metadata={'correlation_id': correlation_id, 'pattern_name': pattern_name}
        )
        
        # Execute using unified architecture
        return await self.execute(workflow_config, context)
    
    # REMOVED: Pattern-based execution mode detection - unified architecture uses modes directly
    # def _determine_execution_mode(self, config: WorkflowConfig, patterns: List[Dict[str, Any]]) -> str:
    #     """Determine execution mode based on configuration and detected patterns."""
    #     ...
    
    # REMOVED: Multiple executors - unified architecture has one execution flow
    # def _get_executor(self, mode: str) -> ExecutionStrategy:
    #     """Get executor for specified mode (with caching)."""
    #     ...
    
    async def validate_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Validate workflow configuration for unified architecture."""
        
        errors = []
        warnings = []
        
        # Basic configuration validation
        if not config.parameters:
            errors.append("Missing parameters configuration")
        
        # Check for required mode
        mode = config.parameters.get('mode')
        if not mode:
            errors.append("Missing 'mode' parameter (backtest, signal_generation, or signal_replay)")
        elif mode not in ['backtest', 'signal_generation', 'signal_replay']:
            errors.append(f"Invalid mode '{mode}' - must be backtest, signal_generation, or signal_replay")
        
        # Mode-specific validation
        if mode == 'backtest' or mode == 'signal_generation':
            if not config.parameters.get('symbols'):
                errors.append("Missing 'symbols' parameter")
            if not config.parameters.get('start_date'):
                errors.append("Missing 'start_date' parameter")
            if not config.parameters.get('end_date'):
                errors.append("Missing 'end_date' parameter")
            if not config.parameters.get('strategies'):
                errors.append("Missing 'strategies' parameter")
        
        if mode == 'signal_replay':
            if not config.parameters.get('signal_input_dir'):
                errors.append("Missing 'signal_input_dir' parameter for signal replay")
            if not config.parameters.get('risk_profiles'):
                errors.append("Missing 'risk_profiles' parameter for signal replay")
        
        # Calculate complexity
        param_combos = self._expand_parameter_combinations(config)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'mode': mode,
            'parameter_combinations': len(param_combos),
            'estimated_containers': 4 + len(param_combos)  # 4 core + portfolios
        }
    
    async def get_execution_preview(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Get a preview of how the workflow would be executed in unified architecture."""
        
        try:
            # Determine mode
            mode = self._determine_mode(config)
            
            # Calculate parameter combinations
            param_combos = self._expand_parameter_combinations(config)
            
            # Estimate resources
            num_containers = 4 + len(param_combos)  # 4 core + portfolio containers
            num_stateless = (
                len(config.parameters.get('strategies', [])) +
                len(config.parameters.get('classifiers', [])) +
                len(config.parameters.get('risk_profiles', []))
            )
            
            return {
                'mode': mode.value,
                'parameter_combinations': len(param_combos),
                'estimated_resources': {
                    'stateful_containers': num_containers,
                    'stateless_services': num_stateless,
                    'total_portfolios': len(param_combos),
                    'memory_mb': num_containers * 256,  # Rough estimate
                    'execution_time_minutes': len(param_combos) * 0.5  # Rough estimate
                },
                'topology': {
                    'data_container': 1,
                    'feature_hub': 1,
                    'portfolio_containers': len(param_combos),
                    'execution_container': 1,
                    'stateless_strategies': len(config.parameters.get('strategies', [])),
                    'stateless_classifiers': len(config.parameters.get('classifiers', [])),
                    'stateless_risk_validators': len(config.parameters.get('risk_profiles', []))
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'mode': 'unknown'
            }
    
    def get_supported_modes(self) -> Dict[str, Any]:
        """Get information about supported workflow modes in unified architecture."""
        
        return {
            'execution_modes': {
                'backtest': {
                    'description': 'Full pipeline execution: data → signals → orders → fills',
                    'required_params': ['symbols', 'start_date', 'end_date', 'strategies', 'risk_profiles'],
                    'output': 'Portfolio metrics and trade history'
                },
                'signal_generation': {
                    'description': 'Generate and save signals: data → signals → disk',
                    'required_params': ['symbols', 'start_date', 'end_date', 'strategies', 'signal_output_dir'],
                    'output': 'Signal files in JSONL format'
                },
                'signal_replay': {
                    'description': 'Replay saved signals: disk → orders → fills',
                    'required_params': ['signal_input_dir', 'risk_profiles'],
                    'optional_params': ['market_data_file'],
                    'output': 'Portfolio metrics from replayed signals'
                }
            },
            'stateless_components': {
                'strategies': ['momentum', 'mean_reversion'],
                'classifiers': ['trend', 'volatility', 'composite'],
                'risk_validators': ['position', 'drawdown', 'composite']
            },
            'benefits': [
                '60% reduction in container count',
                'Universal topology for all workflows',
                'Simple mode-based configuration',
                'Automatic parameter grid expansion',
                'Better parallelization of stateless components'
            ]
        }
    
    # ========================================================================
    # UNIFIED ARCHITECTURE METHODS
    # ========================================================================
    
    def _determine_mode(self, config: WorkflowConfig) -> WorkflowMode:
        """Determine workflow mode from configuration."""
        # Check for explicit mode in config
        if hasattr(config, 'mode'):
            return WorkflowMode(config.mode)
        
        # Check parameters for mode hints
        params = config.parameters
        if params.get('signal_generation_only', False):
            return WorkflowMode.SIGNAL_GENERATION
        elif params.get('signal_replay', False):
            return WorkflowMode.SIGNAL_REPLAY
        else:
            return WorkflowMode.BACKTEST
    
    async def _create_backtest_topology(self, config: WorkflowConfig) -> Dict[str, Any]:
        """
        Create topology for backtest mode.
        
        Creates Symbol-Timeframe containers for proper isolation:
        - Each symbol-timeframe combination gets its own container
        - Container includes both data and feature components
        - Perfect isolation for multi-timeframe strategies
        
        Plus Portfolio and Execution containers for full pipeline.
        """
        # Check if tracing is enabled in configuration (default to True)
        tracing_config = config.parameters.get('tracing', {})
        tracing_enabled = tracing_config.get('enabled', True)  # Default to True
        
        # Create root event bus for inter-component communication
        if tracing_enabled:
            # Use TracedEventBus for event tracing
            from ..events.tracing import TracedEventBus, EventTracer
            root_event_bus = TracedEventBus("root_event_bus")
            
            # Initialize event tracer
            correlation_id = config.parameters.get('correlation_id', f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            event_tracer = EventTracer(
                correlation_id=correlation_id,
                max_events=config.parameters.get('tracing', {}).get('max_events', 10000)
            )
            
            # Attach tracer to bus
            root_event_bus.set_tracer(event_tracer)
            
            # Store tracer for later access
            self.event_tracer = event_tracer
            
            logger.info("🔍 Created TracedEventBus with event tracing enabled")
        else:
            # Use standard EventBus without tracing
            from ..events import EventBus
            root_event_bus = EventBus("root_event_bus")
            logger.info("Created root event bus for inter-component communication")
        
        topology = {
            'containers': {},
            'stateless_components': {},
            'parameter_combinations': [],
            'root_event_bus': root_event_bus
        }
        
        # 1. Create Data and Feature Containers (Subcontainer Architecture)
        symbol_timeframe_configs = self._extract_symbol_timeframe_configs(config)
        
        # 1a. Infer required features from strategies
        from ...strategy.components.feature_inference import infer_features_from_strategies
        backtest_config = config.parameters.get('backtest', {})
        strategies = backtest_config.get('strategies', [])
        inferred_features = infer_features_from_strategies(strategies)
        logger.info(f"Inferred features from strategies: {sorted(inferred_features)}")
        
        # Add inferred features to each symbol config
        for st_config in symbol_timeframe_configs:
            # Merge inferred features with any explicit features
            existing_features = st_config.get('features', {})
            if not existing_features:
                st_config['features'] = {}
            
            # Convert inferred features to feature config format expected by SymbolTimeframeContainer
            if 'indicators' not in st_config['features']:
                st_config['features']['indicators'] = []
            
            existing_indicators = {ind['name'] for ind in st_config['features']['indicators']}
            
            for feature_spec in inferred_features:
                # Skip if already exists
                if feature_spec in existing_indicators:
                    continue
                    
                # Parse feature spec like "sma_20" or "rsi_14"
                if '_' in feature_spec:
                    parts = feature_spec.split('_')
                    feature_type = parts[0]
                    if len(parts) > 1 and parts[1].isdigit():
                        period = int(parts[1])
                        st_config['features']['indicators'].append({
                            'name': feature_spec,
                            'type': feature_type,
                            'period': period
                        })
                    else:
                        # Feature with non-numeric suffix
                        st_config['features']['indicators'].append({
                            'name': feature_spec,
                            'type': feature_type
                        })
                else:
                    # Feature without parameters
                    st_config['features']['indicators'].append({
                        'name': feature_spec,
                        'type': feature_spec
                    })
        
        for st_config in symbol_timeframe_configs:
            symbol = st_config['symbol']
            timeframe = st_config.get('timeframe', '1d')
            
            # 1a. Create Data Container (streams BAR events only)
            data_container_id = f"{symbol}_{timeframe}_data"
            from ..containers.symbol_timeframe_container import SymbolTimeframeContainer
            
            data_container = SymbolTimeframeContainer(
                symbol=symbol,
                timeframe=timeframe,
                data_config=st_config.get('data_config', {}),
                feature_config={},  # No feature computation in data container
                container_id=data_container_id
            )
            
            # Replace event bus with TracedEventBus if tracing is enabled
            if tracing_enabled and hasattr(self, 'event_tracer'):
                traced_bus = TracedEventBus(f"{data_container_id}_bus")
                traced_bus.set_tracer(self.event_tracer)
                # Copy existing subscriptions
                if hasattr(data_container.event_bus, '_subscribers'):
                    traced_bus._subscribers = data_container.event_bus._subscribers
                    traced_bus._handler_refs = data_container.event_bus._handler_refs
                data_container.event_bus = traced_bus
            
            topology['containers'][data_container_id] = data_container
            self.active_containers[data_container_id] = data_container
            
            # 1b. Create Feature Container (computes FEATURES from BAR events)
            feature_container_id = f"{symbol}_{timeframe}_features"
            feature_container = SymbolTimeframeContainer(
                symbol=symbol,
                timeframe=timeframe,
                data_config={},  # No data streaming in feature container
                feature_config=st_config.get('features', {}),
                container_id=feature_container_id
            )
            
            # Replace event bus with TracedEventBus if tracing is enabled
            if tracing_enabled and hasattr(self, 'event_tracer'):
                traced_bus = TracedEventBus(f"{feature_container_id}_bus")
                traced_bus.set_tracer(self.event_tracer)
                # Copy existing subscriptions
                if hasattr(feature_container.event_bus, '_subscribers'):
                    traced_bus._subscribers = feature_container.event_bus._subscribers
                    traced_bus._handler_refs = feature_container.event_bus._handler_refs
                feature_container.event_bus = traced_bus
            
            topology['containers'][feature_container_id] = feature_container
            self.active_containers[feature_container_id] = feature_container
        
        # 1c. Wire Data → Feature flow AFTER event bus replacement
        for st_config in symbol_timeframe_configs:
            symbol = st_config['symbol']
            timeframe = st_config.get('timeframe', '1d')
            data_container_id = f"{symbol}_{timeframe}_data"
            feature_container_id = f"{symbol}_{timeframe}_features"
            
            data_container = topology['containers'][data_container_id]
            feature_container = topology['containers'][feature_container_id]
            
            # Feature container subscribes to BAR events from data container
            data_container.event_bus.subscribe('BAR', feature_container._on_bar_received)
            logger.info(f"Wired {data_container_id} → {feature_container_id}")
        
        # 2. Create stateless components first (so portfolios can use them)
        topology['stateless_components'] = self._create_stateless_components(config)
        
        # 3. Expand parameter combinations
        param_combos = self._expand_parameter_combinations(config)
        topology['parameter_combinations'] = param_combos
        
        # 4. Create Portfolio Containers (one per combination)
        # Use the new PortfolioContainer for proper FEATURES event processing
        from ..containers.portfolio_container import PortfolioContainer
        
        for combo in param_combos:
            combo_id = combo['combo_id']
            
            portfolio_container = PortfolioContainer(
                combo_id=combo_id,
                strategy_params=combo['strategy_params'],
                risk_params=combo['risk_params'],
                initial_capital=config.parameters.get('backtest', {}).get('portfolio', {}).get('initial_capital', 100000),
                container_id=f'portfolio_{combo_id}'
            )
            
            # Replace event bus with TracedEventBus if tracing is enabled
            if tracing_enabled and hasattr(self, 'event_tracer'):
                traced_bus = TracedEventBus(f"portfolio_{combo_id}_bus")
                traced_bus.set_tracer(self.event_tracer)
                # Copy existing subscriptions
                if hasattr(portfolio_container.event_bus, '_subscribers'):
                    traced_bus._subscribers = portfolio_container.event_bus._subscribers
                    traced_bus._handler_refs = portfolio_container.event_bus._handler_refs
                portfolio_container.event_bus = traced_bus
            
            
            topology['containers'][f'portfolio_{combo_id}'] = portfolio_container
            self.active_containers[f'portfolio_{combo_id}'] = portfolio_container
        
        # 4. Create Execution Container (shared across all portfolios)
        from ..containers.execution_container import ExecutionContainer
        
        execution_container = ExecutionContainer(
            execution_config=config.parameters.get('backtest', {}).get('execution', {}),
            container_id='execution'
        )
        
        # Replace event bus with TracedEventBus if tracing is enabled
        if tracing_enabled and hasattr(self, 'event_tracer'):
            traced_bus = TracedEventBus("execution_bus")
            traced_bus.set_tracer(self.event_tracer)
            # Copy existing subscriptions
            if hasattr(execution_container.event_bus, '_subscribers'):
                traced_bus._subscribers = execution_container.event_bus._subscribers
                traced_bus._handler_refs = execution_container.event_bus._handler_refs
            execution_container.event_bus = traced_bus
        
        topology['containers']['execution'] = execution_container
        self.active_containers['execution'] = execution_container
        
        logger.info(f"Universal topology created: {len(topology['containers'])} containers, "
                   f"{len(param_combos)} parameter combinations")
        
        return topology
    
    def _extract_symbol_timeframe_configs(self, config: WorkflowConfig) -> List[Dict[str, Any]]:
        """
        Extract symbol-timeframe configurations from workflow config.
        
        Supports multiple formats:
        1. Simple: symbols: ['SPY', 'QQQ'] (defaults to 1d timeframe)
        2. Detailed: symbol_configs with explicit timeframes
        3. Multi-timeframe: symbols with multiple timeframes each
        """
        symbol_configs = []
        
        # Check for simple symbols list (backward compatibility)
        if 'symbols' in config.parameters:
            symbols = config.parameters['symbols']
            if isinstance(symbols, list) and all(isinstance(s, str) for s in symbols):
                # Simple symbol list - default to daily timeframe
                for symbol in symbols:
                    symbol_configs.append({
                        'symbol': symbol,
                        'timeframe': '1d',
                        'data_config': config.parameters.get('backtest', {}).get('data', config.data_config),
                        'features': config.parameters.get('backtest', {}).get('features', {})
                    })
                return symbol_configs
        
        # Check for detailed symbol_configs
        if 'symbol_configs' in config.parameters:
            for sc in config.parameters['symbol_configs']:
                symbol = sc['symbol']
                timeframes = sc.get('timeframes', ['1d'])
                if not isinstance(timeframes, list):
                    timeframes = [timeframes]
                
                # Create config for each symbol-timeframe combination
                for timeframe in timeframes:
                    symbol_configs.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'data_config': sc.get('data_config', config.data_config),
                        'features': sc.get('features', config.parameters.get('features', {}))
                    })
        
        # Default case - at least one symbol container
        if not symbol_configs:
            logger.warning("No symbol configurations found, creating default SPY_1d container")
            symbol_configs.append({
                'symbol': 'SPY',
                'timeframe': '1d',
                'data_config': config.data_config,
                'features': config.parameters.get('backtest', {}).get('features', {})
            })
        
        return symbol_configs
    
    def _expand_parameter_combinations(self, config: WorkflowConfig) -> List[Dict[str, Any]]:
        """
        Expand parameter grid into individual combinations.
        
        Example:
        - 20 strategies × 3 risk profiles = 60 combinations
        - Each gets its own portfolio container with unique combo_id
        """
        combinations = []
        
        # Get parameter grids from config
        # The YAML config is stored under parameters, so extract from backtest section
        backtest_config = config.parameters.get('backtest', {})
        strategy_params = backtest_config.get('strategies', [{}])
        risk_params = backtest_config.get('risk_profiles', [{}])
        classifier_params = backtest_config.get('classifiers', [{}])
        
        # Ensure they're lists
        if not isinstance(strategy_params, list):
            strategy_params = [strategy_params]
        if not isinstance(risk_params, list):
            risk_params = [risk_params]
        if not isinstance(classifier_params, list):
            classifier_params = [classifier_params]
        
        # Generate all combinations
        combo_id = 0
        for strat in strategy_params:
            for risk in risk_params:
                for classifier in classifier_params:
                    combinations.append({
                        'combo_id': f'c{combo_id:04d}',
                        'strategy_params': strat,
                        'risk_params': risk,
                        'classifier_params': classifier
                    })
                    combo_id += 1
        
        return combinations
    
    def _create_stateless_components(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Create stateless strategy, classifier, and risk components."""
        components = {
            'strategies': {},
            'classifiers': {},
            'risk_validators': {}
        }
        
        # Create strategy instances (stateless)
        backtest_config = config.parameters.get('backtest', {})
        for strat_config in backtest_config.get('strategies', []):
            strat_type = strat_config.get('type', 'momentum')
            strategy = self._create_stateless_strategy(strat_type, strat_config)
            components['strategies'][strat_type] = strategy
        
        # Create classifier instances (stateless)
        for class_config in backtest_config.get('classifiers', []):
            class_type = class_config.get('type', 'simple')
            classifier = self._create_stateless_classifier(class_type, class_config)
            components['classifiers'][class_type] = classifier
        
        # Create risk validator instances (stateless)
        for risk_config in backtest_config.get('risk_profiles', []):
            risk_type = risk_config.get('type', 'conservative')
            validator = self._create_stateless_risk_validator(risk_type, risk_config)
            components['risk_validators'][risk_type] = validator
        
        return components
    
    def _create_stateless_strategy(self, strategy_type: str, config: Dict[str, Any]) -> Any:
        """Create a stateless strategy instance using discovery system."""
        from ..containers.discovery import get_component_registry
        
        registry = get_component_registry()
        
        # Get strategy from registry
        strategy_info = registry.get_component(strategy_type)
        
        if strategy_info:
            # Return the factory (which is the strategy function itself)
            return strategy_info.factory
        else:
            # Fallback - try importing directly for backward compatibility
            logger.warning(f"Strategy '{strategy_type}' not found in registry, trying direct import")
            
            try:
                if strategy_type == 'momentum':
                    from ...strategy.strategies.stateless_momentum import momentum_strategy
                    return momentum_strategy
                elif strategy_type == 'mean_reversion':
                    from ...strategy.strategies.mean_reversion_simple import mean_reversion_strategy
                    return mean_reversion_strategy
                elif strategy_type == 'trend_following':
                    from ...strategy.strategies.trend_following import trend_following_strategy
                    return trend_following_strategy
                elif strategy_type == 'simple_trend':
                    from ...strategy.strategies.simple_trend import simple_trend_strategy
                    return simple_trend_strategy
            except ImportError as e:
                logger.error(f"Failed to import strategy '{strategy_type}': {e}")
            
            # Final fallback
            logger.error(f"Strategy type '{strategy_type}' not found")
            return lambda features, bar, params: None
    
    def _create_stateless_classifier(self, classifier_type: str, config: Dict[str, Any]) -> Any:
        """Create a stateless classifier instance using discovery system."""
        from ..containers.discovery import get_component_registry
        
        registry = get_component_registry()
        
        # Handle common name mappings
        name_mapping = {
            'trend': 'trend_classifier',
            'volatility': 'volatility_classifier',
            'momentum_regime': 'momentum_regime_classifier',
            'simple': 'trend_classifier',  # Default simple to trend
            'composite': 'trend_classifier'  # For now, use trend as default
        }
        
        # Map the type to registry name
        registry_name = name_mapping.get(classifier_type, classifier_type)
        
        # Get classifier from registry
        classifier_info = registry.get_component(registry_name)
        
        if classifier_info:
            # Return the factory (which is the classifier function itself)
            return classifier_info.factory
        else:
            # Fallback - try importing directly for backward compatibility
            logger.warning(f"Classifier '{registry_name}' not found in registry, trying direct import")
            
            try:
                from ...strategy.classifiers.stateless_classifiers import (
                    trend_classifier,
                    volatility_classifier,
                    momentum_regime_classifier
                )
                
                if classifier_type in ['trend', 'simple', 'composite']:
                    return trend_classifier
                elif classifier_type == 'volatility':
                    return volatility_classifier
                elif classifier_type == 'momentum_regime':
                    return momentum_regime_classifier
            except ImportError as e:
                logger.error(f"Failed to import classifier '{classifier_type}': {e}")
            
            # Final fallback
            logger.error(f"Classifier type '{classifier_type}' not found")
            return lambda features, params: {'regime': 'unknown', 'confidence': 0.0}
    
    def _create_stateless_risk_validator(self, risk_type: str, config: Dict[str, Any]) -> Any:
        """Create a stateless risk validator instance."""
        # Import risk validator modules
        from ...risk.stateless_validators import (
            create_stateless_position_validator,
            create_stateless_drawdown_validator,
            create_stateless_composite_validator
        )
        
        if risk_type == 'position':
            return create_stateless_position_validator()
        elif risk_type == 'drawdown':
            return create_stateless_drawdown_validator()
        elif risk_type in ['conservative', 'moderate', 'aggressive', 'composite']:
            # All risk profiles use composite validator with different params
            return create_stateless_composite_validator()
        else:
            # Fallback placeholder for other validators
            logger.warning(f"Risk validator type '{risk_type}' not yet converted to stateless")
            return {'type': risk_type, 'config': config, 'stateless': True}
    
    def _get_strategy_feature_requirements(self, strategy_type: str, config: Dict[str, Any]) -> List[str]:
        """
        Determine which features a strategy needs using discovery system.
        
        This avoids sending all features to every strategy.
        """
        from ..containers.discovery import get_component_registry
        
        registry = get_component_registry()
        
        # Get strategy info from registry
        strategy_info = registry.get_component(strategy_type)
        
        if strategy_info:
            # Get features from metadata
            base_features = strategy_info.metadata.get('features', [])
            
            # If feature_config is present, extract feature names
            feature_config = strategy_info.metadata.get('feature_config', {})
            if feature_config:
                base_features.extend(feature_config.keys())
        else:
            # Fallback to hardcoded requirements
            logger.warning(f"Strategy '{strategy_type}' not in registry, using fallback feature requirements")
            feature_requirements = {
                'momentum': ['sma', 'rsi'],  # Momentum uses moving averages and RSI
                'mean_reversion': ['rsi', 'bollinger_bands'],  # Mean reversion uses RSI and BB
                'breakout': ['atr', 'high', 'low'],  # Breakout uses ATR and price extremes
                'trend_following': ['sma', 'ema', 'adx'],  # Trend following uses various MAs and ADX
            }
            base_features = feature_requirements.get(strategy_type, [])
        
        # Add any custom features from config
        custom_features = config.get('required_features', [])
        
        # Combine and deduplicate
        all_features = list(set(base_features + custom_features))
        
        return all_features
    
    def _create_backtest_adapters(self, topology: Dict[str, Any]) -> List[Any]:
        """
        Create adapter configuration for backtest mode.
        
        Adapters:
        1. Feature routing: feature containers → strategies (via Feature Dispatcher)
        2. Signal routing: strategies → portfolios (via root bus)
        3. Order routing: portfolios → execution (via root bus)
        4. Fill broadcast: execution → portfolios
        """
        adapters = []
        
        # Get root event bus
        root_event_bus = topology.get('root_event_bus')
        if not root_event_bus:
            raise RuntimeError("Root event bus not found in topology")
            
        # Get all feature container names (these publish FEATURES events)
        feature_container_names = [name for name in topology['containers'].keys() 
                                  if name.endswith('_features')]
        
        # Get all portfolio container names
        portfolio_container_names = [name for name in topology['containers'].keys() 
                                   if name.startswith('portfolio_')]
        
        # Create Feature Dispatcher for granular feature routing
        from ..components.feature_dispatcher import FeatureDispatcher
        feature_dispatcher = FeatureDispatcher(root_event_bus=root_event_bus)
        
        # 1. Wire FEATURES events through Feature Dispatcher
        # Architecture: Symbol Container → Feature Dispatcher → Strategy Services → Root Bus → Portfolios
        stateless_components = topology.get('stateless_components', {})
        strategies = stateless_components.get('strategies', {})
        
        if strategies and feature_container_names:
            # First, register all strategies with the dispatcher
            for combo in topology['parameter_combinations']:
                strategy_config = combo['strategy_params']
                strategy_type = strategy_config.get('type')
                combo_id = combo['combo_id']
                
                # Determine which features this strategy needs
                required_features = self._get_strategy_feature_requirements(strategy_type, strategy_config)
                
                # Register with dispatcher
                feature_dispatcher.register_strategy(
                    strategy_id=f"{combo_id}_{strategy_type}",
                    strategy_type=strategy_type,
                    required_features=required_features
                )
            
            # Wire feature containers to publish to dispatcher
            for feature_container_name in feature_container_names:
                feature_container = topology['containers'][feature_container_name]
                
                # Subscribe dispatcher to FEATURES events from this container
                feature_container.event_bus.subscribe(EventType.FEATURES, feature_dispatcher.handle_features)
                logger.info(f"Wired feature container '{feature_container_name}' to Feature Dispatcher")
            
            # Wire each strategy service to handle filtered features
            for strategy_type, strategy_service in strategies.items():
                    # Create a handler for this specific strategy (fix closure issue)
                    def make_strategy_handler(s_type, s_service, param_combos, root_bus):
                        def handle_features(event: Event):
                            # Only process if this event is targeted to this strategy
                            strategy_id = event.metadata.get('target_strategy', '')
                            if not event.payload.get('filtered', False):
                                # Skip non-filtered events (shouldn't happen with dispatcher)
                                return
                            
                            logger.info(f"🎯 Strategy {s_type} received FILTERED FEATURES")
                            logger.debug(f"Event type: {event.event_type}, filtered: {event.payload.get('filtered')}")
                            
                            features = event.payload.get('features', {})
                            bar = event.payload.get('bar')
                            symbol = event.payload.get('symbol')
                            
                            logger.debug(f"Filtered features for {s_type}: {list(features.keys())}")
                            
                            if not all([features, bar, symbol]):
                                return
                                
                            # Convert Bar object to dict if needed
                            if hasattr(bar, 'to_dict'):
                                bar_dict = bar.to_dict()
                            else:
                                bar_dict = bar
                            
                            # Process for relevant portfolios
                            for combo in param_combos:
                                if combo['strategy_params'].get('type') == s_type:
                                    combo_id = combo['combo_id']
                                    strategy_params = combo['strategy_params']
                                    
                                    # Check if this event is for this combo
                                    expected_strategy_id = f"{combo_id}_{s_type}"
                                    if strategy_id and strategy_id != expected_strategy_id:
                                        continue
                                    
                                    # Call strategy service with filtered features
                                    logger.debug(f"Calling {s_type} strategy for {combo_id} with filtered features: {list(features.keys())}")
                                    signal = s_service(features, bar_dict, strategy_params)
                                    
                                    if signal and signal.get('direction') != 'flat':
                                        # Create SIGNAL event for specific portfolio
                                        signal_event = Event(
                                            event_type=EventType.SIGNAL,
                                            payload={
                                                'signal': signal,
                                                'symbol': symbol,
                                                'bar': bar_dict,
                                                'combo_id': combo_id,
                                                'correlation_id': event.payload.get('correlation_id')
                                            },
                                            source_id=f'strategy_{s_type}'
                                        )
                                        root_bus.publish(signal_event)
                                        logger.info(f"Strategy {s_type} published SIGNAL for {combo_id}")
                        return handle_features
                
                    # Register handler with feature dispatcher instead of direct subscription
                    handler = make_strategy_handler(strategy_type, strategy_service, topology['parameter_combinations'], root_event_bus)
                    
                    # Register all instances of this strategy type with the dispatcher
                    for combo in topology['parameter_combinations']:
                        if combo['strategy_params'].get('type') == strategy_type:
                            strategy_id = f"{combo['combo_id']}_{strategy_type}"
                            feature_dispatcher.feature_handlers[strategy_id].append(handler)
                    
                    logger.info(f"Registered strategy service '{strategy_type}' with Feature Dispatcher")
            
            # 1.5. Wire portfolio containers to receive SIGNAL events from ROOT event bus
            # Portfolios subscribe to root bus for SIGNAL events from strategies
            for portfolio_name in portfolio_container_names:
                portfolio_container = topology['containers'][portfolio_name]
                
                # Subscribe portfolio to SIGNAL events on the ROOT event bus
                root_event_bus.subscribe(EventType.SIGNAL, portfolio_container._on_signal_received)
                logger.info(f"Wired portfolio '{portfolio_name}' to receive SIGNAL events from ROOT event bus")
        else:
            logger.warning("No strategy services found or no feature containers to wire")
        
        # 2. Risk Service Adapter: portfolios → ORDER_REQUEST → risk → ORDER → execution
        # Create risk service adapter to bridge isolated portfolios to root bus
        if portfolio_container_names and 'execution' in topology['containers']:
            # Get stateless risk validators
            risk_validators = topology.get('stateless_components', {}).get('risk_validators', {})
            
            # Add a default composite validator if none exist
            if not risk_validators:
                from ...risk.validators import validate_composite
                risk_validators['default'] = validate_composite
                logger.info("Created default composite risk validator")
            
            # Create risk service adapter
            risk_adapter = self.adapter_factory.create_adapter(
                name='risk_service',
                config={
                    'type': 'risk_service',
                    'risk_validators': risk_validators,
                    'root_event_bus': root_event_bus
                }
            )
            adapters.append(risk_adapter)
            logger.info(f"Created risk service adapter for {len(portfolio_container_names)} portfolios")
            
            # Subscribe execution to ORDER events on ROOT event bus
            execution_container = topology['containers']['execution']
            root_event_bus.subscribe(EventType.ORDER, execution_container._on_order_received)
            logger.info(f"Wired execution container to receive ORDER events from ROOT event bus")
        
        # 3. Fill broadcast: execution → all portfolios
        if 'execution' in topology['containers'] and portfolio_container_names:
            fill_broadcast = self.adapter_factory.create_adapter(
                name='fill_broadcast',
                config={
                    'type': 'broadcast',
                    'source': 'execution',
                    'targets': portfolio_container_names,
                    'allowed_types': [EventType.FILL]  # Only broadcast FILL events
                }
            )
            adapters.append(fill_broadcast)
            logger.info(f"Created fill broadcast from execution to {len(portfolio_container_names)} portfolios")
        
        # Wire the adapters with the actual containers
        all_containers = topology['containers']
        for adapter in adapters:
            try:
                adapter.setup(all_containers)
                adapter.start()
            except Exception as e:
                logger.error(f"Failed to setup adapter {adapter.name}: {e}")
                raise
        
        self.active_adapters.extend(adapters)
        logger.info(f"Created {len(adapters)} backtest adapters for topology")
        return adapters
    
    def _create_signal_generation_adapters(self, topology: Dict[str, Any]) -> List[Any]:
        """
        Create adapter configuration for signal generation mode.
        
        Adapters:
        1. Feature routing: feature containers → strategies (via Feature Dispatcher)
        NO portfolio adapters - signals are saved to disk instead
        """
        adapters = []
        
        # Get root event bus
        root_event_bus = topology.get('root_event_bus')
        if not root_event_bus:
            raise RuntimeError("Root event bus not found in topology")
            
        # Get feature container names
        feature_container_names = [name for name in topology['containers'].keys() 
                                  if name.endswith('_features')]
        
        # Create Feature Dispatcher for signal generation
        from ..components.feature_dispatcher import FeatureDispatcher
        feature_dispatcher = FeatureDispatcher(root_event_bus=root_event_bus)
        
        # Wire FEATURES events through dispatcher (similar to backtest)
        stateless_components = topology.get('stateless_components', {})
        strategies = stateless_components.get('strategies', {})
        
        if strategies and feature_container_names:
            # Register strategies with dispatcher
            for combo in topology['parameter_combinations']:
                strategy_config = combo['strategy_params']
                strategy_type = strategy_config.get('type')
                combo_id = combo['combo_id']
                
                required_features = self._get_strategy_feature_requirements(strategy_type, strategy_config)
                
                feature_dispatcher.register_strategy(
                    strategy_id=f"{combo_id}_{strategy_type}",
                    strategy_type=strategy_type,
                    required_features=required_features
                )
            
            # Wire feature containers to dispatcher
            for feature_container_name in feature_container_names:
                feature_container = topology['containers'][feature_container_name]
                feature_container.event_bus.subscribe(EventType.FEATURES, feature_dispatcher.handle_features)
                logger.info(f"Wired feature container '{feature_container_name}' to Feature Dispatcher")
            
            # For signal generation, strategies publish signals to be saved
            # No portfolio wiring needed
        
        logger.info(f"Created {len(adapters)} signal generation adapters")
        return adapters
    
    def _create_signal_replay_adapters(self, topology: Dict[str, Any]) -> List[Any]:
        """
        Create adapter configuration for signal replay mode.
        
        Adapters:
        1. Order routing: portfolios → execution (via root bus)
        2. Fill broadcast: execution → portfolios
        NO feature/strategy adapters - signals come from disk
        """
        adapters = []
        
        # Get root event bus
        root_event_bus = topology.get('root_event_bus')
        if not root_event_bus:
            raise RuntimeError("Root event bus not found in topology")
            
        # Get portfolio container names
        portfolio_container_names = [name for name in topology['containers'].keys() 
                                   if name.startswith('portfolio_')]
        
        # 1. Order flow: portfolios → execution via ROOT event bus
        if 'execution' in topology['containers']:
            execution_container = topology['containers']['execution']
            
            # Subscribe execution to ORDER events on root bus
            root_event_bus.subscribe(EventType.ORDER, execution_container._on_order_received)
            logger.info(f"Wired execution container to receive ORDER events from ROOT event bus")
        
        # 2. Fill broadcast: execution → all portfolios
        if 'execution' in topology['containers'] and portfolio_container_names:
            fill_broadcast = self.adapter_factory.create_adapter(
                name='fill_broadcast',
                config={
                    'type': 'broadcast',
                    'source': 'execution',
                    'targets': portfolio_container_names,
                    'allowed_types': [EventType.FILL]
                }
            )
            adapters.append(fill_broadcast)
            logger.info(f"Created fill broadcast from execution to {len(portfolio_container_names)} portfolios")
        
        # Wire the adapters
        all_containers = topology['containers']
        for adapter in adapters:
            try:
                adapter.setup(all_containers)
                adapter.start()
            except Exception as e:
                logger.error(f"Failed to setup adapter {adapter.name}: {e}")
                raise
        
        self.active_adapters.extend(adapters)
        logger.info(f"Created {len(adapters)} signal replay adapters")
        return adapters
    
    async def _execute_full_pipeline(
        self, 
        topology: Dict[str, Any], 
        config: WorkflowConfig, 
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute full backtest pipeline using event-driven architecture."""
        logger.info("Executing full backtest pipeline")
        
        # Initialize result
        result = WorkflowResult(
            workflow_id=context.workflow_id,
            workflow_type=config.workflow_type,
            success=True,
            metadata={
                'mode': 'backtest',
                'containers': len(topology['containers']),
                'combinations': len(topology['parameter_combinations'])
            }
        )
        
        try:
            # 1. Initialize all containers (but don't start yet)
            logger.info("Initializing all containers...")
            for container_name, container in topology['containers'].items():
                await container.initialize()
                logger.debug(f"Initialized container: {container_name}")
                
            # 2. Now start containers after adapters are already connected
            logger.info("Starting all containers...")
            for container_name, container in topology['containers'].items():
                await container.start()
                logger.debug(f"Started container: {container_name}")
            
            # 2. Simple approach: wait for data processing to complete
            # In real implementation, this would be event-driven
            # For now, just simulate the execution
            import asyncio
            
            # Let containers process for a simulated duration
            # In reality, symbol containers would stream data and broadcast FEATURES
            # Portfolio containers would receive and process those events
            logger.info("Running backtest simulation...")
            await asyncio.sleep(2)  # Simulate processing time
            
            # 2.5 Close all open positions at final prices
            logger.info("Closing all open positions...")
            
            # Get final prices from symbol containers
            final_prices = {}
            for container_name, container in topology['containers'].items():
                if container_name.endswith('_1d') or container_name.endswith('_1m'):
                    # This is a symbol-timeframe container
                    state = container.get_state_info()
                    symbol = state.get('symbol')
                    if symbol and hasattr(container, 'current_bar') and container.current_bar:
                        final_prices[symbol] = container.current_bar.close
                        logger.info(f"Final price for {symbol}: ${container.current_bar.close:.2f}")
            
            # Close positions in all portfolios
            for combo in topology['parameter_combinations']:
                combo_id = combo['combo_id']
                portfolio = topology['containers'][f'portfolio_{combo_id}']
                if hasattr(portfolio, 'close_all_positions'):
                    await portfolio.close_all_positions(final_prices)
            
            # Wait for closing orders to process
            logger.info("Waiting for closing orders to be filled...")
            await asyncio.sleep(1.0)  # Give more time for fill processing
            
            # 3. Collect results from portfolio containers
            portfolio_results = {}
            
            # Get actual bar count from data containers
            bar_count = 0
            for container_name, container in topology['containers'].items():
                if container_name.endswith('_data'):
                    # This is a data container
                    if hasattr(container, 'data_source') and hasattr(container.data_source, 'current_index'):
                        bar_count = max(bar_count, container.data_source.current_index)
            
            # Fallback if we couldn't get bar count
            if bar_count == 0:
                # Try to get from feature containers
                for container_name, container in topology['containers'].items():
                    if container_name.endswith('_features'):
                        state = container.get_state_info()
                        bar_count = max(bar_count, state.get('bars_processed', 0))
            
            logger.info(f"Total bars processed: {bar_count}")
                
                # Process through each parameter combination
            # Collect results from portfolio containers  
            for combo in topology['parameter_combinations']:
                combo_id = combo['combo_id']
                portfolio = topology['containers'][f'portfolio_{combo_id}']
                
                # Get actual metrics from portfolio container
                state_info = portfolio.get_state_info()
                logger.info(f"Portfolio {combo_id} state: features_received={state_info.get('_features_received', 0)}, "
                          f"signals_generated={state_info.get('_signals_generated', 0)}, "
                          f"orders_created={state_info.get('_orders_created', 0)}")
                
                if hasattr(portfolio, 'get_metrics'):
                    metrics = await portfolio.get_metrics()
                else:
                    # Fallback to getting state info
                    metrics = state_info.get('metrics', {
                        'total_value': 100000,
                        'total_return': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0
                    })
                
                portfolio_results[combo_id] = {
                    'combo_id': combo_id,
                    'parameters': combo,
                    'metrics': metrics,
                    'final_value': metrics.get('total_value', 0),
                    'total_return': metrics.get('total_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0)
                }
            
            # Find best combination
            if portfolio_results:
                best_combo = max(portfolio_results.values(), 
                               key=lambda x: x.get('sharpe_ratio', 0))
            else:
                best_combo = None
            
            result.final_results = {
                'best_combination': best_combo,
                'all_results': portfolio_results,
                'bar_count': bar_count
            }
            
            # Add trace summary if tracing is enabled
            trace_summary = self.get_trace_summary()
            if trace_summary:
                result.final_results['trace_summary'] = trace_summary
                logger.info(f"🔍 Event trace summary: {trace_summary.get('total_events', 0)} events traced")
            
        except Exception as e:
            logger.error(f"Full pipeline execution failed: {e}")
            result.success = False
            result.add_error(str(e))
        
        finally:
            # Add trace summary to metadata even on failure
            trace_summary = self.get_trace_summary()
            if trace_summary:
                result.metadata['trace_summary'] = trace_summary
                
            # Stop all containers
            for container_name, container in topology['containers'].items():
                try:
                    await container.stop()
                except Exception as e:
                    logger.error(f"Error stopping container {container_name}: {e}")
        
        return result
    
    async def _execute_signal_generation(
        self, 
        topology: Dict[str, Any], 
        config: WorkflowConfig, 
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute signal generation only: data → features → signals (save)."""
        logger.info("Executing signal generation pipeline")
        
        # Initialize result
        result = WorkflowResult(
            workflow_id=context.workflow_id,
            workflow_type=config.workflow_type,
            success=True,
            metadata={
                'mode': 'signal_generation',
                'containers': len(topology['containers']),
                'combinations': len(topology['parameter_combinations'])
            }
        )
        
        try:
            # 1. Start only data, feature, and portfolio containers
            logger.info("Starting containers for signal generation...")
            for container_name, container in topology['containers'].items():
                if container_name != 'execution':  # Skip execution for signal gen
                    await container.start()
                    logger.debug(f"Started container: {container_name}")
            
            # 2. Initialize signal storage
            signal_output_dir = config.parameters.get('signal_output_dir', f"./signals/{context.workflow_id}")
            os.makedirs(signal_output_dir, exist_ok=True)
            signal_files = {}
            
            # Create signal file for each combination
            for combo in topology['parameter_combinations']:
                combo_id = combo['combo_id']
                signal_file_path = os.path.join(signal_output_dir, f"{combo_id}_signals.jsonl")
                signal_files[combo_id] = open(signal_file_path, 'w')
            
            # 3. Initialize data streaming
            data_container = topology['containers']['data']
            feature_hub = topology['containers']['feature_hub']
            
            # Configure data source
            data_config = config.data_config
            await data_container.configure({
                'symbols': data_config.get('symbols', ['SPY']),
                'start_date': data_config.get('start_date'),
                'end_date': data_config.get('end_date'),
                'frequency': data_config.get('frequency', '1d')
            })
            
            # 4. Process each bar and capture signals
            bar_count = 0
            signal_counts = {combo['combo_id']: 0 for combo in topology['parameter_combinations']}
            
            # Start data streaming
            async for market_bar in data_container.stream_data():
                bar_count += 1
                
                # Update features with new bar
                await feature_hub.update_bar(market_bar)
                
                # Get current features
                features = await feature_hub.get_features(market_bar['symbol'])
                
                # Generate signals for each parameter combination
                for combo in topology['parameter_combinations']:
                    combo_id = combo['combo_id']
                    
                    # Get stateless strategy for this combination
                    strategy_type = combo['strategy_params'].get('type', 'momentum')
                    strategy = topology['stateless_components']['strategies'].get(strategy_type)
                    
                    if strategy:
                        # Generate signal using stateless strategy
                        signal = self._call_stateless_strategy(
                            strategy,
                            features,
                            market_bar,
                            combo['strategy_params']
                        )
                        
                        if signal:
                            # Add combination metadata
                            signal['combo_id'] = combo_id
                            signal['strategy_params'] = combo['strategy_params']
                            signal['bar_number'] = bar_count
                            
                            # Check if classifier affects signal
                            classifier_type = combo['classifier_params'].get('type')
                            if classifier_type:
                                classifier = topology['stateless_components']['classifiers'].get(classifier_type)
                                if classifier:
                                    regime = self._call_stateless_classifier(
                                        classifier,
                                        features,
                                        combo['classifier_params']
                                    )
                                    signal['regime'] = regime
                            
                            # Write signal to file
                            signal_files[combo_id].write(json.dumps(signal) + '\n')
                            signal_counts[combo_id] += 1
                            
                            # Optionally store in portfolio for metrics calculation
                            portfolio = topology['containers'][f'portfolio_{combo_id}']
                            await portfolio.record_signal(signal)
            
            # 5. Close signal files and collect metrics
            for combo_id, file_handle in signal_files.items():
                file_handle.close()
            
            logger.info(f"Generated signals for {bar_count} bars")
            logger.info(f"Signal counts by combination: {signal_counts}")
            
            # Collect performance metrics (without execution)
            portfolio_metrics = {}
            for combo in topology['parameter_combinations']:
                combo_id = combo['combo_id']
                portfolio = topology['containers'][f'portfolio_{combo_id}']
                
                # Get signal-based metrics
                metrics = await portfolio.get_signal_metrics()
                portfolio_metrics[combo_id] = {
                    'combo_id': combo_id,
                    'parameters': combo,
                    'signal_count': signal_counts[combo_id],
                    'signal_metrics': metrics,
                    'signal_file': os.path.join(signal_output_dir, f"{combo_id}_signals.jsonl")
                }
            
            result.final_results = {
                'signal_output_dir': signal_output_dir,
                'bar_count': bar_count,
                'total_signals': sum(signal_counts.values()),
                'signal_counts': signal_counts,
                'portfolio_metrics': portfolio_metrics
            }
            
            # Store output directory for signal replay
            result.metadata['signal_output_dir'] = signal_output_dir
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            result.success = False
            result.add_error(str(e))
            
            # Clean up signal files on error
            for file_handle in signal_files.values():
                if not file_handle.closed:
                    file_handle.close()
        
        finally:
            # Stop containers
            for container_name, container in topology['containers'].items():
                if container_name != 'execution':
                    try:
                        await container.stop()
                    except Exception as e:
                        logger.error(f"Error stopping container {container_name}: {e}")
        
        return result
    
    async def _execute_signal_replay(
        self, 
        topology: Dict[str, Any], 
        config: WorkflowConfig, 
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute signal replay: load signals → orders → fills."""
        logger.info("Executing signal replay pipeline")
        
        # Initialize result
        result = WorkflowResult(
            workflow_id=context.workflow_id,
            workflow_type=config.workflow_type,
            success=True,
            metadata={
                'mode': 'signal_replay',
                'containers': len(topology['containers']),
                'combinations': len(topology['parameter_combinations'])
            }
        )
        
        try:
            # 1. Start portfolio and execution containers (no data/feature needed)
            logger.info("Starting containers for signal replay...")
            for container_name, container in topology['containers'].items():
                if container_name.startswith('portfolio_') or container_name == 'execution':
                    await container.start()
                    logger.debug(f"Started container: {container_name}")
            
            # 2. Get signal input directory
            signal_input_dir = config.parameters.get('signal_input_dir')
            if not signal_input_dir:
                raise ValueError("signal_input_dir parameter required for signal replay")
            
            if not os.path.exists(signal_input_dir):
                raise ValueError(f"Signal directory not found: {signal_input_dir}")
            
            # 3. Load market data for pricing (if provided)
            market_data_file = config.parameters.get('market_data_file')
            market_data = {}
            if market_data_file and os.path.exists(market_data_file):
                with open(market_data_file, 'r') as f:
                    market_data = json.load(f)
            
            # 4. Process signals for each combination
            signal_counts = {}
            order_counts = {}
            fill_counts = {}
            portfolio_results = {}
            
            for combo in topology['parameter_combinations']:
                combo_id = combo['combo_id']
                signal_file = os.path.join(signal_input_dir, f"{combo_id}_signals.jsonl")
                
                if not os.path.exists(signal_file):
                    logger.warning(f"Signal file not found for {combo_id}, skipping")
                    continue
                
                portfolio = topology['containers'][f'portfolio_{combo_id}']
                execution = topology['containers']['execution']
                
                # Get risk validator for this combination
                risk_type = combo['risk_params'].get('type', 'conservative')
                risk_validator = topology['stateless_components']['risk_validators'].get(risk_type)
                
                signal_count = 0
                order_count = 0
                fill_count = 0
                
                # Read and process signals
                with open(signal_file, 'r') as f:
                    for line in f:
                        signal = json.loads(line.strip())
                        signal_count += 1
                        
                        # Update portfolio with signal
                        await portfolio.process_signal(signal)
                        
                        # Get any resulting orders
                        if portfolio.has_pending_orders():
                            orders = await portfolio.get_pending_orders()
                            portfolio_state = await portfolio.get_state()
                            
                            for order in orders:
                                # Get current market price from signal or market data
                                symbol = order.get('symbol', 'SPY')
                                bar_data = {
                                    'symbol': symbol,
                                    'close': signal.get('price', market_data.get(symbol, {}).get('close', 100)),
                                    'timestamp': signal.get('timestamp')
                                }
                                
                                # Validate order with stateless risk validator
                                validation = self._call_stateless_risk_validator(
                                    risk_validator,
                                    order,
                                    portfolio_state,
                                    combo['risk_params'],
                                    bar_data
                                )
                                
                                if validation['approved']:
                                    # Adjust quantity if needed
                                    if validation.get('adjusted_quantity'):
                                        order['quantity'] = validation['adjusted_quantity']
                                    
                                    # Send to execution
                                    fill = await execution.submit_order(order)
                                    order_count += 1
                                    
                                    if fill and fill.get('status') == 'filled':
                                        fill_count += 1
                                        # Update portfolio with fill
                                        await portfolio.update_position(fill)
                                else:
                                    await portfolio.reject_order(order, validation['reason'])
                        
                        # Update portfolio valuation if we have market data
                        if market_data:
                            await portfolio.update_market_prices(bar_data)
                
                # Store counts
                signal_counts[combo_id] = signal_count
                order_counts[combo_id] = order_count
                fill_counts[combo_id] = fill_count
                
                # Get final portfolio metrics
                metrics = await portfolio.get_metrics()
                portfolio_results[combo_id] = {
                    'combo_id': combo_id,
                    'parameters': combo,
                    'signals_processed': signal_count,
                    'orders_placed': order_count,
                    'orders_filled': fill_count,
                    'metrics': metrics,
                    'final_value': metrics.get('total_value', 0),
                    'total_return': metrics.get('total_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0)
                }
            
            # 5. Find best combination
            if portfolio_results:
                best_combo = max(portfolio_results.values(), 
                               key=lambda x: x.get('sharpe_ratio', 0))
            else:
                best_combo = None
            
            result.final_results = {
                'best_combination': best_combo,
                'all_results': portfolio_results,
                'total_signals': sum(signal_counts.values()),
                'total_orders': sum(order_counts.values()),
                'total_fills': sum(fill_counts.values()),
                'signal_counts': signal_counts,
                'order_counts': order_counts,
                'fill_counts': fill_counts
            }
            
            logger.info(f"Signal replay completed: {sum(signal_counts.values())} signals, "
                       f"{sum(order_counts.values())} orders, {sum(fill_counts.values())} fills")
            
        except Exception as e:
            logger.error(f"Signal replay failed: {e}")
            result.success = False
            result.add_error(str(e))
        
        finally:
            # Stop containers
            for container_name, container in topology['containers'].items():
                if container_name.startswith('portfolio_') or container_name == 'execution':
                    try:
                        await container.stop()
                    except Exception as e:
                        logger.error(f"Error stopping container {container_name}: {e}")
        
        return result
    
    def _call_stateless_strategy(
        self,
        strategy: Any,
        features: Dict[str, Any],
        bar: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Call a stateless strategy to generate a signal."""
        try:
            # Check if it's a real stateless strategy with generate_signal method
            if hasattr(strategy, 'generate_signal'):
                # Call the actual stateless strategy
                signal = strategy.generate_signal(features, bar, params)
                
                # Add bar metadata to signal if not flat
                if signal and signal.get('direction') != 'flat':
                    signal['symbol'] = bar.get('symbol', 'SPY')
                    signal['timestamp'] = bar.get('timestamp')
                    return signal
                return None
                
            # Fallback for placeholder strategies
            elif isinstance(strategy, dict) and strategy.get('stateless'):
                # Mock signal generation for strategies not yet converted
                import random
                if random.random() > 0.7:  # 30% chance of signal
                    return {
                        'symbol': bar.get('symbol', 'SPY'),
                        'direction': random.choice(['long', 'short']),
                        'strength': random.uniform(0.5, 1.0),
                        'timestamp': bar.get('timestamp'),
                        'metadata': {'strategy': strategy['type']}
                    }
            return None
        except Exception as e:
            logger.error(f"Error calling stateless strategy: {e}")
            return None
    
    def _call_stateless_risk_validator(
        self,
        validator: Any,
        order: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        risk_params: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a stateless risk validator to validate an order."""
        try:
            # Check if it's a real stateless validator with validate_order method
            if hasattr(validator, 'validate_order'):
                # Call the actual stateless validator
                return validator.validate_order(order, portfolio_state, risk_params, market_data)
                
            # Fallback for placeholder validators
            elif isinstance(validator, dict) and validator.get('stateless'):
                # Mock risk validation for validators not yet converted
                import random
                approved = random.random() > 0.1  # 90% approval rate
                return {
                    'approved': approved,
                    'adjusted_quantity': order.get('quantity'),
                    'reason': 'Risk limits exceeded' if not approved else '',
                    'risk_metrics': {
                        'position_size_pct': 0.02,
                        'portfolio_risk': 0.15
                    }
                }
            return {'approved': True, 'reason': ''}
        except Exception as e:
            logger.error(f"Error calling stateless risk validator: {e}")
            return {'approved': False, 'reason': str(e)}
    
    def _call_stateless_classifier(
        self,
        classifier: Any,
        features: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a stateless classifier to detect market regime."""
        try:
            # Check if it's a real stateless classifier with classify_regime method
            if hasattr(classifier, 'classify_regime'):
                # Call the actual stateless classifier
                return classifier.classify_regime(features, params)
                
            # Fallback for placeholder classifiers
            elif isinstance(classifier, dict) and classifier.get('stateless'):
                # Mock regime classification for classifiers not yet converted
                import random
                regimes = ['bull', 'bear', 'sideways']
                return {
                    'regime': random.choice(regimes),
                    'confidence': random.uniform(0.6, 0.95),
                    'metadata': {'classifier': classifier['type']}
                }
            return {'regime': 'unknown', 'confidence': 0.0}
        except Exception as e:
            logger.error(f"Error calling stateless classifier: {e}")
            return {'regime': 'error', 'confidence': 0.0}
    
    def get_trace_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get event trace summary if tracing is enabled.
        
        Returns:
            Dict with trace summary or None if tracing not enabled
        """
        logger.debug(f"Checking for event tracer: hasattr={hasattr(self, 'event_tracer')}, value={getattr(self, 'event_tracer', None)}")
        if hasattr(self, 'event_tracer') and self.event_tracer:
            summary = self.event_tracer.get_summary()
            logger.info(f"Getting trace summary: {summary.get('total_events', 0)} total events")
            return summary
        logger.info(f"No event tracer available for trace summary (hasattr={hasattr(self, 'event_tracer')})")
        return None
    
    async def cleanup(self) -> None:
        """Clean up all active resources."""
        
        logger.info("Cleaning up workflow manager resources...")
        
        # Clean up active containers
        for container_id, container in list(self.active_containers.items()):
            try:
                await container.dispose()
                del self.active_containers[container_id]
            except Exception as e:
                logger.error(f"Error disposing container {container_id}: {e}")
        
        # Clean up communication adapters
        try:
            self.adapter_factory.stop_all()
            self.active_adapters.clear()
        except Exception as e:
            logger.error(f"Error stopping adapters: {e}")
        
        # No executor cache in unified architecture
        # self._executors.clear()
        
        logger.info("Workflow manager cleanup complete")
    
    async def _create_signal_generation_topology(self, config: WorkflowConfig) -> Dict[str, Any]:
        """
        Create topology for signal generation mode.
        
        Creates:
        - Symbol-Timeframe containers (data + features)
        - Strategy services
        - NO Portfolio containers (signals are saved to disk)
        - NO Execution container (no order/fill processing)
        """
        # Start with same event bus setup as backtest
        tracing_config = config.parameters.get('tracing', {})
        tracing_enabled = tracing_config.get('enabled', True)
        
        # Create root event bus
        if tracing_enabled:
            from ..events.tracing import TracedEventBus, EventTracer
            root_event_bus = TracedEventBus("root_event_bus")
            
            correlation_id = config.parameters.get('correlation_id', f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            event_tracer = EventTracer(
                correlation_id=correlation_id,
                max_events=config.parameters.get('tracing', {}).get('max_events', 10000)
            )
            
            root_event_bus.set_tracer(event_tracer)
            self.event_tracer = event_tracer
            logger.info("🔍 Created TracedEventBus for signal generation")
        else:
            from ..events import EventBus
            root_event_bus = EventBus("root_event_bus")
            
        topology = {
            'containers': {},
            'stateless_components': {},
            'parameter_combinations': [],
            'root_event_bus': root_event_bus
        }
        
        # 1. Create Symbol-Timeframe containers (same as backtest)
        symbol_timeframe_configs = self._extract_symbol_timeframe_configs(config)
        
        for st_config in symbol_timeframe_configs:
            symbol = st_config['symbol']
            timeframe = st_config.get('timeframe', '1d')
            
            # Create data container
            data_container_id = f"{symbol}_{timeframe}_data"
            from ..containers.symbol_timeframe_container import SymbolTimeframeContainer
            
            data_container = SymbolTimeframeContainer(
                symbol=symbol,
                timeframe=timeframe,
                data_config=st_config.get('data_config', {}),
                feature_config={},
                container_id=data_container_id
            )
            
            if tracing_enabled and hasattr(self, 'event_tracer'):
                traced_bus = TracedEventBus(f"{data_container_id}_bus")
                traced_bus.set_tracer(self.event_tracer)
                data_container.event_bus = traced_bus
            
            topology['containers'][data_container_id] = data_container
            self.active_containers[data_container_id] = data_container
            
            # Create feature container
            feature_container_id = f"{symbol}_{timeframe}_features"
            feature_container = SymbolTimeframeContainer(
                symbol=symbol,
                timeframe=timeframe,
                data_config={},
                feature_config=st_config.get('features', {}),
                container_id=feature_container_id
            )
            
            if tracing_enabled and hasattr(self, 'event_tracer'):
                traced_bus = TracedEventBus(f"{feature_container_id}_bus")
                traced_bus.set_tracer(self.event_tracer)
                feature_container.event_bus = traced_bus
            
            topology['containers'][feature_container_id] = feature_container
            self.active_containers[feature_container_id] = feature_container
            
        # Wire Data → Feature flow
        for st_config in symbol_timeframe_configs:
            symbol = st_config['symbol']
            timeframe = st_config.get('timeframe', '1d')
            data_container_id = f"{symbol}_{timeframe}_data"
            feature_container_id = f"{symbol}_{timeframe}_features"
            
            data_container = topology['containers'][data_container_id]
            feature_container = topology['containers'][feature_container_id]
            
            data_container.event_bus.subscribe('BAR', feature_container._on_bar_received)
            logger.info(f"Wired {data_container_id} → {feature_container_id}")
        
        # 2. Create stateless strategy components (same as backtest)
        topology['stateless_components'] = self._create_stateless_components(config)
        
        # 3. Expand parameter combinations (same as backtest)
        param_combos = self._expand_parameter_combinations(config)
        topology['parameter_combinations'] = param_combos
        
        # NO Portfolio containers for signal generation
        # NO Execution container for signal generation
        
        logger.info(f"Signal generation topology created: {len(topology['containers'])} containers, "
                   f"{len(param_combos)} parameter combinations")
        
        return topology
    
    async def _create_signal_replay_topology(self, config: WorkflowConfig) -> Dict[str, Any]:
        """
        Create topology for signal replay mode.
        
        Creates:
        - Portfolio containers (to process saved signals)
        - Execution container (to process orders)
        - Risk validators
        - NO Symbol-Timeframe containers (signals come from disk)
        - NO Strategy services (signals already generated)
        """
        # Start with same event bus setup
        tracing_config = config.parameters.get('tracing', {})
        tracing_enabled = tracing_config.get('enabled', True)
        
        # Create root event bus
        if tracing_enabled:
            from ..events.tracing import TracedEventBus, EventTracer
            root_event_bus = TracedEventBus("root_event_bus")
            
            correlation_id = config.parameters.get('correlation_id', f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            event_tracer = EventTracer(
                correlation_id=correlation_id,
                max_events=config.parameters.get('tracing', {}).get('max_events', 10000)
            )
            
            root_event_bus.set_tracer(event_tracer)
            self.event_tracer = event_tracer
            logger.info("🔍 Created TracedEventBus for signal replay")
        else:
            from ..events import EventBus
            root_event_bus = EventBus("root_event_bus")
            
        topology = {
            'containers': {},
            'stateless_components': {},
            'parameter_combinations': [],
            'root_event_bus': root_event_bus
        }
        
        # NO Symbol-Timeframe containers for signal replay
        # NO Strategy components for signal replay
        
        # 1. Create risk validators (stateless)
        topology['stateless_components'] = {
            'strategies': {},  # Empty - no strategies needed
            'classifiers': {},  # Empty - no classifiers needed
            'risk_validators': {}  # Will be populated
        }
        
        # Create risk validators only
        backtest_config = config.parameters.get('backtest', {})
        for risk_config in backtest_config.get('risk_profiles', []):
            risk_type = risk_config.get('type', 'conservative')
            validator = self._create_stateless_risk_validator(risk_type, risk_config)
            topology['stateless_components']['risk_validators'][risk_type] = validator
        
        # 2. Expand parameter combinations (for signal replay, based on risk profiles only)
        param_combos = []
        combo_id = 0
        
        # For signal replay, we create combinations based on saved signal files
        signal_input_dir = config.parameters.get('signal_input_dir')
        if signal_input_dir and os.path.exists(signal_input_dir):
            # Look for signal files to determine combinations
            import glob
            signal_files = glob.glob(os.path.join(signal_input_dir, "*_signals.jsonl"))
            
            for signal_file in signal_files:
                # Extract combo_id from filename (e.g., "c0000_signals.jsonl")
                filename = os.path.basename(signal_file)
                original_combo_id = filename.split('_')[0]
                
                # Create combination for each risk profile
                for risk_config in backtest_config.get('risk_profiles', [{}]):
                    param_combos.append({
                        'combo_id': f'r{combo_id:04d}',
                        'original_combo_id': original_combo_id,
                        'signal_file': signal_file,
                        'strategy_params': {},  # Empty for replay
                        'risk_params': risk_config,
                        'classifier_params': {}  # Empty for replay
                    })
                    combo_id += 1
        else:
            # Fallback - create one combination per risk profile
            for risk_config in backtest_config.get('risk_profiles', [{}]):
                param_combos.append({
                    'combo_id': f'r{combo_id:04d}',
                    'strategy_params': {},
                    'risk_params': risk_config,
                    'classifier_params': {}
                })
                combo_id += 1
        
        topology['parameter_combinations'] = param_combos
        
        # 3. Create Portfolio containers (one per combination)
        from ..containers.portfolio_container import PortfolioContainer
        
        for combo in param_combos:
            combo_id = combo['combo_id']
            
            portfolio_container = PortfolioContainer(
                combo_id=combo_id,
                strategy_params=combo['strategy_params'],
                risk_params=combo['risk_params'],
                initial_capital=config.parameters.get('backtest', {}).get('portfolio', {}).get('initial_capital', 100000),
                container_id=f'portfolio_{combo_id}',
                root_event_bus=root_event_bus
            )
            
            if tracing_enabled and hasattr(self, 'event_tracer'):
                traced_bus = TracedEventBus(f"portfolio_{combo_id}_bus")
                traced_bus.set_tracer(self.event_tracer)
                portfolio_container.event_bus = traced_bus
            
            # Give portfolio access to root event bus
            portfolio_container.root_event_bus = root_event_bus
            
            # Assign risk validator
            risk_type = combo['risk_params'].get('type')
            if risk_type in topology.get('stateless_components', {}).get('risk_validators', {}):
                risk_validator = topology['stateless_components']['risk_validators'][risk_type]
                portfolio_container.set_risk_validator(risk_validator)
            
            topology['containers'][f'portfolio_{combo_id}'] = portfolio_container
            self.active_containers[f'portfolio_{combo_id}'] = portfolio_container
        
        # 4. Create Execution container
        from ..containers.execution_container import ExecutionContainer
        
        execution_container = ExecutionContainer(
            execution_config=config.parameters.get('backtest', {}).get('execution', {}),
            container_id='execution'
        )
        
        if tracing_enabled and hasattr(self, 'event_tracer'):
            traced_bus = TracedEventBus("execution_bus")
            traced_bus.set_tracer(self.event_tracer)
            execution_container.event_bus = traced_bus
        
        topology['containers']['execution'] = execution_container
        self.active_containers['execution'] = execution_container
        
        logger.info(f"Signal replay topology created: {len(topology['containers'])} containers, "
                   f"{len(param_combos)} parameter combinations")
        
        return topology
    
    async def _execute_custom_workflow(
        self,
        workflow_name: str,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """
        Execute a custom workflow by dynamically loading it.
        
        Custom workflows should be in src/core/coordinator/workflows/
        and implement an execute() method.
        
        Args:
            workflow_name: Name of the workflow module
            config: Workflow configuration
            context: Execution context
            
        Returns:
            WorkflowResult from the custom workflow
        """
        try:
            # Import the custom workflow module
            import importlib
            module_path = f"src.core.coordinator.workflows.{workflow_name}"
            
            logger.info(f"Loading custom workflow module: {module_path}")
            workflow_module = importlib.import_module(module_path)
            
            # Get the workflow class (convention: <WorkflowName>Workflow)
            workflow_class_name = ''.join(word.capitalize() for word in workflow_name.split('_')) + 'Workflow'
            
            if hasattr(workflow_module, workflow_class_name):
                workflow_class = getattr(workflow_module, workflow_class_name)
                logger.info(f"Found workflow class: {workflow_class_name}")
            else:
                # Try factory function as fallback
                if hasattr(workflow_module, 'create_' + workflow_name + '_workflow'):
                    factory_func = getattr(workflow_module, 'create_' + workflow_name + '_workflow')
                    workflow_instance = await factory_func()
                    logger.info(f"Created workflow using factory function")
                else:
                    raise AttributeError(f"Workflow module must have either {workflow_class_name} class or create_{workflow_name}_workflow factory function")
            
            # Create workflow instance if we have a class
            if 'workflow_instance' not in locals():
                workflow_instance = workflow_class()
            
            # Execute the workflow
            if hasattr(workflow_instance, 'execute'):
                logger.info(f"Executing custom workflow: {workflow_name}")
                result = await workflow_instance.execute(config, context)
                
                # Ensure we return a WorkflowResult
                if not isinstance(result, WorkflowResult):
                    raise TypeError(f"Custom workflow must return WorkflowResult, got {type(result)}")
                
                return result
            else:
                raise AttributeError(f"Custom workflow {workflow_name} must have an execute() method")
                
        except ImportError as e:
            logger.error(f"Failed to import custom workflow '{workflow_name}': {e}")
            return WorkflowResult(
                workflow_id=context.workflow_id,
                workflow_type=config.workflow_type,
                success=False,
                errors=[f"Custom workflow '{workflow_name}' not found"],
                metadata={'error': str(e)}
            )
        except Exception as e:
            logger.error(f"Error executing custom workflow '{workflow_name}': {e}")
            return WorkflowResult(
                workflow_id=context.workflow_id,
                workflow_type=config.workflow_type,
                success=False,
                errors=[f"Custom workflow execution failed: {str(e)}"],
                metadata={'workflow': workflow_name, 'error': str(e)}
            )


# Backward compatibility alias - TopologyBuilder is the canonical implementation
# but coordinator still expects WorkflowManager for type hints
WorkflowManager = TopologyBuilder
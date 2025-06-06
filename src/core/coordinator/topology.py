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
from ..containers.protocols import Container as ContainerProtocol, ContainerRole
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
        self.active_containers: Dict[str, ContainerProtocol] = {}
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
    
    def execute(
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
            return self._execute_custom_workflow(custom_workflow, config, context)
        
        try:
            # 1. Determine workflow mode (backtest, signal gen, or replay)
            mode = self._determine_mode(config)
            logger.info(f"Using workflow mode: {mode.value}")
            
            # 2. Create topology based on mode
            topology = await self._create_topology(mode, config)
            logger.info(f"Created {mode.value} topology with {len(topology['containers'])} containers")
            
            # Note: Adapters are already created and wired in the topology modules
            # The topology includes the 'adapters' key with all configured adapters
            
            # 3. Execute the topology (generic execution for all modes)
            result = await self._execute_topology(topology, config, context, mode)
            
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
    
    def execute_pattern(
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
        return self.execute(workflow_config, context)
    
    # REMOVED: Pattern-based execution mode detection - unified architecture uses modes directly
    # def _determine_execution_mode(self, config: WorkflowConfig, patterns: List[Dict[str, Any]]) -> str:
    #     """Determine execution mode based on configuration and detected patterns."""
    #     ...
    
    # REMOVED: Multiple executors - unified architecture has one execution flow
    # def _get_executor(self, mode: str) -> ExecutionStrategy:
    #     """Get executor for specified mode (with caching)."""
    #     ...
    
    def validate_config(self, config: WorkflowConfig) -> Dict[str, Any]:
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
    
    def get_execution_preview(self, config: WorkflowConfig) -> Dict[str, Any]:
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
    
    def _create_topology(self, mode: WorkflowMode, config: WorkflowConfig) -> Dict[str, Any]:
        """
        Create topology based on workflow mode using modular topology system.
        
        The topology modules return everything needed:
        - containers: Dict of container instances
        - adapters: List of configured adapters
        - parameter_combinations: List of parameter combos
        - root_event_bus: The root event bus
        - event_tracer: Optional event tracer
        """
        # Import topology modules
        from .topologies import (
            create_backtest_topology,
            create_signal_generation_topology,
            create_signal_replay_topology
        )
        
        # Map mode to topology function
        topology_functions = {
            WorkflowMode.BACKTEST: create_backtest_topology,
            WorkflowMode.SIGNAL_GENERATION: create_signal_generation_topology,
            WorkflowMode.SIGNAL_REPLAY: create_signal_replay_topology
        }
        
        if mode not in topology_functions:
            raise ValueError(f"Unknown workflow mode: {mode}")
        
        # Extract config parameters
        tracing_enabled = config.parameters.get('tracing', {}).get('enabled', True)
        
        # Convert WorkflowConfig to dict for topology modules
        topology_config = config.parameters.copy()
        
        # Create topology using appropriate function
        topology_function = topology_functions[mode]
        topology = topology_function(topology_config, tracing_enabled)
        
        # Store references for cleanup
        if topology.get('event_tracer'):
            self.event_tracer = topology['event_tracer']
            
        self.active_containers.update(topology['containers'])
        
        if topology.get('adapters'):
            self.active_adapters.extend(topology['adapters'])
        
        logger.info(f"Created {mode.value} topology: "
                   f"{len(topology['containers'])} containers, "
                   f"{len(topology.get('adapters', []))} adapters")
        
        return topology
    
    def _execute_topology(
        self, 
        topology: Dict[str, Any], 
        config: WorkflowConfig, 
        context: ExecutionContext
    ) -> WorkflowResult:
        """
        Execute a topology generically.
        
        The topology itself defines what needs to happen - we just:
        1. Start the containers
        2. Let them run (event-driven)
        3. Collect results
        4. Stop containers
        """
        logger.info("Executing topology")
        
        # Initialize result
        result = WorkflowResult(
            workflow_id=context.workflow_id,
            workflow_type=config.workflow_type,
            success=True,
            metadata={
                'containers': len(topology['containers']),
                'adapters': len(topology.get('adapters', [])),
                'combinations': len(topology.get('parameter_combinations', []))
            }
        )
        
        try:
            # The topology modules have already created and wired everything
            # We just need to run the simulation
            
            # For now, simulate execution
            # In a real system, this would be event-driven
            import time
            logger.info("Running topology simulation...")
            time.sleep(2)  # Simulate processing
            
            # Collect results from containers
            portfolio_results = {}
            for container_name, container in topology['containers'].items():
                if container_name.startswith('portfolio_'):
                    # Get metrics from portfolio container
                    state = container.get_state_info() if hasattr(container, 'get_state_info') else {}
                    portfolio_results[container_name] = {
                        'container': container_name,
                        'state': state,
                        'metrics': state.get('metrics', {})
                    }
            
            result.final_results = {
                'portfolios': portfolio_results,
                'total_containers': len(topology['containers']),
                'execution_time': 2.0  # Simulated
            }
            
            # Add trace summary if available
            trace_summary = self.get_trace_summary()
            if trace_summary:
                result.metadata['trace_summary'] = trace_summary
                logger.info(f"🔍 Event trace summary: {trace_summary.get('total_events', 0)} events traced")
                
        except Exception as e:
            logger.error(f"Topology execution failed: {e}")
            result.success = False
            result.add_error(str(e))
        
        return result
    
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
        - 20 strategies × 3 risk profiles × 2 execution models = 120 combinations
        - Each gets its own portfolio container with unique combo_id
        """
        combinations = []
        
        # Get parameter grids from config
        # The YAML config is stored under parameters, so extract from backtest section
        backtest_config = config.parameters.get('backtest', {})
        strategy_params = backtest_config.get('strategies', [{}])
        risk_params = backtest_config.get('risk_profiles', [{}])
        classifier_params = backtest_config.get('classifiers', [{}])
        execution_params = backtest_config.get('execution_models', [{}])
        
        # Ensure they're lists
        if not isinstance(strategy_params, list):
            strategy_params = [strategy_params]
        if not isinstance(risk_params, list):
            risk_params = [risk_params]
        if not isinstance(classifier_params, list):
            classifier_params = [classifier_params]
        if not isinstance(execution_params, list):
            execution_params = [execution_params]
        
        # Generate all combinations
        combo_id = 0
        for strat in strategy_params:
            for risk in risk_params:
                for classifier in classifier_params:
                    for execution in execution_params:
                        combinations.append({
                            'combo_id': f'c{combo_id:04d}',
                            'strategy_params': strat,
                            'risk_params': risk,
                            'classifier_params': classifier,
                            'execution_params': execution
                        })
                        combo_id += 1
        
        return combinations
    
    def _create_stateless_components(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Create stateless strategy, classifier, risk, and execution components."""
        components = {
            'strategies': {},
            'classifiers': {},
            'risk_validators': {},
            'execution_models': {}
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
        
        # Create execution model instances (stateless)
        for exec_config in backtest_config.get('execution_models', []):
            exec_type = exec_config.get('type', 'standard')
            exec_models = self._create_stateless_execution_models(exec_type, exec_config)
            components['execution_models'][exec_type] = exec_models
        
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
                    from ...strategy.strategies.momentum import momentum_strategy
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
                from ...strategy.classifiers.classifiers import (
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
        from ...risk.validators import (
            validate_max_position,
            validate_drawdown,
            validate_composite
        )
        
        if risk_type == 'position':
            return validate_max_position
        elif risk_type == 'drawdown':
            return validate_drawdown
        elif risk_type in ['conservative', 'moderate', 'aggressive', 'composite']:
            # All risk profiles use composite validator with different params
            return validate_composite
        else:
            # Fallback placeholder for other validators
            logger.warning(f"Risk validator type '{risk_type}' not yet converted to stateless")
            return {'type': risk_type, 'config': config, 'stateless': True}
    
    def _create_stateless_execution_models(self, exec_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create stateless execution models (slippage, commission) using discovery system."""
        from ..containers.discovery import get_component_registry
        
        registry = get_component_registry()
        
        models = {}
        
        # Create slippage model
        slippage_config = config.get('slippage', {})
        slippage_type = slippage_config.get('type', 'percentage')
        
        # Try to get from registry first
        slippage_info = registry.get_component(slippage_type)
        if not slippage_info:
            # Try with model suffix
            slippage_info = registry.get_component(f'{slippage_type}_slippage')
        
        if slippage_info:
            # Use factory to create instance with config params
            params = slippage_config.get('params', {})
            models['slippage'] = slippage_info.factory(**params)
        else:
            # Fallback - import directly
            logger.warning(f"Slippage model '{slippage_type}' not found in registry, using fallback")
            from ...execution.brokers.slippage import PercentageSlippageModel, ZeroSlippageModel
            
            if slippage_type == 'zero':
                models['slippage'] = ZeroSlippageModel()
            else:
                models['slippage'] = PercentageSlippageModel()
        
        # Create commission model
        commission_config = config.get('commission', {})
        commission_type = commission_config.get('type', 'zero')
        
        # Try to get from registry first
        commission_info = registry.get_component(commission_type)
        if not commission_info:
            # Try with model suffix
            commission_info = registry.get_component(f'{commission_type}_commission')
        
        if commission_info:
            # Use factory to create instance with config params
            params = commission_config.get('params', {})
            models['commission'] = commission_info.factory(**params)
        else:
            # Fallback - import directly
            logger.warning(f"Commission model '{commission_type}' not found in registry, using fallback")
            from ...execution.brokers.commission import ZeroCommissionModel, PerShareCommissionModel
            
            if commission_type == 'per_share':
                models['commission'] = PerShareCommissionModel()
            else:
                models['commission'] = ZeroCommissionModel()
        
        # Add liquidity model if specified
        liquidity_config = config.get('liquidity', {})
        if liquidity_config:
            liquidity_type = liquidity_config.get('type', 'perfect')
            # Similar pattern for liquidity models when they have decorators
            models['liquidity'] = {'type': liquidity_type, 'config': liquidity_config}
        
        return models
    
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
    
    def _call_stateless_strategy(
        self,
        strategy: Any,
        features: Dict[str, Any],
        bar: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Call a stateless strategy to generate a signal."""
        try:
            # Check if it's a callable function (pure function strategy)
            if callable(strategy):
                # Call the pure function strategy directly
                signal = strategy(features, bar, params)
                
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
            # Check if it's a callable function (pure function validator)
            if callable(validator):
                # Call the pure function validator directly
                return validator(order, portfolio_state, risk_params, market_data)
                
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
            # Check if it's a callable function (pure function classifier)
            if callable(classifier):
                # Call the pure function classifier directly
                return classifier(features, params)
                
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
    
    def cleanup(self) -> None:
        """Clean up all active resources."""
        
        logger.info("Cleaning up workflow manager resources...")
        
        # Clean up active containers
        for container_id, container in list(self.active_containers.items()):
            try:
                if hasattr(container, 'dispose'):
                    container.dispose()
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
    
    def _execute_custom_workflow(
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
                    workflow_instance = factory_func()
                    logger.info(f"Created workflow using factory function")
                else:
                    raise AttributeError(f"Workflow module must have either {workflow_class_name} class or create_{workflow_name}_workflow factory function")
            
            # Create workflow instance if we have a class
            if 'workflow_instance' not in locals():
                workflow_instance = workflow_class()
            
            # Execute the workflow
            if hasattr(workflow_instance, 'execute'):
                logger.info(f"Executing custom workflow: {workflow_name}")
                result = workflow_instance.execute(config, context)
                
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
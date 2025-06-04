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

from ..containers.factory import get_global_factory, get_global_registry
from ..communication import AdapterFactory
from ..types.workflow import WorkflowConfig, WorkflowType, ExecutionContext, WorkflowResult
from ..containers.protocols import ComposableContainer

from .workflows.config import PatternDetector, ParameterAnalyzer, ConfigBuilder
from .workflows.execution import get_executor, ExecutionStrategy

logger = logging.getLogger(__name__)


class WorkflowMode(str, Enum):
    """Execution modes for unified architecture."""
    BACKTEST = "backtest"              # Full pipeline: data → signals → orders → fills
    SIGNAL_GENERATION = "signal_generation"  # Stop after signals, save for replay
    SIGNAL_REPLAY = "signal_replay"     # Start from saved signals → orders → fills


class WorkflowManager:
    """
    Modular workflow manager using execution strategies and pattern detection.
    
    This manager orchestrates workflows by:
    1. Detecting appropriate patterns based on configuration
    2. Delegating execution to specialized execution strategies
    3. Coordinating multi-parameter workflows
    4. Managing container lifecycle and communication
    """
    
    def __init__(
        self,
        container_id: Optional[str] = None,
        shared_services: Optional[Dict[str, Any]] = None,
        coordinator: Optional[Any] = None,
        execution_mode: str = 'standard',
        enable_nesting: bool = False,
        enable_pipeline_communication: bool = False
    ):
        """Initialize modular workflow manager."""
        self.container_id = container_id
        self.shared_services = shared_services or {}
        self.coordinator = coordinator
        self.execution_mode = execution_mode
        self.enable_nesting = enable_nesting
        self.enable_pipeline_communication = enable_pipeline_communication
        
        # Core factories
        self.factory = get_global_factory()
        self.registry = get_global_registry()
        self.adapter_factory = AdapterFactory()
        
        # Modular components
        self.pattern_detector = PatternDetector()
        self.parameter_analyzer = ParameterAnalyzer()
        self.config_builder = ConfigBuilder()
        
        # Execution strategy cache
        self._executors: Dict[str, ExecutionStrategy] = {}
        
        # Active resources
        self.active_containers: Dict[str, ComposableContainer] = {}
        self.active_adapters = []
        
        logger.info(f"WorkflowManager initialized (mode: {execution_mode}, nesting: {enable_nesting}, pipeline: {enable_pipeline_communication})")
    
    async def execute(
        self,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute workflow using unified architecture with universal topology."""
        
        logger.info(f"Executing {config.workflow_type.value} workflow")
        
        try:
            # 1. Determine workflow mode (backtest, signal gen, or replay)
            mode = self._determine_mode(config)
            logger.info(f"Using workflow mode: {mode.value}")
            
            # 2. Create universal topology (same for all modes!)
            topology = await self._create_universal_topology(config)
            logger.info(f"Created universal topology with {len(topology['containers'])} containers")
            
            # 3. Wire universal adapters
            adapters = self._create_universal_adapters(topology)
            logger.info(f"Created {len(adapters)} universal adapters")
            
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
                metadata={'execution_mode': 'failed'}
            )
    
    async def execute_pattern(
        self,
        pattern_name: str,
        config: Dict[str, Any],
        correlation_id: str
    ) -> WorkflowResult:
        """
        Execute a specific workflow pattern.
        
        This method is called by the Coordinator for single-phase workflows.
        It delegates to the appropriate execution strategy based on the pattern.
        
        Args:
            pattern_name: Name of the pattern to execute
            config: Configuration for the pattern
            correlation_id: Correlation ID for tracking
            
        Returns:
            WorkflowResult with execution details
        """
        logger.info(f"Executing pattern '{pattern_name}' with correlation_id: {correlation_id}")
        
        try:
            # Get pattern from registry
            pattern = self.registry.get_pattern(pattern_name)
            if not pattern:
                raise ValueError(f"Unknown pattern: {pattern_name}")
            
            # Create pattern info structure
            pattern_info = {
                'name': pattern_name,
                'config': config,
                'pattern': pattern
            }
            
            # Create execution context with correlation ID
            context = ExecutionContext(
                workflow_id=correlation_id.split('_')[1],  # Extract workflow ID from correlation ID
                workflow_type=WorkflowType.BACKTEST,  # Default, will be overridden if needed
                metadata={'correlation_id': correlation_id}
            )
            
            # Get executor and execute pattern
            executor = self._get_executor(self.execution_mode)
            result = await executor.execute_single_pattern(pattern_info, config, context)
            
            # Add correlation ID to result metadata
            result.metadata['correlation_id'] = correlation_id
            
            logger.info(f"Pattern execution completed: {pattern_name}, success={result.success}")
            return result
            
        except Exception as e:
            logger.error(f"Pattern execution failed: {e}")
            return WorkflowResult(
                workflow_id=correlation_id.split('_')[1],
                workflow_type=WorkflowType.BACKTEST,
                success=False,
                errors=[str(e)],
                metadata={
                    'pattern_name': pattern_name,
                    'correlation_id': correlation_id,
                    'error': str(e)
                }
            )
    
    def _determine_execution_mode(self, config: WorkflowConfig, patterns: List[Dict[str, Any]]) -> str:
        """Determine execution mode based on configuration and detected patterns."""
        
        # Check for explicit execution mode configuration
        if hasattr(config, 'execution_mode') and config.execution_mode:
            return config.execution_mode
        
        # Use instance configuration
        if self.enable_nesting:
            return 'nested'
        elif self.enable_pipeline_communication:
            return 'pipeline'
        
        # Auto-detect based on patterns
        pattern_names = [p['name'] for p in patterns]
        
        if any('multi_parameter' in name or 'optimization_grid' in name for name in pattern_names):
            return 'multi_parameter'
        elif len(patterns) > 1:
            return 'multi_pattern'
        else:
            return self.execution_mode
    
    def _get_executor(self, mode: str) -> ExecutionStrategy:
        """Get executor for specified mode (with caching)."""
        if mode not in self._executors:
            self._executors[mode] = get_executor(mode, self)
        return self._executors[mode]
    
    async def validate_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Validate workflow configuration using modular components."""
        
        errors = []
        warnings = []
        
        # Basic configuration validation
        if not config.data_config:
            errors.append("Missing data configuration")
        
        # Workflow-specific validation
        if config.workflow_type == WorkflowType.BACKTEST:
            if not config.backtest_config:
                warnings.append("Missing backtest configuration, using defaults")
        elif config.workflow_type == WorkflowType.OPTIMIZATION:
            if not config.optimization_config:
                errors.append("Missing optimization configuration")
        
        # Pattern validation
        try:
            patterns = self.pattern_detector.determine_patterns(config)
            pattern_info = []
            
            for pattern_data in patterns:
                pattern_name = pattern_data['name']
                pattern = self.registry.get_pattern(pattern_name)
                
                if not pattern:
                    errors.append(f"Unknown container pattern: {pattern_name}")
                elif not self.factory.validate_pattern(pattern):
                    errors.append(f"Invalid container pattern: {pattern_name}")
                else:
                    pattern_info.append({
                        'name': pattern_name,
                        'description': pattern.description,
                        'required_capabilities': list(pattern.required_capabilities)
                    })
            
        except Exception as e:
            errors.append(f"Pattern detection failed: {e}")
            pattern_info = []
        
        # Multi-parameter analysis
        complexity_analysis = None
        if self.parameter_analyzer.requires_multi_parameter(config):
            try:
                complexity_analysis = self.parameter_analyzer.estimate_execution_complexity(config)
            except Exception as e:
                warnings.append(f"Could not analyze multi-parameter complexity: {e}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'patterns': pattern_info,
            'complexity_analysis': complexity_analysis
        }
    
    async def get_execution_preview(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Get a preview of how the workflow would be executed."""
        
        try:
            # Detect patterns
            patterns = self.pattern_detector.determine_patterns(config)
            execution_mode = self._determine_execution_mode(config, patterns)
            
            # Analyze complexity
            complexity_analysis = None
            if self.parameter_analyzer.requires_multi_parameter(config):
                complexity_analysis = self.parameter_analyzer.estimate_execution_complexity(config)
            
            # Get available patterns info
            available_patterns = self.registry.list_available_patterns()
            
            return {
                'detected_patterns': [p['name'] for p in patterns],
                'execution_mode': execution_mode,
                'pattern_details': patterns,
                'complexity_analysis': complexity_analysis,
                'available_patterns': available_patterns,
                'estimated_resources': {
                    'containers': complexity_analysis['estimated_containers'] if complexity_analysis else 5,
                    'duration_minutes': complexity_analysis['estimated_duration_minutes'] if complexity_analysis else 1
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'detected_patterns': [],
                'execution_mode': 'unknown'
            }
    
    def get_supported_patterns(self) -> Dict[str, Any]:
        """Get information about supported workflow patterns."""
        
        available_patterns = self.registry.list_available_patterns()
        pattern_info = {}
        
        for pattern_name in available_patterns:
            pattern = self.registry.get_pattern(pattern_name)
            if pattern:
                pattern_info[pattern_name] = {
                    'description': pattern.description,
                    'required_capabilities': list(pattern.required_capabilities),
                    'default_config': pattern.default_config
                }
        
        return {
            'available_patterns': pattern_info,
            'execution_modes': ['standard', 'nested', 'pipeline', 'multi_pattern'],
            'communication_patterns': ['pipeline', 'broadcast', 'hierarchical', 'selective']
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
    
    async def _create_universal_topology(self, config: WorkflowConfig) -> Dict[str, Any]:
        """
        Create universal topology for all workflow modes.
        
        Always creates the same 4 stateful containers:
        1. Data Container - streaming and caching
        2. FeatureHub Container - indicator calculations
        3. Portfolio Container(s) - one per parameter combination
        4. Execution Container - order management
        
        Plus stateless components for strategies, classifiers, and risk.
        """
        topology = {
            'containers': {},
            'stateless_components': {},
            'parameter_combinations': []
        }
        
        # 1. Create Data Container (always needed)
        data_container = await self.factory.create_container(
            container_type='data',
            config=config.data_config
        )
        topology['containers']['data'] = data_container
        self.active_containers['data'] = data_container
        
        # 2. Create FeatureHub Container (always needed for indicators)
        feature_hub = await self.factory.create_container(
            container_type='feature_hub',
            config=config.parameters.get('features', {})
        )
        topology['containers']['feature_hub'] = feature_hub
        self.active_containers['feature_hub'] = feature_hub
        
        # 3. Expand parameter combinations
        param_combos = self._expand_parameter_combinations(config)
        topology['parameter_combinations'] = param_combos
        
        # 4. Create Portfolio Containers (one per combination)
        for combo in param_combos:
            combo_id = combo['combo_id']
            portfolio_container = await self.factory.create_container(
                container_type='portfolio',
                config={
                    'combo_id': combo_id,
                    'strategy_params': combo['strategy_params'],
                    'risk_params': combo['risk_params'],
                    'classifier_params': combo.get('classifier_params', {})
                }
            )
            topology['containers'][f'portfolio_{combo_id}'] = portfolio_container
            self.active_containers[f'portfolio_{combo_id}'] = portfolio_container
        
        # 5. Create Execution Container (always needed, even for signal gen)
        execution_container = await self.factory.create_container(
            container_type='execution',
            config=config.parameters.get('execution', {})
        )
        topology['containers']['execution'] = execution_container
        self.active_containers['execution'] = execution_container
        
        # 6. Create stateless components (shared across all portfolios)
        topology['stateless_components'] = self._create_stateless_components(config)
        
        logger.info(f"Universal topology created: {len(topology['containers'])} containers, "
                   f"{len(param_combos)} parameter combinations")
        
        return topology
    
    def _expand_parameter_combinations(self, config: WorkflowConfig) -> List[Dict[str, Any]]:
        """
        Expand parameter grid into individual combinations.
        
        Example:
        - 20 strategies × 3 risk profiles = 60 combinations
        - Each gets its own portfolio container with unique combo_id
        """
        combinations = []
        
        # Get parameter grids from config
        strategy_params = config.parameters.get('strategies', [{}])
        risk_params = config.parameters.get('risk_profiles', [{}])
        classifier_params = config.parameters.get('classifiers', [{}])
        
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
        for strat_config in config.parameters.get('strategies', []):
            strat_type = strat_config.get('type', 'momentum')
            strategy = self._create_stateless_strategy(strat_type, strat_config)
            components['strategies'][strat_type] = strategy
        
        # Create classifier instances (stateless)
        for class_config in config.parameters.get('classifiers', []):
            class_type = class_config.get('type', 'simple')
            classifier = self._create_stateless_classifier(class_type, class_config)
            components['classifiers'][class_type] = classifier
        
        # Create risk validator instances (stateless)
        for risk_config in config.parameters.get('risk_profiles', []):
            risk_type = risk_config.get('type', 'conservative')
            validator = self._create_stateless_risk_validator(risk_type, risk_config)
            components['risk_validators'][risk_type] = validator
        
        return components
    
    def _create_stateless_strategy(self, strategy_type: str, config: Dict[str, Any]) -> Any:
        """Create a stateless strategy instance."""
        # Import strategy modules
        if strategy_type == 'momentum':
            from ...strategy.strategies.momentum import create_stateless_momentum
            return create_stateless_momentum()
        elif strategy_type == 'mean_reversion':
            from ...strategy.strategies.mean_reversion_simple import create_stateless_mean_reversion
            return create_stateless_mean_reversion()
        else:
            # Fallback placeholder for other strategies
            logger.warning(f"Strategy type '{strategy_type}' not yet converted to stateless")
            return {'type': strategy_type, 'config': config, 'stateless': True}
    
    def _create_stateless_classifier(self, classifier_type: str, config: Dict[str, Any]) -> Any:
        """Create a stateless classifier instance."""
        # Import classifier modules
        from ...strategy.classifiers.stateless_classifiers import (
            create_stateless_trend_classifier,
            create_stateless_volatility_classifier,
            create_stateless_composite_classifier
        )
        
        if classifier_type == 'trend':
            return create_stateless_trend_classifier()
        elif classifier_type == 'volatility':
            return create_stateless_volatility_classifier()
        elif classifier_type in ['composite', 'simple']:
            return create_stateless_composite_classifier()
        else:
            # Fallback placeholder for other classifiers
            logger.warning(f"Classifier type '{classifier_type}' not yet converted to stateless")
            return {'type': classifier_type, 'config': config, 'stateless': True}
    
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
    
    def _create_universal_adapters(self, topology: Dict[str, Any]) -> List[Any]:
        """
        Create universal adapter configuration.
        
        Always the same 4 adapters regardless of workflow:
        1. BroadcastAdapter: feature_hub → strategies/classifiers
        2. RoutingAdapter: strategies → portfolios (by combo_id)
        3. PipelineAdapter: portfolios → execution
        4. BroadcastAdapter: execution → portfolios
        """
        adapters = []
        
        # 1. Feature broadcast to strategies/classifiers
        feature_broadcast = self.adapter_factory.create_adapter(
            adapter_type='broadcast',
            source_container=topology['containers']['feature_hub'],
            targets=list(topology['stateless_components']['strategies'].values()) +
                   list(topology['stateless_components']['classifiers'].values())
        )
        adapters.append(feature_broadcast)
        
        # 2. Strategy routing to portfolios by combo_id
        portfolio_containers = [c for name, c in topology['containers'].items() 
                              if name.startswith('portfolio_')]
        
        strategy_router = self.adapter_factory.create_adapter(
            adapter_type='routing',
            sources=list(topology['stateless_components']['strategies'].values()),
            targets=portfolio_containers,
            routing_key='combo_id'
        )
        adapters.append(strategy_router)
        
        # 3. Pipeline from portfolios to execution
        portfolio_pipeline = self.adapter_factory.create_adapter(
            adapter_type='pipeline',
            containers=portfolio_containers + [topology['containers']['execution']]
        )
        adapters.append(portfolio_pipeline)
        
        # 4. Execution broadcast back to portfolios
        execution_broadcast = self.adapter_factory.create_adapter(
            adapter_type='broadcast',
            source_container=topology['containers']['execution'],
            targets=portfolio_containers
        )
        adapters.append(execution_broadcast)
        
        self.active_adapters.extend(adapters)
        return adapters
    
    async def _execute_full_pipeline(
        self, 
        topology: Dict[str, Any], 
        config: WorkflowConfig, 
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute full backtest pipeline: data → features → signals → orders → fills."""
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
            # 1. Start all containers
            logger.info("Starting all containers...")
            for container_name, container in topology['containers'].items():
                await container.start()
                logger.debug(f"Started container: {container_name}")
            
            # 2. Initialize data streaming
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
            
            # 3. Process each bar through the pipeline
            bar_count = 0
            portfolio_results = {}
            
            # Start data streaming
            async for market_bar in data_container.stream_data():
                bar_count += 1
                
                # Update features with new bar
                await feature_hub.update_bar(market_bar)
                
                # Get current features
                features = await feature_hub.get_features(market_bar['symbol'])
                
                # Process through each parameter combination
                for combo in topology['parameter_combinations']:
                    combo_id = combo['combo_id']
                    portfolio = topology['containers'][f'portfolio_{combo_id}']
                    
                    # Get stateless components for this combination
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
                            # Process signal through portfolio
                            await portfolio.process_signal(signal)
                    
                    # Check for risk validation
                    risk_type = combo['risk_params'].get('type', 'conservative')
                    risk_validator = topology['stateless_components']['risk_validators'].get(risk_type)
                    
                    if risk_validator and portfolio.has_pending_orders():
                        # Validate orders with stateless risk validator
                        orders = await portfolio.get_pending_orders()
                        portfolio_state = await portfolio.get_state()
                        
                        for order in orders:
                            validation = self._call_stateless_risk_validator(
                                risk_validator,
                                order,
                                portfolio_state,
                                combo['risk_params'],
                                market_bar
                            )
                            
                            if validation['approved']:
                                # Send to execution
                                execution = topology['containers']['execution']
                                await execution.submit_order(order)
                            else:
                                await portfolio.reject_order(order, validation['reason'])
                
                # Update portfolio valuations
                for combo in topology['parameter_combinations']:
                    combo_id = combo['combo_id']
                    portfolio = topology['containers'][f'portfolio_{combo_id}']
                    await portfolio.update_market_prices(market_bar)
            
            # 4. Collect final results
            logger.info(f"Processed {bar_count} bars")
            
            for combo in topology['parameter_combinations']:
                combo_id = combo['combo_id']
                portfolio = topology['containers'][f'portfolio_{combo_id}']
                
                # Get final portfolio metrics
                metrics = await portfolio.get_metrics()
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
            best_combo = max(portfolio_results.values(), 
                           key=lambda x: x.get('sharpe_ratio', 0))
            
            result.final_results = {
                'best_combination': best_combo,
                'all_results': portfolio_results,
                'bar_count': bar_count
            }
            
        except Exception as e:
            logger.error(f"Full pipeline execution failed: {e}")
            result.success = False
            result.add_error(str(e))
        
        finally:
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
        
        # Clear executor cache
        self._executors.clear()
        
        logger.info("Workflow manager cleanup complete")


# No backward compatibility aliases - use WorkflowManager directly
# Following STYLE.md: ONE canonical implementation per concept
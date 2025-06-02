"""
Composable Workflow Manager

This integrates composable containers with the existing workflow manager protocol,
allowing seamless use of container patterns within the traditional coordinator architecture.
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
import logging
from datetime import datetime

from .simple_types import WorkflowConfig, ExecutionContext, WorkflowType, WorkflowPhase
from .protocols import WorkflowManager
from .simple_types import WorkflowResult

from ..containers.composition_engine import get_global_composition_engine
from ..containers.composable import (
    ComposableContainerProtocol, ContainerRole, ContainerPattern
)
from ..events.routing import EventRouter, EventScope
from ..events.routing.debugging import EventFlowVisualizer, EventRoutingMonitor


logger = logging.getLogger(__name__)


class ComposableWorkflowManager:
    """
    Workflow manager that uses composable container patterns.
    
    This bridges the gap between the traditional workflow manager protocol
    and the new composable container system.
    """
    
    def __init__(self, container_id: str, shared_services: Optional[Dict[str, Any]] = None):
        """Initialize composable workflow manager."""
        self.container_id = container_id
        self.shared_services = shared_services or {}
        self.composition_engine = get_global_composition_engine()
        self.active_containers: Dict[str, ComposableContainerProtocol] = {}
        self.event_router: Optional[EventRouter] = None
        self.event_monitor: Optional[EventRoutingMonitor] = None
        
    async def execute(
        self,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute workflow using composable containers."""
        
        logger.info(f"Executing {config.workflow_type.value} workflow with composable containers")
        
        try:
            # Determine which pattern(s) to use
            patterns = self._determine_patterns(config)
            
            if len(patterns) == 1:
                # Single pattern execution
                return await self._execute_single_pattern(patterns[0], config, context)
            else:
                # Multi-pattern execution
                return await self._execute_multi_pattern(patterns, config, context)
                
        except Exception as e:
            logger.error(f"Composable workflow execution failed: {e}")
            return WorkflowResult(
                workflow_id=context.workflow_id,
                workflow_type=config.workflow_type,
                success=False,
                errors=[str(e)]
            )
    
    def _determine_patterns(self, config: WorkflowConfig) -> List[Dict[str, Any]]:
        """Determine which container patterns to use based on workflow config."""
        
        patterns = []
        
        if config.workflow_type == WorkflowType.BACKTEST:
            # Check for explicit pattern specification
            explicit_pattern = getattr(config, 'pattern', None)
            if explicit_pattern:
                # Use the explicitly specified pattern
                patterns.append({
                    'name': explicit_pattern,
                    'config': self._build_pattern_config(config, explicit_pattern)
                })
            elif self._is_multi_strategy_backtest(config):
                # Multi-strategy backtest - use simple_backtest with ensemble config
                patterns.append({
                    'name': 'simple_backtest',
                    'config': self._build_simple_backtest_config(config)
                })
            elif self._is_simple_backtest(config):
                # Simple single strategy backtest
                patterns.append({
                    'name': 'simple_backtest',
                    'config': self._build_simple_backtest_config(config)
                })
            else:
                # Full backtest with classifiers/advanced features
                patterns.append({
                    'name': 'full_backtest',
                    'config': self._build_full_backtest_config(config)
                })
        
        elif config.workflow_type == WorkflowType.OPTIMIZATION:
            # Optimization workflow - could be multiple patterns
            optimization_config = config.optimization_config or {}
            
            if optimization_config.get('use_signal_generation', False):
                # Phase 1: Signal generation
                patterns.append({
                    'name': 'signal_generation',
                    'config': self._build_signal_generation_config(config)
                })
                
                # Phase 2: Signal replay for optimization
                patterns.append({
                    'name': 'signal_replay',
                    'config': self._build_signal_replay_config(config)
                })
            else:
                # Direct optimization
                patterns.append({
                    'name': 'full_backtest',
                    'config': self._build_optimization_config(config)
                })
        
        elif config.workflow_type == WorkflowType.ANALYSIS:
            # Analysis workflow
            analysis_mode = config.analysis_config.get('mode', 'signal_generation')
            
            if analysis_mode == 'signal_generation':
                patterns.append({
                    'name': 'signal_generation',
                    'config': self._build_signal_generation_config(config)
                })
            else:
                patterns.append({
                    'name': 'full_backtest',
                    'config': self._build_full_backtest_config(config)
                })
        
        else:
            # Default to full backtest
            patterns.append({
                'name': 'full_backtest',
                'config': self._build_full_backtest_config(config)
            })
        
        return patterns
    
    def _is_simple_backtest(self, config: WorkflowConfig) -> bool:
        """Determine if this is a simple backtest (minimal containers)."""
        
        # Simple if no classifiers or risk profiles specified
        optimization_config = config.optimization_config or {}
        
        has_classifiers = bool(optimization_config.get('classifiers'))
        has_risk_profiles = bool(optimization_config.get('risk_profiles'))
        has_portfolios = bool(optimization_config.get('portfolios'))
        
        # Check for multiple strategies
        has_multiple_strategies = self._has_multiple_strategies(config)
        
        # Simple if only basic strategy configuration and single strategy
        return not (has_classifiers or has_risk_profiles or has_portfolios or has_multiple_strategies)
    
    def _is_multi_strategy_backtest(self, config: WorkflowConfig) -> bool:
        """Determine if this is a multi-strategy backtest."""
        return self._has_multiple_strategies(config)
    
    def _has_multiple_strategies(self, config: WorkflowConfig) -> bool:
        """Check if config specifies multiple strategies."""
        strategies = []
        
        # Check top-level strategies attribute
        if hasattr(config, 'strategies') and config.strategies:
            strategies.extend(config.strategies)
        
        # Check backtest.strategies section
        backtest_config = getattr(config, 'backtest_config', {})
        if backtest_config and 'strategies' in backtest_config:
            backtest_strategies = backtest_config['strategies']
            if isinstance(backtest_strategies, list):
                strategies.extend(backtest_strategies)
        
        return len(strategies) > 1
    
    def _build_simple_backtest_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build config for simple backtest pattern."""
        
        container_config = {}
        
        # Data configuration
        if config.data_config:
            container_config['data'] = self._extract_data_config(config)
        
        # Strategy configuration - check multiple locations
        strategies = []
        
        # Check top-level strategies attribute
        if hasattr(config, 'strategies') and config.strategies:
            strategies.extend(config.strategies)
        
        # Check backtest.strategies 
        if config.backtest_config and config.backtest_config.get('strategies'):
            strategies.extend(config.backtest_config.get('strategies', []))
            
        # Check optimization_config.strategies (fallback)
        optimization_config = config.optimization_config or {}
        if optimization_config.get('strategies'):
            strategies.extend(optimization_config.get('strategies', []))
        
        logger.info(f"Found {len(strategies)} strategies from config")
        for i, strategy in enumerate(strategies):
            logger.info(f"  Strategy {i+1}: {strategy}")
        
        # CRITICAL: Automatic Indicator Inference
        # This is the missing piece that breaks the architecture!
        required_indicators = self._infer_indicators_from_strategies(strategies)
        logger.info(f"Inferred {len(required_indicators)} indicators from strategies: {required_indicators}")
        
        # Add indicator configuration with inferred indicators
        container_config['indicator'] = {
            'required_indicators': list(required_indicators),
            'cache_size': 1000
        }
        
        # Pass ALL strategies to composition engine for indicator inference
        if strategies:
            container_config['strategies'] = strategies
            
            # Configure strategy container - automatically detects single vs multi-strategy
            if len(strategies) > 1:
                # Multiple strategies - pass all strategies for automatic sub-container creation
                container_config['strategy'] = {
                    'strategies': strategies,  # StrategyContainer will auto-create sub-containers
                    'aggregation': {
                        'method': config.parameters.get('signal_aggregation', {}).get('method', 'weighted_voting'),
                        'min_confidence': config.parameters.get('signal_aggregation', {}).get('min_confidence', 0.6)
                    }
                }
            else:
                # Single strategy - use traditional config
                strategy_config = strategies[0]
                container_config['strategy'] = {
                    'type': strategy_config.get('type', 'momentum'),
                    'parameters': strategy_config.get('parameters', {})
                }
        
        # Risk configuration
        risk_config = {}
        if hasattr(config, 'parameters') and config.parameters.get('risk'):
            risk_config.update(config.parameters['risk'])
        if config.backtest_config and config.backtest_config.get('risk'):
            risk_config.update(config.backtest_config['risk'])
        if config.optimization_config and config.optimization_config.get('risk'):
            risk_config.update(config.optimization_config['risk'])
        
        # Add initial_capital to risk config
        initial_capital = None
        if config.backtest_config and config.backtest_config.get('initial_capital'):
            initial_capital = config.backtest_config['initial_capital']
        elif hasattr(config, 'parameters') and config.parameters.get('portfolio', {}).get('initial_capital'):
            initial_capital = config.parameters['portfolio']['initial_capital']
        elif hasattr(config, 'parameters') and config.parameters.get('initial_capital'):
            initial_capital = config.parameters['initial_capital']
        
        if initial_capital:
            risk_config['initial_capital'] = initial_capital
        
        if risk_config:
            container_config['risk'] = risk_config
            logger.info(f"Added risk config to simple_backtest: {risk_config}")
        
        # Portfolio configuration
        portfolio_config = {}
        if hasattr(config, 'parameters') and config.parameters.get('portfolio'):
            portfolio_config.update(config.parameters['portfolio'])
        if config.backtest_config and config.backtest_config.get('portfolio'):
            portfolio_config.update(config.backtest_config['portfolio'])
        
        if initial_capital and 'initial_capital' not in portfolio_config:
            portfolio_config['initial_capital'] = initial_capital
            
        if portfolio_config:
            container_config['portfolio'] = portfolio_config
            logger.info(f"Added portfolio config to simple_backtest: {portfolio_config}")
        
        # Execution configuration
        if config.backtest_config:
            container_config['execution'] = self._extract_execution_config(config)
        
        return container_config
    
    def _build_full_backtest_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build config for full backtest pattern."""
        
        container_config = {}
        
        # Data configuration
        if config.data_config:
            container_config['data'] = self._extract_data_config(config)
        
        # Indicator configuration
        container_config['indicator'] = {
            'max_indicators': config.parameters.get('max_indicators', 100)
        }
        
        # Classifier configuration
        optimization_config = config.optimization_config or {}
        classifiers = optimization_config.get('classifiers', [])
        if classifiers:
            classifier_config = classifiers[0]
            container_config['classifier'] = {
                'type': classifier_config.get('type', 'hmm'),
                'parameters': classifier_config.get('parameters', {})
            }
        
        # Risk configuration
        risk_profiles = optimization_config.get('risk_profiles', [])
        if risk_profiles:
            risk_config = risk_profiles[0]
            container_config['risk'] = {
                'profile': risk_config.get('name', 'conservative'),
                'max_position_size': risk_config.get('max_position_size', 0.02),
                'max_total_exposure': risk_config.get('max_total_exposure', 0.10)
            }
        
        # Portfolio configuration
        portfolios = optimization_config.get('portfolios', [])
        if portfolios:
            portfolio_config = portfolios[0]
            container_config['portfolio'] = {
                'allocation': portfolio_config.get('allocation', 100000),
                'rebalance_frequency': portfolio_config.get('rebalance_frequency', 'daily')
            }
        
        # Strategy configuration
        strategies = optimization_config.get('strategies', [])
        if strategies:
            strategy_config = strategies[0]
            container_config['strategy'] = {
                'type': strategy_config.get('type', 'momentum'),
                'parameters': strategy_config.get('parameters', {})
            }
        
        # Execution configuration
        if config.backtest_config:
            container_config['execution'] = self._extract_execution_config(config)
        
        return container_config
    
    def _build_signal_generation_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build config for signal generation pattern."""
        
        container_config = {}
        
        # Data configuration
        if config.data_config:
            container_config['data'] = self._extract_data_config(config)
        
        # Indicator configuration
        container_config['indicator'] = {
            'max_indicators': config.parameters.get('max_indicators', 100)
        }
        
        # Classifier configuration
        optimization_config = config.optimization_config or {}
        classifiers = optimization_config.get('classifiers', [])
        if classifiers:
            classifier_config = classifiers[0]
            container_config['classifier'] = {
                'type': classifier_config.get('type', 'hmm'),
                'parameters': classifier_config.get('parameters', {})
            }
        
        # Strategy configuration
        strategies = optimization_config.get('strategies', [])
        if strategies:
            strategy_config = strategies[0]
            container_config['strategy'] = {
                'type': strategy_config.get('type', 'momentum'),
                'parameters': strategy_config.get('parameters', {})
            }
        
        # Analysis configuration
        analysis_config = config.analysis_config or {}
        container_config['analysis'] = {
            'mode': 'signal_generation',
            'output_path': analysis_config.get('output_path', './signals/'),
            'capture_mae_mfe': analysis_config.get('capture_mae_mfe', True),
            'signal_quality_metrics': analysis_config.get('signal_quality_metrics', True)
        }
        
        return container_config
    
    def _build_signal_replay_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build config for signal replay pattern."""
        
        container_config = {}
        
        # Signal log configuration
        analysis_config = config.analysis_config or {}
        container_config['signal_log'] = {
            'source': analysis_config.get('signal_log_path', './signals/'),
            'format': 'jsonl'
        }
        
        # Ensemble configuration
        optimization_config = config.optimization_config or {}
        container_config['ensemble'] = {
            'optimization_method': optimization_config.get('ensemble_method', 'grid_search'),
            'weight_constraints': optimization_config.get('weight_constraints', {}),
            'objective_function': optimization_config.get('objective_function', 'sharpe_ratio')
        }
        
        # Risk configuration
        risk_profiles = optimization_config.get('risk_profiles', [])
        if risk_profiles:
            risk_config = risk_profiles[0]
            container_config['risk'] = {
                'profile': risk_config.get('name', 'conservative'),
                'max_position_size': risk_config.get('max_position_size', 0.02),
                'max_total_exposure': risk_config.get('max_total_exposure', 0.10)
            }
        
        # Portfolio configuration
        portfolios = optimization_config.get('portfolios', [])
        if portfolios:
            portfolio_config = portfolios[0]
            container_config['portfolio'] = {
                'allocation': portfolio_config.get('allocation', 100000)
            }
        
        # Execution configuration
        if config.backtest_config:
            container_config['execution'] = self._extract_execution_config(config)
        
        return container_config
    
    def _build_optimization_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build config for optimization workflow."""
        
        # For now, use full backtest config
        # In future, this could create multiple container instances for parallel optimization
        return self._build_full_backtest_config(config)
    
    def _extract_data_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Extract data configuration for containers."""
        data_config = config.data_config
        
        return {
            'source': data_config.get('source', 'historical'),
            'symbols': data_config.get('symbols', ['SPY']),
            'start_date': data_config.get('start_date'),
            'end_date': data_config.get('end_date'),
            'frequency': data_config.get('frequency', '1d'),
            'file_path': data_config.get('file_path'),
            'data_path': data_config.get('data_path'),
            'data_dir': data_config.get('data_dir', 'data'),
            'max_bars': data_config.get('max_bars')
        }
    
    def _extract_execution_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Extract execution configuration for containers."""
        backtest_config = config.backtest_config
        
        return {
            'mode': 'backtest',
            'initial_capital': backtest_config.get('initial_capital', 100000),
            'commission': backtest_config.get('commission', 0.001),
            'slippage': backtest_config.get('slippage', 0.0005),
            'enable_shorting': backtest_config.get('enable_shorting', True)
        }
    
    async def _execute_single_pattern(
        self,
        pattern_info: Dict[str, Any],
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute a single container pattern."""
        
        pattern_name = pattern_info['name']
        pattern_config = pattern_info['config']
        
        logger.info(f"Executing single pattern: {pattern_name}")
        
        try:
            # Create event router for this workflow if not exists
            if not self.event_router:
                self.event_router = EventRouter(
                    enable_debugging=context.debug_mode if hasattr(context, 'debug_mode') else False
                )
                self.event_monitor = EventRoutingMonitor(self.event_router)
            
            # Compose container pattern
            root_container = self.composition_engine.compose_pattern(
                pattern_name=pattern_name,
                config_overrides=pattern_config
            )
            
            # Register all containers with event router
            await self._register_containers_with_router(root_container)
            
            # Store container
            container_key = f"{context.workflow_id}_{pattern_name}"
            self.active_containers[container_key] = root_container
            
            # Validate event topology
            validation = self.event_router.validate_topology()
            if not validation.is_valid:
                logger.warning(f"Event topology validation issues: {validation.errors}")
            
            # Generate debug visualization if enabled
            if context.debug_mode if hasattr(context, 'debug_mode') else False:
                await self._generate_event_visualization(context.workflow_id)
            
            # Execute pattern
            result = await self._execute_container(root_container, config, context)
            result.metadata['container_pattern'] = pattern_name
            
            # Add event routing metrics to result
            if self.event_monitor:
                result.metadata['event_routing_metrics'] = self.event_router.get_metrics()
            
            return result
            
        except Exception as e:
            logger.error(f"Single pattern execution failed: {e}")
            raise e
        finally:
            # Clean up
            if container_key in self.active_containers:
                await self.active_containers[container_key].dispose()
                del self.active_containers[container_key]
    
    async def _execute_multi_pattern(
        self,
        patterns: List[Dict[str, Any]],
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute multiple container patterns in sequence."""
        
        logger.info(f"Executing multi-pattern workflow with {len(patterns)} patterns")
        
        pattern_results = {}
        overall_success = True
        
        for i, pattern_info in enumerate(patterns):
            pattern_name = pattern_info['name']
            pattern_config = pattern_info['config']
            
            logger.info(f"Executing pattern {i+1}/{len(patterns)}: {pattern_name}")
            
            try:
                # Compose container pattern
                root_container = self.composition_engine.compose_pattern(
                    pattern_name=pattern_name,
                    config_overrides=pattern_config
                )
                
                # Store container
                container_key = f"{context.workflow_id}_{pattern_name}_{i}"
                self.active_containers[container_key] = root_container
                
                # Execute pattern
                pattern_result = await self._execute_container(root_container, config, context)
                pattern_results[f"{pattern_name}_{i}"] = pattern_result
                
                if not pattern_result.success:
                    overall_success = False
                
                # For multi-pattern workflows, pass results between phases
                if i < len(patterns) - 1:
                    # Update next pattern config with results from this pattern
                    next_pattern = patterns[i + 1]
                    next_pattern['config'] = self._update_config_with_results(
                        next_pattern['config'],
                        pattern_result
                    )
                
            except Exception as e:
                logger.error(f"Pattern {pattern_name} failed: {e}")
                pattern_results[f"{pattern_name}_{i}"] = WorkflowResult(
                    workflow_id=context.workflow_id,
                    workflow_type=config.workflow_type,
                    success=False,
                    errors=[str(e)]
                )
                overall_success = False
            finally:
                # Clean up pattern container
                if container_key in self.active_containers:
                    await self.active_containers[container_key].dispose()
                    del self.active_containers[container_key]
        
        # Aggregate results
        result = WorkflowResult(
            workflow_id=context.workflow_id,
            workflow_type=config.workflow_type,
            success=overall_success,
            final_results={
                'pattern_results': {
                    name: result.final_results for name, result in pattern_results.items()
                },
                'execution_order': [p['name'] for p in patterns]
            },
            metadata={
                'execution_mode': 'multi_pattern',
                'patterns_executed': len(patterns),
                'successful_patterns': sum(1 for r in pattern_results.values() if r.success)
            }
        )
        
        return result
    
    def _update_config_with_results(
        self,
        next_config: Dict[str, Any],
        previous_result: WorkflowResult
    ) -> Dict[str, Any]:
        """Update next pattern config with results from previous pattern."""
        
        # For signal generation → signal replay workflows
        if 'signal_log' in next_config and previous_result.success:
            # Update signal log path with output from previous phase
            signal_output = previous_result.final_results.get('signal_output_path')
            if signal_output:
                next_config['signal_log']['source'] = signal_output
        
        return next_config
    
    async def _execute_container(
        self,
        root_container: ComposableContainerProtocol,
        config: WorkflowConfig,
        context: ExecutionContext
    ) -> WorkflowResult:
        """Execute a single container and return results."""
        
        try:
            # Initialize container
            await root_container.initialize()
            
            # Start execution
            await root_container.start()
            
            # Wait for completion based on workflow type
            await self._wait_for_completion(root_container, config)
            
            # Position closure is now handled by END_OF_BACKTEST event in containers
            # No need for explicit force close here
            
            # Collect results
            container_status = root_container.get_status()
            
            # Extract comprehensive backtest data
            backtest_data = self._extract_backtest_data(root_container)
            
            # Create result
            container_state = root_container.state.value
            logger.info(f"Container final state: {container_state}")
            
            result = WorkflowResult(
                workflow_id=context.workflow_id,
                workflow_type=config.workflow_type,
                success=root_container.state.value in ['stopped', 'initialized', 'running'],
                final_results={
                    'container_status': container_status,
                    'container_structure': self._get_container_structure(root_container),
                    'metrics': container_status.get('metrics', {}),
                    'final_state': container_state,
                    'backtest_data': backtest_data  # Add actual backtest results
                }
            )
            
            # Stop container
            await root_container.stop()
            
            return result
            
        except Exception as e:
            logger.error(f"Container execution failed: {e}")
            raise e
    
    async def _wait_for_completion(self, root_container: ComposableContainerProtocol, config: WorkflowConfig) -> None:
        """Wait for container execution to complete intelligently."""
        
        # For backtest workflows, detect when data streaming completes
        if config.workflow_type == WorkflowType.BACKTEST:
            await self._wait_for_data_streaming_completion(root_container)
        else:
            # For other workflows, use a shorter default wait
            await asyncio.sleep(1.0)
    
    async def _wait_for_data_streaming_completion(self, root_container: ComposableContainerProtocol) -> None:
        """Wait for data streaming to complete by monitoring DataContainer state."""
        
        # Much simpler approach - just wait a bit for data to finish streaming, then check
        # if container is still actively publishing events
        await asyncio.sleep(2.0)  # Give time for initial data processing
        
        # Find the DataContainer (root should be DataContainer)
        data_container = root_container
        if data_container.metadata.role.value != 'data':
            logger.warning("Root container is not DataContainer, using default completion detection")
            await asyncio.sleep(1.0)
            return
        
        # Monitor for completion with shorter intervals
        max_additional_wait = 5.0  # Maximum additional wait time
        check_interval = 0.2  # Check every 200ms
        idle_threshold = 1.0  # Consider complete if idle for 1 second
        
        start_time = asyncio.get_event_loop().time()
        last_events_published = 0
        stable_count = 0
        
        while True:
            current_time = asyncio.get_event_loop().time()
            
            # Check timeout
            if current_time - start_time > max_additional_wait:
                logger.info(f"Container completion detection finished after max wait time")
                break
            
            # Get current event count
            status = data_container.get_status()
            metrics = status.get('metrics', {})
            events_published = metrics.get('events_published', 0)
            
            # Check if event publishing has stopped
            if events_published == last_events_published:
                stable_count += 1
                # If stable for multiple checks, consider complete
                if stable_count >= (idle_threshold / check_interval):
                    logger.info(f"Data streaming complete (no new events for {stable_count * check_interval:.1f}s)")
                    break
            else:
                # Reset stability counter
                stable_count = 0
                last_events_published = events_published
            
            await asyncio.sleep(check_interval)
    
    def _infer_indicators_from_strategies(self, strategies: List[Dict[str, Any]]) -> Set[str]:
        """Infer required indicators from strategy configurations.
        
        This is the core indicator inference logic that automatically determines
        what indicators are needed based on strategy configurations.
        """
        required_indicators = set()
        
        for strategy_config in strategies:
            strategy_class = strategy_config.get('class', strategy_config.get('type'))
            strategy_params = strategy_config.get('parameters', {})
            
            if strategy_class in ['MomentumStrategy', 'momentum']:
                # MomentumStrategy needs SMA for momentum and RSI for signals
                lookback_period = strategy_params.get('lookback_period', 20)
                rsi_period = strategy_params.get('rsi_period', 14)
                
                required_indicators.add(f'SMA_{lookback_period}')
                required_indicators.add('RSI')
                
                logger.info(f"MomentumStrategy requires: SMA_{lookback_period}, RSI")
                
            elif strategy_class in ['MeanReversionStrategy', 'mean_reversion']:
                # MeanReversionStrategy typically needs Bollinger Bands, RSI
                period = strategy_params.get('period', 20)
                required_indicators.add(f'BB_{period}')
                required_indicators.add('RSI')
                
                logger.info(f"MeanReversionStrategy requires: BB_{period}, RSI")
                
            elif strategy_class in ['moving_average_crossover', 'momentum_crossover']:
                # For crossover strategies, infer from parameter names
                for param_name, param_value in strategy_params.items():
                    if 'fast_period' in param_name:
                        required_indicators.add(f'SMA_{param_value}')
                    elif 'slow_period' in param_name:
                        required_indicators.add(f'SMA_{param_value}')
                    elif 'rsi_period' in param_name:
                        required_indicators.add('RSI')
                        
            else:
                # Default indicators for unknown strategies
                logger.warning(f"Unknown strategy class {strategy_class}, using default indicators")
                required_indicators.update(['SMA_20', 'RSI'])
        
        # If no strategies found, add default indicators to prevent empty indicator hub
        if not required_indicators:
            logger.warning("No strategies found, using default indicators")
            required_indicators.update(['SMA_20', 'RSI'])
            
        return required_indicators
    
    def _get_container_structure(self, container: ComposableContainerProtocol) -> Dict[str, Any]:
        """Get hierarchical structure of container."""
        structure = {
            'id': container.metadata.container_id,
            'role': container.metadata.role.value,
            'name': container.metadata.name,
            'state': container.state.value,
            'children': []
        }
        
        for child in container.child_containers:
            structure['children'].append(self._get_container_structure(child))
        
        return structure
    
    def _extract_backtest_data(self, root_container) -> Dict[str, Any]:
        """Extract comprehensive backtest data from containers"""
        backtest_data = {
            'trades': [],
            'portfolio_value': 0,
            'positions': {},
            'performance_metrics': {},
            'signals_generated': 0,
            'bars_processed': 0
        }
        
        try:
            # Find the risk container (which has portfolio state)
            risk_container = self._find_container_by_role(root_container, 'risk')
            if risk_container:
                logger.info(f"Found risk container: {type(risk_container)}")
                
                # Extract portfolio state from risk container
                portfolio_extracted = False
                
                # Try multiple paths to get portfolio state from risk container
                # First try: risk_manager attribute (RiskPortfolioContainer)
                if hasattr(risk_container, 'risk_manager') and risk_container.risk_manager:
                    risk_manager = risk_container.risk_manager
                    logger.info(f"Found risk_manager: {type(risk_manager)}")
                    
                    if hasattr(risk_manager, 'get_portfolio_state'):
                        state = risk_manager.get_portfolio_state()
                        logger.info(f"Portfolio state object: {type(state)}")
                        
                        # Use proper method calls instead of attributes
                        cash = 0
                        total_portfolio_value = 0
                        positions = {}
                        position_value = 0
                        
                        # Get cash balance using method call
                        if hasattr(state, 'get_cash_balance'):
                            cash = float(state.get_cash_balance())
                            logger.info(f"Cash balance: ${cash}")
                        
                        # Get total portfolio value using method call
                        if hasattr(state, 'get_total_value'):
                            total_portfolio_value = float(state.get_total_value())
                            logger.info(f"Total portfolio value: ${total_portfolio_value}")
                        
                        # Get positions using method call
                        if hasattr(state, 'get_all_positions'):
                            all_positions = state.get_all_positions()
                            logger.info(f"Found {len(all_positions)} positions")
                            
                            for symbol, position in all_positions.items():
                                # Position objects use 'quantity', not 'shares'
                                if hasattr(position, 'quantity') and position.quantity != 0:
                                    shares = float(position.quantity)
                                    avg_price = float(position.average_price) if hasattr(position, 'average_price') else 0
                                    current_price = float(position.current_price) if hasattr(position, 'current_price') else avg_price
                                    market_value = shares * current_price
                                    position_value += market_value
                                    
                                    positions[symbol] = {
                                        'shares': shares,
                                        'avg_price': avg_price,
                                        'current_price': current_price,
                                        'market_value': market_value
                                    }
                                    logger.info(f"Position {symbol}: {shares} shares @ ${current_price} = ${market_value}")
                        
                        backtest_data['portfolio_value'] = total_portfolio_value
                        backtest_data['cash'] = cash
                        backtest_data['position_value'] = position_value
                        backtest_data['positions'] = positions
                        portfolio_extracted = True
                        
                        logger.info(f"Portfolio extracted from risk_manager - Total: ${total_portfolio_value} (cash: ${cash}, positions: ${position_value})")
                
                # Second try: risk_portfolio attribute (legacy)
                if not portfolio_extracted and hasattr(risk_container, 'risk_portfolio') and risk_container.risk_portfolio:
                    risk_portfolio = risk_container.risk_portfolio
                    logger.info(f"Found risk_portfolio: {type(risk_portfolio)}")
                    
                    if hasattr(risk_portfolio, 'get_portfolio_state'):
                        state = risk_portfolio.get_portfolio_state()
                        logger.info(f"Portfolio state - total_value: {getattr(state, 'total_value', 'N/A')}, cash: {getattr(state, 'cash', 'N/A')}")
                        
                        # Get current market prices from last trades for position valuation
                        current_prices = {}
                        
                        # Calculate portfolio value with proper position valuation
                        cash = float(state.cash) if hasattr(state, 'cash') else 0
                        position_value = 0
                        positions = {}
                        
                        if hasattr(state, 'positions') and state.positions:
                            # Need current prices to value positions - get from state's price cache
                            if hasattr(state, 'get_current_prices'):
                                current_prices = state.get_current_prices()
                                logger.info(f"Current prices from portfolio state: {current_prices}")
                            
                            for symbol, position in state.positions.items():
                                if hasattr(position, 'shares') and position.shares != 0:
                                    shares = float(position.shares)
                                    avg_price = float(position.avg_price) if hasattr(position, 'avg_price') else 0
                                    current_price = current_prices.get(symbol, avg_price)  # Use avg_price as fallback
                                    market_value = shares * current_price
                                    position_value += market_value
                                    
                                    positions[symbol] = {
                                        'shares': shares,
                                        'avg_price': avg_price,
                                        'current_price': current_price,
                                        'market_value': market_value
                                    }
                                    logger.info(f"Position {symbol}: {shares} shares @ ${current_price} = ${market_value}")
                        
                        total_portfolio_value = cash + position_value
                        backtest_data['portfolio_value'] = total_portfolio_value
                        backtest_data['cash'] = cash
                        backtest_data['position_value'] = position_value
                        backtest_data['positions'] = positions
                        portfolio_extracted = True
                        
                        logger.info(f"Portfolio extracted from risk container - Total: ${total_portfolio_value} (cash: ${cash}, positions: ${position_value})")
                
                # Fallback: try portfolio_container
                if not portfolio_extracted and hasattr(risk_container, 'portfolio_container') and risk_container.portfolio_container:
                    portfolio = risk_container.portfolio_container
                    logger.info(f"Trying portfolio_container: {type(portfolio)}")
                    
                    # Get portfolio state if available
                    if hasattr(portfolio, 'get_status'):
                        portfolio_status = portfolio.get_status()
                        backtest_data['portfolio_value'] = portfolio_status.get('portfolio_value', 0)
                        backtest_data['positions'] = portfolio_status.get('positions', {})
                        portfolio_extracted = True
                        logger.info(f"Portfolio extracted from portfolio_container: ${backtest_data['portfolio_value']}")
                
                if not portfolio_extracted:
                    logger.warning("Could not extract portfolio state from risk container")
            else:
                logger.warning("No risk container found for portfolio extraction")
            
            # Find execution container for trade data
            execution_container = self._find_container_by_role(root_container, 'execution')
            if execution_container:
                exec_status = execution_container.get_status()
                logger.info(f"Execution container status: {exec_status}")
                
                # Try to get broker data from execution engine
                if hasattr(execution_container, 'execution_engine') and execution_container.execution_engine:
                    engine = execution_container.execution_engine
                    logger.info(f"Found execution engine: {type(engine)}")
                    
                    if hasattr(engine, 'broker') and engine.broker:
                        broker = engine.broker
                        logger.info(f"Found broker: {type(broker)}")
                        
                        # Get fills/trades from broker order tracker
                        fills_found = False
                        if hasattr(broker, 'order_tracker') and broker.order_tracker:
                            if hasattr(broker.order_tracker, 'fills') and broker.order_tracker.fills:
                                logger.info(f"Found {len(broker.order_tracker.fills)} fills in broker.order_tracker")
                                trades = []
                                for fill in broker.order_tracker.fills:
                                    trade = {
                                        'timestamp': str(fill.timestamp) if hasattr(fill, 'timestamp') else '',
                                        'symbol': fill.symbol if hasattr(fill, 'symbol') else '',
                                        'side': str(fill.side) if hasattr(fill, 'side') else '',
                                        'quantity': float(fill.quantity) if hasattr(fill, 'quantity') else 0,
                                        'price': float(fill.price) if hasattr(fill, 'price') else 0,
                                        'commission': float(fill.commission) if hasattr(fill, 'commission') else 0
                                    }
                                    trades.append(trade)
                                    logger.info(f"Extracted trade: {trade}")
                                backtest_data['trades'] = trades
                                fills_found = True
                        
                        # Also try direct fills attribute as fallback
                        if not fills_found and hasattr(broker, 'fills') and broker.fills:
                            logger.info(f"Found {len(broker.fills)} fills in broker.fills")
                            trades = []
                            for fill in broker.fills:
                                trade = {
                                    'timestamp': str(fill.timestamp) if hasattr(fill, 'timestamp') else '',
                                    'symbol': fill.symbol if hasattr(fill, 'symbol') else '',
                                    'side': str(fill.side) if hasattr(fill, 'side') else '',
                                    'quantity': float(fill.quantity) if hasattr(fill, 'quantity') else 0,
                                    'price': float(fill.price) if hasattr(fill, 'price') else 0,
                                    'commission': float(fill.commission) if hasattr(fill, 'commission') else 0
                                }
                                trades.append(trade)
                                logger.info(f"Extracted trade: {trade}")
                            backtest_data['trades'] = trades
                            fills_found = True
                        
                        if not fills_found:
                            logger.warning("No fills found in broker.order_tracker.fills or broker.fills")
                    else:
                        logger.warning("No broker found in execution engine")
                else:
                    logger.warning("No execution engine found in execution container")
                
                # Note: Portfolio value is extracted from risk container above
                # Execution container only handles trade execution, not portfolio state
            
            # Get data container metrics
            data_container = root_container  # Root should be data container
            if data_container:
                data_status = data_container.get_status()
                metrics = data_status.get('metrics', {})
                backtest_data['bars_processed'] = metrics.get('events_published', 0)
            
            logger.info(f"Extracted backtest data: {len(backtest_data.get('trades', []))} trades, {backtest_data.get('bars_processed', 0)} bars")
            
        except Exception as e:
            logger.error(f"Error extracting backtest data: {e}")
        
        return backtest_data
    
    def _find_container_by_role(self, container, role: str):
        """Recursively find container by role"""
        if hasattr(container, 'metadata') and container.metadata.role.value == role:
            return container
        
        if hasattr(container, 'child_containers'):
            for child in container.child_containers:
                result = self._find_container_by_role(child, role)
                if result:
                    return result
        
        return None
    
    async def validate_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Validate workflow configuration for composable execution."""
        
        errors = []
        warnings = []
        
        # Check data configuration
        if not config.data_config:
            errors.append("Missing data configuration")
        
        # Check workflow-specific configuration
        if config.workflow_type == WorkflowType.BACKTEST:
            if not config.backtest_config:
                warnings.append("Missing backtest configuration, using defaults")
        
        elif config.workflow_type == WorkflowType.OPTIMIZATION:
            if not config.optimization_config:
                errors.append("Missing optimization configuration")
        
        elif config.workflow_type == WorkflowType.ANALYSIS:
            if not config.analysis_config:
                warnings.append("Missing analysis configuration, using defaults")
        
        # Validate container patterns
        patterns = self._determine_patterns(config)
        for pattern_info in patterns:
            pattern_name = pattern_info['name']
            pattern = self.composition_engine.registry.get_pattern(pattern_name)
            
            if not pattern:
                errors.append(f"Unknown container pattern: {pattern_name}")
            elif not self.composition_engine.validate_pattern(pattern):
                errors.append(f"Invalid container pattern: {pattern_name}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'patterns': [p['name'] for p in patterns]
        }
    
    def get_required_capabilities(self) -> Dict[str, Any]:
        """Get required infrastructure capabilities."""
        return {
            'composable_containers': True,
            'container_patterns': True,
            'event_bus': True
        }
    
    async def _register_containers_with_router(
        self, 
        root_container: ComposableContainerProtocol
    ) -> None:
        """Recursively register all containers with the event router."""
        
        # Register root container
        root_container.register_with_router(self.event_router)
        logger.debug(f"Registered container {root_container.metadata.name} with event router")
        
        # Build container hierarchy for scope resolution
        hierarchy = {}
        
        def build_hierarchy(container: ComposableContainerProtocol, parent_id: Optional[str] = None):
            container_id = container.metadata.container_id
            children_ids = [child.metadata.container_id for child in container.child_containers]
            
            hierarchy[container_id] = {
                'parent_id': parent_id,
                'children_ids': children_ids,
                'role': container.metadata.role.value,
                'name': container.metadata.name
            }
            
            # Recursively process children
            for child in container.child_containers:
                build_hierarchy(child, container_id)
        
        # Build the hierarchy starting from root
        build_hierarchy(root_container)
        
        # Set hierarchy in router for scope resolution
        self.event_router.set_container_hierarchy(hierarchy)
        
        # Recursively register all child containers
        async def register_children(container: ComposableContainerProtocol):
            for child in container.child_containers:
                child.register_with_router(self.event_router)
                logger.debug(f"Registered child container {child.metadata.name} with event router")
                await register_children(child)
        
        await register_children(root_container)
        
        logger.info(f"Registered {len(hierarchy)} containers with event router")
    
    async def _generate_event_visualization(self, workflow_id: str) -> None:
        """Generate event flow visualization for debugging."""
        
        try:
            from ..events.routing.debugging import EventFlowVisualizer
            
            visualizer = EventFlowVisualizer(self.event_router)
            
            # Generate Mermaid diagram
            mermaid_path = f"debug/{workflow_id}_event_flow.mmd"
            visualizer.save_visualization(mermaid_path, format="mermaid")
            logger.info(f"Generated Mermaid event flow diagram: {mermaid_path}")
            
            # Generate DOT diagram
            dot_path = f"debug/{workflow_id}_event_flow.dot"
            visualizer.save_visualization(dot_path, format="dot")
            logger.info(f"Generated Graphviz event flow diagram: {dot_path}")
            
            # Log topology summary
            topology = self.event_router.get_topology()
            logger.info(f"Event topology: {len(topology['nodes'])} nodes, {len(topology['edges'])} edges")
            
        except Exception as e:
            logger.warning(f"Failed to generate event visualization: {e}")
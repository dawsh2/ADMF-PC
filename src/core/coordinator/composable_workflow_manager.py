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
            # Single backtest - determine complexity level
            if self._is_simple_backtest(config):
                patterns.append({
                    'name': 'simple_backtest',
                    'config': self._build_simple_backtest_config(config)
                })
            else:
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
        
        # Simple if only basic strategy configuration
        return not (has_classifiers or has_risk_profiles or has_portfolios)
    
    def _build_simple_backtest_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build config for simple backtest pattern."""
        
        container_config = {}
        
        # Data configuration
        if config.data_config:
            container_config['data'] = self._extract_data_config(config)
        
        # Strategy configuration - check multiple locations
        strategies = []
        
        # Check top-level strategies
        if hasattr(config, 'strategies') and config.strategies:
            strategies.extend(config.strategies)
        
        # Check backtest.strategies 
        if config.backtest_config and config.backtest_config.get('strategies'):
            strategies.extend(config.backtest_config.get('strategies', []))
            
        # Check optimization_config.strategies (fallback)
        optimization_config = config.optimization_config or {}
        if optimization_config.get('strategies'):
            strategies.extend(optimization_config.get('strategies', []))
        
        # Pass ALL strategies to composition engine for indicator inference
        if strategies:
            container_config['strategies'] = strategies
            # Also set primary strategy for backward compatibility
            strategy_config = strategies[0]  # Use first strategy
            container_config['strategy'] = {
                'type': strategy_config.get('type', 'momentum'),
                'parameters': strategy_config.get('parameters', {})
            }
        
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
            # Compose container pattern
            root_container = self.composition_engine.compose_pattern(
                pattern_name=pattern_name,
                config_overrides=pattern_config
            )
            
            # Store container
            container_key = f"{context.workflow_id}_{pattern_name}"
            self.active_containers[container_key] = root_container
            
            # Execute pattern
            result = await self._execute_container(root_container, config, context)
            result.metadata['container_pattern'] = pattern_name
            
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
            
            # Wait for completion
            # For data streaming workflows, wait longer to let data processing complete
            # TODO: This should listen for completion events instead of fixed time
            await asyncio.sleep(5.0)  # Give time for data streaming to complete
            
            # Collect results
            container_status = root_container.get_status()
            
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
                    'final_state': container_state
                }
            )
            
            # Stop container
            await root_container.stop()
            
            return result
            
        except Exception as e:
            logger.error(f"Container execution failed: {e}")
            raise e
    
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
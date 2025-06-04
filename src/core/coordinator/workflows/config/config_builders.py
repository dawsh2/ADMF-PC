"""
Configuration builders for different workflow patterns.

Builds pattern-specific configurations from WorkflowConfig objects.
"""

import logging
from typing import Dict, Any, List, Optional

from ....types.workflow import WorkflowConfig

logger = logging.getLogger(__name__)


class ConfigBuilder:
    """Builds pattern-specific configurations from workflow configs."""
    
    def build_simple_backtest_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build config for simple backtest pattern."""
        
        container_config = {}
        
        # Data configuration
        if config.data_config:
            container_config['data'] = self._extract_data_config(config)
        
        # Strategy configuration - check multiple locations
        strategies = self._extract_strategies(config)
        
        # Automatic indicator inference
        if strategies:
            from ....strategy.components.indicator_inference import infer_indicators_from_strategies
            required_indicators = infer_indicators_from_strategies(strategies)
            
            container_config['indicator'] = {
                'required_indicators': list(required_indicators),
                'cache_size': 1000
            }
            
            container_config['strategies'] = strategies
            
            # Configure strategy container
            if len(strategies) > 1:
                container_config['strategy'] = {
                    'strategies': strategies,
                    'aggregation': {
                        'method': config.parameters.get('signal_aggregation', {}).get('method', 'weighted_voting'),
                        'min_confidence': config.parameters.get('signal_aggregation', {}).get('min_confidence', 0.6)
                    }
                }
            else:
                strategy_config = strategies[0]
                container_config['strategy'] = {
                    'type': strategy_config.get('type', 'momentum'),
                    'parameters': strategy_config.get('parameters', {})
                }
        
        # Risk configuration
        risk_config = self._extract_risk_config(config)
        if risk_config:
            container_config['risk'] = risk_config
        
        # Portfolio configuration
        portfolio_config = self._extract_portfolio_config(config)
        if portfolio_config:
            container_config['portfolio'] = portfolio_config
        
        # Execution configuration
        if config.backtest_config:
            container_config['execution'] = self._extract_execution_config(config)
        
        return container_config
    
    def build_full_backtest_config(self, config: WorkflowConfig) -> Dict[str, Any]:
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
        
        # Strategy configuration
        strategies = optimization_config.get('strategies', [])
        if strategies:
            strategy_config = strategies[0]
            container_config['strategy'] = {
                'type': strategy_config.get('type', 'momentum'),
                'parameters': strategy_config.get('parameters', {})
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
        
        # Execution configuration
        if config.backtest_config:
            container_config['execution'] = self._extract_execution_config(config)
        
        return container_config
    
    def build_signal_generation_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build config for signal generation pattern."""
        
        container_config = {}
        
        # Data configuration
        if config.data_config:
            container_config['data'] = self._extract_data_config(config)
        
        # Indicator configuration
        container_config['indicator'] = {
            'max_indicators': config.parameters.get('max_indicators', 100)
        }
        
        # Classifier configuration (if available)
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
    
    def build_signal_replay_config(self, config: WorkflowConfig) -> Dict[str, Any]:
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
    
    def build_multi_parameter_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build config for multi-parameter backtest."""
        
        from .parameter_analysis import ParameterAnalyzer
        
        analyzer = ParameterAnalyzer()
        combinations = analyzer.generate_parameter_combinations(config)
        
        container_config = {
            'hub': {
                'role': 'hub',
                'parameter_combinations': combinations,
                'shared_data_config': self._extract_data_config(config) if config.data_config else {}
            },
            'portfolios': []
        }
        
        # Create configuration for each parameter combination
        for combo in combinations:
            portfolio_config = {
                'combination_id': combo['combination_id'],
                'strategy_config': {
                    'type': combo['strategy_type'],
                    'parameters': combo['parameters']
                },
                'risk_config': self._extract_risk_config(config),
                'execution_config': self._extract_execution_config(config) if config.backtest_config else {}
            }
            container_config['portfolios'].append(portfolio_config)
        
        return container_config
    
    def build_optimization_grid_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build config for optimization grid."""
        
        # Similar to multi-parameter but with optimization-specific settings
        base_config = self.build_multi_parameter_config(config)
        
        # Add optimization-specific configuration
        optimization_config = config.optimization_config or {}
        base_config['optimization'] = {
            'objective_function': optimization_config.get('objective_function', 'sharpe_ratio'),
            'optimization_method': optimization_config.get('method', 'grid_search'),
            'parallel_execution': optimization_config.get('parallel_execution', True),
            'early_stopping': optimization_config.get('early_stopping', False)
        }
        
        return base_config
    
    # Helper methods for extracting common configurations
    
    def _extract_strategies(self, config: WorkflowConfig) -> List[Dict[str, Any]]:
        """Extract all strategies from configuration."""
        strategies = []
        
        # Check top-level strategies
        if hasattr(config, 'strategies') and config.strategies:
            strategies.extend(config.strategies)
        
        # Check backtest config strategies
        if config.backtest_config and config.backtest_config.get('strategies'):
            strategies.extend(config.backtest_config['strategies'])
        
        # Check optimization config strategies
        if config.optimization_config and config.optimization_config.get('strategies'):
            strategies.extend(config.optimization_config['strategies'])
        
        return strategies
    
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
    
    def _extract_risk_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Extract risk configuration from multiple sources."""
        risk_config = {}
        
        # From parameters.risk
        if hasattr(config, 'parameters') and config.parameters.get('risk'):
            risk_config.update(config.parameters['risk'])
        
        # From backtest_config.risk
        if config.backtest_config and config.backtest_config.get('risk'):
            risk_config.update(config.backtest_config['risk'])
        
        # From optimization_config.risk
        if config.optimization_config and config.optimization_config.get('risk'):
            risk_config.update(config.optimization_config['risk'])
        
        # Add initial_capital if available
        initial_capital = self._extract_initial_capital(config)
        if initial_capital:
            risk_config['initial_capital'] = initial_capital
        
        return risk_config
    
    def _extract_portfolio_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Extract portfolio configuration."""
        portfolio_config = {}
        
        # From parameters.portfolio
        if hasattr(config, 'parameters') and config.parameters.get('portfolio'):
            portfolio_config.update(config.parameters['portfolio'])
        
        # From backtest_config.portfolio
        if config.backtest_config and config.backtest_config.get('portfolio'):
            portfolio_config.update(config.backtest_config['portfolio'])
        
        # Add initial_capital if not present
        initial_capital = self._extract_initial_capital(config)
        if initial_capital and 'initial_capital' not in portfolio_config:
            portfolio_config['initial_capital'] = initial_capital
        
        return portfolio_config
    
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
    
    def _extract_initial_capital(self, config: WorkflowConfig) -> Optional[float]:
        """Extract initial capital from various configuration sources."""
        
        # Check backtest_config first
        if config.backtest_config and config.backtest_config.get('initial_capital'):
            return config.backtest_config['initial_capital']
        
        # Check parameters.portfolio
        if hasattr(config, 'parameters') and config.parameters.get('portfolio', {}).get('initial_capital'):
            return config.parameters['portfolio']['initial_capital']
        
        # Check parameters.initial_capital
        if hasattr(config, 'parameters') and config.parameters.get('initial_capital'):
            return config.parameters['initial_capital']
        
        return None
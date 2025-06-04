"""
Pattern detection logic for workflow configurations.

Analyzes WorkflowConfig to determine which container patterns should be used.
"""

import logging
from typing import Dict, Any, List

from ....types.workflow import WorkflowConfig, WorkflowType
from .parameter_analysis import ParameterAnalyzer

logger = logging.getLogger(__name__)


class PatternDetector:
    """Detects appropriate container patterns based on workflow configuration."""
    
    def __init__(self):
        self.parameter_analyzer = ParameterAnalyzer()
    
    def determine_patterns(self, config: WorkflowConfig) -> List[Dict[str, Any]]:
        """Determine which container patterns to use based on workflow config."""
        
        patterns = []
        
        # Check for explicit pattern specification
        if hasattr(config, 'pattern') and config.pattern:
            patterns.append({
                'name': config.pattern,
                'config': self._build_pattern_config(config, config.pattern)
            })
            return patterns
        
        # Check if this requires multi-parameter support
        if self.parameter_analyzer.requires_multi_parameter(config):
            if config.workflow_type == WorkflowType.OPTIMIZATION:
                patterns.append({
                    'name': 'optimization_grid',
                    'config': self._build_optimization_grid_config(config)
                })
            else:
                patterns.append({
                    'name': 'multi_parameter_backtest',
                    'config': self._build_multi_parameter_config(config)
                })
            return patterns
        
        # Standard pattern selection based on workflow type
        if config.workflow_type == WorkflowType.BACKTEST:
            pattern_name = self._detect_backtest_pattern(config)
            patterns.append({
                'name': pattern_name,
                'config': self._build_pattern_config(config, pattern_name)
            })
        
        elif config.workflow_type == WorkflowType.OPTIMIZATION:
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
            analysis_mode = (config.analysis_config or {}).get('mode', 'signal_generation')
            
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
    
    def _detect_backtest_pattern(self, config: WorkflowConfig) -> str:
        """Detect specific backtest pattern based on configuration complexity."""
        
        # Check for explicit pattern specification
        if hasattr(config, 'pattern') and config.pattern:
            return config.pattern
        
        # Analyze configuration complexity
        optimization_config = config.optimization_config or {}
        
        has_classifiers = bool(optimization_config.get('classifiers'))
        has_risk_profiles = bool(optimization_config.get('risk_profiles'))
        has_portfolios = bool(optimization_config.get('portfolios'))
        has_multiple_strategies = self._has_multiple_strategies(config)
        
        # Simple backtest if only basic strategy configuration
        if not (has_classifiers or has_risk_profiles or has_portfolios):
            if has_multiple_strategies:
                return 'simple_backtest'  # Can handle multiple strategies
            else:
                return 'simple_backtest'
        
        # Full backtest for complex configurations
        return 'full_backtest'
    
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
    
    def _build_pattern_config(self, config: WorkflowConfig, pattern_name: str) -> Dict[str, Any]:
        """Build configuration for a specific pattern."""
        from .config_builders import ConfigBuilder
        
        builder = ConfigBuilder()
        
        if pattern_name == 'simple_backtest':
            return builder.build_simple_backtest_config(config)
        elif pattern_name == 'full_backtest':
            return builder.build_full_backtest_config(config)
        elif pattern_name == 'signal_generation':
            return builder.build_signal_generation_config(config)
        elif pattern_name == 'signal_replay':
            return builder.build_signal_replay_config(config)
        else:
            # Default to simple backtest config
            return builder.build_simple_backtest_config(config)
    
    def _build_signal_generation_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build config for signal generation pattern."""
        from .config_builders import ConfigBuilder
        return ConfigBuilder().build_signal_generation_config(config)
    
    def _build_signal_replay_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build config for signal replay pattern."""
        from .config_builders import ConfigBuilder
        return ConfigBuilder().build_signal_replay_config(config)
    
    def _build_full_backtest_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build config for full backtest pattern."""
        from .config_builders import ConfigBuilder
        return ConfigBuilder().build_full_backtest_config(config)
    
    def _build_optimization_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build config for optimization workflow."""
        from .config_builders import ConfigBuilder
        return ConfigBuilder().build_full_backtest_config(config)
    
    def _build_multi_parameter_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build config for multi-parameter backtest."""
        from .config_builders import ConfigBuilder
        return ConfigBuilder().build_multi_parameter_config(config)
    
    def _build_optimization_grid_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Build config for optimization grid."""
        from .config_builders import ConfigBuilder
        return ConfigBuilder().build_optimization_grid_config(config)
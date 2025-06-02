"""
File: src/strategy/multi_strategy_config.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#multi-strategy-config
Step: 4 - Multiple Strategies
Dependencies: dataclasses, typing, yaml, json

Multi-strategy configuration system for coordinated strategy execution.
Supports YAML/JSON configuration with validation and factory creation.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json
from pathlib import Path

from ..core.logging.structured import ComponentLogger


class AggregationMethod(Enum):
    """Signal aggregation methods"""
    WEIGHTED_VOTING = "weighted_voting"
    MAJORITY_VOTING = "majority_voting"
    ENSEMBLE = "ensemble"
    UNANIMOUS = "unanimous"


class StrategyType(Enum):
    """Supported strategy types"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    PAIRS_TRADING = "pairs_trading"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"


@dataclass
class StrategyConfig:
    """Configuration for individual strategy"""
    type: str
    name: str
    params: Dict[str, Any]
    weight: float = 1.0
    enabled: bool = True
    min_confidence: float = 0.5
    max_signal_frequency: Optional[int] = None
    performance_tracking: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate strategy configuration"""
        if self.type not in [st.value for st in StrategyType]:
            raise ValueError(f"Unknown strategy type: {self.type}")
        
        if not (0.0 <= self.weight <= 10.0):
            raise ValueError(f"Strategy weight must be between 0 and 10: {self.weight}")
        
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(f"Min confidence must be between 0 and 1: {self.min_confidence}")
        
        if self.max_signal_frequency is not None and self.max_signal_frequency <= 0:
            raise ValueError("Max signal frequency must be positive")


@dataclass
class CoordinatorConfig:
    """Configuration for strategy coordinator"""
    aggregation_method: str = AggregationMethod.WEIGHTED_VOTING.value
    min_consensus_confidence: float = 0.6
    performance_tracking_enabled: bool = True
    dynamic_weight_adjustment: bool = False
    weight_update_frequency: int = 100
    max_concurrent_strategies: int = 10
    signal_timeout_seconds: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate coordinator configuration"""
        if self.aggregation_method not in [am.value for am in AggregationMethod]:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        if not (0.0 <= self.min_consensus_confidence <= 1.0):
            raise ValueError(f"Min consensus confidence must be between 0 and 1: {self.min_consensus_confidence}")
        
        if self.max_concurrent_strategies <= 0:
            raise ValueError("Max concurrent strategies must be positive")


@dataclass
class RiskConfig:
    """Risk management configuration for multi-strategy system"""
    max_portfolio_risk: float = 0.02
    max_position_size: float = 0.1
    max_correlation: float = 0.8
    position_sizing_method: str = "equal_weight"
    risk_budget_allocation: Dict[str, float] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate risk configuration"""
        if not (0.0 < self.max_portfolio_risk <= 1.0):
            raise ValueError("Max portfolio risk must be between 0 and 1")
        
        if not (0.0 < self.max_position_size <= 1.0):
            raise ValueError("Max position size must be between 0 and 1")


@dataclass
class MultiStrategyConfig:
    """Complete multi-strategy system configuration"""
    strategies: Dict[str, StrategyConfig]
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: Dict[str, Any] = field(default_factory=dict)
    optimization: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate complete configuration"""
        if not self.strategies:
            raise ValueError("At least one strategy must be configured")
        
        # Validate individual components
        for strategy_id, strategy_config in self.strategies.items():
            try:
                strategy_config.validate()
            except Exception as e:
                raise ValueError(f"Strategy '{strategy_id}' validation failed: {e}")
        
        self.coordinator.validate()
        self.risk.validate()
        
        # Validate strategy weights sum to reasonable value
        total_weight = sum(s.weight for s in self.strategies.values() if s.enabled)
        if total_weight == 0:
            raise ValueError("No enabled strategies with positive weight")
    
    def get_enabled_strategies(self) -> Dict[str, StrategyConfig]:
        """Get only enabled strategies"""
        return {
            name: config for name, config in self.strategies.items()
            if config.enabled
        }
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get normalized strategy weights"""
        enabled = self.get_enabled_strategies()
        total_weight = sum(s.weight for s in enabled.values())
        
        if total_weight == 0:
            # Equal weights if all zero
            return {name: 1.0 / len(enabled) for name in enabled}
        
        return {name: s.weight / total_weight for name, s in enabled.items()}


class MultiStrategyConfigLoader:
    """Loads and validates multi-strategy configurations"""
    
    def __init__(self, logger: Optional[ComponentLogger] = None):
        self.logger = logger or ComponentLogger("MultiStrategyConfigLoader", "global")
    
    def load_from_file(self, file_path: Union[str, Path]) -> MultiStrategyConfig:
        """Load configuration from YAML or JSON file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yml', '.yaml']:
                    data = yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            return self.load_from_dict(data)
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {file_path}: {e}")
            raise
    
    def load_from_dict(self, data: Dict[str, Any]) -> MultiStrategyConfig:
        """Load configuration from dictionary"""
        try:
            # Parse strategies
            strategies = {}
            for name, strategy_data in data.get('strategies', {}).items():
                strategies[name] = StrategyConfig(
                    type=strategy_data['type'],
                    name=name,
                    params=strategy_data.get('params', {}),
                    weight=strategy_data.get('weight', 1.0),
                    enabled=strategy_data.get('enabled', True),
                    min_confidence=strategy_data.get('min_confidence', 0.5),
                    max_signal_frequency=strategy_data.get('max_signal_frequency'),
                    performance_tracking=strategy_data.get('performance_tracking', True),
                    metadata=strategy_data.get('metadata', {})
                )
            
            # Parse coordinator config
            coordinator_data = data.get('coordinator', {})
            coordinator = CoordinatorConfig(
                aggregation_method=coordinator_data.get('aggregation_method', 'weighted_voting'),
                min_consensus_confidence=coordinator_data.get('min_consensus_confidence', 0.6),
                performance_tracking_enabled=coordinator_data.get('performance_tracking_enabled', True),
                dynamic_weight_adjustment=coordinator_data.get('dynamic_weight_adjustment', False),
                weight_update_frequency=coordinator_data.get('weight_update_frequency', 100),
                max_concurrent_strategies=coordinator_data.get('max_concurrent_strategies', 10),
                signal_timeout_seconds=coordinator_data.get('signal_timeout_seconds', 1.0),
                metadata=coordinator_data.get('metadata', {})
            )
            
            # Parse risk config
            risk_data = data.get('risk', {})
            risk = RiskConfig(
                max_portfolio_risk=risk_data.get('max_portfolio_risk', 0.02),
                max_position_size=risk_data.get('max_position_size', 0.1),
                max_correlation=risk_data.get('max_correlation', 0.8),
                position_sizing_method=risk_data.get('position_sizing_method', 'equal_weight'),
                risk_budget_allocation=risk_data.get('risk_budget_allocation', {})
            )
            
            # Create complete config
            config = MultiStrategyConfig(
                strategies=strategies,
                coordinator=coordinator,
                risk=risk,
                backtest=data.get('backtest', {}),
                optimization=data.get('optimization', {}),
                metadata=data.get('metadata', {})
            )
            
            # Validate
            config.validate()
            
            self.logger.info(
                f"Loaded multi-strategy config with {len(strategies)} strategies, "
                f"aggregation: {coordinator.aggregation_method}"
            )
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to parse configuration: {e}")
            raise
    
    def save_to_file(self, config: MultiStrategyConfig, file_path: Union[str, Path]) -> None:
        """Save configuration to file"""
        file_path = Path(file_path)
        
        # Convert to dictionary
        data = self._config_to_dict(config)
        
        try:
            with open(file_path, 'w') as f:
                if file_path.suffix.lower() in ['.yml', '.yaml']:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
                elif file_path.suffix.lower() == '.json':
                    json.dump(data, f, indent=2)
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            self.logger.info(f"Saved configuration to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config to {file_path}: {e}")
            raise
    
    def _config_to_dict(self, config: MultiStrategyConfig) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'strategies': {
                name: {
                    'type': strategy.type,
                    'params': strategy.params,
                    'weight': strategy.weight,
                    'enabled': strategy.enabled,
                    'min_confidence': strategy.min_confidence,
                    'max_signal_frequency': strategy.max_signal_frequency,
                    'performance_tracking': strategy.performance_tracking,
                    'metadata': strategy.metadata
                }
                for name, strategy in config.strategies.items()
            },
            'coordinator': {
                'aggregation_method': config.coordinator.aggregation_method,
                'min_consensus_confidence': config.coordinator.min_consensus_confidence,
                'performance_tracking_enabled': config.coordinator.performance_tracking_enabled,
                'dynamic_weight_adjustment': config.coordinator.dynamic_weight_adjustment,
                'weight_update_frequency': config.coordinator.weight_update_frequency,
                'max_concurrent_strategies': config.coordinator.max_concurrent_strategies,
                'signal_timeout_seconds': config.coordinator.signal_timeout_seconds,
                'metadata': config.coordinator.metadata
            },
            'risk': {
                'max_portfolio_risk': config.risk.max_portfolio_risk,
                'max_position_size': config.risk.max_position_size,
                'max_correlation': config.risk.max_correlation,
                'position_sizing_method': config.risk.position_sizing_method,
                'risk_budget_allocation': config.risk.risk_budget_allocation
            },
            'backtest': config.backtest,
            'optimization': config.optimization,
            'metadata': config.metadata
        }


def create_sample_multi_strategy_config() -> MultiStrategyConfig:
    """Create a sample multi-strategy configuration"""
    strategies = {
        'momentum_short': StrategyConfig(
            type='momentum',
            name='momentum_short',
            params={'lookback_period': 10, 'momentum_threshold': 0.001},
            weight=0.3,
            min_confidence=0.6
        ),
        'momentum_long': StrategyConfig(
            type='momentum',
            name='momentum_long',
            params={'lookback_period': 20, 'momentum_threshold': 0.0005},
            weight=0.3,
            min_confidence=0.6
        ),
        'mean_reversion': StrategyConfig(
            type='mean_reversion',
            name='mean_reversion',
            params={'period': 15, 'threshold': 2.0},
            weight=0.2,
            min_confidence=0.7
        ),
        'trend_following': StrategyConfig(
            type='trend_following',
            name='trend_following',
            params={'period': 50, 'min_trend_strength': 0.7},
            weight=0.2,
            min_confidence=0.65
        )
    }
    
    coordinator = CoordinatorConfig(
        aggregation_method='weighted_voting',
        min_consensus_confidence=0.6,
        dynamic_weight_adjustment=True,
        weight_update_frequency=50
    )
    
    risk = RiskConfig(
        max_portfolio_risk=0.02,
        max_position_size=0.08,
        position_sizing_method='risk_parity'
    )
    
    return MultiStrategyConfig(
        strategies=strategies,
        coordinator=coordinator,
        risk=risk,
        metadata={'description': 'Sample multi-strategy configuration'}
    )


def load_config_file(file_path: str) -> MultiStrategyConfig:
    """Convenience function to load configuration file"""
    loader = MultiStrategyConfigLoader()
    return loader.load_from_file(file_path)
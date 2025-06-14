"""
Configuration schema and utilities for strategy-specific risk management.

Provides structured configuration for per-strategy risk parameters,
exit criteria, and performance-based adjustments.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json


class ExitType(Enum):
    """Types of exit signals/criteria."""
    TIME_BASED = "time_based"
    STOP_LOSS = "stop_loss"
    PROFIT_TAKING = "profit_taking"
    SIGNAL_BASED = "signal_based"
    VOLATILITY_BASED = "volatility_based"
    CORRELATION_BASED = "correlation_based"


class StrategyType(Enum):
    """Strategy classification types."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MA_CROSSOVER = "ma_crossover"
    TREND_FOLLOWING = "trend_following"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    VOLATILITY = "volatility"


@dataclass
class ExitRules:
    """Exit criteria configuration for a strategy."""
    
    # Time-based exits
    max_holding_bars: Optional[int] = None
    max_holding_hours: Optional[float] = None
    
    # Price-based exits
    max_adverse_excursion_pct: Optional[float] = None  # Stop loss
    min_favorable_excursion_pct: Optional[float] = None  # Profit target
    profit_take_at_mfe_pct: Optional[float] = None  # Partial profit taking
    
    # Signal-based exits
    min_exit_signal_strength: float = 0.5
    require_exit_signal: bool = False
    
    # Volatility-based exits
    max_volatility_multiplier: Optional[float] = None
    min_volatility_threshold: Optional[float] = None
    
    # Risk-adjusted exits
    max_position_heat: Optional[float] = None  # % of portfolio at risk
    correlation_exit_threshold: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class PositionSizingRules:
    """Position sizing configuration for a strategy."""
    
    # Base sizing
    base_position_percent: float = 0.02  # 2% of portfolio
    max_position_percent: float = 0.10   # 10% max
    min_position_value: float = 100.0    # Minimum $ position
    
    # Strategy multipliers
    strategy_type_multiplier: float = 1.0
    volatility_adjustment: bool = True
    
    # Signal-based adjustments
    use_signal_strength: bool = True
    signal_strength_multiplier: float = 1.0
    
    # Performance-based adjustments
    performance_lookback_trades: int = 20
    performance_adjustment_factor: float = 0.5  # How much to adjust based on performance
    min_adjustment_factor: float = 0.2
    max_adjustment_factor: float = 2.0
    
    # Risk adjustments
    correlation_penalty: bool = True
    max_correlation_exposure: float = 0.3
    drawdown_scaling: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class PerformanceTracking:
    """Performance tracking and adjustment configuration."""
    
    # Tracking windows
    short_term_window: int = 10  # Recent trades
    medium_term_window: int = 50
    long_term_window: int = 200
    
    # Performance metrics
    track_win_rate: bool = True
    track_avg_return: bool = True
    track_return_volatility: bool = True
    track_max_drawdown: bool = True
    track_sharpe_ratio: bool = True
    
    # Adjustment triggers
    min_trades_for_adjustment: int = 10
    performance_review_frequency: int = 20  # Every N trades
    
    # Thresholds for adjustments
    poor_performance_threshold: float = -0.02  # -2% avg return
    good_performance_threshold: float = 0.03   # 3% avg return
    high_volatility_threshold: float = 0.3     # 30% volatility
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class StrategyRiskProfile:
    """Complete risk profile for a strategy."""
    
    strategy_id: str
    strategy_type: StrategyType
    description: str = ""
    
    # Core configuration
    position_sizing: PositionSizingRules = field(default_factory=PositionSizingRules)
    exit_rules: ExitRules = field(default_factory=ExitRules)
    performance_tracking: PerformanceTracking = field(default_factory=PerformanceTracking)
    
    # Advanced settings
    correlation_matrix: Dict[str, float] = field(default_factory=dict)
    custom_validators: List[str] = field(default_factory=list)
    risk_override_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_date: Optional[str] = None
    last_updated: Optional[str] = None
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['strategy_type'] = self.strategy_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyRiskProfile':
        """Create from dictionary."""
        # Handle strategy_type enum
        if isinstance(data.get('strategy_type'), str):
            data['strategy_type'] = StrategyType(data['strategy_type'])
        
        # Handle nested dataclasses
        if 'position_sizing' in data and isinstance(data['position_sizing'], dict):
            data['position_sizing'] = PositionSizingRules(**data['position_sizing'])
        
        if 'exit_rules' in data and isinstance(data['exit_rules'], dict):
            data['exit_rules'] = ExitRules(**data['exit_rules'])
            
        if 'performance_tracking' in data and isinstance(data['performance_tracking'], dict):
            data['performance_tracking'] = PerformanceTracking(**data['performance_tracking'])
        
        return cls(**data)


class StrategyRiskConfigManager:
    """Manager for strategy risk configurations."""
    
    def __init__(self):
        self.profiles: Dict[str, StrategyRiskProfile] = {}
        self.templates: Dict[str, StrategyRiskProfile] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default risk profile templates."""
        
        # Aggressive momentum template
        self.templates['aggressive_momentum'] = StrategyRiskProfile(
            strategy_id='aggressive_momentum_template',
            strategy_type=StrategyType.MOMENTUM,
            description='High-risk, high-reward momentum strategy',
            position_sizing=PositionSizingRules(
                base_position_percent=0.04,
                max_position_percent=0.15,
                strategy_type_multiplier=1.5,
                performance_adjustment_factor=0.8
            ),
            exit_rules=ExitRules(
                max_holding_bars=30,
                max_adverse_excursion_pct=0.08,
                min_favorable_excursion_pct=0.15,
                profit_take_at_mfe_pct=0.12
            )
        )
        
        # Conservative mean reversion template
        self.templates['conservative_mean_reversion'] = StrategyRiskProfile(
            strategy_id='conservative_mean_reversion_template',
            strategy_type=StrategyType.MEAN_REVERSION,
            description='Low-risk mean reversion with tight stops',
            position_sizing=PositionSizingRules(
                base_position_percent=0.015,
                max_position_percent=0.08,
                strategy_type_multiplier=0.8,
                performance_adjustment_factor=0.3
            ),
            exit_rules=ExitRules(
                max_holding_bars=8,
                max_adverse_excursion_pct=0.025,
                min_favorable_excursion_pct=0.03,
                profit_take_at_mfe_pct=0.025
            )
        )
        
        # Scalping breakout template
        self.templates['scalping_breakout'] = StrategyRiskProfile(
            strategy_id='scalping_breakout_template',
            strategy_type=StrategyType.BREAKOUT,
            description='Fast scalping on breakout signals',
            position_sizing=PositionSizingRules(
                base_position_percent=0.01,
                max_position_percent=0.05,
                strategy_type_multiplier=2.0,
                use_signal_strength=True,
                signal_strength_multiplier=1.5
            ),
            exit_rules=ExitRules(
                max_holding_bars=5,
                max_adverse_excursion_pct=0.015,
                min_favorable_excursion_pct=0.02,
                profit_take_at_mfe_pct=0.015,
                min_exit_signal_strength=0.8
            )
        )
    
    def create_profile(
        self,
        strategy_id: str,
        strategy_type: Union[StrategyType, str],
        template: Optional[str] = None,
        **overrides
    ) -> StrategyRiskProfile:
        """
        Create a new strategy risk profile.
        
        Args:
            strategy_id: Unique identifier for the strategy
            strategy_type: Type of strategy
            template: Optional template to base the profile on
            **overrides: Override specific configuration values
            
        Returns:
            New StrategyRiskProfile instance
        """
        if isinstance(strategy_type, str):
            strategy_type = StrategyType(strategy_type)
        
        # Start with template if provided
        if template and template in self.templates:
            base_profile = self.templates[template]
            profile = StrategyRiskProfile(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                position_sizing=PositionSizingRules(**asdict(base_profile.position_sizing)),
                exit_rules=ExitRules(**asdict(base_profile.exit_rules)),
                performance_tracking=PerformanceTracking(**asdict(base_profile.performance_tracking))
            )
        else:
            profile = StrategyRiskProfile(
                strategy_id=strategy_id,
                strategy_type=strategy_type
            )
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        self.profiles[strategy_id] = profile
        return profile
    
    def get_profile(self, strategy_id: str) -> Optional[StrategyRiskProfile]:
        """Get risk profile for a strategy."""
        return self.profiles.get(strategy_id)
    
    def update_profile(self, strategy_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing risk profile."""
        if strategy_id not in self.profiles:
            return False
        
        profile = self.profiles[strategy_id]
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        return True
    
    def save_to_file(self, filepath: str):
        """Save all profiles to a JSON file."""
        data = {
            'profiles': {sid: profile.to_dict() for sid, profile in self.profiles.items()},
            'templates': {tid: template.to_dict() for tid, template in self.templates.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_from_file(self, filepath: str):
        """Load profiles from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.profiles = {
            sid: StrategyRiskProfile.from_dict(pdata)
            for sid, pdata in data.get('profiles', {}).items()
        }
        
        if 'templates' in data:
            self.templates.update({
                tid: StrategyRiskProfile.from_dict(tdata)
                for tid, tdata in data['templates'].items()
            })
    
    def get_risk_params_for_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get risk parameters in the format expected by validators.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dictionary of risk parameters for use with validators
        """
        profile = self.get_profile(strategy_id)
        if not profile:
            # Return default parameters
            return {
                'base_position_percent': 0.02,
                'max_position_percent': 0.10,
                'strategy_configs': {}
            }
        
        return {
            'base_position_percent': profile.position_sizing.base_position_percent,
            'max_position_percent': profile.position_sizing.max_position_percent,
            'strategy_configs': {
                strategy_id: {
                    'base_position_percent': profile.position_sizing.base_position_percent,
                    'strategy_type_multiplier': profile.position_sizing.strategy_type_multiplier,
                    'use_signal_strength': profile.position_sizing.use_signal_strength,
                    'exit_rules': profile.exit_rules.to_dict(),
                    'performance_tracking': profile.performance_tracking.to_dict()
                }
            },
            'performance_adjustment': {
                'target_win_rate': 0.5,
                'min_adjustment_factor': profile.position_sizing.min_adjustment_factor,
                'max_adjustment_factor': profile.position_sizing.max_adjustment_factor,
                'max_volatility': profile.performance_tracking.high_volatility_threshold
            },
            'strategy_correlations': profile.correlation_matrix,
            'max_correlation_exposure': profile.position_sizing.max_correlation_exposure
        }


# Example usage
def create_example_configurations():
    """Create example configurations for common strategy types."""
    
    manager = StrategyRiskConfigManager()
    
    # Create profiles for different strategy types
    momentum_configs = [
        ('momentum_aggressive', 'aggressive_momentum'),
        ('momentum_conservative', None),
    ]
    
    for strategy_id, template in momentum_configs:
        profile = manager.create_profile(
            strategy_id=strategy_id,
            strategy_type=StrategyType.MOMENTUM,
            template=template
        )
    
    # Create mean reversion strategies
    manager.create_profile(
        strategy_id='mean_reversion_tight',
        strategy_type=StrategyType.MEAN_REVERSION,
        template='conservative_mean_reversion',
        position_sizing__base_position_percent=0.01  # Even more conservative
    )
    
    # Create breakout strategies  
    manager.create_profile(
        strategy_id='breakout_volume',
        strategy_type=StrategyType.BREAKOUT,
        template='scalping_breakout',
        exit_rules__max_holding_bars=15,  # Hold longer than scalping
        position_sizing__base_position_percent=0.025
    )
    
    return manager


if __name__ == '__main__':
    # Example usage
    manager = create_example_configurations()
    
    # Save to file
    manager.save_to_file('strategy_risk_configs.json')
    
    # Print example configuration
    profile = manager.get_profile('momentum_aggressive')
    if profile:
        print("Example aggressive momentum profile:")
        print(json.dumps(profile.to_dict(), indent=2, default=str))
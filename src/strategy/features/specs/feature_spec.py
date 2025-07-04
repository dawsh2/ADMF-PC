"""
Feature specification system with deterministic naming and strict validation.

This module provides the foundation for the new feature system that replaces
the fragile inference-based approach with explicit, validated specifications.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set, Callable
from enum import Enum


class FeatureType(Enum):
    """Supported feature types."""
    # Trend indicators
    SMA = "sma"
    EMA = "ema"
    DEMA = "dema"
    TEMA = "tema"
    WMA = "wma"
    HMA = "hma"
    VWMA = "vwma"
    ICHIMOKU = "ichimoku"
    PSAR = "psar"
    PARABOLIC_SAR = "parabolic_sar"
    
    # Oscillator indicators
    RSI = "rsi"
    STOCHASTIC = "stochastic"
    WILLIAMS_R = "williams_r"
    CCI = "cci"
    STOCHASTIC_RSI = "stochastic_rsi"
    MFI = "mfi"
    ULTIMATE_OSCILLATOR = "ultimate_oscillator"
    
    # Volatility indicators
    ATR = "atr"
    BOLLINGER = "bollinger"
    BOLLINGER_BANDS = "bollinger_bands"
    BB = "bb"
    KELTNER = "keltner"
    KELTNER_CHANNEL = "keltner_channel"
    DONCHIAN = "donchian"
    DONCHIAN_CHANNEL = "donchian_channel"
    VOLATILITY = "volatility"
    SUPERTREND = "supertrend"
    VWAP = "vwap"
    
    # Volume indicators
    VOLUME = "volume"
    VOLUME_SMA = "volume_sma"
    VOLUME_RATIO = "volume_ratio"
    OBV = "obv"
    VPT = "vpt"
    CMF = "cmf"
    CHAIKIN_MONEY_FLOW = "chaikin_money_flow"
    AD = "ad"
    AD_LINE = "ad_line"
    ACCUMULATION_DISTRIBUTION = "accumulation_distribution"
    VROC = "vroc"
    VOLUME_ROC = "volume_roc"
    VOLUME_MOMENTUM = "volume_momentum"
    
    # Momentum indicators
    MACD = "macd"
    MOMENTUM = "momentum"
    ROC = "roc"
    RATE_OF_CHANGE = "rate_of_change"
    ADX = "adx"
    AROON = "aroon"
    VORTEX = "vortex"
    
    # Structure indicators
    PIVOT_POINTS = "pivot_points"
    SUPPORT_RESISTANCE = "support_resistance"
    SR = "sr"
    SWING_POINTS = "swing_points"
    SWING = "swing"
    LINEAR_REGRESSION = "linear_regression"
    LINREG = "linreg"
    FIBONACCI_RETRACEMENT = "fibonacci_retracement"
    FIBONACCI = "fibonacci"
    FIB = "fib"
    TRENDLINES = "trendlines"
    TRENDLINE = "trendline"
    DIAGONAL_CHANNEL = "diagonal_channel"
    CHANNEL = "channel"
    PRICE_PEAKS = "price_peaks"
    PEAKS = "peaks"
    
    # Divergence features
    RSI_DIVERGENCE = "rsi_divergence"
    BB_RSI_DIVERGENCE = "bb_rsi_divergence"
    BB_RSI_TRACKER = "bb_rsi_tracker"
    BB_RSI_DIVERGENCE_PROPER = "bb_rsi_divergence_proper"
    BB_RSI_DEPENDENT = "bb_rsi_dependent"
    BB_RSI_DIVERGENCE_EXACT = "bb_rsi_divergence_exact"
    BB_RSI_DIVERGENCE_SELF = "bb_rsi_divergence_self"


@dataclass(frozen=True)
class FeatureSpec:
    """
    Immutable feature specification with deterministic naming.
    
    This replaces the old string-based feature inference system with
    explicit, validated feature requirements.
    """
    feature_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    output_component: Optional[str] = None
    
    def __post_init__(self):
        """Validate parameters at creation time."""
        self._validate_params()
    
    @property 
    def canonical_name(self) -> str:
        """
        Generate deterministic feature name.
        
        Standard format: {type}_{param1}_{param2}_{component}
        Parameters are sorted alphabetically for consistency.
        """
        # Sort parameters for consistent naming
        sorted_params = sorted(self.params.items())
        param_str = '_'.join(str(v) for _, v in sorted_params)
        
        base_name = f"{self.feature_type}_{param_str}" if param_str else self.feature_type
        
        if self.output_component:
            return f"{base_name}_{self.output_component}"
        return base_name
    
    def _validate_params(self):
        """Strict parameter validation against registry."""
        registry_entry = FEATURE_REGISTRY.get(self.feature_type)
        if not registry_entry:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
        
        # Check all required parameters present
        missing = set(registry_entry.required_params) - set(self.params.keys())
        if missing:
            raise ValueError(
                f"Missing required parameters for {self.feature_type}: {missing}"
            )
        
        # Check no extra parameters
        all_params = set(registry_entry.required_params) | set(registry_entry.optional_params.keys())
        extra = set(self.params.keys()) - all_params
        if extra:
            raise ValueError(
                f"Unknown parameters for {self.feature_type}: {extra}"
            )
        
        # Validate parameter values
        for param_name, value in self.params.items():
            validator = registry_entry.param_validators.get(param_name)
            if validator and not validator(value):
                raise ValueError(
                    f"Invalid value {value} for parameter {param_name} in {self.feature_type}"
                )
        
        # Validate output component
        if self.output_component:
            if self.output_component not in registry_entry.output_components:
                raise ValueError(
                    f"Unknown output component '{self.output_component}' for {self.feature_type}. "
                    f"Valid components: {registry_entry.output_components}"
                )


@dataclass
class FeatureRegistryEntry:
    """Complete specification for a feature type."""
    name: str
    required_params: List[str]
    optional_params: Dict[str, Any] = field(default_factory=dict)
    param_validators: Dict[str, Callable[[Any], bool]] = field(default_factory=dict)
    output_components: List[str] = field(default_factory=list)
    computation_func: Optional[Callable] = None
    description: str = ""
    
    @property
    def all_params(self) -> Set[str]:
        """All valid parameter names."""
        return set(self.required_params) | set(self.optional_params.keys())


# Centralized registry with strict validation
FEATURE_REGISTRY: Dict[str, FeatureRegistryEntry] = {
    # ========== TREND INDICATORS ==========
    'sma': FeatureRegistryEntry(
        name='sma',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 500
        },
        description='Simple moving average'
    ),
    
    'ema': FeatureRegistryEntry(
        name='ema',
        required_params=['period'],
        optional_params={'smoothing': 2.0},
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 500,
            'smoothing': lambda x: isinstance(x, (int, float)) and x > 0
        },
        description='Exponential moving average'
    ),
    
    'dema': FeatureRegistryEntry(
        name='dema',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 500
        },
        description='Double exponential moving average'
    ),
    
    'tema': FeatureRegistryEntry(
        name='tema',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 500
        },
        description='Triple exponential moving average'
    ),
    
    'wma': FeatureRegistryEntry(
        name='wma',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 500
        },
        description='Weighted moving average'
    ),
    
    'hma': FeatureRegistryEntry(
        name='hma',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 4 <= x <= 500
        },
        description='Hull moving average'
    ),
    
    'vwma': FeatureRegistryEntry(
        name='vwma',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 500
        },
        description='Volume weighted moving average'
    ),
    
    'ichimoku': FeatureRegistryEntry(
        name='ichimoku',
        required_params=[],
        optional_params={
            'conversion_period': 9,
            'base_period': 26,
            'lagging_span_period': 52,
            'displacement': 26
        },
        param_validators={
            'conversion_period': lambda x: isinstance(x, int) and x > 0,
            'base_period': lambda x: isinstance(x, int) and x > 0,
            'lagging_span_period': lambda x: isinstance(x, int) and x > 0,
            'displacement': lambda x: isinstance(x, int) and x > 0
        },
        output_components=['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span'],
        description='Ichimoku cloud indicator'
    ),
    
    'psar': FeatureRegistryEntry(
        name='psar',
        required_params=[],
        optional_params={'af_start': 0.02, 'af_max': 0.2},
        param_validators={
            'af_start': lambda x: isinstance(x, (int, float)) and 0 < x < 1,
            'af_max': lambda x: isinstance(x, (int, float)) and 0 < x < 1
        },
        description='Parabolic SAR'
    ),
    
    'parabolic_sar': FeatureRegistryEntry(
        name='parabolic_sar',
        required_params=[],
        optional_params={'af_start': 0.02, 'af_max': 0.2},
        param_validators={
            'af_start': lambda x: isinstance(x, (int, float)) and 0 < x < 1,
            'af_max': lambda x: isinstance(x, (int, float)) and 0 < x < 1
        },
        description='Parabolic SAR (alias)'
    ),
    
    # ========== OSCILLATOR INDICATORS ==========
    'rsi': FeatureRegistryEntry(
        name='rsi',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 2 <= x <= 100
        },
        description='Relative strength index'
    ),
    
    'stochastic': FeatureRegistryEntry(
        name='stochastic',
        required_params=['k_period', 'd_period'],
        param_validators={
            'k_period': lambda x: isinstance(x, int) and 1 <= x <= 100,
            'd_period': lambda x: isinstance(x, int) and 1 <= x <= 20
        },
        output_components=['k', 'd'],
        description='Stochastic oscillator with %K and %D components'
    ),
    
    'williams_r': FeatureRegistryEntry(
        name='williams_r',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 2 <= x <= 100
        },
        description='Williams %R'
    ),
    
    'cci': FeatureRegistryEntry(
        name='cci',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 2 <= x <= 100
        },
        description='Commodity channel index'
    ),
    
    'stochastic_rsi': FeatureRegistryEntry(
        name='stochastic_rsi',
        required_params=['rsi_period', 'stoch_period', 'd_period'],
        param_validators={
            'rsi_period': lambda x: isinstance(x, int) and 2 <= x <= 100,
            'stoch_period': lambda x: isinstance(x, int) and 2 <= x <= 100,
            'd_period': lambda x: isinstance(x, int) and 1 <= x <= 20
        },
        output_components=['k', 'd'],
        description='Stochastic RSI'
    ),
    
    'mfi': FeatureRegistryEntry(
        name='mfi',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 2 <= x <= 100
        },
        description='Money flow index'
    ),
    
    'ultimate_oscillator': FeatureRegistryEntry(
        name='ultimate_oscillator',
        required_params=[],
        optional_params={'period1': 7, 'period2': 14, 'period3': 28},
        param_validators={
            'period1': lambda x: isinstance(x, int) and x > 0,
            'period2': lambda x: isinstance(x, int) and x > 0,
            'period3': lambda x: isinstance(x, int) and x > 0
        },
        description='Ultimate oscillator'
    ),
    
    # ========== VOLATILITY INDICATORS ==========
    'atr': FeatureRegistryEntry(
        name='atr',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 100
        },
        description='Average true range'
    ),
    
    'bollinger': FeatureRegistryEntry(
        name='bollinger',
        required_params=['period'],
        optional_params={'std_dev': 2.0},
        param_validators={
            'period': lambda x: isinstance(x, int) and 2 <= x <= 200,
            'std_dev': lambda x: isinstance(x, (int, float)) and 0.5 <= x <= 4
        },
        output_components=['upper', 'middle', 'lower'],
        description='Bollinger bands'
    ),
    
    'bollinger_bands': FeatureRegistryEntry(
        name='bollinger_bands',
        required_params=['period'],
        optional_params={'std_dev': 2.0},
        param_validators={
            'period': lambda x: isinstance(x, int) and 2 <= x <= 200,
            'std_dev': lambda x: isinstance(x, (int, float)) and 0.5 <= x <= 4
        },
        output_components=['upper', 'middle', 'lower'],
        description='Bollinger bands (alias)'
    ),
    
    'bb': FeatureRegistryEntry(
        name='bb',
        required_params=['period'],
        optional_params={'std_dev': 2.0},
        param_validators={
            'period': lambda x: isinstance(x, int) and 2 <= x <= 200,
            'std_dev': lambda x: isinstance(x, (int, float)) and 0.5 <= x <= 4
        },
        output_components=['upper', 'middle', 'lower'],
        description='Bollinger bands (short alias)'
    ),
    
    'keltner': FeatureRegistryEntry(
        name='keltner',
        required_params=['period'],
        optional_params={'multiplier': 2.0},
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 100,
            'multiplier': lambda x: isinstance(x, (int, float)) and x > 0
        },
        output_components=['upper', 'middle', 'lower'],
        description='Keltner channel'
    ),
    
    'keltner_channel': FeatureRegistryEntry(
        name='keltner_channel',
        required_params=['period'],
        optional_params={'multiplier': 2.0},
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 100,
            'multiplier': lambda x: isinstance(x, (int, float)) and x > 0
        },
        output_components=['upper', 'middle', 'lower'],
        description='Keltner channel (alias)'
    ),
    
    'donchian': FeatureRegistryEntry(
        name='donchian',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 200
        },
        output_components=['upper', 'middle', 'lower'],
        description='Donchian channel'
    ),
    
    'donchian_channel': FeatureRegistryEntry(
        name='donchian_channel',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 200
        },
        output_components=['upper', 'middle', 'lower'],
        description='Donchian channel (alias)'
    ),
    
    'volatility': FeatureRegistryEntry(
        name='volatility',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 2 <= x <= 200
        },
        description='Price volatility (standard deviation of returns)'
    ),
    
    'supertrend': FeatureRegistryEntry(
        name='supertrend',
        required_params=['period'],
        optional_params={'multiplier': 3.0},
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 100,
            'multiplier': lambda x: isinstance(x, (int, float)) and x > 0
        },
        output_components=['supertrend', 'trend', 'upper', 'lower'],
        description='SuperTrend indicator'
    ),
    
    'vwap': FeatureRegistryEntry(
        name='vwap',
        required_params=[],  # No parameters for intraday VWAP
        description='Volume-weighted average price'
    ),
    
    # ========== VOLUME INDICATORS ==========
    'volume': FeatureRegistryEntry(
        name='volume',
        required_params=[],
        description='Raw volume from bar data'
    ),
    
    'volume_sma': FeatureRegistryEntry(
        name='volume_sma',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 200
        },
        description='Volume simple moving average'
    ),
    
    'volume_ratio': FeatureRegistryEntry(
        name='volume_ratio',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 200
        },
        description='Volume ratio (current volume / average volume)'
    ),
    
    'obv': FeatureRegistryEntry(
        name='obv',
        required_params=[],  # No parameters
        description='On-balance volume'
    ),
    
    'vpt': FeatureRegistryEntry(
        name='vpt',
        required_params=[],
        description='Volume price trend'
    ),
    
    'cmf': FeatureRegistryEntry(
        name='cmf',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 100
        },
        description='Chaikin money flow'
    ),
    
    'chaikin_money_flow': FeatureRegistryEntry(
        name='chaikin_money_flow',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 100
        },
        description='Chaikin money flow (alias)'
    ),
    
    'ad': FeatureRegistryEntry(
        name='ad',
        required_params=[],
        description='Accumulation/distribution line (short alias)'
    ),
    
    'ad_line': FeatureRegistryEntry(
        name='ad_line',
        required_params=[],
        description='Accumulation/distribution line'
    ),
    
    'accumulation_distribution': FeatureRegistryEntry(
        name='accumulation_distribution',
        required_params=[],
        description='Accumulation/distribution line (full alias)'
    ),
    
    'vroc': FeatureRegistryEntry(
        name='vroc',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 200
        },
        description='Volume rate of change'
    ),
    
    'volume_roc': FeatureRegistryEntry(
        name='volume_roc',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 200
        },
        description='Volume rate of change (alias)'
    ),
    
    'volume_momentum': FeatureRegistryEntry(
        name='volume_momentum',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 200
        },
        description='Volume momentum'
    ),
    
    # ========== MOMENTUM INDICATORS ==========
    'macd': FeatureRegistryEntry(
        name='macd', 
        required_params=['fast_period', 'slow_period', 'signal_period'],
        param_validators={
            'fast_period': lambda x: isinstance(x, int) and 1 <= x <= 50,
            'slow_period': lambda x: isinstance(x, int) and 10 <= x <= 200,
            'signal_period': lambda x: isinstance(x, int) and 1 <= x <= 50
        },
        output_components=['macd', 'signal', 'histogram'],
        description='MACD with signal line and histogram'
    ),
    
    'momentum': FeatureRegistryEntry(
        name='momentum',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 200
        },
        description='Price momentum'
    ),
    
    'roc': FeatureRegistryEntry(
        name='roc',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 200
        },
        description='Rate of change'
    ),
    
    'rate_of_change': FeatureRegistryEntry(
        name='rate_of_change',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 1 <= x <= 200
        },
        description='Rate of change (alias)'
    ),
    
    'adx': FeatureRegistryEntry(
        name='adx',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 2 <= x <= 100
        },
        output_components=['adx', 'di_plus', 'di_minus', 'dx'],
        description='Average directional index with DI lines'
    ),
    
    'aroon': FeatureRegistryEntry(
        name='aroon',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 2 <= x <= 100
        },
        output_components=['up', 'down', 'oscillator'],
        description='Aroon indicator'
    ),
    
    'vortex': FeatureRegistryEntry(
        name='vortex',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 2 <= x <= 100
        },
        output_components=['vi_plus', 'vi_minus'],
        description='Vortex indicator'
    ),
    
    # ========== STRUCTURE INDICATORS ==========
    'pivot_points': FeatureRegistryEntry(
        name='pivot_points',
        required_params=[],
        optional_params={'type': 'standard', 'timeframe': 'D'},
        param_validators={
            'type': lambda x: x in ['standard', 'fibonacci', 'woodie', 'camarilla', 'demark'],
            'timeframe': lambda x: x in ['1m', '5m', '15m', '30m', '1h', '4h', 'D', 'W', 'M']
        },
        output_components=['pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3'],
        description='Classic pivot points'
    ),
    
    'support_resistance': FeatureRegistryEntry(
        name='support_resistance',
        required_params=[],
        optional_params={'lookback': 50, 'min_touches': 2},
        param_validators={
            'lookback': lambda x: isinstance(x, int) and x > 0,
            'min_touches': lambda x: isinstance(x, int) and x > 0
        },
        output_components=['resistance', 'support'],
        description='Support and resistance levels'
    ),
    
    'sr': FeatureRegistryEntry(
        name='sr',
        required_params=[],
        optional_params={'lookback': 50, 'min_touches': 2},
        param_validators={
            'lookback': lambda x: isinstance(x, int) and x > 0,
            'min_touches': lambda x: isinstance(x, int) and x > 0
        },
        output_components=['resistance', 'support'],
        description='Support and resistance levels (alias)'
    ),
    
    'swing_points': FeatureRegistryEntry(
        name='swing_points',
        required_params=[],
        optional_params={'lookback': 5},
        param_validators={
            'lookback': lambda x: isinstance(x, int) and x > 0
        },
        output_components=['swing_high', 'swing_low'],
        description='Swing high/low points'
    ),
    
    'swing': FeatureRegistryEntry(
        name='swing',
        required_params=[],
        optional_params={'lookback': 5},
        param_validators={
            'lookback': lambda x: isinstance(x, int) and x > 0
        },
        output_components=['swing_high', 'swing_low'],
        description='Swing high/low points (alias)'
    ),
    
    'linear_regression': FeatureRegistryEntry(
        name='linear_regression',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 2 <= x <= 500
        },
        output_components=['value', 'slope', 'intercept', 'r2'],
        description='Linear regression channel'
    ),
    
    'linreg': FeatureRegistryEntry(
        name='linreg',
        required_params=['period'],
        param_validators={
            'period': lambda x: isinstance(x, int) and 2 <= x <= 500
        },
        output_components=['value', 'slope', 'intercept', 'r2'],
        description='Linear regression channel (alias)'
    ),
    
    'fibonacci_retracement': FeatureRegistryEntry(
        name='fibonacci_retracement',
        required_params=[],
        optional_params={'lookback': 50},
        param_validators={
            'lookback': lambda x: isinstance(x, int) and x > 0
        },
        output_components=['fib_0', 'fib_23', 'fib_38', 'fib_50', 'fib_61', 'fib_78', 'fib_100'],
        description='Fibonacci retracement levels'
    ),
    
    'fibonacci': FeatureRegistryEntry(
        name='fibonacci',
        required_params=[],
        optional_params={'lookback': 50},
        param_validators={
            'lookback': lambda x: isinstance(x, int) and x > 0
        },
        output_components=['fib_0', 'fib_23', 'fib_38', 'fib_50', 'fib_61', 'fib_78', 'fib_100'],
        description='Fibonacci retracement levels (alias)'
    ),
    
    'fib': FeatureRegistryEntry(
        name='fib',
        required_params=[],
        optional_params={'lookback': 50},
        param_validators={
            'lookback': lambda x: isinstance(x, int) and x > 0
        },
        output_components=['fib_0', 'fib_23', 'fib_38', 'fib_50', 'fib_61', 'fib_78', 'fib_100'],
        description='Fibonacci retracement levels (short alias)'
    ),
    
    'trendlines': FeatureRegistryEntry(
        name='trendlines',
        required_params=[],
        optional_params={
            'pivot_lookback': 20,
            'min_touches': 2,
            'tolerance': 0.002,
            'max_lines': 10,
            'num_pivots': 3
        },
        param_validators={
            'pivot_lookback': lambda x: isinstance(x, int) and x > 0,
            'min_touches': lambda x: isinstance(x, int) and x > 0,
            'tolerance': lambda x: isinstance(x, (int, float)) and 0 < x < 1,
            'max_lines': lambda x: isinstance(x, int) and x > 0,
            'num_pivots': lambda x: isinstance(x, int) and x > 0
        },
        output_components=['valid_uptrends', 'valid_downtrends', 'nearest_support', 
                           'nearest_resistance', 'strongest_uptrend', 'strongest_downtrend'],
        description='Dynamic trendline detection'
    ),
    
    'trendline': FeatureRegistryEntry(
        name='trendline',
        required_params=[],
        optional_params={
            'pivot_lookback': 20,
            'min_touches': 2,
            'tolerance': 0.002,
            'max_lines': 10,
            'num_pivots': 3
        },
        param_validators={
            'pivot_lookback': lambda x: isinstance(x, int) and x > 0,
            'min_touches': lambda x: isinstance(x, int) and x > 0,
            'tolerance': lambda x: isinstance(x, (int, float)) and 0 < x < 1,
            'max_lines': lambda x: isinstance(x, int) and x > 0,
            'num_pivots': lambda x: isinstance(x, int) and x > 0
        },
        output_components=['valid_uptrends', 'valid_downtrends', 'nearest_support', 
                           'nearest_resistance', 'strongest_uptrend', 'strongest_downtrend'],
        description='Dynamic trendline detection (alias)'
    ),
    
    'diagonal_channel': FeatureRegistryEntry(
        name='diagonal_channel',
        required_params=[],
        optional_params={
            'lookback': 20,
            'min_points': 3,
            'channel_tolerance': 0.02,
            'parallel_tolerance': 0.1
        },
        param_validators={
            'lookback': lambda x: isinstance(x, int) and x > 0,
            'min_points': lambda x: isinstance(x, int) and x > 0,
            'channel_tolerance': lambda x: isinstance(x, (int, float)) and 0 < x < 1,
            'parallel_tolerance': lambda x: isinstance(x, (int, float)) and 0 < x < 1
        },
        output_components=['upper_channel', 'lower_channel', 'channel_width', 'channel_angle'],
        description='Diagonal channel detection with parallel trendlines'
    ),
    
    # ========== DIVERGENCE FEATURES ==========
    'rsi_divergence': FeatureRegistryEntry(
        name='rsi_divergence',
        required_params=[],
        optional_params={
            'rsi_period': 14,
            'lookback_bars': 50,
            'min_bars_between': 5,
            'rsi_divergence_threshold': 5.0,
            'price_threshold_pct': 0.001
        },
        param_validators={
            'rsi_period': lambda x: isinstance(x, int) and 1 <= x <= 100,
            'lookback_bars': lambda x: isinstance(x, int) and 10 <= x <= 200,
            'min_bars_between': lambda x: isinstance(x, int) and 1 <= x <= 50,
            'rsi_divergence_threshold': lambda x: isinstance(x, (int, float)) and x > 0,
            'price_threshold_pct': lambda x: isinstance(x, (int, float)) and 0 < x < 0.1
        },
        output_components=['has_bullish_divergence', 'has_bearish_divergence', 'divergence_strength', 'bars_since_divergence'],
        description='True RSI divergence detection comparing price extremes with RSI extremes over time'
    ),
    
    'bb_rsi_divergence': FeatureRegistryEntry(
        name='bb_rsi_divergence',
        required_params=[],
        optional_params={
            'lookback': 20,
            'rsi_divergence_threshold': 5.0,
            'confirmation_bars': 10
        },
        param_validators={
            'lookback': lambda x: isinstance(x, int) and x > 0,
            'rsi_divergence_threshold': lambda x: isinstance(x, (int, float)) and x > 0,
            'confirmation_bars': lambda x: isinstance(x, int) and x > 0
        },
        output_components=['has_bullish_divergence', 'has_bearish_divergence', 'confirmed_long', 'confirmed_short'],
        description='Bollinger Band + RSI divergence detection'
    ),
    
    'bb_rsi_tracker': FeatureRegistryEntry(
        name='bb_rsi_tracker',
        required_params=[],
        optional_params={
            'lookback_bars': 20,
            'rsi_divergence_threshold': 5.0,
            'confirmation_bars': 10
        },
        param_validators={
            'lookback_bars': lambda x: isinstance(x, int) and x > 0,
            'rsi_divergence_threshold': lambda x: isinstance(x, (int, float)) and x > 0,
            'confirmation_bars': lambda x: isinstance(x, int) and x > 0
        },
        output_components=['confirmed_long', 'confirmed_short', 'has_bullish_divergence', 'has_bearish_divergence'],
        description='Bollinger Band + RSI divergence tracker with multi-bar pattern detection'
    ),
    
    'bb_rsi_divergence_proper': FeatureRegistryEntry(
        name='bb_rsi_divergence_proper',
        required_params=[],
        optional_params={
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'lookback': 20,
            'rsi_divergence_threshold': 5.0,
            'confirmation_bars': 10
        },
        param_validators={
            'bb_period': lambda x: isinstance(x, int) and 5 <= x <= 50,
            'bb_std': lambda x: isinstance(x, (int, float)) and 0.5 <= x <= 4.0,
            'rsi_period': lambda x: isinstance(x, int) and 2 <= x <= 30,
            'lookback': lambda x: isinstance(x, int) and x > 0,
            'rsi_divergence_threshold': lambda x: isinstance(x, (int, float)) and x > 0,
            'confirmation_bars': lambda x: isinstance(x, int) and x > 0
        },
        output_components=['confirmed_long', 'confirmed_short', 'has_bullish_divergence', 'has_bearish_divergence', 'upper_band', 'middle_band', 'lower_band', 'rsi'],
        description='Self-contained BB + RSI divergence detector that computes indicators internally'
    ),
    
    'bb_rsi_dependent': FeatureRegistryEntry(
        name='bb_rsi_dependent',
        required_params=[],
        optional_params={
            'lookback': 20,
            'rsi_divergence_threshold': 5.0,
            'confirmation_bars': 10,
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14
        },
        param_validators={
            'lookback': lambda x: isinstance(x, int) and x > 0,
            'rsi_divergence_threshold': lambda x: isinstance(x, (int, float)) and x > 0,
            'confirmation_bars': lambda x: isinstance(x, int) and x > 0,
            'bb_period': lambda x: isinstance(x, int) and 5 <= x <= 50,
            'bb_std': lambda x: isinstance(x, (int, float)) and 0.5 <= x <= 4.0,
            'rsi_period': lambda x: isinstance(x, int) and 2 <= x <= 30
        },
        output_components=['confirmed_long', 'confirmed_short', 'has_bullish_divergence', 'has_bearish_divergence', 'divergence_strength', 'bars_since_divergence'],
        description='BB + RSI divergence with proper feature dependencies'
    ),
    
    'bb_rsi_divergence_exact': FeatureRegistryEntry(
        name='bb_rsi_divergence_exact',
        required_params=[],
        optional_params={},  # All parameters hardcoded for exact pattern
        param_validators={},
        output_components=['value', 'signal', 'reason', 'stage', 'position_type', 'divergence_active', 'in_position', 'extremes_tracked'],
        description='EXACT implementation of profitable BB + RSI divergence pattern'
    ),
    
    'bb_rsi_divergence_self': FeatureRegistryEntry(
        name='bb_rsi_divergence_self',
        required_params=[],
        optional_params={},
        param_validators={},
        output_components=['value', 'signal', 'reason', 'stage', 'position_type', 'signals_generated', 'bar_count'],
        description='Self-contained BB + RSI divergence that computes indicators internally'
    ),
}


class FeatureValidationError(Exception):
    """Raised when feature validation fails."""
    pass


class ValidatedFeatures:
    """
    Container that guarantees all requested features exist.
    
    This replaces the old approach of runtime feature discovery
    with compile-time validation and guarantees.
    """
    
    def __init__(self, raw_features: Dict[str, Any], required_specs: List[FeatureSpec]):
        self.features = raw_features
        self.required_specs = required_specs
        self._validate_all_features()
    
    def _validate_all_features(self):
        """Strict validation - fail if ANY required feature missing."""
        missing_features = []
        
        for spec in self.required_specs:
            feature_name = spec.canonical_name
            if feature_name not in self.features:
                missing_features.append(feature_name)
        
        if missing_features:
            available = list(self.features.keys())
            raise FeatureValidationError(
                f"Missing required features: {missing_features}. "
                f"Available features: {available[:10]}..."  # Show first 10 for debugging
            )
    
    def __getitem__(self, feature_name: str) -> Any:
        """Dictionary-style access with guarantee feature exists."""
        return self.features[feature_name]
    
    def get(self, feature_name: str, default=None) -> Any:
        """Optional access for non-required features."""
        return self.features.get(feature_name, default)
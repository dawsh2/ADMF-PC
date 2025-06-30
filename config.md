# Declarative Regime-Adaptive Trading Strategies: YAML System Design

## Executive Summary

This document outlines a complete redesign of the YAML-based trading strategy system to support regime-adaptive strategies and complex conditional logic. The design enables users to create sophisticated trading strategies entirely through YAML configuration without writing any Python code. 

Key capabilities include:
- **Composable ensembles**: Combine any strategies into weighted ensembles
- **Regime-adaptive behavior**: Switch strategies based on market conditions
- **Conditional logic**: If-then-else strategy selection
- **Parameter optimization**: Automatic parameter space expansion with `--optimize` flag
- **Nested composition**: Ensembles within ensembles, unlimited depth

The goal is for users to express any trading logic declaratively in YAML, provided the underlying indicators and features exist in the system.

## Core Design Principles

### 1. **Clean Architecture First**
Legacy patterns and backward compatibility are explicitly rejected to prevent technical debt accumulation. All existing strategies must be migrated to the new system, ensuring new developers learn correct patterns from day one.

### 2. **Recursive Composability**
Every strategy, regardless of complexity, implements the same interface. Complex strategies are built by composing simpler strategies, creating a fractal architecture where any strategy can contain any other strategy type.

### 3. **Centralized Feature Computation**
The centralized computation system (where `sma(5)` is computed once and shared) extends to regime detectors and conditional evaluators, with strict validation ensuring all required features are available.

### 4. **Progressive Enhancement**
Users can start with simple indicator lists and gradually add regime detection, conditions, or decision trees only where needed. Configurations can evolve from simple to complex without structural rewrites.

### 5. **Zero-Code Strategy Composition**
**Users never need to write Python code to create new trading strategies.** All composition, regime adaptation, conditional logic, and ensemble creation is done purely through YAML configuration. If the required indicators and base strategies exist, any combination can be created declaratively.

## Strategy Type Hierarchy

### Base Strategy (Current System)
```yaml
strategy:
  indicators:
    - rsi(20, 80), weight: 0.33
    - macd(30, 60), weight: 0.33
    - sma_cross(5, 20), weight: 0.33
  weight_threshold: 0.66
```

### Ensemble Strategy
```yaml
strategy:
  type: ensemble
  strategies:
    - name: trend_follower
      weight: 0.4
      indicators:
        - sma_cross(10, 30), weight: 0.6
        - momentum(14), weight: 0.4
      weight_threshold: 0.5
    
    - name: mean_reverter
      weight: 0.3
      indicators:
        - rsi(20, 80), weight: 0.7
        - bb_squeeze(), weight: 0.3
      weight_threshold: 0.6
    
    - name: breakout_catcher
      weight: 0.3
      type: conditional
      condition: atr(14) > atr_sma(14, 20)
      if_true:
        indicators: [breakout(20), volume_spike(2.0)]
      if_false:
        indicators: [range_bound(20)]
  
  # How to combine signals from ensemble members
  combination: weighted_vote  # or 'average', 'majority', 'unanimous', 'ranked'
  
  # Optional: adaptive weighting
  adaptive_weights:
    method: performance_based  # Adjust weights based on recent performance
    lookback: 20
    update_frequency: daily
```

### Regime-Switching Strategy
```yaml
strategy:
  type: regime_switch
  detector: volatility_percentile(20, [30, 70])
  regimes:
    low:
      indicators:
        - sma_cross(20, 50), weight: 0.7
        - rsi(30, 70), weight: 0.3
      weight_threshold: 0.5
    medium:
      # Can reference another strategy type
      type: conditional
      condition: trend_strength(14) > 0.5
      if_true:
        indicators: [momentum(14), weight: 1.0]
      if_false:
        indicators: [mean_reversion(20), weight: 1.0]
    high:
      indicators:
        - rsi(20, 80), weight: 0.5
        - atr_stop(14), weight: 0.5
      weight_threshold: 0.7
```

### Multi-Regime Strategy
```yaml
strategy:
  type: multi_regime
  detectors:
    - name: volatility
      detector: volatility_regime(20)
      weight: 0.5
    - name: trend
      detector: trend_regime(50)
      weight: 0.5
  combination: weighted_vote  # or 'unanimous', 'majority'
  strategies:
    volatility:
      low:
        trend:
          bullish: { indicators: [sma_cross(10, 30), momentum(14)] }
          bearish: { indicators: [rsi(20, 80), mean_reversion(20)] }
      high:
        trend:
          bullish: { indicators: [breakout(20), atr_stop(14)] }
          bearish: { indicators: [volatility_squeeze(), rsi(15, 85)] }
```

### Conditional Strategy
```yaml
strategy:
  type: conditional
  conditions:
    - market_hours() == 'pre_market'
    - volatility_percentile(5) > 80
  operator: AND  # or 'OR'
  if_true:
    indicators: [gap_fade(0.02), weight: 1.0]
  if_false:
    # Nested conditional
    type: conditional
    condition: day_of_week() in ['Mon', 'Fri']
    if_true:
      indicators: [rsi(25, 75), sma_cross(5, 20)]
    if_false:
      indicators: [standard_strategy()]
```

### Decision Tree Strategy
```yaml
strategy:
  type: decision_tree
  root:
    split: volatility_percentile(20)
    branches:
      - condition: "< 30"
        node:
          split: trend_direction(50)
          branches:
            - condition: "== 'bullish'"
              strategy: { indicators: [trend_follow_low_vol()] }
            - condition: "== 'bearish'"
              strategy: { indicators: [short_trend_low_vol()] }
            - condition: "== 'neutral'"
              strategy: { indicators: [mean_reversion(20)] }
      - condition: ">= 30 and < 70"
        strategy: { ref: 'strategies.balanced_approach' }
      - condition: ">= 70"
        node:
          split: time_until_close()
          branches:
            - condition: "< 30"  # Last 30 minutes
              strategy: { indicators: [close_position()] }
            - condition: ">= 30"
              strategy: { indicators: [volatility_harvest()] }
```

## Advanced Features

### Natural Strategy Composition

The system supports multiple ways to compose strategies, making complex combinations feel natural and readable:

#### Using YAML Anchors and Merging
```yaml
# Define reusable strategy components
definitions:
  base_strategies:
    trend_follow: &trend_follow
      indicators:
        - sma_cross(10, 30), weight: 0.6
        - momentum(14), weight: 0.4
      weight_threshold: 0.5
    
    mean_revert: &mean_revert
      indicators:
        - rsi(20, 80), weight: 0.7
        - bb_squeeze(), weight: 0.3
      weight_threshold: 0.6
    
    volatility_capture: &vol_capture
      indicators:
        - atr_breakout(14), weight: 0.5
        - volume_spike(2.0), weight: 0.5

# Compose them naturally
strategy:
  type: ensemble
  strategies:
    # Direct inclusion
    - <<: *trend_follow
      name: primary_trend
      weight: 0.5
    
    # Merge and override
    - <<: *mean_revert
      name: adjusted_reversion
      weight: 0.3
      weight_threshold: 0.7  # Override the threshold
    
    # Composition within regime switching
    - name: adaptive_vol
      weight: 0.2
      type: regime_switch
      detector: volatility_regime(20)
      regimes:
        low: *mean_revert
        high: *vol_capture
```

#### Strategy Pipelines and Modular Composition
```yaml
strategy:
  type: pipeline
  stages:
    # First stage: Get signals from multiple strategies
    - stage: generate_signals
      parallel:
        - *trend_follow
        - *mean_revert
        - { indicators: [vwap_deviation(20)] }
    
    # Second stage: Filter based on regime
    - stage: regime_filter
      type: conditional
      condition: market_regime() != 'choppy'
      pass_through: true
      filter_weight: 0.5
    
    # Third stage: Risk adjustment
    - stage: risk_adjust
      type: position_sizer
      base_on: portfolio_heat(0.06)
```

#### Nested Ensembles for Complex Strategies
```yaml
strategy:
  type: ensemble
  strategies:
    # Trend ensemble
    - name: trend_ensemble
      weight: 0.4
      type: ensemble
      strategies:
        - { indicators: [sma_cross(5, 20)], weight: 0.3 }
        - { indicators: [sma_cross(10, 30)], weight: 0.4 }
        - { indicators: [sma_cross(20, 50)], weight: 0.3 }
      combination: ranked  # Use ranking instead of weighted average
    
    # Momentum ensemble with regime adaptation
    - name: momentum_ensemble  
      weight: 0.3
      type: regime_switch
      detector: trend_strength(14)
      regimes:
        strong:
          type: ensemble
          strategies:
            - { indicators: [momentum(7)], weight: 0.5 }
            - { indicators: [momentum(14)], weight: 0.5 }
        weak:
          indicators: [mean_reversion(20)]
    
    # Volatility strategies
    - name: volatility_strategies
      weight: 0.3
      <<: *vol_capture
```

### Strategy References and Libraries
```yaml
# Define reusable components
definitions:
  detectors:
    vol_regime: &vol_regime
      volatility_percentile(20, [30, 70])
    
    trend_regime: &trend_regime
      composite:
        - sma_slope(50) 
        - adx(14) > 25
  
  strategies:
    conservative: &conservative
      indicators:
        - rsi(30, 70), weight: 0.4
        - sma_cross(20, 50), weight: 0.6
      weight_threshold: 0.8
    
    aggressive: &aggressive
      indicators:
        - momentum(14), weight: 0.5
        - breakout(20), weight: 0.5
      weight_threshold: 0.5

# Use references in main strategy
strategy:
  type: regime_switch
  detector: *vol_regime
  regimes:
    low: *conservative
    medium:
      type: regime_switch
      detector: *trend_regime
      regimes:
        trending: *aggressive
        ranging: *conservative
    high:
      indicators: [atr_stop(14), weight: 1.0]
```

### Inline Adaptations
```yaml
# Simple strategy with inline regime adaptation
strategy:
  indicators:
    - rsi(20, 80), weight: 0.33
    - macd(30, 60), weight: 0.33
    - sma_cross(5, 20), weight: 0.33
  weight_threshold: 0.66
  # Optional adaptation without changing type
  adapt:
    - when: volatility_percentile(20) > 70
      set:
        weight_threshold: 0.8
        weights: [0.25, 0.25, 0.5]  # More weight on SMA cross
    - when: market_hours() == 'after_hours'
      set:
        enabled: false
```

### Dynamic Parameters
```yaml
strategy:
  indicators:
    - rsi(oversold, overbought), weight: 0.5
    - atr_stop(period), weight: 0.5
  parameters:
    oversold:
      formula: "20 + 10 * (1 - volatility_percentile(20) / 100)"
    overbought:
      formula: "80 - 10 * (1 - volatility_percentile(20) / 100)"
    period:
      formula: "max(10, min(20, 14 * volatility_ratio(20)))"
  weight_threshold: 0.6
```

## Implementation Architecture

### Strategy Interface
All strategy types implement a common interface:

```python
class Strategy:
    def evaluate(self, market_data: MarketData, context: TradingContext) -> SignalSet:
        """Returns trading signals based on current market state"""
        pass
    
    def get_required_features(self) -> List[Feature]:
        """Returns all features needed by this strategy and its children"""
        pass
    
    def get_parameter_space(self) -> Dict[str, ParameterSpec]:
        """Returns parameter specifications for optimization"""
        pass
```

### Parameter Space Specification

Each strategy defines its optimizable parameter space:

```python
@dataclass
class ParameterSpec:
    """Specification for an optimizable parameter."""
    param_type: str  # 'int', 'float', 'categorical'
    default: Any
    range: Optional[Tuple[Any, Any]] = None  # For numeric types
    choices: Optional[List[Any]] = None      # For categorical
    constraint: Optional[str] = None         # e.g., "fast_period < slow_period"
    description: Optional[str] = None

# Example strategy implementation
def get_parameter_space(self) -> Dict[str, ParameterSpec]:
    return {
        'fast_period': ParameterSpec(
            param_type='int',
            default=10,
            range=(3, 50),
            constraint='fast_period < slow_period',
            description='Fast moving average period'
        ),
        'slow_period': ParameterSpec(
            param_type='int',
            default=30,
            range=(10, 200),
            description='Slow moving average period'
        )
    }
```

### Feature Registry Extension
Your centralized computation system extends to regime detectors:

```python
class FeatureRegistry:
    def register_indicator(self, indicator_spec: str) -> Feature
    def register_detector(self, detector_spec: str) -> RegimeDetector
    def register_condition(self, condition_spec: str) -> Condition
    
    def compute_all(self, market_data: MarketData) -> FeatureCache
```

### Parser Architecture
The parser uses a factory pattern to create appropriate strategy objects:

```python
class StrategyParser:
    def parse(self, config: dict) -> Strategy:
        # Auto-detect type based on keys
        if 'type' in config:
            return self._parse_typed_strategy(config)
        elif 'indicators' in config:
            return self._parse_indicator_strategy(config)
        elif 'strategies' in config:
            # Implicit ensemble when multiple strategies without explicit type
            return self._parse_ensemble_strategy(config)
        else:
            raise ValueError("Unknown strategy format")
    
    def _parse_typed_strategy(self, config: dict) -> Strategy:
        strategy_type = config['type']
        if strategy_type == 'ensemble':
            return self._parse_ensemble(config)
        elif strategy_type == 'regime_switch':
            return self._parse_regime_switch(config)
        elif strategy_type == 'conditional':
            return self._parse_conditional(config)
        elif strategy_type == 'pipeline':
            return self._parse_pipeline(config)
        # ... etc
```

### Recursive Parameter Extraction for Optimization

The topology builder handles recursive parameter extraction for ensemble optimization:

```python
class EnsembleParameterExtractor:
    """Extracts parameter spaces recursively from ensemble configurations."""
    
    def extract_parameter_space(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively extract all parameter spaces from ensemble configuration.
        
        Returns hierarchical parameter space that preserves ensemble structure
        while allowing optimization of all nested strategies.
        """
        param_space = {}
        
        # Handle optimization space overrides from YAML
        overrides = config.get('optimization_space', {})
        
        if config.get('type') == 'ensemble' or 'strategies' in config:
            # Ensemble case - recurse into each strategy
            for idx, strategy_config in enumerate(config.get('strategies', [])):
                strategy_name = strategy_config.get('name', f'strategy_{idx}')
                
                if strategy_config.get('type') == 'ensemble':
                    # Nested ensemble - recursive extraction
                    param_space[strategy_name] = self.extract_parameter_space(strategy_config)
                else:
                    # Atomic strategy - get its parameter space
                    param_space[strategy_name] = self._get_atomic_strategy_space(
                        strategy_config, 
                        overrides.get(strategy_name, {})
                    )
        else:
            # Single strategy - get its parameter space
            param_space = self._get_atomic_strategy_space(config, overrides)
            
        return param_space
    
    def _get_atomic_strategy_space(self, 
                                   strategy_config: Dict[str, Any],
                                   overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameter space for atomic (non-ensemble) strategy."""
        strategy_type = strategy_config['type']
        
        # Get default parameter space from strategy decorator metadata
        strategy_info = get_component_registry().get_component(strategy_type)
        default_space = strategy_info.metadata.get('parameter_space', {})
        
        # Apply YAML overrides
        final_space = {}
        for param_name, param_spec in default_space.items():
            if param_name in overrides:
                # Override can be a list (explicit values) or dict (spec override)
                override_value = overrides[param_name]
                if isinstance(override_value, list):
                    # Explicit value list
                    final_space[param_name] = override_value
                elif isinstance(override_value, dict):
                    # Spec override - merge with defaults
                    final_space[param_name] = {**param_spec.__dict__, **override_value}
                else:
                    # Single value - treat as fixed parameter
                    final_space[param_name] = [override_value]
            elif overrides.get('use_defaults', True):
                # Use default parameter space
                final_space[param_name] = self._sample_from_spec(
                    param_spec,
                    overrides.get('sample_points', 5)
                )
                
        return final_space
```

### YAML Configuration with Optimization Space Overrides

```yaml
# Ensemble with optimization configuration
strategy:
  type: ensemble
  name: adaptive_ensemble
  
  # Optional: Override default parameter spaces for optimization
  optimization_space:
    # Strategy-specific overrides
    trend_follower:
      fast_period: [5, 10, 20, 30]  # Explicit values
      slow_period: 
        range: [20, 100]            # Override range
        sample_points: 4            # Sample 4 points
        
    momentum_strategy:
      use_defaults: true            # Use all defaults
      sample_points: 3              # But only sample 3 points
      
    # For nested ensembles, use hierarchical paths
    "mean_reversion_ensemble.rsi_bands":
      rsi_period: [14, 21, 28]
      
  # The actual ensemble composition
  strategies:
    - name: trend_follower
      type: sma_crossover
      weight: 0.3
      params:
        fast_period: 10  # Current value
        slow_period: 30
        
    - name: momentum_strategy  
      type: momentum_breakout
      weight: 0.3
      params:
        momentum_period: 14
        breakout_threshold: 0.01
        
    - name: mean_reversion_ensemble
      type: ensemble
      weight: 0.4
      strategies:
        - name: rsi_bands
          type: rsi_bands
          weight: 0.6
          params:
            rsi_period: 14
            oversold: 30
            overbought: 70
            
        - name: bollinger
          type: bollinger_bands
          weight: 0.4
          params:
            period: 20
            num_std: 2
```

### Topology Builder Integration

The topology builder uses the parameter extractor during optimization workflows:

```python
class TopologyBuilder:
    def __init__(self):
        self.parameter_extractor = EnsembleParameterExtractor()
        
    def build_optimization_topology(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build topology for parameter optimization workflow."""
        # Extract full parameter space recursively
        param_space = self.parameter_extractor.extract_parameter_space(
            config['strategy']
        )
        
        # Expand parameter combinations
        expanded_configs = self._expand_parameter_combinations(
            config['strategy'],
            param_space
        )
        
        # Build topology with all parameter variants
        return self._build_expanded_topology(expanded_configs)
```

### Natural Composition Examples

The parser supports intuitive shortcuts for common patterns:

```python
# These three are equivalent:

# Explicit ensemble
config1 = {
    'type': 'ensemble',
    'strategies': [
        {'indicators': ['rsi(20, 80)']},
        {'indicators': ['sma_cross(10, 30)']}
    ]
}

# Implicit ensemble (no type specified, but has strategies list)
config2 = {
    'strategies': [
        {'indicators': ['rsi(20, 80)']},
        {'indicators': ['sma_cross(10, 30)']}
    ]
}

# Compact notation for simple ensembles
config3 = {
    'strategies': [
        'rsi(20, 80)',
        'sma_cross(10, 30)'
    ]
}
```

## Implementation Philosophy

### Backward Compatibility vs Progressive Enhancement

**Backward Compatibility (REJECTED)**: Supporting old code formats and legacy patterns to avoid migration work.
- ❌ Keeps technical debt in the system
- ❌ New developers learn bad patterns  
- ❌ Prevents architectural improvements

**Progressive Enhancement (EMBRACED)**: Allowing configurations to evolve from simple to complex structures.
- ✅ Simple strategies stay simple: `indicators: [rsi(20, 80)]`
- ✅ Can add regime detection: `type: regime_switch` + `detector: volatility_regime(20)`
- ✅ Can add nesting: `type: ensemble` + `strategies: [...]`
- ✅ Clean evolution path without structural rewrites

### Migration Path - Clean Slate Approach

### Phase 1: System Replacement (Week 1)
- Replace feature inference system with validation-first approach
- Implement new strategy parser with explicit declarations
- Delete legacy code patterns to prevent confusion

### Phase 2: Strategy Migration (Week 2)  
- Convert all existing strategies to new format
- Standardize parameter naming across all strategies
- Validate all configurations work with new system

### Phase 3: Advanced Composition Features (Week 3)
- Implement regime_switch, conditional, ensemble, and decision_tree types
- Add strategy nesting and composition support
- Enable progressive enhancement from simple to complex

### Phase 4: Documentation and Training (Week 4)
- Document progressive enhancement patterns
- Create examples showing evolution from simple to complex
- Train team on new composition capabilities

## Example: Progressive Enhancement

Starting simple:
```yaml
# Version 1: Basic strategy
strategy:
  indicators:
    - rsi(20, 80), weight: 0.5
    - sma_cross(10, 30), weight: 0.5
  weight_threshold: 0.6
```

Adding ensemble:
```yaml
# Version 2: Ensemble of simple strategies
strategy:
  type: ensemble
  strategies:
    - indicators: [rsi(20, 80), sma_cross(10, 30)]
      weight: 0.5
    - indicators: [momentum(14), breakout(20)]
      weight: 0.5
  combination: weighted_vote
```

Adding regime awareness:
```yaml
# Version 3: Regime-aware ensemble
strategy:
  type: ensemble
  strategies:
    - name: trend_component
      weight: 0.4
      type: regime_switch
      detector: trend_strength(50)
      regimes:
        strong: { indicators: [momentum(14), sma_cross(10, 30)] }
        weak: { indicators: [mean_reversion(20)] }
    
    - name: volatility_component
      weight: 0.6
      type: regime_switch
      detector: volatility_regime(20)
      regimes:
        low: { indicators: [rsi(30, 70), bb_squeeze()] }
        high: { indicators: [atr_stop(14), breakout(20)] }
```

Full composition:
```yaml
# Version 4: Full compositional power
strategy:
  type: ensemble
  strategies:
    # Trend following ensemble
    - name: trend_suite
      weight: 0.3
      type: ensemble
      strategies:
        - indicators: [sma_cross(5, 20)]
        - indicators: [sma_cross(10, 30)]
        - indicators: [ema_cross(12, 26)]
      combination: majority
    
    # Adaptive mean reversion
    - name: adaptive_reversion
      weight: 0.3
      type: conditional
      condition: market_efficiency(20) > 0.7
      if_true:
        type: ensemble
        strategies:
          - indicators: [rsi(14, 86)]
          - indicators: [bb_reversion(20, 2)]
        combination: unanimous  # Both must agree
      if_false:
        indicators: [simple_mean_reversion(20)]
    
    # Regime-based strategy
    - name: regime_adaptive
      weight: 0.4
      type: regime_switch
      detector: composite_regime(trend, volatility, correlation)
      regimes:
        trending_calm: { ref: 'libraries.trend_strategies.smooth_trend' }
        trending_volatile: { ref: 'libraries.trend_strategies.robust_trend' }
        ranging_calm: { ref: 'libraries.mean_reversion.tight_range' }
        ranging_volatile: { ref: 'libraries.volatility.squeeze_play' }
  
  # Ensemble-level adaptation
  adaptive_weights:
    method: sharpe_optimization
    lookback: 30
    constraints:
      min_weight: 0.1
      max_weight: 0.6
```

### Composition Patterns

#### Mix and Match Pattern
```yaml
# Freely combine different strategy types
strategy:
  type: ensemble
  strategies:
    - indicators: [simple_momentum(14)]  # Basic strategy
    - type: regime_switch               # Regime adaptive
      detector: volatility_regime(20)
      regimes:
        low: { indicators: [mean_reversion(20)] }
        high: { indicators: [breakout(20)] }
    - type: conditional                 # Conditional
      condition: market_hours() == 'regular'
      if_true: { indicators: [vwap_deviation()] }
      if_false: { indicators: [overnight_momentum()] }
    - type: ensemble                    # Nested ensemble
      strategies:
        - indicators: [rsi(14, 86)]
        - indicators: [rsi(28, 72)]
```

#### Strategy Factory Pattern
```yaml
# Define strategy templates with parameters
templates:
  trend_strategy:
    params: [fast, slow]
    definition:
      indicators:
        - sma_cross(${fast}, ${slow}), weight: 0.6
        - momentum(${fast}), weight: 0.4

# Use templates in ensemble
strategy:
  type: ensemble
  strategies:
    - from_template: trend_strategy
      params: { fast: 5, slow: 20 }
      weight: 0.33
    - from_template: trend_strategy
      params: { fast: 10, slow: 30 }
      weight: 0.33
    - from_template: trend_strategy
      params: { fast: 20, slow: 50 }
      weight: 0.34
```

## Benefits

1. **No Breaking Changes**: Existing strategies work unchanged
2. **Gradual Adoption**: Add complexity only where needed
3. **Reusability**: Define strategies once, use everywhere
4. **Testability**: Each component can be tested in isolation
5. **Performance**: Centralized computation prevents redundant calculations
6. **Flexibility**: Mix and match different strategy types freely

## Next Steps

1. Prototype the parser extensions with regime_switch type
2. Design the regime detector library interface  
3. Create a test suite with progressively complex strategies
4. Benchmark performance with deeply nested strategies
5. Document best practices for strategy composition

## Zero-Code Trading: Complete Examples

### Example 1: Market-Adaptive Ensemble
This sophisticated strategy adapts to market conditions without any Python code:

```yaml
strategy:
  type: ensemble
  name: market_adaptive_system
  
  strategies:
    # Trend following component - active in trending markets
    - name: trend_component
      type: regime_switch
      weight: 0.4
      detector: trend_strength(50)
      regimes:
        strong_up:
          type: ensemble
          strategies:
            - { type: sma_crossover, params: {fast: 10, slow: 30} }
            - { type: momentum, params: {period: 14} }
            - { type: breakout, params: {lookback: 20} }
          combination: majority
        strong_down:
          type: inverse
          base_strategy: { ref: trend_component.regimes.strong_up }
        weak:
          type: mean_reversion
          params: {period: 20, threshold: 2.0}
    
    # Volatility harvesting - active in high volatility
    - name: volatility_component  
      type: conditional
      weight: 0.3
      condition: volatility_percentile(20) > 70
      if_true:
        type: ensemble
        strategies:
          - { type: bollinger_squeeze, params: {period: 20, num_std: 2} }
          - { type: atr_breakout, params: {period: 14, multiplier: 1.5} }
      if_false:
        type: disabled  # No trading in low volatility
    
    # Time-based component
    - name: time_component
      type: decision_tree
      weight: 0.3
      root:
        split: market_hours()
        branches:
          - condition: "== 'pre_market'"
            strategy: { type: gap_fade, params: {threshold: 0.02} }
          - condition: "== 'regular'"
            node:
              split: time_until_close()
              branches:
                - condition: "< 30"
                  strategy: { type: close_positions }
                - condition: ">= 30"
                  strategy: { ref: strategies.intraday_momentum }
          - condition: "== 'after_hours'"
            strategy: { type: overnight_carry }
```

### Example 2: Self-Optimizing Strategy
This configuration automatically adjusts parameters based on recent performance:

```yaml
strategy:
  type: adaptive_ensemble
  
  # Base strategies with performance tracking
  strategies:
    - name: fast_momentum
      type: momentum
      params: {period: 7}
      track_performance: true
      
    - name: medium_momentum  
      type: momentum
      params: {period: 14}
      track_performance: true
      
    - name: slow_momentum
      type: momentum  
      params: {period: 28}
      track_performance: true
  
  # Adaptive weight allocation
  adaptive_weights:
    method: sharpe_maximization
    lookback: 30
    update_frequency: daily
    constraints:
      min_weight: 0.1
      max_weight: 0.6
      sum_to_one: true
  
  # Regime override - in trending markets, prefer faster momentum
  regime_overrides:
    - detector: trend_strength(50)
      condition: "> 0.7"
      weight_adjustments:
        fast_momentum: 1.5   # 50% weight boost
        slow_momentum: 0.5   # 50% weight reduction
```

### Example 3: Complete Trading System
A production-ready system combining multiple approaches:

```yaml
# Dual-purpose configuration
name: complete_trading_system
symbols: ["SPY", "QQQ", "IWM"]
timeframes: ["5m"]

# For optimization runs
parameter_space:
  indicators:
    crossover: "*"
    momentum: ["macd_crossover", "rsi_momentum", "roc_trend"]
    oscillator: "*"
    structure: ["pivot_points", "support_resistance_breakout"]
  
  classifiers:
    - market_regime_classifier
    - volatility_regime_classifier
    - trend_strength_classifier

# Production strategy
strategy:
  type: master_ensemble
  
  # Risk management wrapper
  risk_management:
    max_positions: 3
    position_sizing: kelly_criterion
    stop_loss: atr_based
    take_profit: dynamic
  
  # Main strategy components  
  strategies:
    # Long-term trend follower
    - name: strategic_trend
      type: regime_switch
      weight: 0.25
      detector: market_regime_classifier(20, 1.5)
      regimes:
        bull_market:
          type: buy_and_hold
          filters: [above_sma(200)]
        bear_market:
          type: inverse
          base_strategy: { type: trend_follow, params: {period: 50} }
        neutral:
          type: disabled
    
    # Tactical allocation
    - name: tactical_trading
      type: multi_regime
      weight: 0.5
      detectors:
        - { name: trend, detector: trend_strength(20), weight: 0.6 }
        - { name: volatility, detector: volatility_regime(20), weight: 0.4 }
      strategies:
        trend:
          strong:
            volatility:
              low: { type: sma_crossover, params: {fast: 10, slow: 30} }
              high: { type: breakout, params: {period: 20, confirm: true} }
          weak:
            volatility:
              low: { type: mean_reversion, params: {period: 20} }
              high: { type: straddle, params: {strikes: 2} }
    
    # Opportunistic trades
    - name: opportunity_catcher
      type: scanner
      weight: 0.25
      scan_for:
        - { pattern: golden_cross, action: { type: momentum, params: {period: 14} } }
        - { pattern: oversold_bounce, action: { type: rsi_reversal, params: {period: 14} } }
        - { pattern: breakout_setup, action: { type: breakout, params: {confirm: true} } }
      max_concurrent: 2
  
  # Performance-based rebalancing
  rebalancing:
    method: risk_parity
    frequency: weekly
    constraints:
      turnover_limit: 0.2
      min_weight: 0.1
```

These examples demonstrate that users can create arbitrarily complex trading systems without writing any code. The YAML configuration supports:
- Unlimited nesting of strategies
- Multiple types of regime detection
- Conditional logic and decision trees
- Performance-based adaptation
- Risk management integration
- Multi-timeframe and multi-asset strategies

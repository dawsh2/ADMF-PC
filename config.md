# Declarative Regime-Adaptive Trading Strategies: YAML System Design

## Executive Summary

This document outlines a complete redesign of the YAML-based trading strategy system to support regime-adaptive strategies and complex conditional logic. The design prioritizes clean architecture and maintainability over backward compatibility, requiring migration of existing strategies to eliminate technical debt and enable robust future development.

## Core Design Principles

### 1. **Clean Architecture First**
Legacy patterns and backward compatibility are explicitly rejected to prevent technical debt accumulation. All existing strategies must be migrated to the new system, ensuring new developers learn correct patterns from day one.

### 2. **Recursive Composability**
Every strategy, regardless of complexity, implements the same interface. Complex strategies are built by composing simpler strategies, creating a fractal architecture where any strategy can contain any other strategy type.

### 3. **Centralized Feature Computation**
The centralized computation system (where `sma(5)` is computed once and shared) extends to regime detectors and conditional evaluators, with strict validation ensuring all required features are available.

### 4. **Progressive Enhancement**
Users can start with simple indicator lists and gradually add regime detection, conditions, or decision trees only where needed. Configurations can evolve from simple to complex without structural rewrites.

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

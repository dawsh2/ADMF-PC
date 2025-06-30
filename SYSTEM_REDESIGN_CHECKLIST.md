# ADMF-PC System Redesign Checklist

## Overview

This document consolidates the feature system redesign and config-driven architecture improvements for ADMF-PC. It serves as a comprehensive checklist and documentation of completed work, enabling any context-unaware agent to understand and continue the implementation.

## Core Problem Statement

The current system has two critical issues:

1. **Feature Inference is Fragile**: Strategies use string formatting, parameter guessing, and runtime discovery that leads to silent failures when features are missing. The system conflates feature discovery (what features do I need?) with feature validation (do those features exist?).

2. **Strategy Composition is Hardcoded**: Creating ensemble strategies requires new Python classes. The system cannot compose strategies from configuration alone, limiting flexibility and requiring code changes for new combinations.

## Architectural Solution

### 1. Separation of Concerns for Features

**Dynamic Discovery** → **Deterministic Validation** → **Guaranteed Execution**

- **Discovery Phase** (strategy loading): Transform parameters into concrete feature requirements
- **Validation Phase** (pre-execution): Verify all requirements can be satisfied  
- **Execution Phase** (runtime): Access guaranteed features with confidence

### 2. Protocol + Composition Architecture

- Strategies are composable by design
- No special code for ensembles - they naturally aggregate sub-strategy requirements
- Configuration-driven composition without new Python classes

## Implementation Status

### ✅ Completed Components

#### 1. Feature Specification System
**Files Created:**
- `src/core/features/feature_spec.py` - Core FeatureSpec class with deterministic naming
- `src/core/coordinator/feature_discovery.py` - Discovery system for strategies

**Key Features:**
```python
# Static requirements
@strategy(
    name='rsi_oversold',
    required_features=[
        FeatureSpec('rsi', {'period': 14})
    ]
)

# Dynamic discovery
@strategy(
    name='sma_crossover',
    feature_discovery=lambda params: [
        FeatureSpec('sma', {'period': params['fast_period']}),
        FeatureSpec('sma', {'period': params['slow_period']})
    ]
)
```

#### 2. Structured Event System
**Files Created:**
- `src/core/events/structured_events.py` - Structured signal events with embedded parameters
- `src/core/coordinator/subscription_helpers.py` - Portfolio subscription utilities

**Key Improvements:**
```python
# Old: "SPY_1m_sma_crossover_grid_5_20"
# New: Structured data
{
    'symbol': 'SPY',
    'timeframe': '1m', 
    'strategy_type': 'sma_crossover',
    'parameters': {'fast_period': 5, 'slow_period': 20}
}
```

#### 3. Updated Topology Builder
**Files Modified:**
- `src/core/coordinator/topology.py` - Now uses FeatureDiscovery, no legacy fallback

**Key Changes:**
- Removed legacy feature inference fallback
- Forces migration to new system (throws clear errors for unmigrated strategies)
- Uses structured event subscriptions

#### 4. Example Migrations
**Files Created:**
- `src/strategy/strategies/indicators/crossovers_migrated.py` - Shows all migration patterns

**Demonstrates:**
- Static feature requirements
- Dynamic feature discovery
- Multi-output features (MACD components)
- Hybrid approaches

### 🚧 Remaining Work

#### Phase 1: Delete Legacy Code (High Priority)
- [x] Delete `_infer_features_from_strategies()` in topology.py (lines 1020-1200+) ✅
- [x] Delete `_create_feature_config_from_id()` in topology.py ✅
- [x] Delete parameter mapping code in `feature_inference_helpers.py` ✅
- [x] Delete feature guessing logic in `strategy/state.py` ✅

#### Phase 2: Migrate All Strategies (High Priority)
- [x] Update all indicator strategies to use FeatureSpec ✅
  - [x] crossovers.py (9 strategies) ✅
  - [x] oscillators.py (8 strategies) ✅
  - [x] momentum.py (7 strategies) ✅
  - [x] trend.py (5 strategies) ✅
  - [x] volatility.py (5 strategies) ✅
  - [x] volume.py (5 strategies) ✅
  - [x] structure.py (9 strategies) ✅
- [x] Remove `param_feature_mapping` from all indicator strategies ✅
- [x] Add `parameter_space` to strategy decorator ✅
- [x] Add default parameter spaces to all indicator strategies ✅
- [x] Standardize parameter names across strategies ✅

#### Phase 3: Config-Driven Optimization & Ensembles (High Priority)
- [x] Implement parameter space expansion in ConfigHandler ✅
  - [x] Support `parameter_space: indicators: *` expansion ✅
  - [x] Support category-based expansion (crossover: *, momentum: [...]) ✅
  - [x] Support explicit strategy lists ✅
- [x] Create EnsembleParameterExtractor for recursive parameter extraction ✅
- [x] Implement --optimize flag behavior ✅
  - [x] Expand parameter_space section when flag is present ✅
  - [x] Use strategy section when flag is absent ✅
- [x] Implement compositional strategy system (better than universal ensemble) ✅
- [x] Add ensemble composition examples to config/ ✅
- [x] Update documentation with compositional strategy design ✅

#### Phase 4: Enhanced Event Routing (Medium Priority)
- [x] Update MultiStrategyTracer to use strategy_type instead of full ID ✅
- [x] Modify portfolio containers to use structured subscriptions ✅
- [x] Update analytics queries to handle new event structure ✅

## Migration Guide

### For Strategy Developers

**Old Pattern (MUST MIGRATE):**
```python
@strategy(
    name='macd_crossover',
    feature_config=['macd'],
    param_feature_mapping={
        'fast_period': 'macd_{fast_period}_{slow_period}_{signal_period}',
        'slow_period': 'macd_{fast_period}_{slow_period}_{signal_period}',
        'signal_period': 'macd_{fast_period}_{slow_period}_{signal_period}'
    }
)
def macd_crossover(features, bar, params):
    # Complex string formatting to find features
    macd_key = f"macd_{params['fast_period']}_{params['slow_period']}_{params['signal_period']}"
    macd_data = features.get(macd_key)  # Might be None!
    if macd_data is None:
        return None  # Silent failure
```

**New Pattern:**
```python
@strategy(
    name='macd_crossover',
    feature_discovery=lambda params: [
        FeatureSpec('macd', {
            'fast_period': params['fast_period'],
            'slow_period': params['slow_period'],
            'signal_period': params['signal_period']
        }, 'macd'),
        FeatureSpec('macd', {
            'fast_period': params['fast_period'],
            'slow_period': params['slow_period'],
            'signal_period': params['signal_period']
        }, 'signal')
    ]
)
def macd_crossover(features: ValidatedFeatures, bar, params):
    # Exact, deterministic names - guaranteed to exist
    macd_line = features[f"macd_{params['fast_period']}_{params['slow_period']}_{params['signal_period']}_macd"]
    signal_line = features[f"macd_{params['fast_period']}_{params['slow_period']}_{params['signal_period']}_signal"]
    # No error handling needed!
```

### For Config Authors

**Unified Config with Dual Purpose:**
```yaml
# unified_config.yaml
name: adaptive_trading_system
symbols: ["SPY"]
timeframes: ["1m"]

# Parameter space defines search space for optimization (--optimize flag)
parameter_space:
  indicators: "*"  # Expand all indicator strategies with default params
  # Or be selective:
  # indicators:
  #   crossover: "*"     # All crossover strategies
  #   momentum: ["macd_crossover", "roc_trend"]  # Specific strategies
  #   oscillator: "*"   # All oscillator strategies
  
  classifiers:
    - multi_timeframe_trend_classifier
    - market_regime_classifier

# Strategy defines production ensemble (no --optimize flag)
strategy:
  type: universal_ensemble
  strategies:
    - name: trend
      type: sma_crossover
      weight: 0.3
      params: {fast_period: 10, slow_period: 30}
      
    - name: momentum
      type: rsi_threshold
      weight: 0.3
      params: {period: 14, threshold: 50}
      
    - name: mean_revert
      type: universal_ensemble  # Nested ensemble!
      weight: 0.4
      strategies:
        - type: rsi_bands
          weight: 0.6
          params: {rsi_period: 14, oversold: 30, overbought: 70}
        - type: bollinger_bands
          weight: 0.4
          params: {period: 20, num_std: 2}
```

**Execution Behavior:**
```bash
# Optimization mode: expands parameter_space
python run.py --config unified_config.yaml --optimize

# Production mode: uses exact strategy specification
python run.py --config unified_config.yaml
```

## Parameter Space Design

### Strategy Parameter Space Definition
Each strategy defines its default parameter space in the decorator:
```python
@strategy(
    name='sma_crossover',
    feature_discovery=lambda params: [
        FeatureSpec('sma', {'period': params['fast_period']}),
        FeatureSpec('sma', {'period': params['slow_period']})
    ],
    parameter_space={
        'fast_period': {'type': 'int', 'range': [5, 50], 'default': 10},
        'slow_period': {'type': 'int', 'range': [20, 200], 'default': 30}
    }
)
```

### Config Parameter Space Expansion
The `parameter_space` section in YAML supports concise expansion patterns:
```yaml
parameter_space:
  # Expand all indicators with their default parameter spaces
  indicators: "*"
  
  # Category-based expansion
  indicators:
    crossover: "*"        # All crossover strategies
    momentum: ["macd", "rsi"]  # Specific momentum strategies
    
  # Explicit parameter overrides
  strategies:
    - type: sma_crossover
      param_overrides:
        fast_period: [5, 10, 20]
        slow_period: [30, 50, 100]
```

### Recursive Ensemble Parameter Extraction
Ensemble strategies automatically aggregate parameter spaces from sub-strategies:
```python
class EnsembleParameterExtractor:
    def extract_parameters(self, ensemble_config):
        """Recursively extract all parameters from ensemble and sub-strategies"""
        all_params = {}
        for sub_strategy in ensemble_config['strategies']:
            if sub_strategy['type'] == 'universal_ensemble':
                # Recursive extraction
                sub_params = self.extract_parameters(sub_strategy)
            else:
                # Get default parameter space from strategy
                sub_params = get_strategy_param_space(sub_strategy['type'])
            # Namespace parameters to avoid conflicts
            for param, spec in sub_params.items():
                all_params[f"{sub_strategy['name']}.{param}"] = spec
        return all_params
```

## Key Design Principles

### 1. Fail Fast, Fail Clear
- Validation errors at strategy load time, not runtime
- Clear error messages: "Strategy 'sma_crossover' uses legacy feature system. Must be migrated to use FeatureSpec"
- No silent failures or fallbacks

### 2. Explicit Over Implicit
- Feature requirements declared explicitly
- No guessing parameter names or trying alternatives
- Canonical naming: `indicator_param1_param2_component`

### 3. Composition Over Inheritance
- Strategies compose naturally without special handling
- Ensembles are just strategies that aggregate other strategies
- No "enhanced" or "improved" versions - use composition

### 4. Configuration Over Code
- New strategy combinations via config
- Parameter optimization without code changes
- Recursive discovery enables nested ensembles

## Testing Strategy

### 1. Feature System Tests
```bash
# Run migrated strategies
python main.py --config config/test_migrated_strategies.yaml

# Verify legacy strategies fail with clear errors
python main.py --config config/complete_grid_search.yaml  # Should fail
```

### 2. Event System Tests
```bash
# Test structured events
python examples/structured_events_example.py

# Verify subscriptions work
python main.py --config config/test_portfolio_subscriptions.yaml
```

### 3. Integration Tests
- Create ensemble via config only
- Run grid search with new event format
- Verify analytics work with structured data

## Success Criteria

1. **No Silent Failures**: All feature mismatches cause immediate, clear errors
2. **Clean Strategy IDs**: No more `strategy_grid_5_20_10_30` in traces
3. **Config-Only Ensembles**: Create new strategy combinations without Python code
4. **Simplified Strategies**: No defensive coding, no None checks for features
5. **Natural Composition**: Ensembles work without special handling

## Architecture Benefits

1. **Pushes Uncertainty to Load Time**: All feature validation happens before execution
2. **Enables Dependency Tracking**: System knows exactly what each strategy needs
3. **Simplifies Strategy Logic**: Guaranteed features mean cleaner code
4. **Supports Complex Composition**: Nested ensembles with recursive discovery
5. **Improves Debugging**: Clear feature requirements and validation errors

## Next Steps for Implementation

1. **Immediate**: Delete legacy feature inference code (prevents confusion)
2. **This Week**: Migrate high-value strategies (most used in production)
3. **Next Week**: Complete all strategy migrations
4. **Future**: Add advanced ensemble features (adaptive weighting, etc.)

This redesign solves the root architectural issues rather than patching symptoms. The separation of discovery and validation, combined with protocol-based composition, creates a more maintainable and flexible system.
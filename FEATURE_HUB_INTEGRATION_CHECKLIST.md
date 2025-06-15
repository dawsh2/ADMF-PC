# Feature Hub Integration Checklist

## Overview
This checklist tracks the critical architectural change to use a centralized FeatureHub for all feature computation instead of having each strategy compute features independently. This will improve performance by 10-100x for large numbers of strategies.

## Current Problem
- Each of 969 strategies has its own ComponentState computing features independently
- Common features (e.g., SMA(20), RSI(14)) are computed hundreds of times
- Performance: 500 bars with 969 strategies takes over 1 minute

## Target Architecture
- ONE FeatureHub computes all unique features once per bar
- All strategies receive features from the shared FeatureHub
- Features are deduplicated at startup (e.g., only one SMA(20) regardless of how many strategies use it)

## Implementation Checklist

### Phase 1: Create FeatureHub Container Component
- [ ] Create a FeatureHub container component that wraps the existing FeatureHub class
- [ ] Make it subscribe to BAR events from the root event bus
- [ ] On each BAR event, update the FeatureHub with new bar data
- [ ] Ensure it computes all configured features once per bar

### Phase 2: Modify ComponentState to Use FeatureHub
- [ ] Add a `feature_hub` reference to ComponentState
- [ ] Replace `_update_features()` method to query FeatureHub instead of computing
- [ ] Remove feature computation logic from ComponentState
- [ ] Keep feature requirement tracking for strategy readiness checks

### Phase 3: Update Topology Builder
- [ ] Update signal_generation topology pattern to create FeatureHub container
- [ ] Ensure FeatureHub is created before strategy containers
- [ ] Pass FeatureHub reference to ComponentState during initialization
- [ ] Wire FeatureHub to subscribe to BAR events

### Phase 4: Feature Deduplication
- [ ] Collect all feature requirements from strategies during topology build
- [ ] Deduplicate features (e.g., multiple SMA(20) → one SMA(20))
- [ ] Configure FeatureHub with the deduplicated feature set
- [ ] Map strategy feature names to FeatureHub feature names

### Phase 5: Testing & Validation
- [ ] Test with a small number of strategies to ensure correctness
- [ ] Verify features are computed only once per bar
- [ ] Ensure all strategies receive correct feature values
- [ ] Benchmark performance improvement with 969 strategies

### Phase 6: Documentation
- [ ] Create strategy development guide with naming conventions
- [ ] Document feature naming patterns for parameter expansion
- [ ] Add examples of how strategies should declare feature requirements
- [ ] Update architecture documentation with FeatureHub role

## Key Files to Modify

### 1. Container Component for FeatureHub
**File**: `src/core/containers/components/feature_hub_component.py` (NEW)
- [ ] Create new component that wraps FeatureHub
- [ ] Implement BAR event handler
- [ ] Provide feature access interface

### 2. ComponentState Changes
**File**: `src/strategy/state.py`
- [ ] Add feature_hub reference
- [ ] Modify `_update_features()` to use FeatureHub
- [ ] Update `get_features()` to query FeatureHub
- [ ] Keep feature requirement inference for readiness

### 3. Topology Pattern
**File**: `config/patterns/topologies/signal_generation.yaml`
- [ ] Add feature_hub container configuration
- [ ] Update to use new pattern with FeatureHub

### 4. Topology Builder
**File**: `src/core/coordinator/topology.py`
- [ ] Add feature_hub container type handling
- [ ] Implement feature deduplication logic
- [ ] Wire FeatureHub into topology

### 5. Feature Hub Class (Minor Updates)
**File**: `src/strategy/components/features/hub.py`
- [ ] Ensure thread-safety if needed
- [ ] Add any missing feature types
- [ ] Optimize for performance

## Success Criteria
- [ ] Performance: 500 bars with 969 strategies completes in < 10 seconds
- [ ] Memory: No duplicate feature computation or storage
- [ ] Correctness: All strategies receive identical feature values for same parameters
- [ ] Maintainability: Clear separation between feature computation and strategy logic

## Testing Plan
1. **Unit Tests**
   - [ ] FeatureHub computes features correctly
   - [ ] ComponentState queries FeatureHub properly
   - [ ] Feature deduplication works correctly

2. **Integration Tests**
   - [ ] Full topology with FeatureHub works
   - [ ] Multiple strategies share features
   - [ ] Signal generation produces same results

3. **Performance Tests**
   - [ ] Benchmark before: Time with current architecture
   - [ ] Benchmark after: Time with FeatureHub
   - [ ] Memory usage comparison

## Rollback Plan
- Keep original signal_generation.yaml pattern
- Add feature flag to choose between architectures
- Gradual migration strategy by strategy

## Notes
- This is a CRITICAL performance fix for the system
- Expected performance improvement: 10-100x for large strategy counts
- Must maintain backward compatibility during transition
- Consider making this the default architecture once proven
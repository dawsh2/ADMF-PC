# Routing Refactor Summary

## Key Insights

1. **Mixed Concerns**: Current specialized routes (RiskServiceRoute, etc.) mix routing logic with business logic

2. **Feature Dispatcher is Fine**: It's not really a route - it's a specialized component that happens to do some routing. Keep it as-is.

3. **Generic Patterns Exist**: We already have FanOutRoute which supports transforms - we should use it!

## Immediate Actions

### 1. Rename "helpers" directory
```bash
mv src/core/coordinator/topologies/helpers src/core/coordinator/topologies/route_builders
```

### 2. Create ProcessingRoute for validation/transformation patterns
This handles the common pattern of:
- Collect events from multiple sources
- Process/validate/transform them
- Publish results to a target

### 3. Refactor specialized routes into processors
- RiskServiceRoute → ProcessingRoute + RiskValidationProcessor
- ExecutionServiceRoute → ProcessingRoute + ExecutionProcessor

### 4. Fix missing SignalSaverRoute
Either:
- Implement it as a proper route
- Or use ProcessingRoute + SignalSaverProcessor

## Final Route Types

### Core Routes (Simple, Reusable)
1. **BroadcastRoute** - One source → many targets (same event)
2. **FanOutRoute** - One source → many targets (transformed events)
3. **SelectiveRoute** - Route based on event content/rules
4. **ProcessingRoute** - Apply processor to events (NEW)

### Processors (Business Logic)
1. **RiskValidationProcessor** - Validates orders
2. **ExecutionProcessor** - Applies execution models
3. **SignalSaverProcessor** - Saves signals to disk
4. **FeatureFilterProcessor** - Filters features (if we refactor FeatureDispatcher)

## Benefits

1. **Separation of Concerns**: Routes handle routing, processors handle business logic
2. **Fewer Route Types**: 4 core types instead of many specialized ones
3. **Reusability**: ProcessingRoute pattern works for many use cases
4. **Testability**: Processors can be tested independently
5. **Clarity**: Clear what each component does

## Questions for You

1. Should we keep FeatureDispatcher as-is or refactor into FanOutRoute + FeatureFilterProcessor?
2. Do you want to proceed with the ProcessingRoute implementation?
3. Should we move forward with renaming helpers → route_builders?
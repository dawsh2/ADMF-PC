# Routing Analysis and Recommendations

## Current Signal Flow

### 1. Data → Features
- **Current**: Direct container reference, no route
- **Pattern**: Direct coupling is fine here since features depend on data

### 2. Features → Strategies  
- **Current**: Feature Dispatcher (custom component, not a route)
- **Issue**: Feature Dispatcher is doing routing logic but isn't a proper route
- **Recommendation**: Create a `SelectiveRoute` or custom `FeatureRoute`

### 3. Strategies → Portfolios
- **Current**: Root event bus pub/sub for SIGNAL events
- **Issue**: No filtering - all portfolios get all signals
- **Recommendation**: Use `SelectiveRoute` to route signals to portfolios based on combo_id

### 4. Portfolios → Risk → Execution
- **Current**: `RiskServiceRoute` for validation
- **Pattern**: Good - maintains isolation while enabling validation

### 5. Execution → Portfolios
- **Current**: `BroadcastRoute` for FILL events
- **Pattern**: Good - all portfolios need fills

## Route Types Analysis

### Currently Used
1. **risk_service** - Custom route for order validation
2. **broadcast** - For FILL distribution
3. **signal_saver** - Not registered! Needs fixing

### Available but Unused
1. **pipeline** - Could be useful for data → features → strategies
2. **selective** - Perfect for signal → portfolio routing
3. **hierarchical** - Not needed for current architecture
4. **fan_out** - Could replace Feature Dispatcher
5. Various variants (filtered_broadcast, etc.) - Overkill for now

## Recommendations

### 1. Rename 'helpers' Directory
Rename to one of:
- `topology_routing` - Clear purpose
- `route_builders` - What it actually does
- `routing_config` - Configuration focused

### 2. Consolidate Route Types
Keep only what we need:
- **Core**: pipeline, broadcast, selective
- **Custom**: risk_service, signal_router (new)
- **Archive**: hierarchical, variants (move to `routing/experimental/`)

### 3. Fix Signal Routing
Replace Feature Dispatcher with proper routes:
```python
# Features → Strategies: Use FanOutRoute
feature_route = routing_factory.create_route(
    'feature_router',
    {
        'type': 'fan_out',
        'source': 'feature_container',
        'targets': [
            {
                'name': strategy_id,
                'filter': lambda e: strategy_needs_features(e)
            }
            for strategy_id in strategies
        ]
    }
)

# Strategies → Portfolios: Use SelectiveRoute  
signal_route = routing_factory.create_route(
    'signal_router',
    {
        'type': 'selective',
        'source': 'root_event_bus',
        'routing_rules': [
            {
                'target': portfolio_id,
                'conditions': [
                    {'field': 'payload.combo_id', 'operator': 'equals', 'value': combo_id}
                ]
            }
            for portfolio_id, combo_id in portfolio_mapping.items()
        ]
    }
)
```

### 4. Create Missing Route Types
```python
# Add to factory registry:
'signal_saver': lambda n, c: create_route_with_logging(SignalSaverRoute, n, c),
'feature_router': lambda n, c: create_route_with_logging(FeatureRoute, n, c),
```

### 5. Directory Structure
```
src/core/routing/
├── __init__.py
├── protocols.py          # Keep - defines interfaces
├── factory.py           # Keep - creates routes
├── helpers.py           # Keep - utility functions
├── core/                # Core route implementations
│   ├── pipeline.py      # Sequential routing
│   ├── broadcast.py     # One-to-many routing
│   └── selective.py     # Rule-based routing
├── specialized/         # Domain-specific routes
│   ├── risk_service.py  # ORDER_REQUEST validation
│   ├── signal_router.py # Signal → Portfolio routing
│   └── feature_router.py # Feature → Strategy routing
└── experimental/        # Unused but potentially useful
    ├── hierarchical.py
    ├── variants.py      # All the variants
    └── README.md        # Explain what these are
```
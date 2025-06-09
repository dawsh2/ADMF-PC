# Topology Pattern Example

## Current (Imperative Code)

```python
# topologies/backtest.py - 200 lines of code!
for symbol in symbols:
    for timeframe in timeframes:
        data_container = factory.create_container(...)
        feature_container = factory.create_container(...)
        # ... lots of logic
```

## Proposed (Declarative Pattern)

```python
# topologies/backtest.py - Just data!
BACKTEST_PATTERN = {
    'name': 'backtest',
    'description': 'Full backtest with data → features → strategies → portfolios → risk → execution',
    
    # Components to create
    'components': [
        {
            'type': 'strategies',
            'from_config': 'strategies'  # Get from user config
        },
        {
            'type': 'risk_validators', 
            'from_config': 'risk_profiles'
        }
    ],
    
    # Containers to create
    'containers': [
        # Data containers - one per symbol/timeframe combo
        {
            'name_template': '{symbol}_{timeframe}_data',
            'type': 'data',
            'foreach': {
                'symbol': {'from_config': 'symbols'},
                'timeframe': {'from_config': 'timeframes'}
            },
            'config': {
                'symbol': '{symbol}',
                'timeframe': '{timeframe}',
                'data_source': {'from_config': 'data_source', 'default': 'file'},
                'data_path': {'from_config': 'data_path'}
            }
        },
        
        # Feature containers - one per data container
        {
            'name_template': '{symbol}_{timeframe}_features',
            'type': 'features',
            'foreach': {
                'symbol': {'from_config': 'symbols'},
                'timeframe': {'from_config': 'timeframes'}
            },
            'config': {
                'data_container': '{symbol}_{timeframe}_data',  # Reference
                'features': {'from_config': 'features', 'default': {}}
            }
        },
        
        # Portfolio containers - one per strategy/risk combo
        {
            'name_template': 'portfolio_{combo_id}',
            'type': 'portfolio',
            'foreach': {
                'strategy': {'from_config': 'strategies'},
                'risk_profile': {'from_config': 'risk_profiles'}
            },
            'config': {
                'strategy_type': '{strategy.type}',
                'strategy_params': '{strategy}',
                'risk_type': '{risk_profile.type}',
                'risk_params': '{risk_profile}',
                'initial_capital': {'from_config': 'initial_capital', 'default': 100000}
            }
        },
        
        # Single execution container
        {
            'name': 'execution',
            'type': 'execution',
            'config': {
                'mode': 'backtest'
            }
        }
    ],
    
    # Routes to create
    'routes': [
        # Risk validation route
        {
            'name': 'risk_validation',
            'type': 'processing',
            'source_pattern': 'portfolio_*',
            'processor': 'risk_validator',
            'target': 'root_event_bus'
        },
        
        # Fill broadcast
        {
            'name': 'fill_broadcast',
            'type': 'broadcast',
            'source': 'execution',
            'target_pattern': 'portfolio_*',
            'event_types': ['FILL']
        }
    ],
    
    # Special behaviors
    'behaviors': [
        {
            'type': 'feature_dispatcher',
            'source_pattern': '*_features',
            'target': 'strategies'
        },
        {
            'type': 'subscribe_to_root_bus',
            'containers': 'portfolio_*',
            'event_type': 'SIGNAL',
            'handler': 'signal_processor.on_signal'
        }
    ]
}
```

## Generic TopologyBuilder

```python
class TopologyBuilder:
    """Interprets topology patterns to build actual topologies."""
    
    def build(self, pattern: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build topology from pattern and user configuration."""
        
        # Create evaluation context
        context = EvaluationContext(user_config)
        
        # Build topology
        topology = {
            'name': pattern['name'],
            'containers': {},
            'routes': [],
            'components': {}
        }
        
        # 1. Create components
        for comp_spec in pattern.get('components', []):
            components = self._create_components(comp_spec, context)
            topology['components'].update(components)
            
        # 2. Create containers
        for cont_spec in pattern.get('containers', []):
            containers = self._create_containers(cont_spec, context)
            topology['containers'].update(containers)
            
        # 3. Create routes
        for route_spec in pattern.get('routes', []):
            route = self._create_route(route_spec, context, topology['containers'])
            topology['routes'].append(route)
            
        # 4. Apply behaviors
        for behavior_spec in pattern.get('behaviors', []):
            self._apply_behavior(behavior_spec, context, topology)
            
        return topology
```

## Benefits

1. **Topology files are just data** - Easy to understand, modify, version
2. **No code duplication** - Generic builder handles all patterns
3. **Extensible** - Add new patterns without coding
4. **Testable** - Can validate patterns without running them
5. **Composable** - Patterns can reference/extend other patterns

## Even Better: YAML Patterns

```yaml
# topologies/backtest.yaml
name: backtest
description: Full backtest pipeline

containers:
  - name_template: "{symbol}_{timeframe}_data"
    type: data
    foreach:
      symbol: $config.symbols
      timeframe: $config.timeframes
    config:
      symbol: $symbol
      timeframe: $timeframe
      
routes:
  - name: risk_validation
    type: processing
    source_pattern: "portfolio_*"
    processor: risk_validator
```
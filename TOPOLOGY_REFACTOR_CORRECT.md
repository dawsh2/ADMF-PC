# Correct Topology Refactor Plan

## Current Problem
- Topology "patterns" are hardcoded as Python functions
- Mixing data (what to build) with code (how to build)
- Can't easily add new patterns without writing code

## Correct Architecture

```
topology.py (Generic TopologyBuilder)
├── build(topology_description) -> builds ANY topology
├── _create_containers(container_specs)
├── _create_stateless_components(component_specs)
└── _create_routes(route_specs)

topologies/ (Topology Descriptions/Patterns)
├── backtest.yaml
├── signal_generation.yaml  
├── signal_replay.yaml
└── custom_patterns.yaml
```

## Topology Description Format

```yaml
# topologies/backtest.yaml
name: backtest
description: Full backtest pipeline

# Stateless components to create
components:
  strategies:
    - type: momentum
      params: ${config.strategies}  # From user config
  risk_validators:
    - type: position_limits
    - type: drawdown_limits
      
# Containers to create
containers:
  # Data containers (one per symbol/timeframe)
  - pattern: "${symbol}_${timeframe}_data"
    type: data
    foreach:
      symbol: ${config.symbols}
      timeframe: ${config.timeframes}
    config:
      data_source: ${config.data_source}
      
  # Feature containers  
  - pattern: "${symbol}_${timeframe}_features"
    type: features
    foreach:
      symbol: ${config.symbols}
      timeframe: ${config.timeframes}
    config:
      data_container: "${symbol}_${timeframe}_data"
      
  # Portfolio containers (one per parameter combo)
  - pattern: "portfolio_${combo_id}"
    type: portfolio
    foreach:
      combo: ${generated.parameter_combinations}
    config:
      strategy_type: ${combo.strategy_type}
      risk_type: ${combo.risk_type}
      initial_capital: ${config.initial_capital}
      
  # Single execution container
  - name: execution
    type: execution
    config:
      mode: backtest

# Routes to create
routes:
  # Risk validation
  - type: processing
    name: risk_validation
    processor: risk_validator
    source_pattern: "portfolio_*"
    target: root_event_bus
    
  # Fill broadcast
  - type: broadcast
    name: fill_distribution
    source: execution
    targets: "portfolio_*"
    event_types: [FILL]
```

## Generic TopologyBuilder Implementation

```python
class TopologyBuilder:
    """Generic builder that constructs topologies from descriptions."""
    
    def build(self, topology_desc: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build any topology from its description.
        
        Args:
            topology_desc: The topology pattern/description
            user_config: User's configuration values
            
        Returns:
            Built topology with containers, routes, etc.
        """
        # Merge configs and resolve variables
        context = self._build_context(topology_desc, user_config)
        
        topology = {
            'containers': {},
            'routes': [],
            'components': {},
            'metadata': {
                'pattern': topology_desc.get('name'),
                'description': topology_desc.get('description')
            }
        }
        
        # 1. Create stateless components
        if 'components' in topology_desc:
            topology['components'] = self._create_components(
                topology_desc['components'], context
            )
            
        # 2. Create containers
        if 'containers' in topology_desc:
            topology['containers'] = self._create_containers(
                topology_desc['containers'], context
            )
            
        # 3. Create routes
        if 'routes' in topology_desc:
            topology['routes'] = self._create_routes(
                topology_desc['routes'], context, topology['containers']
            )
            
        return topology
        
    def _create_containers(self, container_specs: List[Dict], context: Dict) -> Dict[str, Container]:
        """Create containers from specifications."""
        containers = {}
        factory = ContainerFactory()
        
        for spec in container_specs:
            if 'foreach' in spec:
                # Create multiple containers from pattern
                containers.update(self._expand_container_pattern(spec, context, factory))
            else:
                # Create single container
                name = self._resolve_value(spec['name'], context)
                config = self._resolve_config(spec.get('config', {}), context)
                containers[name] = factory.create_container(name, config)
                
        return containers
        
    def _expand_container_pattern(self, spec: Dict, context: Dict, factory) -> Dict[str, Container]:
        """Expand a container pattern with foreach loops."""
        containers = {}
        
        # Get iteration variables
        foreach_vars = spec['foreach']
        
        # Generate all combinations
        import itertools
        keys = list(foreach_vars.keys())
        values = [self._resolve_value(foreach_vars[k], context) for k in keys]
        
        for combo in itertools.product(*values):
            # Build context for this iteration
            iter_context = context.copy()
            for i, key in enumerate(keys):
                iter_context[key] = combo[i]
                
            # Create container
            name = self._resolve_value(spec['pattern'], iter_context)
            config = self._resolve_config(spec.get('config', {}), iter_context)
            config['type'] = spec['type']
            
            containers[name] = factory.create_container(name, config)
            
        return containers
```

## Benefits

1. **Data-driven**: Topologies are data, not code
2. **Extensible**: Add new patterns without coding
3. **Generic**: One builder handles all patterns
4. **Clear separation**: Pattern (what) vs Builder (how)
5. **Testable**: Can test builder with any pattern
6. **Reusable**: Patterns can inherit/compose

## Migration Path

1. Extract patterns from current Python files into YAML/JSON
2. Implement generic TopologyBuilder
3. Delete topology Python files
4. Update imports to use TopologyBuilder.build()

## Even Better: Pattern Registry

```python
class TopologyBuilder:
    def __init__(self):
        self.patterns = self._load_patterns()
        
    def _load_patterns(self) -> Dict[str, Dict]:
        """Load all topology patterns from files."""
        patterns = {}
        pattern_dir = Path(__file__).parent / 'topologies'
        
        for pattern_file in pattern_dir.glob('*.yaml'):
            name = pattern_file.stem
            patterns[name] = yaml.safe_load(pattern_file.read_text())
            
        return patterns
        
    def build_from_pattern(self, pattern_name: str, user_config: Dict) -> Dict:
        """Build topology from a named pattern."""
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
            
        return self.build(self.patterns[pattern_name], user_config)
```

This way the system is completely data-driven and extensible!
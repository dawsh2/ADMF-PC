# Smart Container Sharing in Multi-Parameter Workflows

## The Problem: Over-Creation vs. Smart Sharing

### ❌ Naive Approach (What We Had)
```
Parameter Combination 1: momentum(lookback=10, threshold=0.01)
├── Portfolio_1 (different: capital=10000)
├── Strategy_1 (different: lookback=10, threshold=0.01)  
├── Risk_1 (SAME: commission=0.001, max_position=1000)     # WASTE!
└── Execution_1 (SAME: commission=0.001, slippage=0.0005) # WASTE!

Parameter Combination 2: momentum(lookback=20, threshold=0.01)
├── Portfolio_2 (different: capital=10000)
├── Strategy_2 (different: lookback=20, threshold=0.01)
├── Risk_2 (SAME: commission=0.001, max_position=1000)     # DUPLICATE!
└── Execution_2 (SAME: commission=0.001, slippage=0.0005) # DUPLICATE!
```

### ✅ Smart Approach (Optimized)
```
Hub Container
├── Shared_Risk_Container (commission=0.001, max_position=1000)
├── Shared_Execution_Container (commission=0.001, slippage=0.0005)
├── Portfolio_1 (capital=10000) → Strategy_1 (lookback=10, threshold=0.01)
├── Portfolio_2 (capital=10000) → Strategy_2 (lookback=20, threshold=0.01)
└── Portfolio_3 (capital=50000) → Strategy_3 (lookback=10, threshold=0.01)
```

## Implementation: Smart Container Analysis

Let me enhance the WorkflowManager to detect shared vs. unique configurations:

```python
def _analyze_container_sharing(self, param_combinations: List[Dict[str, Any]], base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze which containers can be shared vs. need separate instances."""
    
    analysis = {
        'risk_containers': {},      # risk_config_hash -> container_id
        'execution_containers': {}, # execution_config_hash -> container_id
        'portfolio_containers': [], # Always separate (different P&L tracking)
        'strategy_containers': []   # Always separate (different parameters)
    }
    
    # Analyze each combination
    for i, combo in enumerate(param_combinations):
        combo_id = f"combo_{i}"
        
        # Portfolio is always separate (different P&L tracking)
        portfolio_config = self._build_portfolio_config(combo, base_config)
        analysis['portfolio_containers'].append({
            'combination_index': i,
            'container_id': f"portfolio_{combo_id}",
            'config': portfolio_config
        })
        
        # Strategy is always separate (different parameters)
        strategy_config = combo.get('strategy', {})
        analysis['strategy_containers'].append({
            'combination_index': i,
            'container_id': f"strategy_{combo_id}",
            'config': strategy_config
        })
        
        # Risk container - check if we can reuse
        risk_config = self._build_risk_config(combo, base_config)
        risk_hash = self._config_hash(risk_config)
        
        if risk_hash not in analysis['risk_containers']:
            analysis['risk_containers'][risk_hash] = {
                'container_id': f"risk_{len(analysis['risk_containers'])}",
                'config': risk_config,
                'serves_combinations': [i]
            }
        else:
            analysis['risk_containers'][risk_hash]['serves_combinations'].append(i)
        
        # Execution container - check if we can reuse
        execution_config = self._build_execution_config(combo, base_config)
        execution_hash = self._config_hash(execution_config)
        
        if execution_hash not in analysis['execution_containers']:
            analysis['execution_containers'][execution_hash] = {
                'container_id': f"execution_{len(analysis['execution_containers'])}",
                'config': execution_config,
                'serves_combinations': [i]
            }
        else:
            analysis['execution_containers'][execution_hash]['serves_combinations'].append(i)
    
    return analysis

def _config_hash(self, config: Dict[str, Any]) -> str:
    """Create hash of configuration for sharing detection."""
    import json
    import hashlib
    
    # Remove combination-specific fields that shouldn't affect sharing
    clean_config = {k: v for k, v in config.items() 
                   if k not in ['combination_id', 'combination_index', 'target_portfolio']}
    
    config_str = json.dumps(clean_config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]
```

## Example Configuration Analysis

### Input Configuration
```yaml
type: optimization
strategies:
  - type: momentum
    parameters:
      lookback_period: [10, 20]     # 2 values
      signal_threshold: [0.01, 0.02] # 2 values = 4 strategy combinations

optimization:
  parameter_grids:
    initial_capital: [10000, 50000]  # 2 values
    # Total: 4 × 2 = 8 total combinations

risk:
  max_position_size: 1000
  commission: 0.001               # SAME for all combinations

execution:
  commission: 0.001               # SAME for all combinations  
  slippage: 0.0005               # SAME for all combinations
```

### Smart Analysis Results
```python
analysis = {
    'risk_containers': {
        'a1b2c3d4': {  # Hash of risk config
            'container_id': 'risk_0',
            'config': {'max_position_size': 1000, 'commission': 0.001},
            'serves_combinations': [0, 1, 2, 3, 4, 5, 6, 7]  # ALL combinations
        }
    },
    'execution_containers': {
        'e5f6g7h8': {  # Hash of execution config
            'container_id': 'execution_0', 
            'config': {'commission': 0.001, 'slippage': 0.0005},
            'serves_combinations': [0, 1, 2, 3, 4, 5, 6, 7]  # ALL combinations
        }
    },
    'portfolio_containers': [
        {'combination_index': 0, 'container_id': 'portfolio_combo_0', 'config': {'initial_capital': 10000}},
        {'combination_index': 1, 'container_id': 'portfolio_combo_1', 'config': {'initial_capital': 10000}},
        {'combination_index': 2, 'container_id': 'portfolio_combo_2', 'config': {'initial_capital': 10000}},
        {'combination_index': 3, 'container_id': 'portfolio_combo_3', 'config': {'initial_capital': 10000}},
        {'combination_index': 4, 'container_id': 'portfolio_combo_4', 'config': {'initial_capital': 50000}},
        {'combination_index': 5, 'container_id': 'portfolio_combo_5', 'config': {'initial_capital': 50000}},
        {'combination_index': 6, 'container_id': 'portfolio_combo_6', 'config': {'initial_capital': 50000}},
        {'combination_index': 7, 'container_id': 'portfolio_combo_7', 'config': {'initial_capital': 50000}}
    ],
    'strategy_containers': [
        {'combination_index': 0, 'container_id': 'strategy_combo_0', 'config': {'type': 'momentum', 'parameters': {'lookback_period': 10, 'signal_threshold': 0.01}}},
        {'combination_index': 1, 'container_id': 'strategy_combo_1', 'config': {'type': 'momentum', 'parameters': {'lookback_period': 10, 'signal_threshold': 0.02}}},
        # ... etc
    ]
}
```

### Optimized Container Structure
```
Hub Container
├── risk_0 (SHARED: serves all 8 combinations)
├── execution_0 (SHARED: serves all 8 combinations)
├── portfolio_combo_0 → strategy_combo_0 (momentum: lookback=10, threshold=0.01, capital=10000)
├── portfolio_combo_1 → strategy_combo_1 (momentum: lookback=10, threshold=0.02, capital=10000)
├── portfolio_combo_2 → strategy_combo_2 (momentum: lookback=20, threshold=0.01, capital=10000)
├── portfolio_combo_3 → strategy_combo_3 (momentum: lookback=20, threshold=0.02, capital=10000)
├── portfolio_combo_4 → strategy_combo_4 (momentum: lookback=10, threshold=0.01, capital=50000)
├── portfolio_combo_5 → strategy_combo_5 (momentum: lookback=10, threshold=0.02, capital=50000)
├── portfolio_combo_6 → strategy_combo_6 (momentum: lookback=20, threshold=0.01, capital=50000)
└── portfolio_combo_7 → strategy_combo_7 (momentum: lookback=20, threshold=0.02, capital=50000)

Result: 1 Hub + 1 Risk + 1 Execution + 8 Portfolios + 8 Strategies = 19 containers
Instead of: 1 Hub + 8 Risk + 8 Execution + 8 Portfolios + 8 Strategies = 33 containers
Savings: 42% fewer containers!
```

## Smart Communication Wiring

### Shared Container Communication
```python
def _get_multi_parameter_communication_optimized(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Optimized communication for shared containers."""
    
    configs = []
    
    # 1. Hub broadcasts data to all strategies
    configs.append({
        'name': 'hub_to_strategies',
        'type': 'broadcast',
        'source': 'hub_container',
        'targets': [s['container_id'] for s in analysis['strategy_containers']],
        'event_types': ['BAR', 'TICK', 'FEATURE']
    })
    
    # 2. All strategies send signals to shared risk container(s)
    for risk_hash, risk_info in analysis['risk_containers'].items():
        strategy_ids = [
            analysis['strategy_containers'][i]['container_id'] 
            for i in risk_info['serves_combinations']
        ]
        
        configs.append({
            'name': f'strategies_to_{risk_info["container_id"]}',
            'type': 'selective',
            'sources': strategy_ids,
            'target': risk_info['container_id'],
            'event_types': ['SIGNAL'],
            'routing_key': 'target_portfolio'  # Route based on target portfolio
        })
    
    # 3. Shared risk containers send orders to shared execution container(s)
    for exec_hash, exec_info in analysis['execution_containers'].items():
        risk_ids = [
            risk_info['container_id'] 
            for risk_info in analysis['risk_containers'].values()
            if any(i in exec_info['serves_combinations'] for i in risk_info['serves_combinations'])
        ]
        
        configs.append({
            'name': f'risk_to_{exec_info["container_id"]}',
            'type': 'pipeline',
            'sources': risk_ids,
            'target': exec_info['container_id'],
            'event_types': ['ORDER']
        })
    
    # 4. Shared execution containers send fills to specific portfolios
    for exec_hash, exec_info in analysis['execution_containers'].items():
        portfolio_ids = [
            analysis['portfolio_containers'][i]['container_id'] 
            for i in exec_info['serves_combinations']
        ]
        
        configs.append({
            'name': f'{exec_info["container_id"]}_to_portfolios',
            'type': 'selective',
            'source': exec_info['container_id'],
            'targets': portfolio_ids,
            'event_types': ['FILL'],
            'routing_key': 'target_portfolio'  # Route fill to correct portfolio
        })
    
    return configs
```

## When NOT to Share Containers

### Different Risk Configurations
```yaml
# Configuration with different risk per strategy type
strategies:
  - type: momentum
    parameters:
      lookback_period: [10, 20]
    risk_config:               # Strategy-specific risk
      max_position_size: 1000
      
  - type: mean_reversion  
    parameters:
      rsi_period: [14, 21]
    risk_config:               # Different risk config
      max_position_size: 2000  # DIFFERENT!
```

Result: Creates **2 risk containers** (one per strategy type), not 1 shared.

### Different Execution Settings
```yaml
optimization:
  parameter_grids:
    commission: [0.001, 0.005]    # DIFFERENT execution configs
    slippage: [0.0005, 0.001]
```

Result: Creates **4 execution containers** (one per commission/slippage combination).

## Benefits of Smart Sharing

### 1. **Resource Efficiency**
- 42% fewer containers in typical scenarios
- Reduced memory usage
- Faster startup time

### 2. **Logical Correctness** 
- Portfolios remain isolated (separate P&L)
- Strategies remain isolated (separate parameters)
- Shared services are truly shared

### 3. **Communication Efficiency**
- Fewer adapters needed
- Simpler routing logic
- Better performance

### 4. **Maintains Isolation**
- Each portfolio still gets separate P&L tracking
- Each strategy still gets separate parameters
- No cross-contamination between combinations

## Implementation in WorkflowManager

```python
async def _execute_multi_parameter_workflow_optimized(self, ...):
    # 1. Analyze what can be shared
    analysis = self._analyze_container_sharing(param_combinations, pattern_config)
    
    # 2. Create shared containers (risk, execution)
    shared_containers = await self._create_shared_containers(analysis)
    
    # 3. Create separate containers (portfolios, strategies)  
    separate_containers = await self._create_separate_containers(analysis)
    
    # 4. Wire everything with optimized communication
    await self._setup_optimized_communication(analysis, shared_containers, separate_containers)
```

This approach gives you the benefits of **complete isolation where needed** (portfolios, strategies) while **efficient sharing where possible** (risk, execution), following the smart container reuse principle.
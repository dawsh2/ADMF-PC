# Dual Portfolio Architecture Design

## Overview
Design for running multiple strategies with completely separate portfolio instances, enabling independent tracking, reporting, and signal-to-portfolio flows.

## Current Architecture (Shared Portfolio)
```
DataContainer → IndicatorContainer → StrategyContainer → RiskContainer → ExecutionContainer
                                          ↓                                        ↓
                                    [SubStrategy1]                              [FILL]
                                    [SubStrategy2]                                ↓
                                          ↓                              PortfolioContainer (Shared)
                                    [SIGNAL events]                               ↓
                                                                          [PORTFOLIO events]
```

## Proposed Architecture (Separate Portfolios)

### Option 1: Parallel Pipelines
```
DataContainer ──┬→ Pipeline1: Indicator1 → Strategy1 → Risk1 → Execution1 → Portfolio1
                │                                                               ↓
                │                                                         [Reports1]
                │
                └→ Pipeline2: Indicator2 → Strategy2 → Risk2 → Execution2 → Portfolio2
                                                                               ↓
                                                                         [Reports2]
```

### Option 2: Sub-Container Hierarchy
```
BacktestContainer
├── DataContainer (shared)
├── TradingSystem1
│   ├── IndicatorContainer1
│   ├── StrategyContainer1
│   ├── RiskContainer1
│   ├── ExecutionContainer1
│   └── PortfolioContainer1
└── TradingSystem2
    ├── IndicatorContainer2
    ├── StrategyContainer2
    ├── RiskContainer2
    ├── ExecutionContainer2
    └── PortfolioContainer2
```

### Option 3: Named Pipeline Routing
```
DataContainer → Router
                  ↓
            ┌─────┴─────┐
            ↓           ↓
      [momentum]    [mean_rev]
            ↓           ↓
    Full Pipeline   Full Pipeline
```

## Implementation Details

### 1. Configuration Structure
```yaml
workflow:
  type: "multi_portfolio_backtest"
  name: "Dual Portfolio Test"

data:
  source: "csv"
  file_path: "data/SPY_1m.csv"
  symbols: ["SPY"]

trading_systems:
  - name: "momentum_system"
    portfolio:
      initial_capital: 50000
      namespace: "momentum"  # For event routing
    
    indicators:
      - "SMA_20"
      - "RSI"
    
    strategy:
      type: "momentum"
      parameters:
        lookback_period: 20
        momentum_threshold: 0.0002
    
    risk:
      position_sizers:
        - type: "fixed"
          size: 1000
      limits:
        - type: "max_position_value"
          value: 5000
    
    execution:
      slippage_bps: 5
      commission: 0.005

  - name: "mean_reversion_system"
    portfolio:
      initial_capital: 50000
      namespace: "mean_rev"  # For event routing
    
    indicators:
      - "BB_20"
      - "RSI"
    
    strategy:
      type: "mean_reversion"
      parameters:
        lookback_period: 15
        entry_threshold: 1.0
    
    risk:
      position_sizers:
        - type: "percentage"
          size: 0.02
      limits:
        - type: "max_total_exposure"
          value: 0.8
    
    execution:
      slippage_bps: 10
      commission: 0.01

reporting:
  separate_reports: true
  combined_summary: true
```

### 2. Event Routing with Namespaces

```python
# Event with namespace
event = Event(
    event_type=EventType.SIGNAL,
    namespace="momentum",  # Routes to momentum pipeline
    payload={...}
)

# Pipeline adapter configuration
pipeline_config = {
    "momentum": {
        "containers": ["indicator_momentum", "strategy_momentum", 
                      "risk_momentum", "execution_momentum", "portfolio_momentum"]
    },
    "mean_rev": {
        "containers": ["indicator_mean_rev", "strategy_mean_rev",
                      "risk_mean_rev", "execution_mean_rev", "portfolio_mean_rev"]
    }
}
```

### 3. Container Creation Pattern

```python
class MultiPortfolioWorkflowManager:
    async def setup_trading_systems(self, config):
        systems = []
        
        for system_config in config['trading_systems']:
            namespace = system_config['portfolio']['namespace']
            
            # Create isolated container set
            containers = {
                'indicator': IndicatorContainer(
                    config=system_config['indicators'],
                    container_id=f"indicator_{namespace}"
                ),
                'strategy': StrategyContainer(
                    config=system_config['strategy'],
                    container_id=f"strategy_{namespace}"
                ),
                'risk': RiskContainer(
                    config=system_config['risk'],
                    container_id=f"risk_{namespace}"
                ),
                'execution': ExecutionContainer(
                    config=system_config['execution'],
                    container_id=f"execution_{namespace}"
                ),
                'portfolio': PortfolioContainer(
                    config=system_config['portfolio'],
                    container_id=f"portfolio_{namespace}"
                )
            }
            
            # Set up pipeline for this system
            await self.setup_pipeline(namespace, containers)
            
            systems.append({
                'namespace': namespace,
                'containers': containers
            })
        
        return systems
```

### 4. Data Distribution

```python
class DataContainer:
    def __init__(self, config, namespaces=None):
        self.namespaces = namespaces or []
        
    async def _stream_data(self):
        for timestamp, row in data.iterrows():
            bar_event = Event(
                event_type=EventType.BAR,
                payload={...},
                timestamp=timestamp
            )
            
            # Broadcast to all namespaces
            for namespace in self.namespaces:
                namespaced_event = bar_event.copy()
                namespaced_event.namespace = namespace
                self.event_bus.publish(namespaced_event)
```

### 5. Separate Reporting

```python
class MultiPortfolioReporter:
    def generate_reports(self, trading_systems):
        reports = {}
        
        for system in trading_systems:
            namespace = system['namespace']
            portfolio = system['containers']['portfolio']
            
            # Generate individual report
            reports[namespace] = {
                'performance': self.calculate_performance(portfolio),
                'trades': self.get_trades(portfolio),
                'positions': portfolio.get_all_positions(),
                'equity_curve': self.get_equity_curve(portfolio)
            }
        
        # Generate combined summary
        combined = self.generate_combined_summary(reports)
        
        return reports, combined
```

## Benefits

1. **Complete Isolation**
   - No position conflicts between strategies
   - Independent risk management
   - Separate capital allocation

2. **Better Analysis**
   - Per-strategy performance metrics
   - Individual drawdown tracking
   - Strategy-specific reports

3. **Flexibility**
   - Different risk parameters per strategy
   - Independent position sizing
   - Custom execution settings

4. **Scalability**
   - Easy to add more strategies
   - Parallel processing potential
   - Resource isolation

## Challenges

1. **Complexity**
   - More containers to manage
   - Complex event routing
   - State synchronization

2. **Resource Usage**
   - Multiple portfolio states
   - Duplicate indicator calculations
   - More memory usage

3. **Coordination**
   - END_OF_DATA handling
   - Synchronized reporting
   - Combined metrics

## Migration Path

### Phase 1: Namespace Support
- Add namespace field to Event class
- Update pipeline adapter for namespace routing
- Test with single pipeline

### Phase 2: Multi-Pipeline Setup
- Create MultiPortfolioWorkflowManager
- Implement container factory pattern
- Add configuration parser

### Phase 3: Reporting Integration
- Separate portfolio tracking
- Individual performance reports
- Combined dashboard

### Phase 4: Optimization
- Shared indicator caching
- Efficient data broadcasting
- Resource pooling

## Example Usage

```python
# Run dual portfolio backtest
python main.py --config config/dual_portfolio_backtest.yaml

# Output structure
results/
├── momentum_system/
│   ├── trades.csv
│   ├── performance.json
│   └── report.html
├── mean_reversion_system/
│   ├── trades.csv
│   ├── performance.json
│   └── report.html
└── combined_summary.html
```

## Next Steps

1. **Prototype Implementation**
   - Start with Option 3 (Named Pipeline Routing)
   - Use existing pipeline adapter with modifications
   - Test with two simple strategies

2. **Validation**
   - Ensure complete isolation
   - Verify no cross-contamination
   - Performance benchmarking

3. **Production Features**
   - Dynamic strategy addition/removal
   - Live portfolio rebalancing
   - Cross-strategy risk limits
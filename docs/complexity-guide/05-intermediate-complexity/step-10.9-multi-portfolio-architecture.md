# Step 10.9: Multi-Portfolio Architecture

## 🎯 Objectives

Implement support for running multiple isolated portfolios simultaneously, enabling:

1. Strategy comparison with different risk profiles
2. Portfolio-level performance attribution
3. Cross-portfolio correlation analysis
4. Independent risk management per portfolio
5. Scalable architecture for N portfolios

## 📋 Prerequisites

Before starting this step:
- [ ] Step 10.8 Event Tracing complete (CRITICAL for debugging)
- [ ] Strong understanding of container isolation
- [ ] Portfolio state management working correctly
- [ ] Risk management fully operational
- [ ] Event bus isolation validated

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                Multi-Portfolio Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  TradingSystemContainer                                      │
│  ├── Portfolio_1 (Isolated)                                  │
│  │   ├── Event Bus (portfolio_1)                            │
│  │   ├── DataContainer                                      │
│  │   ├── StrategyContainer(s)                               │
│  │   ├── RiskContainer                                      │
│  │   ├── PortfolioContainer                                 │
│  │   └── ExecutionContainer                                 │
│  │                                                           │
│  ├── Portfolio_2 (Isolated)                                  │
│  │   ├── Event Bus (portfolio_2)                            │
│  │   └── ... (complete stack)                               │
│  │                                                           │
│  └── CrossPortfolioAnalytics                                │
│      ├── Correlation Tracker                                │
│      ├── Performance Aggregator                             │
│      └── Risk Consolidator                                  │
└─────────────────────────────────────────────────────────────┘
```

## 📚 Implementation Guide

### 1. Portfolio Namespace Design

```python
# src/core/portfolio/portfolio_namespace.py
class PortfolioNamespace:
    """Ensures complete isolation between portfolios"""
    
    def __init__(self, portfolio_id: str):
        self.portfolio_id = portfolio_id
        self.namespace = f"portfolio_{portfolio_id}"
        
    def wrap_event(self, event: Event) -> Event:
        """Add portfolio namespace to event"""
        event.metadata['portfolio_id'] = self.portfolio_id
        event.metadata['namespace'] = self.namespace
        return event
        
    def create_correlation_id(self) -> str:
        """Portfolio-specific correlation ID"""
        return f"{self.namespace}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
```

### 2. Trading System Container

```python
# src/core/containers/trading_system_container.py
class TradingSystemContainer(ComposableContainer):
    """Container that manages multiple portfolio containers"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.portfolios = {}
        self.analytics = CrossPortfolioAnalytics()
        
        # Create isolated portfolio containers
        for portfolio_config in config['portfolios']:
            portfolio = self._create_portfolio_container(portfolio_config)
            self.portfolios[portfolio.portfolio_id] = portfolio
            
    def _create_portfolio_container(self, config: Dict) -> PortfolioContainer:
        """Create completely isolated portfolio"""
        # CRITICAL: Each portfolio gets its own event bus!
        portfolio = PortfolioSystemContainer(
            portfolio_id=config['id'],
            event_bus=TracedEventBus(f"portfolio_{config['id']}")
        )
        
        # Build complete trading stack
        portfolio.add_component(DataContainer(config['data']))
        portfolio.add_component(StrategyContainer(config['strategies']))
        portfolio.add_component(RiskContainer(config['risk']))
        portfolio.add_component(PortfolioStateContainer(config['portfolio']))
        portfolio.add_component(ExecutionContainer(config['execution']))
        
        return portfolio
```

### 3. Portfolio System Container

```python
# src/core/containers/portfolio_system_container.py
class PortfolioSystemContainer(ComposableContainer):
    """Isolated container for a single portfolio"""
    
    def __init__(self, portfolio_id: str, event_bus: EventBus):
        super().__init__()
        self.portfolio_id = portfolio_id
        self.namespace = PortfolioNamespace(portfolio_id)
        self.event_bus = event_bus
        self.components = {}
        
        # Set correlation ID for all events
        self.correlation_id = self.namespace.create_correlation_id()
        self.event_bus.set_correlation_id(self.correlation_id)
        
    def publish_event(self, event: Event):
        """Publish event with portfolio namespace"""
        namespaced_event = self.namespace.wrap_event(event)
        self.event_bus.publish(namespaced_event, source=self)
```

### 4. Cross-Portfolio Analytics

```python
# src/analytics/cross_portfolio_analytics.py
class CrossPortfolioAnalytics:
    """Analyzes relationships between portfolios"""
    
    def __init__(self):
        self.portfolio_returns = defaultdict(list)
        self.correlation_window = 252  # 1 year
        
    def update_returns(self, portfolio_id: str, returns: pd.Series):
        """Track returns for correlation analysis"""
        self.portfolio_returns[portfolio_id].append(returns)
        
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate rolling correlations between portfolios"""
        returns_df = pd.DataFrame(self.portfolio_returns)
        return returns_df.rolling(self.correlation_window).corr()
        
    def detect_correlation_breaks(self):
        """Alert when portfolio correlations change significantly"""
        # Critical for risk management
```

### 5. Portfolio-Specific Reporting

```python
# src/reporting/multi_portfolio_reporter.py
class MultiPortfolioReporter:
    """Generate individual and consolidated reports"""
    
    def generate_portfolio_report(self, portfolio_id: str):
        """Individual portfolio performance report"""
        return {
            'portfolio_id': portfolio_id,
            'performance': self.calculate_performance(portfolio_id),
            'attribution': self.calculate_attribution(portfolio_id),
            'risk_metrics': self.calculate_risk_metrics(portfolio_id),
            'trades': self.get_portfolio_trades(portfolio_id)
        }
        
    def generate_consolidated_report(self):
        """System-wide performance report"""
        return {
            'total_aum': self.calculate_total_aum(),
            'portfolio_performance': self.get_all_portfolio_performance(),
            'correlation_matrix': self.calculate_correlations(),
            'risk_consolidation': self.consolidate_risk(),
            'cross_portfolio_analysis': self.analyze_portfolio_relationships()
        }
```

### 6. Event Routing Configuration

```yaml
# config/multi_portfolio_example.yaml
workflow:
  type: multi_portfolio_backtest
  
portfolios:
  - id: conservative
    initial_capital: 500000
    strategies:
      - type: mean_reversion
        parameters:
          lookback: 30
          entry_threshold: 1.5
    risk:
      max_position_pct: 2.0
      max_drawdown_pct: 10.0
      
  - id: aggressive  
    initial_capital: 500000
    strategies:
      - type: momentum
        parameters:
          fast_period: 5
          slow_period: 20
    risk:
      max_position_pct: 5.0
      max_drawdown_pct: 20.0
      
  - id: balanced
    initial_capital: 500000
    strategies:
      - type: ensemble
        components: [momentum, mean_reversion]
        weights: [0.5, 0.5]
    risk:
      max_position_pct: 3.0
      max_drawdown_pct: 15.0

cross_portfolio:
  rebalancing:
    frequency: monthly
    method: risk_parity
  correlation_monitoring:
    enabled: true
    alert_threshold: 0.8
```

## ✅ Implementation Checklist

### Core Infrastructure
- [ ] Create PortfolioNamespace class
- [ ] Implement TradingSystemContainer
- [ ] Create PortfolioSystemContainer
- [ ] Add portfolio-specific event buses
- [ ] Implement namespace wrapping

### Event Isolation
- [ ] Verify complete event isolation between portfolios
- [ ] Test no cross-portfolio contamination
- [ ] Validate independent correlation IDs
- [ ] Ensure separate event stores per portfolio
- [ ] Test parallel execution safety

### Analytics & Reporting
- [ ] Implement CrossPortfolioAnalytics
- [ ] Create correlation tracking
- [ ] Build performance aggregation
- [ ] Add risk consolidation
- [ ] Generate multi-portfolio reports

### Configuration & Management
- [ ] Design multi-portfolio YAML schema
- [ ] Create portfolio factory methods
- [ ] Implement portfolio lifecycle management
- [ ] Add dynamic portfolio creation/removal
- [ ] Build portfolio state persistence

### Testing
- [ ] Test 2-portfolio scenarios
- [ ] Test 10-portfolio scenarios  
- [ ] Verify memory scaling (linear, not exponential)
- [ ] Test portfolio isolation under stress
- [ ] Validate cross-portfolio analytics

## 🧪 Testing Requirements

### Isolation Tests
```python
def test_portfolio_isolation():
    """Verify complete isolation between portfolios"""
    system = TradingSystemContainer(config)
    
    # Generate events in portfolio 1
    portfolio1_events = generate_test_events()
    
    # Verify portfolio 2 receives ZERO events from portfolio 1
    assert len(portfolio2_received_events) == 0
    
def test_parallel_execution():
    """Test multiple portfolios running in parallel"""
    # Should have linear performance scaling
```

### Performance Tests
```python
def test_multi_portfolio_performance():
    """Benchmark multi-portfolio overhead"""
    # Single portfolio baseline
    single_time = run_single_portfolio()
    
    # 10 portfolio test
    multi_time = run_ten_portfolios()
    
    # Should be < 10x single (due to shared resources)
    assert multi_time < single_time * 8
```

### Analytics Tests
```python
def test_cross_portfolio_correlation():
    """Test correlation calculation accuracy"""
    # Known correlation scenario
    # Verify accurate calculation
    # Test correlation break detection
```

## 🎯 Success Criteria

### Functionality
- [ ] Can run 2+ portfolios simultaneously
- [ ] Complete isolation between portfolios
- [ ] Accurate cross-portfolio analytics
- [ ] Portfolio-specific reporting works
- [ ] Can scale to 10+ portfolios

### Performance  
- [ ] Linear memory scaling with portfolio count
- [ ] <20% overhead per additional portfolio
- [ ] Correlation calculations < 100ms
- [ ] Report generation < 5s for 10 portfolios

### Quality
- [ ] Zero cross-portfolio event leakage
- [ ] Reproducible results per portfolio
- [ ] Accurate performance attribution
- [ ] Stable under high load

## 🔗 Integration Points

### With Event Tracing (Step 10.8)
- Portfolio-specific correlation IDs
- Namespace in event metadata
- Portfolio attribution in patterns
- Cross-portfolio pattern analysis

### With Existing System
- Coordinator manages TradingSystemContainer
- Each portfolio has full component stack
- Shared data sources (with isolation)
- Consolidated reporting

## 📊 Use Case Examples

### Strategy Comparison
```python
# Run same strategy with different risk profiles
portfolios = [
    {"id": "conservative", "risk": {"max_position": 1000}},
    {"id": "moderate", "risk": {"max_position": 5000}},
    {"id": "aggressive", "risk": {"max_position": 10000}}
]
```

### Diversification Analysis
```python
# Run uncorrelated strategies
portfolios = [
    {"id": "trend", "strategy": "momentum"},
    {"id": "revert", "strategy": "mean_reversion"},
    {"id": "arb", "strategy": "pairs_trading"}
]
```

### Risk Segregation
```python
# Separate portfolios by asset class
portfolios = [
    {"id": "equities", "symbols": ["SPY", "QQQ"]},
    {"id": "commodities", "symbols": ["GLD", "USO"]},
    {"id": "forex", "symbols": ["EURUSD", "GBPUSD"]}
]
```

## 🚨 Common Pitfalls

### 1. Event Leakage
**Problem**: Events from one portfolio affect another
**Solution**: Strict namespace enforcement and validation

### 2. Memory Explosion
**Problem**: Memory usage grows exponentially
**Solution**: Shared immutable data, efficient state management

### 3. Correlation Calculation Cost
**Problem**: O(n²) correlation calculations
**Solution**: Incremental updates, sparse matrices

### 4. Report Generation Time
**Problem**: Slow consolidated reporting
**Solution**: Parallel report generation, caching

## 📈 Performance Optimization

### Memory Management
- Share immutable market data
- Use copy-on-write for state
- Implement state compression
- Regular garbage collection

### Computation Efficiency
- Parallel portfolio execution
- Vectorized correlation calculations
- Incremental metric updates
- Smart caching strategies

## 🎯 Next Steps

After completing multi-portfolio support:
1. Implement portfolio rebalancing strategies
2. Add inter-portfolio capital flows
3. Create portfolio optimization algorithms
4. Build risk parity allocation
5. Implement correlation-based hedging

## 📚 Additional Resources

- [Container Architecture](../../architecture/02-CONTAINER-HIERARCHY.md)
- [Event Isolation Guide](../validation-framework/event-bus-isolation.md)
- [Performance Testing](../testing-framework/README.md)
- [Risk Management Patterns](../../references/risk-management-patterns.md)
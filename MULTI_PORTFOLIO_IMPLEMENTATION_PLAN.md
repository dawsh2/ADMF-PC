# Multi-Portfolio Backtesting Implementation Plan

## Executive Summary

This plan provides a phased approach to implementing multi-portfolio backtesting in the ADMF-PC system. The plan prioritizes risk reduction, incremental progress, and building on the existing architecture while addressing current gaps.

## Current State Analysis

### Strengths
1. **Strong Portfolio State Management**: `PortfolioState` class already tracks positions, P&L, and risk metrics
2. **Event-Driven Architecture**: Event system supports isolation and routing
3. **Container Architecture**: Modular design allows for multiple instances
4. **Risk Management**: `RiskPortfolioContainer` manages signal-to-order conversion
5. **Reporting Infrastructure**: Basic HTML report generation exists

### Gaps and Challenges
1. **Event Isolation**: Need to validate complete event bus isolation between containers
2. **Testing Coverage**: Limited integration tests for multi-container scenarios
3. **Event Tracing**: Event flow tracking exists but needs enhancement for debugging
4. **Performance Tracking**: No comprehensive performance attribution per strategy
5. **Report Aggregation**: Current reporting focused on single portfolio

## Phased Implementation Approach

### Phase 1: Strengthen Foundation (Weeks 1-2)

#### Objectives
- Ensure rock-solid single-portfolio system
- Add comprehensive testing
- Implement event tracing infrastructure

#### Tasks

1. **Event Bus Isolation Validation**
   ```python
   # Create enhanced_isolation.py
   class IsolationValidator:
       def validate_container_isolation(self, containers: List[Container]):
           """Ensure no event leakage between containers"""
   ```
   - Implement isolation tests
   - Add strict mode enforcement
   - Create violation detection

2. **Comprehensive Testing Suite**
   ```python
   # tests/integration/test_portfolio_isolation.py
   class TestPortfolioIsolation:
       def test_parallel_backtests_no_interference(self):
           """Run parallel backtests and verify no cross-contamination"""
   ```
   - Test parallel execution
   - Verify state isolation
   - Check memory boundaries

3. **Enhanced Event Tracing**
   ```python
   # Enhance EventFlowTracer
   class EnhancedEventFlowTracer(EventFlowTracer):
       def trace_portfolio_event(self, portfolio_id: str, event: Event):
           """Track events specific to portfolio instances"""
   ```
   - Add portfolio-specific tracking
   - Create correlation analysis
   - Build debugging tools

#### Deliverables
- ✅ Isolation validation framework
- ✅ 20+ isolation test cases
- ✅ Event tracing dashboard
- ✅ Performance baseline metrics

#### Risk Mitigation
- Run existing backtests to establish baseline
- Create rollback procedures
- Document all changes

### Phase 2: Event Routing Infrastructure (Weeks 3-4)

#### Objectives
- Implement namespace-based event routing
- Add portfolio identification to events
- Create routing configuration

#### Tasks

1. **Namespace Support in Events**
   ```python
   @dataclass
   class Event:
       event_type: EventType
       payload: Dict[str, Any]
       namespace: Optional[str] = None  # Add namespace
       portfolio_id: Optional[str] = None  # Add portfolio tracking
   ```

2. **Enhanced Pipeline Adapter**
   ```python
   class NamespacedPipelineAdapter(PipelineAdapter):
       def route_event(self, event: Event):
           """Route based on namespace/portfolio_id"""
           if event.namespace:
               pipeline = self.pipelines.get(event.namespace)
               if pipeline:
                   self._process_pipeline(event, pipeline)
   ```

3. **Portfolio-Aware Data Distribution**
   ```python
   class MultiPortfolioDataContainer(DataContainer):
       def broadcast_to_portfolios(self, bar_event: Event, portfolio_ids: List[str]):
           """Broadcast market data to specific portfolios"""
   ```

#### Deliverables
- ✅ Namespaced event system
- ✅ Portfolio-aware routing
- ✅ Configuration schema updates
- ✅ Integration tests

#### Risk Mitigation
- Maintain backward compatibility
- Test with existing single-portfolio configs
- Performance benchmarking

### Phase 3: Multi-Portfolio Core (Weeks 5-6)

#### Objectives
- Implement multi-portfolio container management
- Create portfolio-specific reporting
- Add performance attribution

#### Tasks

1. **Trading System Container**
   ```python
   class TradingSystemContainer(UniversalScopedContainer):
       """Encapsulates complete trading pipeline with isolated portfolio"""
       def __init__(self, config: Dict, portfolio_id: str):
           self.portfolio_id = portfolio_id
           self.portfolio_state = PortfolioState(
               initial_capital=config['initial_capital']
           )
   ```

2. **Multi-Portfolio Workflow Manager**
   ```python
   class MultiPortfolioWorkflowManager:
       async def execute_multi_portfolio_backtest(self, config):
           """Orchestrate multiple portfolio backtests"""
           trading_systems = []
           for system_config in config['trading_systems']:
               system = await self.create_trading_system(system_config)
               trading_systems.append(system)
   ```

3. **Portfolio-Specific Reporting**
   ```python
   class MultiPortfolioReporter:
       def generate_portfolio_report(self, portfolio_id: str, state: PortfolioState):
           """Generate individual portfolio report"""
       
       def generate_combined_dashboard(self, portfolios: Dict[str, PortfolioState]):
           """Create combined performance dashboard"""
   ```

#### Deliverables
- ✅ TradingSystemContainer implementation
- ✅ Multi-portfolio configuration support
- ✅ Individual portfolio reports
- ✅ Combined dashboard

#### Risk Mitigation
- Start with 2 portfolios maximum
- Extensive logging and monitoring
- Memory usage profiling

### Phase 4: Advanced Features (Weeks 7-8)

#### Objectives
- Add cross-portfolio analytics
- Implement portfolio rebalancing
- Create correlation analysis

#### Tasks

1. **Cross-Portfolio Analytics**
   ```python
   class PortfolioAnalyzer:
       def calculate_correlation_matrix(self, portfolios: Dict[str, PortfolioState]):
           """Calculate return correlations between portfolios"""
       
       def attribution_analysis(self, portfolios: Dict[str, PortfolioState]):
           """Performance attribution across portfolios"""
   ```

2. **Dynamic Portfolio Management**
   ```python
   class PortfolioRebalancer:
       def rebalance_allocations(self, portfolios: Dict[str, PortfolioState],
                                target_weights: Dict[str, float]):
           """Rebalance capital across portfolios"""
   ```

3. **Advanced Reporting**
   - Correlation heatmaps
   - Risk contribution analysis
   - Portfolio efficiency frontier
   - Drawdown comparison

#### Deliverables
- ✅ Analytics dashboard
- ✅ Rebalancing engine
- ✅ Advanced visualizations
- ✅ Performance comparison tools

## Configuration Example

```yaml
# config/multi_portfolio_backtest.yaml
workflow:
  type: "multi_portfolio_backtest"
  name: "Dual Strategy Test"

data:
  source: "csv"
  file_path: "data/SPY_1m.csv"
  symbols: ["SPY"]

trading_systems:
  - portfolio_id: "momentum_portfolio"
    initial_capital: 50000
    namespace: "momentum"
    
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

  - portfolio_id: "mean_reversion_portfolio"
    initial_capital: 50000
    namespace: "mean_rev"
    
    strategy:
      type: "mean_reversion"
      parameters:
        lookback_period: 15
        entry_threshold: 1.0
    
    risk:
      position_sizers:
        - type: "percentage"
          size: 0.02

reporting:
  individual_reports: true
  combined_dashboard: true
  correlation_analysis: true
```

## Testing Strategy

### Unit Tests
```python
# tests/unit/core/test_multi_portfolio.py
class TestMultiPortfolio:
    def test_portfolio_isolation(self):
        """Test complete isolation between portfolios"""
    
    def test_event_routing(self):
        """Test namespace-based routing"""
    
    def test_performance_attribution(self):
        """Test per-portfolio performance tracking"""
```

### Integration Tests
```python
# tests/integration/test_multi_portfolio_flow.py
class TestMultiPortfolioFlow:
    def test_dual_portfolio_backtest(self):
        """Run complete dual portfolio backtest"""
    
    def test_portfolio_reporting(self):
        """Test individual and combined reporting"""
```

### Performance Tests
```python
# tests/performance/test_multi_portfolio_scale.py
class TestMultiPortfolioScale:
    def test_10_portfolios_parallel(self):
        """Test system with 10 parallel portfolios"""
    
    def test_memory_usage(self):
        """Monitor memory usage during execution"""
```

## Success Criteria

### Phase 1
- ✅ All isolation tests passing
- ✅ Event tracing operational
- ✅ Zero event leakage detected
- ✅ Performance baseline established

### Phase 2
- ✅ Namespace routing working
- ✅ Backward compatibility maintained
- ✅ Configuration validation passing
- ✅ Integration tests green

### Phase 3
- ✅ Dual portfolio backtest running
- ✅ Individual reports generated
- ✅ Combined dashboard created
- ✅ Memory usage acceptable

### Phase 4
- ✅ Correlation analysis working
- ✅ Rebalancing tested
- ✅ Advanced reports generated
- ✅ System scalable to 10+ portfolios

## Risk Mitigation Strategies

1. **Incremental Development**
   - Small, testable changes
   - Continuous integration
   - Regular code reviews

2. **Comprehensive Testing**
   - Unit tests for each component
   - Integration tests for workflows
   - Performance benchmarking

3. **Monitoring and Logging**
   - Enhanced event tracing
   - Performance metrics
   - Memory profiling

4. **Rollback Procedures**
   - Version control discipline
   - Feature flags for new code
   - Backward compatibility

5. **Documentation**
   - Update architecture docs
   - Create migration guide
   - Document configuration changes

## Implementation Timeline

```
Week 1-2: Phase 1 - Foundation
  - Event isolation validation
  - Testing infrastructure
  - Event tracing enhancement

Week 3-4: Phase 2 - Event Routing
  - Namespace implementation
  - Pipeline adapter updates
  - Configuration support

Week 5-6: Phase 3 - Multi-Portfolio Core
  - Trading system containers
  - Portfolio-specific reporting
  - Basic multi-portfolio support

Week 7-8: Phase 4 - Advanced Features
  - Cross-portfolio analytics
  - Rebalancing capabilities
  - Advanced visualizations
```

## Next Steps

1. **Immediate Actions**
   - Create `enhanced_isolation.py`
   - Write isolation test suite
   - Set up event tracing dashboard

2. **Week 1 Deliverables**
   - Isolation validation framework
   - 10+ isolation tests
   - Performance baseline

3. **Communication**
   - Weekly progress updates
   - Risk assessment reports
   - Demo sessions

## Conclusion

This plan provides a systematic approach to implementing multi-portfolio backtesting while minimizing risk and ensuring system stability. By building incrementally and testing thoroughly, we can deliver a robust multi-portfolio system that maintains the architectural principles of ADMF-PC.
# Step 15: Scale to Institutional - Multi-Account/Multi-PM

## 📋 Status: Advanced (95% Complexity)
**Estimated Time**: 4-6 weeks
**Difficulty**: Very High
**Prerequisites**: Steps 1-14 completed, production infrastructure

## 🎯 Objectives

Transform the system to handle institutional-scale operations with multiple portfolio managers, accounts, and complex allocation strategies.

## 🔗 Architecture References

- **Container Hierarchy**: [Container Architecture](../../architecture/02-CONTAINER-HIERARCHY.md)
- **Protocol Design**: [Protocol Composition](../../architecture/03-PROTOCOL-COMPOSITION.md)
- **Risk Management**: [src/risk/README.md](../../../src/risk/README.md)
- **Execution**: [src/execution/README.md](../../../src/execution/README.md)

## 📚 Required Reading

1. **System Architecture**: Review container isolation patterns
2. **Risk Module**: Understand hierarchical risk management
3. **Execution Module**: Study order routing and allocation
4. **Coordinator**: Learn multi-phase workflow orchestration

## 🏗️ Implementation Tasks

### 1. Multi-Portfolio Manager Architecture

```python
# src/portfolio/multi_pm_container.py
from src.core.containers.protocols import ContainerProtocol
from src.risk.protocols import RiskProtocol
from typing import Dict, List, Optional

class MultiPMContainer(ContainerProtocol):
    """
    Container managing multiple portfolio managers with isolated risk.
    
    Architecture:
    Institution Container
        ├── PM Container A (isolated)
        │   ├── Strategy Containers
        │   ├── Risk Container (PM limits)
        │   └── Account Containers
        ├── PM Container B (isolated)
        │   ├── Strategy Containers
        │   ├── Risk Container (PM limits)
        │   └── Account Containers
        └── Master Risk Container (firm limits)
    """
    
    def __init__(self, config: Dict):
        self.pm_containers: Dict[str, PMContainer] = {}
        self.master_risk = MasterRiskContainer(config['firm_limits'])
        self.allocation_engine = AllocationEngine(config['allocation'])
        
    def add_portfolio_manager(self, pm_id: str, pm_config: Dict):
        """Add new PM with isolated container and risk limits"""
        pm_container = PMContainer(
            pm_id=pm_id,
            capital_allocation=pm_config['capital'],
            risk_limits=pm_config['limits'],
            allowed_strategies=pm_config['strategies']
        )
        
        # Register with master risk
        self.master_risk.register_pm(pm_id, pm_container.risk_container)
        
        # Wire event routing
        self._setup_pm_event_routing(pm_container)
        
        self.pm_containers[pm_id] = pm_container
```

### 2. Capital Allocation Engine

```python
# src/portfolio/allocation_engine.py
class AllocationEngine:
    """
    Manages capital allocation across PMs and strategies.
    
    Features:
    - Dynamic capital allocation
    - Performance-based rebalancing
    - Risk-adjusted allocation
    - Drawdown-based reduction
    """
    
    def __init__(self, allocation_config: Dict):
        self.allocation_method = allocation_config['method']
        self.rebalance_frequency = allocation_config['rebalance_freq']
        self.performance_window = allocation_config['perf_window']
        
    def calculate_allocations(self, 
                            pm_performance: Dict[str, float],
                            pm_risk_metrics: Dict[str, Dict],
                            firm_capital: float) -> Dict[str, float]:
        """
        Calculate capital allocations for each PM.
        
        Methods:
        - Equal weight
        - Risk parity
        - Performance weighted
        - Sharpe optimized
        - Custom formula
        """
        if self.allocation_method == 'risk_parity':
            return self._risk_parity_allocation(pm_risk_metrics, firm_capital)
        elif self.allocation_method == 'performance_weighted':
            return self._performance_weighted(pm_performance, firm_capital)
        # ... other methods
        
    def _risk_parity_allocation(self, risk_metrics: Dict, capital: float):
        """Allocate capital for equal risk contribution"""
        # Calculate inverse volatility weights
        total_inv_vol = sum(1/metrics['volatility'] 
                          for metrics in risk_metrics.values())
        
        allocations = {}
        for pm_id, metrics in risk_metrics.items():
            weight = (1/metrics['volatility']) / total_inv_vol
            allocations[pm_id] = capital * weight
            
        return allocations
```

### 3. Multi-Account Management

```python
# src/portfolio/account_management.py
class AccountContainer:
    """
    Manages individual account within PM's allocation.
    
    Features:
    - Account-specific restrictions
    - Tax lot tracking
    - Regulatory compliance
    - Client-specific rules
    """
    
    def __init__(self, account_id: str, account_config: Dict):
        self.account_id = account_id
        self.restrictions = AccountRestrictions(account_config['restrictions'])
        self.tax_lot_tracker = TaxLotTracker(account_config['tax_method'])
        self.compliance = ComplianceEngine(account_config['regulations'])
        
class AccountRestrictions:
    """Enforces account-specific trading restrictions"""
    
    def __init__(self, restrictions: Dict):
        self.prohibited_symbols = set(restrictions.get('prohibited', []))
        self.max_position_size = restrictions.get('max_position', 0.05)
        self.allowed_asset_classes = set(restrictions.get('asset_classes', []))
        self.min_market_cap = restrictions.get('min_market_cap', 0)
        
    def check_order(self, order: Order) -> Tuple[bool, Optional[str]]:
        """Validate order against account restrictions"""
        if order.symbol in self.prohibited_symbols:
            return False, f"Symbol {order.symbol} prohibited"
            
        if order.asset_class not in self.allowed_asset_classes:
            return False, f"Asset class {order.asset_class} not allowed"
            
        # Additional checks...
        return True, None
```

### 4. Order Allocation System

```python
# src/execution/order_allocation.py
class OrderAllocationSystem:
    """
    Allocates orders across multiple accounts based on rules.
    
    Allocation methods:
    - Pro-rata by capital
    - Equal shares
    - Custom allocation
    - Priority-based
    """
    
    def allocate_order(self, 
                      master_order: Order,
                      accounts: List[Account],
                      method: str = 'pro_rata') -> List[Order]:
        """Split master order into account-specific orders"""
        
        # Filter eligible accounts
        eligible_accounts = [
            acc for acc in accounts 
            if acc.restrictions.check_order(master_order)[0]
        ]
        
        if method == 'pro_rata':
            return self._pro_rata_allocation(master_order, eligible_accounts)
        elif method == 'equal_shares':
            return self._equal_allocation(master_order, eligible_accounts)
        # ... other methods
        
    def _pro_rata_allocation(self, order: Order, accounts: List[Account]):
        """Allocate based on account capital"""
        total_capital = sum(acc.capital for acc in accounts)
        
        allocated_orders = []
        remaining_shares = order.quantity
        
        for i, account in enumerate(accounts):
            if i == len(accounts) - 1:
                # Last account gets remaining shares (handles rounding)
                shares = remaining_shares
            else:
                weight = account.capital / total_capital
                shares = int(order.quantity * weight)
                remaining_shares -= shares
                
            if shares > 0:
                allocated_orders.append(
                    order.copy(account_id=account.id, quantity=shares)
                )
                
        return allocated_orders
```

### 5. Master Risk Aggregation

```python
# src/risk/master_risk_container.py
class MasterRiskContainer:
    """
    Aggregates risk across all PMs and enforces firm-wide limits.
    
    Monitors:
    - Total firm exposure
    - Concentration risk
    - Correlation risk
    - Liquidity risk
    - Regulatory limits
    """
    
    def __init__(self, firm_limits: Dict):
        self.firm_limits = firm_limits
        self.pm_risks: Dict[str, PMRiskContainer] = {}
        self.aggregated_positions = PositionAggregator()
        
    def check_firm_limits(self, proposed_order: Order) -> Tuple[bool, str]:
        """Check if order would breach firm-wide limits"""
        
        # Aggregate current positions
        total_exposure = self.aggregated_positions.calculate_gross_exposure()
        concentration = self.aggregated_positions.calculate_concentration()
        
        # Simulate order impact
        simulated_exposure = total_exposure + proposed_order.value
        simulated_concentration = self._simulate_concentration(proposed_order)
        
        # Check limits
        if simulated_exposure > self.firm_limits['max_gross_exposure']:
            return False, "Would exceed firm gross exposure limit"
            
        if simulated_concentration > self.firm_limits['max_single_name']:
            return False, "Would exceed single name concentration limit"
            
        return True, "Approved"
```

## 🧪 Testing Requirements

### Unit Tests

```python
# tests/test_multi_pm.py
def test_pm_isolation():
    """Test that PMs are truly isolated"""
    institution = MultiPMContainer(config)
    
    # Add two PMs
    institution.add_portfolio_manager('PM_A', pm_a_config)
    institution.add_portfolio_manager('PM_B', pm_b_config)
    
    # PM A generates signal
    pm_a_signal = Signal('AAPL', 'BUY', 0.8)
    institution.pm_containers['PM_A'].process_signal(pm_a_signal)
    
    # Verify PM B doesn't see it
    assert len(institution.pm_containers['PM_B'].signals) == 0

def test_capital_allocation():
    """Test various allocation methods"""
    engine = AllocationEngine({'method': 'risk_parity'})
    
    pm_metrics = {
        'PM_A': {'volatility': 0.15, 'sharpe': 1.2},
        'PM_B': {'volatility': 0.10, 'sharpe': 1.5}
    }
    
    allocations = engine.calculate_allocations(
        pm_performance={},
        pm_risk_metrics=pm_metrics,
        firm_capital=100_000_000
    )
    
    # PM B should get more capital (lower vol)
    assert allocations['PM_B'] > allocations['PM_A']
```

### Integration Tests

```python
def test_full_institutional_flow():
    """Test complete multi-PM trading flow"""
    # 1. Setup institution with multiple PMs
    institution = setup_test_institution()
    
    # 2. Each PM runs different strategies
    for pm_id, pm in institution.pm_containers.items():
        pm.start_strategies()
    
    # 3. Process market data
    market_data = load_test_data()
    institution.process_market_data(market_data)
    
    # 4. Verify orders allocated correctly
    orders = institution.get_all_orders()
    verify_order_allocation(orders)
    
    # 5. Check risk aggregation
    assert institution.master_risk.check_all_limits()
```

### System Tests

```python
def test_institutional_scale():
    """Test with realistic institutional setup"""
    # 10 PMs, 100 accounts, 500 strategies
    config = create_large_scale_config(
        num_pms=10,
        accounts_per_pm=10,
        strategies_per_pm=50
    )
    
    institution = MultiPMContainer(config)
    
    # Run full day simulation
    results = run_institutional_backtest(
        institution,
        data='full_universe',
        date='2023-01-01'
    )
    
    # Verify performance at scale
    assert results['avg_latency'] < 100  # milliseconds
    assert results['orders_processed'] > 10000
    assert results['risk_breaches'] == 0
```

## 🎮 Validation Checklist

### Isolation Validation
- [ ] PMs cannot see each other's positions
- [ ] PMs cannot exceed individual capital allocations
- [ ] PM failure doesn't affect other PMs
- [ ] Strategies isolated within PM containers

### Scale Validation
- [ ] System handles 10+ PMs efficiently
- [ ] 100+ accounts managed without degradation
- [ ] 1000+ strategies execute in parallel
- [ ] Order allocation completes in milliseconds

### Risk Validation
- [ ] Firm-wide limits enforced
- [ ] PM-specific limits enforced
- [ ] Account restrictions respected
- [ ] Real-time risk aggregation works

### Performance Validation
- [ ] Sub-second order generation
- [ ] Efficient capital allocation
- [ ] Low-latency risk checks
- [ ] Scalable to more PMs

## 💾 Memory Management

```python
# Memory considerations for institutional scale
class InstitutionalMemoryManager:
    def __init__(self):
        self.position_cache_size = 10000  # positions
        self.order_buffer_size = 50000   # orders/day
        self.signal_retention = 3600     # seconds
        
    def estimate_memory_usage(self, num_pms: int, num_accounts: int):
        """Estimate memory requirements"""
        base_per_pm = 500  # MB
        per_account = 10   # MB
        per_strategy = 50  # MB
        
        total_mb = (
            base_per_pm * num_pms +
            per_account * num_accounts +
            per_strategy * num_pms * 50  # avg strategies
        )
        
        return total_mb
```

## 🔧 Common Issues

1. **Capital Allocation Drift**: Monitor and rebalance regularly
2. **Order Allocation Rounding**: Use remainder handling
3. **Risk Aggregation Lag**: Implement caching strategies
4. **PM Isolation Leaks**: Strict event bus boundaries

## ✅ Success Criteria

- [ ] Multi-PM architecture fully functional
- [ ] Capital allocation engine working
- [ ] Order allocation system operational
- [ ] Risk aggregation in real-time
- [ ] Performance meets institutional standards
- [ ] Complete audit trail maintained

## 📚 Next Steps

Once institutional multi-PM scale is achieved:
1. Proceed to [Step 16: Scale to 1000+ Symbols](step-16-massive-universe.md)
2. Implement additional allocation strategies
3. Add more sophisticated risk aggregation
4. Enhance compliance monitoring
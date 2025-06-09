## Order Lifecycle Management: Complete Event Flow 🔄

### **Complete Order Journey: Risk Manager → Portfolio State Update**

This section traces the complete order lifecycle from when the Risk Manager creates an order through final portfolio state updates.

### **Step 1: Portfolio Receives Order from Risk Manager**
```python
class PortfolioContainer:
    def process_signal(self, signal: Signal, regime: RegimeState):
        """Portfolio orchestrates risk decision and handles result."""
        
        # Risk manager creates order (or None)
        order = StatelessRiskManager.process_signal_to_order(
            signal=signal,
            regime=regime,
            portfolio_state=self.portfolio_state,
            risk_config=self.get_risk_config(),
            market_data=self.get_market_data()
        )
        
        if order:
            self._handle_approved_order(order)
        else:
            self._handle_rejected_signal(signal, regime)
    
    def _handle_approved_order(self, order: Order):
        """Portfolio processes approved order from risk manager."""
        
        # 1. Track pending order (stateful)
        self.pending_orders[order.order_id] = order
        self.portfolio_state.record_pending_order(order)
        
        # 2. Route order to execution
        self.send_order_to_execution(order)
        
        # 3. Record order creation for analytics
        self.analytics.record_order_created(
            order=order,
            portfolio_id=self.portfolio_id,
            signal_context={'signal': signal, 'regime': regime},
            correlation_id=self.correlation_id
        )
    
    def _handle_rejected_signal(self, signal: Signal, regime: RegimeState):
        """Portfolio handles signals rejected by risk manager."""
        
        # Record rejection for analytics
        self.rejected_signals.append({
            'signal': signal,
            'regime': regime,
            'timestamp': datetime.now(),
            'reason': 'risk_manager_rejection'
        })
        
        self.analytics.record_signal_rejection(
            signal=signal,
            regime=regime,
            portfolio_id=self.portfolio_id,
            correlation_id=self.correlation_id
        )
```

### **Step 2: Execution Container Processes Order**
```python
class ExecutionContainer:
    """Stateful container managing order lifecycle with stateless execution logic."""
    
    def receive_order(self, order: Order):
        """Execution container receives order from portfolio."""
        
        try:
            # Update stateful order tracking
            self.active_orders.add(order.order_id)
            self.order_history.append(order)
            
            # Execute order using stateless simulation
            fill = self._execute_order_with_simulation(order)
            
            # Update stateful execution state
            self.active_orders.remove(order.order_id)
            self.execution_stats.record_fill(fill)
            
            # Route fill back to originating portfolio
            self._route_fill_to_portfolio(fill)
            
        except ExecutionError as e:
            # Handle execution failure
            self._handle_execution_failure(order, e)
    
    def _execute_order_with_simulation(self, order: Order) -> Fill:
        """Execute order using stateless execution logic."""
        
        # Get current market data
        market_data = self.get_current_market_data()
        
        # Use stateless execution simulator (pure function)
        fill = StatelessExecutionSimulator.simulate_execution(
            order=order,
            market_data=market_data,
            execution_config=self.execution_config
        )
        
        return fill
    
    def _route_fill_to_portfolio(self, fill: Fill):
        """Route fill back to the portfolio that created the order."""
        
        # Get the correct portfolio container
        target_portfolio = self.portfolio_containers[fill.portfolio_id]
        
        # Send fill to originating portfolio
        target_portfolio.receive_fill(fill)
    
    def _handle_execution_failure(self, order: Order, error: ExecutionError):
        """Handle order execution failures."""
        
        # Remove from active orders
        self.active_orders.discard(order.order_id)
        
        # Create rejection message
        rejection = OrderRejection(
            order_id=order.order_id,
            original_order=order,
            reason=str(error),
            timestamp=datetime.now(),
            portfolio_id=order.portfolio_id
        )
        
        # Route rejection back to originating portfolio
        target_portfolio = self.portfolio_containers[order.portfolio_id]
        target_portfolio.receive_order_rejection(rejection)
```

### **Step 3: Portfolio Receives Fill and Updates State**
```python
class PortfolioContainer:
    def receive_fill(self, fill: Fill):
        """Portfolio receives fill from execution container."""
        
        # 1. Remove from pending orders (stateful)
        original_order = None
        if fill.order_id in self.pending_orders:
            original_order = self.pending_orders.pop(fill.order_id)
        
        # 2. Update portfolio state (stateful operations)
        self.portfolio_state.update_position(fill)        # Position tracking
        self.portfolio_state.update_cash_balance(fill)    # Cash tracking
        self.portfolio_state.record_transaction(fill)     # Transaction history
        
        # 3. Recalculate portfolio metrics (stateful)
        current_prices = self.get_current_market_prices()
        self.portfolio_state.calculate_unrealized_pnl(current_prices)
        self.portfolio_state.update_performance_metrics()
        
        # 4. Record fill for analytics (correlation tracking)
        self.analytics.record_fill_processed(
            fill=fill,
            original_order=original_order,
            portfolio_state=self.portfolio_state,
            correlation_id=self.correlation_id
        )
        
        # 5. Optional: Background analytics storage (async I/O)
        if self.analytics_enabled:
            asyncio.create_task(
                self.analytics_db.store_portfolio_update(
                    portfolio_id=self.portfolio_id,
                    fill=fill,
                    portfolio_state=self.portfolio_state,
                    correlation_id=self.correlation_id
                )
            )
        
        # 6. Optional: Trigger portfolio rebalancing
        if self.should_rebalance_after_fill(fill):
            self.schedule_rebalance_check()
    
    def receive_order_rejection(self, rejection: OrderRejection):
        """Portfolio handles order execution failures."""
        
        # Remove from pending orders
        if rejection.order_id in self.pending_orders:
            failed_order = self.pending_orders.pop(rejection.order_id)
        
        # Record execution failure for analytics
        self.analytics.record_execution_failure(
            rejection=rejection,
            portfolio_id=self.portfolio_id,
            correlation_id=self.correlation_id
        )
        
        # Optional: Retry logic or alternative action
        if self.should_retry_failed_order(rejection):
            self.schedule_order_retry(rejection.original_order)
```

### **Step 4: Portfolio State Consistency and Isolation**
```python
# Each portfolio maintains complete isolation
class PortfolioContainer:
    def __init__(self, portfolio_id: str, combo_config: Dict):
        # Stateful components (isolated per portfolio)
        self.portfolio_id = portfolio_id
        self.portfolio_state = PortfolioState(portfolio_id)
        self.pending_orders: Dict[str, Order] = {}         # Orders awaiting fills
        self.completed_orders: List[Order] = []            # Historical orders
        self.rejected_signals: List[Dict] = []             # Rejected signals
        self.execution_failures: List[OrderRejection] = [] # Failed orders
        
        # Configuration (unique per combination)
        self.strategy_config = combo_config['strategy']
        self.classifier_config = combo_config['classifier']
        self.risk_config = combo_config['risk']
        self.correlation_id = combo_config['correlation_id']
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get complete portfolio state for monitoring."""
        return {
            'portfolio_id': self.portfolio_id,
            'total_value': self.portfolio_state.get_total_value(),
            'cash_balance': self.portfolio_state.get_cash_balance(),
            'positions': self.portfolio_state.get_all_positions(),
            'pending_orders': len(self.pending_orders),
            'completed_orders': len(self.completed_orders),
            'rejected_signals': len(self.rejected_signals),
            'unrealized_pnl': self.portfolio_state.get_unrealized_pnl(),
            'realized_pnl': self.portfolio_state.get_realized_pnl()
        }
```

### **Complete Order Lifecycle Flow Diagram**

```
┌─────────────────┐    Signal + Regime    ┌─────────────────┐
│   Strategy/     │────────────────────────▶│   Portfolio     │
│   Classifier    │                        │   Container     │
│   Services      │                        │   (Stateful)    │
└─────────────────┘                        └─────────┬───────┘
                                                     │
                                           ┌─────────▼───────┐
                                           │ Risk Manager    │
                                           │ (Stateless)     │
                                           │ Creates ORDER   │
                                           └─────────┬───────┘
                                                     │
                                           ┌─────────▼───────┐
                                           │   Portfolio     │
                                           │   Tracks Order  │
                                           │   Routes to     │
                                           │   Execution     │
                                           └─────────┬───────┘
                                                     │
                                           ┌─────────▼───────┐
                                           │   Execution     │
                                           │   Container     │
                                           │   (Stateful +   │
                                           │   Stateless)    │
                                           └─────────┬───────┘
                                                     │
                                             FILL or REJECTION
                                                     │
                                           ┌─────────▼───────┐
                                           │   Portfolio     │
                                           │   Updates State │
                                           │   Records       │
                                           │   Analytics     │
                                           └─────────────────┘
```

### **Key Lifecycle Management Benefits** ✅

1. **Complete Portfolio Isolation**: Each portfolio only receives its own fills
2. **Order State Tracking**: Pending orders tracked until filled or rejected
3. **Error Handling**: Execution failures handled gracefully with portfolio notification
4. **Analytics Integration**: Complete order lifecycle tracked with correlation IDs
5. **State Consistency**: Portfolio state always reflects actual positions
6. **Async Optimization**: Only I/O operations (analytics) are async
7. **Retry Logic**: Failed orders can be retried or handled with alternative actions

This complete lifecycle ensures that **every signal is tracked from generation through final portfolio state update**, providing comprehensive observability and reliable state management across all portfolio combinations.

## Execution Engine: Stateful vs Stateless Breakdown 🔧

### **Execution Engine Components Classification**

```python
# STATEFUL Components (Must Be Containers)
class ExecutionContainer:
    def __init__(self):
        self.active_orders: Set[str] = set()           # ✅ STATEFUL: Order lifecycle tracking
        self.execution_stats = ExecutionStatistics()   # ✅ STATEFUL: Performance metrics
        self.order_history: List[Order] = []          # ✅ STATEFUL: Historical order log
        self.pending_fills: Dict[str, Fill] = {}      # ✅ STATEFUL: Pending fill tracking

# STATELESS Components (Pure Functions)
class StatelessExecutionSimulator:
    @staticmethod
    def simulate_execution(order, market_data, config) -> Fill:
        """❌ STATELESS: Pure execution simulation logic"""
        pass
    
    @staticmethod
    def calculate_market_impact(quantity, volume, config) -> float:
        """❌ STATELESS: Pure market impact calculation"""
        pass
    
    @staticmethod
    def calculate_slippage(order_type, spread, impact) -> float:
        """❌ STATELESS: Pure slippage calculation"""
        pass

class StatelessOrderValidator:
    @staticmethod
    def validate_order_format(order) -> ValidationResult:
        """❌ STATELESS: Pure order format validation"""
        pass
```

### **Why This Split Makes Sense**

**Stateful Container Responsibilities:**
- **Order Lifecycle Management**: Track which orders are active, pending, completed
- **Execution Statistics**: Aggregate performance metrics over time
- **Order History**: Maintain record of all orders for compliance/audit
- **State Coordination**: Manage order state transitions (submitted → filled → settled)

**Stateless Service Responsibilities:**
- **Execution Simulation**: Calculate fills based on market conditions (pure function)
- **Market Impact Models**: Mathematical models for price impact (deterministic)
- **Slippage Calculations**: Bid-ask spread and impact calculations (pure math)
- **Order Validation**: Format and business rule checks (no state needed)

This separation allows the execution logic to be **perfectly parallelizable** while maintaining essential order lifecycle state in containers.

# Coordinator and Functional Architecture Refactor

## Executive Summary 🎯

This document outlines the evolution from pure container-based architecture to a **hybrid stateful containers + stateless services** approach. The key insight is moving functional/stateless components (strategies, classifiers, risk validation) into pure functions while maintaining stateful components (data, portfolio, execution, feature_hub) as containers. This approach:

- **Reduces container overhead by 60%** while maintaining complete isolation
- **Enables automatic parameter expansion** from simple YAML configurations
- **Improves parallelization** through pure function safety
- **Enhances event tracing** with service-level granularity
- **Maintains composability** through the existing WorkflowManager patterns

## Motivating Examples: Simple → Powerful 🚀

### What Users Should Write (Simple, Clean)
```yaml
# Simple user configuration - no container plumbing knowledge required
workflow:
  type: optimization
  base_pattern: full_backtest

strategies:
  - type: momentum
    parameters:
      lookback_period: [10, 20, 30]      # System auto-detects parameter grid
      signal_threshold: [0.01, 0.02]     # 6 combinations (3×2)

classifiers:
  - type: hmm
    parameters:
      model_type: ['hmm', 'svm']         # System auto-detects parameter grid  
      lookback_days: [30, 60]            # 4 combinations (2×2)

# Optional: Traditional workflow modes still supported
data:
  symbols: ['AAPL', 'MSFT']
  start_date: '2022-01-01'
  end_date: '2023-12-31'
```

### What System Automatically Generates (Powerful)
```python
# System automatically creates optimal execution topology
ExecutableWorkflow(
    total_combinations=24,  # 6 strategy × 4 classifier = 24 cross-combinations
    
    # Minimal stateful containers (only what needs state)
    stateful_containers={
        'data': DataContainer(shared=True),                     # Streaming position, timeline
        'feature_hub': FeatureHubContainer(shared=True),        # Indicator cache
        'execution': ExecutionContainer(shared=True),           # Order tracking
        'portfolios': [
            PortfolioContainer(combo_id=f"combo_{i}")          # Position state per combo
            for i in range(24)
        ]
    },
    
    # Stateless services (pure functions - no containers needed)
    stateless_services={
        'strategy_services': [
            MomentumStrategy(lookback=10, threshold=0.01),
            MomentumStrategy(lookback=10, threshold=0.02),
            MomentumStrategy(lookback=20, threshold=0.01),
            # ... 6 total strategy configurations
        ],
        'classifier_services': [
            HMMClassifier(model='hmm', lookback=30),
            HMMClassifier(model='hmm', lookback=60),
            HMMClassifier(model='svm', lookback=30),
            HMMClassifier(model='svm', lookback=60)
        ],
        'risk_service': StatelessRiskValidator()
    },
    
    # Auto-generated analytics tracking
    analytics=AnalyticsConfig(
        correlation_ids=[f"workflow_combo_{i}" for i in range(24)],
        parameter_tracking=True,
        performance_comparison=True
    )
)
```

**Resource Efficiency Comparison:**
- **Before**: 24 strategy + 24 classifier + 24 portfolio + shared containers = ~75 containers
- **After**: 24 portfolio + 3 shared containers + 24 lightweight service instances = ~27 containers
- **Result**: 60% reduction in container overhead while maintaining complete isolation! 🎯

## Architectural Evolution: Current → Target 📈

### Current State Analysis

**Problems with Pure Container Architecture:**
1. **Hardcoded Container Roles**: Only handles specific types (strategy, risk, execution)
2. **Fixed Container Hierarchy**: Can't dynamically add classifier, data variants, or custom containers  
3. **Container Overhead**: Everything containerized, even stateless logic
4. **Limited Scalability**: No support for arbitrary parameter combinations
5. **Monolithic Coordinator**: Bypasses WorkflowManager delegation patterns

### Target Architecture: Stateful Containers + Stateless Services

```
Simple YAML Config
    ↓ (auto-parameter detection)
Enhanced WorkflowManager 
    ↓ (pattern-based topology generation)
Minimal Stateful Containers + Stateless Service Orchestration
    ↓ (event-driven execution)
Analytics Integration with Correlation Tracking
```

## Stateful vs Stateless Component Classification 🔍

### Definitively Stateful (Must Be Containers)

```python
STATEFUL_COMPONENTS = {
    'data': {
        'why': 'Streaming position, timeline coordination, loaded data cache',
        'state': ['current_indices', 'timeline_idx', 'data_cache', 'splits'],
        'evidence': 'SimpleHistoricalDataHandler tracks position across symbols'
    },
    'portfolio': {
        'why': 'Position tracking, cash balance, P&L history, risk metrics',
        'state': ['positions', 'cash_balance', 'value_history', 'returns_history'],
        'evidence': 'PortfolioState maintains essential trading state'
    },
    'feature_hub': {
        'why': 'Calculated indicators cache, computation optimization',
        'state': ['indicator_cache', 'calculation_history', 'dependencies'],
        'evidence': 'Expensive indicator calculations need caching'
    },
    'execution': {
        'why': 'Active orders tracking, execution state, order lifecycle',
        'state': ['active_orders', 'pending_fills', 'execution_stats'],
        'evidence': 'DefaultExecutionEngine tracks order lifecycles'
    }
}
```

### Can Be Stateless (Pure Functions)

```python
STATELESS_SERVICES = {
    'strategy': {
        'why': 'Pure signal generation based on features',
        'pure_function': 'generate_signal(features, parameters) -> Signal',
        'evidence': 'MomentumStrategy logic is deterministic calculation'
    },
    'classifier': {
        'why': 'Pure regime detection based on features', 
        'pure_function': 'classify_regime(features, parameters) -> Regime',
        'evidence': 'HMM classification is stateless pattern recognition'
    },
    'risk_validator': {
        'why': 'Pure validation based on portfolio state',
        'pure_function': 'validate_order(order, portfolio_state, limits) -> Decision',
        'evidence': 'Risk limits are pure calculations given portfolio state'
    },
    'order_validator': {
        'why': 'Pure order format/business rule validation',
        'pure_function': 'validate_order_format(order) -> ValidationResult',
        'evidence': 'Order validation is pure business logic'
    },
    'market_simulator': {
        'why': 'Pure execution simulation based on market conditions',
        'pure_function': 'simulate_execution(order, market_data) -> Fill',
        'evidence': 'Market simulation is deterministic given inputs'
    }
}
```

## Enhanced WorkflowManager with Auto-Parameter Expansion 🛠️

### Step 1: Enhanced Pattern Detection

```python
# src/core/coordinator/workflows/workflow_manager.py (enhance existing)
class WorkflowManager:
    """Enhanced with automatic parameter expansion and stateless services."""

    def _determine_pattern(self, config: WorkflowConfig) -> str:
        """Enhanced pattern detection with auto parameter expansion."""
        
        # Auto-detect parameter grids
        if self._has_parameter_grids(config):
            if config.workflow_type == WorkflowType.OPTIMIZATION:
                return 'auto_stateless_optimization'  # New auto-expansion pattern
            elif config.workflow_type == WorkflowType.SIGNAL_GENERATION:
                return 'auto_stateless_signal_generation'
            elif config.workflow_type == WorkflowType.SIGNAL_REPLAY:
                return 'auto_stateless_signal_replay'
            else:
                return 'auto_stateless_backtest'      # Default auto-expansion
        
        # Existing pattern detection for traditional workflows
        return self._existing_pattern_detection(config)

    def _has_parameter_grids(self, config: WorkflowConfig) -> bool:
        """Auto-detect parameter grids in strategies/classifiers."""
        
        # Check strategies for parameter grids
        for strategy in getattr(config, 'strategies', []):
            if self._contains_parameter_grid(strategy.get('parameters', {})):
                return True
                
        # Check classifiers for parameter grids  
        for classifier in getattr(config, 'classifiers', []):
            if self._contains_parameter_grid(classifier.get('parameters', {})):
                return True
                
        # Check any arbitrary component for parameter grids (future extensibility)
        for component_type in ['risk', 'execution', 'data', 'custom_components']:
            for component in getattr(config, component_type, []):
                if self._contains_parameter_grid(component.get('parameters', {})):
                    return True
        
        return False

    def _contains_parameter_grid(self, params: Dict[str, Any]) -> bool:
        """Check if parameters contain grids like [10, 20, 30]."""
        for value in params.values():
            if isinstance(value, list) and len(value) > 1:
                return True
        return False
```

### Step 2: Auto-Expansion Patterns (Stateless Services)

```python
# src/core/coordinator/workflows/patterns/optimization_patterns.py (enhance existing)
def get_optimization_patterns() -> Dict[str, Any]:
    """Enhanced with auto-expansion patterns for stateless services."""
    
    # Replace all existing patterns with stateless service patterns
    auto_patterns = {
        'auto_stateless_optimization': {
            'description': 'Auto-detects parameter grids, minimal stateful containers',
            'stateful_containers': ['data', 'feature_hub', 'portfolio', 'execution'],
            'stateless_services': ['strategy', 'classifier', 'risk'],
            'communication_pattern': 'broadcast_to_stateless_services',
            'supports_multi_parameter': True,
            'auto_expansion': True,
            'traditional_modes': {
                'backtest': True,
                'signal_generation': True,
                'signal_replay': True
            }
        },
        'auto_stateless_signal_generation': {
            'description': 'Stateless signal generation with parameter expansion',
            'stateful_containers': ['data', 'feature_hub'],
            'stateless_services': ['strategy', 'classifier'],
            'communication_pattern': 'broadcast_signal_generation',
            'supports_multi_parameter': True,
            'auto_expansion': True
        },
        'auto_stateless_signal_replay': {
            'description': 'Stateless signal replay with parameter expansion',
            'stateful_containers': ['data', 'portfolio', 'execution'],
            'stateless_services': ['risk'],
            'communication_pattern': 'signal_replay_validation',
            'supports_multi_parameter': True,
            'auto_expansion': True
        }
    }
    
    return auto_patterns
```

### Step 3: Auto-Expansion Config Builder

```python
# src/core/coordinator/workflows/config/config_builders.py (enhance existing)
class ConfigBuilder:
    """Enhanced with auto-expansion capability for stateless services."""

    def build_auto_stateless_optimization_config(self, config: WorkflowConfig) -> Dict[str, Any]:
        """Auto-generate optimal container topology from simple config."""
        
        # 1. Auto-detect all parameter combinations
        param_analysis = self._analyze_parameter_combinations(config)
        
        # 2. Create minimal stateful container topology
        container_config = self._create_stateful_topology(param_analysis, config)
        
        # 3. Configure stateless services (no containers needed)
        service_config = self._configure_stateless_services(param_analysis, config)
        
        # 4. Auto-wire communication patterns
        communication_config = self._generate_stateless_communication(container_config, service_config)
        
        return {
            'containers': container_config,
            'stateless_services': service_config,
            'communication': communication_config,
            'analytics': {
                'total_combinations': param_analysis.total_combinations,
                'topology_type': 'auto_optimized_stateless',
                'correlation_tracking': True
            }
        }

    def _analyze_parameter_combinations(self, config: WorkflowConfig) -> ParameterAnalysis:
        """Auto-analyze parameter grids and calculate combinations."""
        
        analysis = ParameterAnalysis()
        
        # Extract strategy parameter combinations
        for strategy in getattr(config, 'strategies', []):
            strategy_combos = self._expand_parameter_grid(strategy.get('parameters', {}))
            analysis.strategy_combinations.extend([
                {'type': strategy['type'], 'parameters': combo}
                for combo in strategy_combos
            ])
        
        # Extract classifier parameter combinations
        for classifier in getattr(config, 'classifiers', []):
            classifier_combos = self._expand_parameter_grid(classifier.get('parameters', {}))
            analysis.classifier_combinations.extend([
                {'type': classifier['type'], 'parameters': combo}
                for combo in classifier_combos
            ])
        
        # Support arbitrary component parameter expansion (future extensibility)
        for component_type in ['risk', 'execution', 'custom_components']:
            for component in getattr(config, component_type, []):
                component_combos = self._expand_parameter_grid(component.get('parameters', {}))
                getattr(analysis, f"{component_type}_combinations", []).extend([
                    {'type': component['type'], 'parameters': combo}
                    for combo in component_combos
                ])
        
        # Calculate total combinations (cross-product)
        analysis.total_combinations = (
            len(analysis.strategy_combinations or [1]) *
            len(analysis.classifier_combinations or [1]) *
            # ... multiply by other component combinations as needed
        )
        
        return analysis

    def _create_stateful_topology(self, analysis: ParameterAnalysis, config: WorkflowConfig) -> Dict[str, Any]:
        """Create minimal container topology - only stateful components."""
        
        # Generate all parameter combinations for portfolio isolation
        all_combinations = self._generate_all_combinations(analysis)
        
        return {
            # Shared stateful containers (efficient resource usage)
            'data': {
                'role': 'data',
                'shared': True,
                'config': self._extract_data_config(config)
            },
            'feature_hub': {
                'role': 'indicator',
                'shared': True,
                'auto_inferred_features': True,
                'required_features': analysis.get_required_features()
            },
            'execution': {
                'role': 'execution',
                'shared': True,
                'config': self._extract_execution_config(config)
            },
            
            # Portfolio containers (one per combination for isolation)
            'portfolios': [
                {
                    'role': 'portfolio',
                    'combination_id': f"combo_{i}",
                    'strategy_config': combo['strategy'],
                    'classifier_config': combo.get('classifier', {}),
                    'correlation_id': f"{base_correlation_id}_combo_{i}",
                    'parameter_fingerprint': self._generate_parameter_fingerprint(combo)
                }
                for i, combo in enumerate(all_combinations)
            ]
        }

    def _configure_stateless_services(self, analysis: ParameterAnalysis, config: WorkflowConfig) -> Dict[str, Any]:
        """Configure stateless services (pure functions, no containers)."""
        
        return {
            'strategy_service': {
                'type': 'stateless_broadcast_service',
                'service_class': 'StatelessStrategyService',
                'implementations': [
                    {
                        'combo_id': f"strategy_{i}",
                        'type': combo['type'], 
                        'parameters': combo['parameters'],
                        'target_portfolios': self._get_target_portfolios_for_strategy(combo, analysis)
                    }
                    for i, combo in enumerate(analysis.strategy_combinations)
                ]
            },
            'classifier_service': {
                'type': 'stateless_broadcast_service',
                'service_class': 'StatelessClassifierService',
                'implementations': [
                    {
                        'combo_id': f"classifier_{i}",
                        'type': combo['type'],
                        'parameters': combo['parameters'],
                        'target_portfolios': self._get_target_portfolios_for_classifier(combo, analysis)
                    }
                    for i, combo in enumerate(analysis.classifier_combinations)
                ]
            },
            'risk_service': {
                'type': 'stateless_validation_service',
                'service_class': 'StatelessRiskValidator',
                'parameters': self._extract_risk_config(config),
                'applies_to': 'all_portfolios'
            }
        }
```

## Event System Integration with Stateless Services ⚡

### Event Flow Architecture

```
FeatureHub Container (stateful)
    ↓ (broadcasts features)
Strategy Services (stateless functions) → Signals
    ↓ (routes signals by combo_id)
Portfolio Containers (stateful) → Orders
    ↓ (sends orders)
Risk Service (stateless function) → Risk Decisions
    ↓ (approved orders)
Execution Container (stateful) → Fills
    ↓ (routes fills back)
Portfolio Containers (stateful)
```

### Enhanced Communication Factory

```python
# src/core/communication/factory.py (enhance existing)
class CommunicationFactory:
    """Enhanced with stateless service broadcasting."""

    def create_auto_stateless_broadcast_adapters(
        self, 
        containers: Dict[str, Any],
        stateless_services: Dict[str, Any]
    ) -> List[Any]:
        """Auto-wire stateless services with broadcast pattern."""
        
        adapters = []
        
        # FeatureHub broadcasts to all stateless services
        feature_hub = containers['feature_hub']
        
        # Strategy services: FeatureHub → Strategy Service → Portfolio Containers
        for strategy_impl in stateless_services['strategy_service']['implementations']:
            adapters.append(self.create_adapter({
                'type': 'stateless_broadcast',
                'source_container': feature_hub,
                'service_config': strategy_impl,
                'service_function': 'generate_signal',
                'target_containers': strategy_impl['target_portfolios'],
                'message_type': 'feature_to_signal',
                'correlation_tracking': True
            }))
        
        # Classifier services: FeatureHub → Classifier Service → Portfolio Containers
        for classifier_impl in stateless_services['classifier_service']['implementations']:
            adapters.append(self.create_adapter({
                'type': 'stateless_broadcast',
                'source_container': feature_hub,
                'service_config': classifier_impl,
                'service_function': 'classify_regime',
                'target_containers': classifier_impl['target_portfolios'],
                'message_type': 'feature_to_regime',
                'correlation_tracking': True
            }))
        
        # Risk service: Portfolio Containers → Risk Service → Portfolio Containers
        risk_config = stateless_services['risk_service']
        adapters.append(self.create_adapter({
            'type': 'stateless_validation',
            'source_containers': containers['portfolios'],
            'service_config': risk_config,
            'service_function': 'validate_order',
            'target_containers': containers['portfolios'],
            'message_type': 'order_to_risk_decision',
            'correlation_tracking': True
        }))
        
        return adapters

    def create_stateless_service_adapter(self, config: Dict[str, Any]) -> StatelessServiceAdapter:
        """Create adapter for stateless service communication."""
        
        return StatelessServiceAdapter(
            service_class=config['service_config']['service_class'],
            service_function=config['service_function'],
            source_containers=config.get('source_containers', []),
            target_containers=config.get('target_containers', []),
            message_transformer=self._create_message_transformer(config['message_type']),
            correlation_tracker=self._create_correlation_tracker(config.get('correlation_tracking', False))
        )
```

### Stateless Service Adapter

```python
class StatelessServiceAdapter:
    """Adapter for stateless service execution in event-driven architecture."""
    
    def __init__(self, service_class, service_function, source_containers, target_containers, 
                 message_transformer, correlation_tracker):
        self.service_class = service_class
        self.service_function = service_function
        self.source_containers = source_containers
        self.target_containers = target_containers
        self.message_transformer = message_transformer
        self.correlation_tracker = correlation_tracker
    
    async def process_event(self, event: Event) -> List[Event]:
        """Process event through stateless service and route results."""
        
        # Extract input data from event
        input_data = self.message_transformer.extract_input(event)
        
        # Execute stateless service (pure function)
        start_time = datetime.now()
        
        try:
            # Call pure function
            result = await self._execute_stateless_service(
                service_class=self.service_class,
                service_function=self.service_function,
                input_data=input_data,
                service_config=event.payload.get('service_config', {})
            )
            
            # Track service execution for analytics
            if self.correlation_tracker:
                await self.correlation_tracker.track_service_call(
                    service_type=self.service_class,
                    input_data=input_data,
                    output_data=result,
                    execution_time=datetime.now() - start_time,
                    correlation_id=event.correlation_id
                )
            
            # Transform result into output events
            output_events = self.message_transformer.create_output_events(
                result=result,
                target_containers=self.target_containers,
                correlation_id=event.correlation_id
            )
            
            return output_events
            
        except Exception as e:
            # Service isolation - failures don't corrupt container state
            await self.correlation_tracker.track_service_error(
                service_type=self.service_class,
                error=e,
                correlation_id=event.correlation_id
            )
            return []  # Failed service call doesn't affect other combinations
    
    async def _execute_stateless_service(self, service_class, service_function, input_data, service_config):
        """Execute stateless service function."""
        
        # Get service class (pure function provider)
        service = getattr(service_class, service_function)
        
        # Execute pure function
        if asyncio.iscoroutinefunction(service):
            return await service(**input_data, **service_config)
        else:
            return service(**input_data, **service_config)
```

## Multi-Portfolio Risk Validation 🛡️

### Stateless Risk Validation for Multiple Portfolios

```python
class StatelessRiskValidator:
    """Pure function risk validation for any portfolio."""

    @staticmethod
    def validate_order(
        order: Order,
        portfolio_state: PortfolioState,  # Specific portfolio instance
        risk_limits: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> RiskDecision:
        """Pure function - no internal state."""
        
        # Get current position for this specific portfolio
        current_position = portfolio_state.get_position(order.symbol)
        current_value = current_position.market_value if current_position else Decimal(0)
        
        # Calculate new position value
        order_value = order.quantity * market_data['prices'][order.symbol]
        new_value = current_value + order_value
        
        # Apply limit from configuration
        max_position = risk_limits.get('max_position_value', Decimal('100000'))
        
        # Perform validation (pure calculation)
        approved = new_value <= max_position
        
        return RiskDecision(
            approved=approved,
            reason=f"Portfolio {portfolio_state.portfolio_id}: Position {new_value} vs limit {max_position}",
            portfolio_id=portfolio_state.portfolio_id,  # Track which portfolio
            risk_metrics={
                'current_value': current_value,
                'order_value': order_value,
                'new_value': new_value,
                'utilization': float(new_value / max_position)
            }
        )

    @staticmethod
    def validate_portfolio_risk(
        portfolio_state: PortfolioState,
        risk_limits: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> PortfolioRiskAssessment:
        """Pure function for overall portfolio risk assessment."""
        
        total_value = portfolio_state.get_total_value()
        cash_balance = portfolio_state.get_cash_balance()
        
        # Calculate risk metrics (pure calculations)
        leverage = total_value / cash_balance if cash_balance > 0 else 0
        max_leverage = risk_limits.get('max_leverage', 2.0)
        
        # Position concentration check
        positions = portfolio_state.get_all_positions()
        max_concentration = max(
            (pos.market_value / total_value for pos in positions),
            default=0
        )
        max_allowed_concentration = risk_limits.get('max_concentration', 0.2)
        
        return PortfolioRiskAssessment(
            portfolio_id=portfolio_state.portfolio_id,
            leverage=leverage,
            leverage_approved=leverage <= max_leverage,
            concentration=max_concentration,
            concentration_approved=max_concentration <= max_allowed_concentration,
            overall_approved=leverage <= max_leverage and max_concentration <= max_allowed_concentration
        )
```

## Event Tracing with Stateless Services 🔍

### Enhanced Event Tracing

```python
class StatelessServiceTracer:
    """Enhanced event tracing for stateless service calls."""
    
    async def trace_service_execution(
        self,
        service_type: str,
        service_function: str,
        input_data: Dict[str, Any],
        service_config: Dict[str, Any],
        correlation_id: str
    ):
        """Trace stateless service execution with detailed parameter tracking."""
        
        # Generate unique trace ID for this service call
        trace_id = f"{correlation_id}_{service_type}_{uuid.uuid4().hex[:8]}"
        
        # Pre-execution trace
        await self.event_tracer.trace_service_call_start(
            trace_id=trace_id,
            service_type=service_type,
            service_function=service_function,
            input_parameters=input_data,
            service_configuration=service_config,
            correlation_id=correlation_id,
            timestamp=datetime.now()
        )
        
        start_time = time.time()
        
        try:
            # Execute service (traced)
            result = await self._execute_traced_service(
                service_type, service_function, input_data, service_config
            )
            
            execution_time = time.time() - start_time
            
            # Post-execution trace (success)
            await self.event_tracer.trace_service_call_success(
                trace_id=trace_id,
                output_data=result,
                execution_time_ms=execution_time * 1000,
                service_metrics=self._calculate_service_metrics(result),
                correlation_id=correlation_id
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Post-execution trace (error)
            await self.event_tracer.trace_service_call_error(
                trace_id=trace_id,
                error_type=type(e).__name__,
                error_message=str(e),
                execution_time_ms=execution_time * 1000,
                correlation_id=correlation_id
            )
            
            raise

    def _calculate_service_metrics(self, result: Any) -> Dict[str, Any]:
        """Calculate service-specific metrics for analytics."""
        
        metrics = {'result_type': type(result).__name__}
        
        # Service-specific metrics
        if hasattr(result, 'confidence'):
            metrics['confidence'] = result.confidence
        if hasattr(result, 'signal_strength'):
            metrics['signal_strength'] = result.signal_strength
        if hasattr(result, 'regime_probability'):
            metrics['regime_probability'] = result.regime_probability
            
        return metrics
```

### Analytics Benefits

**Clear Service Boundaries**: Each stateless service call is a discrete, traceable unit
- Service parameters are explicitly tracked in each trace
- Input/output data lineage is clear
- Performance analysis per service type is straightforward

**Enhanced Pattern Discovery**: Analytics can discover patterns across service combinations
- Which strategy + classifier combinations perform best
- Which parameter ranges are most effective
- How service execution times correlate with market conditions

## Parallelization with Stateless Services 🚀

### Perfect Parallelization Safety

```python
class ParallelStatelessExecutor:
    """Execute stateless services in parallel with complete safety."""
    
    async def process_features_parallel(
        self,
        features: Dict[str, Any],
        strategy_configs: List[Dict[str, Any]],
        correlation_id: str
    ):
        """Process all strategy combinations in parallel."""
        
        # Create parallel tasks for each strategy configuration
        tasks = [
            self._execute_strategy_service_traced(
                features=features,
                config=config,
                correlation_id=f"{correlation_id}_strategy_{config['combo_id']}"
            )
            for config in strategy_configs
        ]
        
        # Execute all strategies in parallel (safe due to pure functions)
        signals = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Route signals to corresponding portfolios
        routing_tasks = []
        for signal, config in zip(signals, strategy_configs):
            if not isinstance(signal, Exception):
                routing_tasks.append(
                    self._route_signal_to_portfolios(signal, config['target_portfolios'])
                )
        
        # Route all signals in parallel
        await asyncio.gather(*routing_tasks)
    
    async def process_risk_validation_parallel(
        self,
        orders: List[Order],
        portfolio_states: Dict[str, PortfolioState],
        risk_configs: Dict[str, Dict[str, Any]],
        market_data: Dict[str, Any]
    ):
        """Validate all orders in parallel."""
        
        # Create parallel validation tasks
        tasks = [
            StatelessRiskValidator.validate_order(
                order=order,
                portfolio_state=portfolio_states[order.portfolio_id],
                risk_limits=risk_configs[order.portfolio_id],
                market_data=market_data
            )
            for order in orders
        ]
        
        # All risk validations run in parallel (safe due to pure functions)
        decisions = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [d for d in decisions if not isinstance(d, Exception)]
    
    async def _execute_strategy_service_traced(self, features, config, correlation_id):
        """Execute strategy service with tracing."""
        
        return await self.service_tracer.trace_service_execution(
            service_type='strategy',
            service_function='generate_signal',
            input_data={'features': features},
            service_config=config,
            correlation_id=correlation_id
        )
```

### Parallel Benefits

**Pure Function Safety**: No shared state means perfect parallelization
- Strategy services can run completely independently
- Risk validation can process multiple orders simultaneously
- No race conditions or state corruption possible

**Resource Efficiency**: Services scale horizontally without container overhead
- CPU-bound services utilize all available cores
- No container creation/destruction overhead
- Memory usage scales linearly with combinations

**Fault Tolerance**: Service failures don't affect other combinations
- Exception in one strategy doesn't stop others
- Service isolation prevents cascading failures
- Analytics can track failure patterns across services

## Traditional Workflow Mode Support 🔄

### Preserving Built-in Patterns and Composability

The stateless services approach **enhances** rather than replaces existing workflow capabilities:

```python
# Enhanced WorkflowManager maintains all existing patterns
class WorkflowManager:
    """Enhanced workflow manager supporting both traditional and stateless patterns."""
    
    def _determine_execution_mode(self, config: WorkflowConfig) -> str:
        """Determine execution mode based on configuration."""
        
        # Auto-detect parameter grids → stateless services
        if self._has_parameter_grids(config):
            return self._determine_stateless_pattern(config)
        
        # Traditional single-configuration → existing patterns
        else:
            return self._determine_traditional_pattern(config)
    
    def _determine_stateless_pattern(self, config: WorkflowConfig) -> str:
        """Map workflow types to stateless service patterns."""
        
        return {
            WorkflowType.BACKTEST: 'auto_stateless_backtest',
            WorkflowType.OPTIMIZATION: 'auto_stateless_optimization', 
            WorkflowType.SIGNAL_GENERATION: 'auto_stateless_signal_generation',
            WorkflowType.SIGNAL_REPLAY: 'auto_stateless_signal_replay'
        }.get(config.workflow_type, 'auto_stateless_backtest')
    
    def _determine_traditional_pattern(self, config: WorkflowConfig) -> str:
        """Use existing pattern determination for traditional workflows."""
        
        # Existing logic preserved - no changes needed
        return self._existing_pattern_determination(config)
```

### Traditional Mode Examples

```yaml
# Traditional backtest (existing pattern - no parameter grids)
workflow:
  type: backtest
  base_pattern: full_backtest

strategy:
  type: momentum
  parameters:
    lookback_period: 20        # Single value → traditional container
    signal_threshold: 0.02

data:
  symbols: ['AAPL']
  start_date: '2022-01-01'
  end_date: '2023-12-31'

# Traditional signal generation (existing pattern)
workflow:
  type: signal_generation
  base_pattern: simple_signal_generation

strategy:
  type: mean_reversion
  parameters:
    lookback_window: 14        # Single value → traditional container

# Traditional signal replay (existing pattern)  
workflow:
  type: signal_replay
  base_pattern: signal_replay_with_risk

signals_file: 'generated_signals.csv'
risk:
  max_position_size: 10000
```

### Composability Preservation

**Pattern Composability**: All existing patterns remain composable
- Multi-phase workflows still work with traditional patterns
- Existing sequencer phase management is preserved
- Cross-phase data flow continues working

**Component Composability**: Existing component composition is enhanced
- Traditional components can be mixed with stateless services
- Existing indicator inference continues working
- Component factories handle both traditional and stateless creation

**Configuration Composability**: YAML configuration flexibility is maintained
- Simple configs automatically get stateless optimization
- Complex configs can specify traditional patterns
- Power users can mix both approaches in multi-phase workflows

## File-Based Configuration Support 📁

### Stateless Services with File-Based Parameters

```python
class FileBasedStatelessStrategy:
    """Stateless strategy that reads configuration files."""
    
    @staticmethod
    def generate_signal(
        features: Dict[str, Any],
        classifier_regime: RegimeState,
        config_file_path: str,  # File path as parameter (not cached)
        strategy_params: Dict[str, Any]
    ) -> Signal:
        """Pure function that reads config fresh each time."""
        
        # Load configuration file (no caching - pure function)
        regime_config = FileBasedStatelessStrategy._load_regime_config(config_file_path)
        
        # Apply regime-based adjustments from file
        adjusted_params = FileBasedStatelessStrategy._apply_regime_adjustments(
            base_params=strategy_params,
            regime=classifier_regime,
            regime_config=regime_config
        )
        
        # Generate signal using adjusted parameters
        return calculate_momentum_signal(features, adjusted_params)
    
    @staticmethod
    def _load_regime_config(config_file_path: str) -> Dict[str, Any]:
        """Pure function - loads config file fresh each time."""
        with open(config_file_path, 'r') as f:
            return yaml.safe_load(f)
```

### File-Based Configuration Examples

```yaml
# config/momentum_regime_rules.yaml
bull_market:
  lookback_multiplier: 0.8    # Shorter lookback in bull markets
  threshold_multiplier: 1.2   # Higher threshold
  risk_multiplier: 1.1

bear_market:
  lookback_multiplier: 1.5    # Longer lookback in bear markets
  threshold_multiplier: 0.6   # Lower threshold
  risk_multiplier: 0.7

neutral:
  lookback_multiplier: 1.0    # No adjustment
  threshold_multiplier: 1.0
  risk_multiplier: 1.0
```

```yaml
# User workflow with file-based configuration
workflow:
  type: optimization
  base_pattern: full_backtest

strategies:
  - type: file_based_momentum
    parameters:
      base_lookback: [10, 20, 30]
      base_threshold: [0.01, 0.02]
      regime_config_file: "config/momentum_regime_rules.yaml"  # File reference

classifiers:
  - type: hmm
    parameters:
      model_type: ['hmm', 'svm']
      lookback_days: [30, 60]
```

**File References Keep Strategies Stateless**: Strategies remain stateless if file paths are passed as parameters and config is loaded fresh each call, maintaining all parallelization and tracing benefits.

## Implementation Roadmap 🗺️

### Phase 1: Stateless Services Foundation (Week 1)

**Enhance Existing WorkflowManager**
1. Add parameter grid detection to `_determine_pattern()`
2. Add auto-expansion patterns to `optimization_patterns.py`
3. Enhance `ConfigBuilder` with stateless service configuration
4. Test parameter expansion with simple YAML configs

**Create Stateless Service Infrastructure**
1. Implement `StatelessServiceAdapter` for event routing
2. Enhance `CommunicationFactory` with stateless broadcasting
3. Create `StatelessRiskValidator` for multi-portfolio validation
4. Test event flow through stateless services

### Phase 2: Event Integration and Tracing (Week 2)

**Event System Integration**
1. Implement stateless service event processing
2. Add correlation ID tracking through service calls
3. Test parallel service execution safety
4. Validate service isolation and fault tolerance

**Enhanced Analytics Integration**
1. Implement `StatelessServiceTracer` for detailed service tracking
2. Add service-level metrics collection
3. Integrate with existing analytics database
4. Test pattern discovery across service combinations

### Phase 3: Multi-Modal Support and Polish (Week 3)

**Traditional Mode Preservation**
1. Ensure backward compatibility with existing workflows
2. Test traditional backtest, signal generation, and signal replay modes
3. Validate composability with multi-phase workflows
4. Document mode selection criteria

**File-Based Configuration Support**
1. Implement file-based stateless strategy patterns
2. Add configuration file validation and caching strategies
3. Test dynamic configuration updates
4. Document file-based configuration patterns

### Phase 4: Performance Optimization and Documentation (Week 4)

**Performance Optimization**
1. Benchmark stateless vs. traditional container performance
2. Optimize parallel service execution
3. Validate 60% container reduction claims
4. Profile memory usage and scaling characteristics

**Comprehensive Documentation**
1. Document stateless service architecture
2. Create migration guide from traditional to stateless patterns
3. Add examples for all supported modes
4. Document troubleshooting and debugging techniques

## Success Criteria 📊

### Functional Requirements
- [ ] **Simple YAML Config**: Users specify parameter grids, system handles topology
- [ ] **Parameter Expansion**: Automatic detection and expansion of parameter grids for arbitrary components
- [ ] **60% Container Reduction**: Minimal stateful containers while maintaining isolation
- [ ] **Multi-Modal Support**: Traditional backtest, signal generation, signal replay modes work
- [ ] **Event System Compatibility**: Stateless services integrate seamlessly with event architecture
- [ ] **Parallel Execution**: Pure function safety enables perfect parallelization
- [ ] **Enhanced Tracing**: Service-level granularity improves debugging and analytics


### Performance Requirements
- [ ] **No Performance Regression**: Stateless services perform as well as containers
- [ ] **Memory Efficiency**: 60% reduction in container overhead validated
- [ ] **Parallel Scaling**: Services utilize available CPU cores effectively
- [ ] **Analytics Overhead**: Service tracing adds <5% execution overhead
- [ ] **Service Isolation**: Failures in one service don't affect others

### Quality Requirements
- [ ] **Protocol Compliance**: All components follow established Protocol + Composition patterns
- [ ] **Configuration Simplicity**: YAML remains simple and concise
- [ ] **Pattern Composability**: Existing workflow patterns remain composable
- [ ] **Component Extensibility**: New component types easily support parameter expansion
- [ ] **Debugging Capability**: Service-level tracing improves troubleshooting

## Risk Mitigation 🛡️

### Performance Risk  
**Risk**: Stateless service overhead degrades performance
**Mitigation**:
- Benchmark stateless vs. container performance early
- Optimize parallel execution patterns
- Monitor analytics overhead and optimize if needed

### Complexity Risk
**Risk**: Stateless service architecture adds complexity
**Mitigation**:
- Follow STYLE.md principles: enhance existing, don't create new files
- Keep YAML configurations simple and concise
- Document clear migration paths and decision criteria

### Integration Risk
**Risk**: Event system integration introduces bugs
**Mitigation**:
- Incremental integration with existing event architecture
- Comprehensive testing of service isolation
- Validate correlation ID tracking and analytics integration

## Expected Benefits 🎯

### Developer Experience
- **Simplified Configuration**: Parameter grids automatically detected and expanded
- **Enhanced Debugging**: Service-level tracing with correlation tracking
- **Improved Testing**: Pure functions easier to test and validate
- **Reduced Complexity**: 60% fewer containers to manage and debug

### System Capabilities  
- **Automatic Optimization**: Parameter expansion works for any component type
- **Perfect Parallelization**: Pure function safety enables optimal CPU utilization
- **Enhanced Analytics**: Service-level granularity improves pattern discovery
- **Multi-Modal Flexibility**: Traditional and stateless modes support different use cases

### Architectural Quality
- **Protocol Compliance**: Maintains established patterns while improving efficiency
- **Component Composability**: Stateless services compose naturally with existing containers
- **Future Extensibility**: Pattern supports arbitrary component parameter expansion
- **Performance Optimization**: Resource efficiency without sacrificing capabilities

### Research Productivity
- **Rapid Experimentation**: Simple YAML configs enable fast iteration
- **Comprehensive Analytics**: Service-level tracking improves research insights
- **Scalable Parameter Search**: Automatic expansion handles large parameter spaces
- **Pattern Discovery**: Analytics automatically identify successful combinations

## Conclusion 🏆

The **Coordinator and Functional Refactor** transforms the ADMF-PC architecture from a pure container-based approach to an optimal **hybrid of stateful containers + stateless services**. This evolution:

1. **Maintains Architectural Principles**: Protocol + Composition patterns are preserved and enhanced
2. **Simplifies User Experience**: Complex parameter expansion hidden behind simple YAML configuration
3. **Improves System Efficiency**: 60% container reduction while maintaining complete isolation
4. **Enhances Research Capabilities**: Automatic parameter expansion and service-level analytics
5. **Preserves Flexibility**: Traditional and stateless modes support different research needs

The key insight is that **only state needs containers**. Pure logic can be stateless services that integrate seamlessly with the event-driven architecture while providing superior resource efficiency, parallelization, and observability.

By enhancing existing components rather than creating new ones, this refactor follows STYLE.md principles while delivering significant architectural improvements that will accelerate quantitative research productivity.

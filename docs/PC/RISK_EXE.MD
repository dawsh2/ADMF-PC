# Execution and Risk Modules with Protocol + Composition Architecture

## Overview

The Execution and Risk modules are critical components that handle order processing, market simulation, position management, and risk control. They integrate seamlessly with our Protocol + Composition architecture, gaining capabilities through composition rather than inheritance.

## 1. Execution Module Protocols

```python
from typing import Protocol, runtime_checkable, Dict, Any, Optional, List
from abc import abstractmethod
from enum import Enum
from decimal import Decimal

# === Core Execution Protocols ===
@runtime_checkable
class OrderProcessor(Protocol):
    """Protocol for components that process orders"""
    
    @abstractmethod
    def process_order(self, order: Dict[str, Any]) -> None:
        """Process an incoming order"""
        ...
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> str:
        """Get status of an order"""
        ...

@runtime_checkable
class Broker(Protocol):
    """Protocol for broker implementations"""
    
    @abstractmethod
    def execute_order(self, order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute order and return fill if successful"""
        ...
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        ...
    
    @abstractmethod
    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get list of pending orders"""
        ...

@runtime_checkable
class SlippageModel(Protocol):
    """Protocol for slippage models"""
    
    @abstractmethod
    def apply_slippage(self, order: Dict[str, Any], 
                      market_price: float) -> float:
        """Apply slippage to get execution price"""
        ...

@runtime_checkable
class CommissionModel(Protocol):
    """Protocol for commission models"""
    
    @abstractmethod
    def calculate_commission(self, order: Dict[str, Any], 
                           fill_price: float) -> float:
        """Calculate commission for order"""
        ...

@runtime_checkable
class BacktestCoordinator(Protocol):
    """Protocol for backtest coordination"""
    
    @abstractmethod
    def run_backtest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a backtest and return results"""
        ...
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from last run"""
        ...
```

## 2. Risk Module Protocols

```python
# === Core Risk Protocols ===
@runtime_checkable
class RiskManager(Protocol):
    """Protocol for risk management"""
    
    @abstractmethod
    def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process signal and return order if approved"""
        ...
    
    @abstractmethod
    def check_risk_limits(self, signal: Dict[str, Any], 
                         quantity: int) -> tuple[bool, str]:
        """Check if trade passes risk limits"""
        ...

@runtime_checkable
class PortfolioManager(Protocol):
    """Protocol for portfolio management"""
    
    @abstractmethod
    def update_position(self, symbol: str, quantity: int, 
                       price: float, commission: float = 0) -> None:
        """Update position from trade"""
        ...
    
    @abstractmethod
    def get_position(self, symbol: str) -> 'Position':
        """Get position for symbol"""
        ...
    
    @abstractmethod
    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        ...
    
    @abstractmethod
    def close_all_positions(self, timestamp: Any, 
                          prices: Dict[str, float]) -> None:
        """Close all open positions"""
        ...

@runtime_checkable
class PositionSizer(Protocol):
    """Protocol for position sizing"""
    
    @abstractmethod
    def calculate_position_size(self, signal: Dict[str, Any], 
                              portfolio: PortfolioManager, 
                              current_position: int = 0) -> int:
        """Calculate appropriate position size"""
        ...

@runtime_checkable
class RiskLimit(Protocol):
    """Protocol for risk limits"""
    
    @abstractmethod
    def check(self, signal: Dict[str, Any], 
              quantity: int, 
              portfolio: PortfolioManager) -> tuple[bool, str]:
        """Check if trade passes limit"""
        ...
    
    @abstractmethod
    def modify_quantity(self, signal: Dict[str, Any], 
                       quantity: int, 
                       portfolio: PortfolioManager) -> Optional[int]:
        """Suggest modified quantity if limit fails"""
        ...
```

## 3. Execution Module Capabilities

### 3.1 Order Management Capability

```python
class OrderManagementCapability(Capability):
    """Adds order management functionality"""
    
    def get_name(self) -> str:
        return "order_management"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        # Initialize order tracking
        if not hasattr(component, 'orders'):
            component.orders = ThreadSafeDict()
            component.order_id_counter = 0
            component.order_history = []
        
        # Add order processing method
        def process_order(order: Dict[str, Any]) -> None:
            """Process incoming order"""
            # Validate order
            if not OrderValidator.validate(order):
                if hasattr(component, 'logger'):
                    component.logger.error("Invalid order", order=order)
                return
            
            # Store order
            order_id = order['order_id']
            component.orders[order_id] = {
                **order,
                'status': 'RECEIVED',
                'received_at': datetime.now()
            }
            
            # Forward to broker if available
            if hasattr(component, 'broker'):
                component.broker.execute_order(order)
            
            # Track in history
            component.order_history.append(order)
            if len(component.order_history) > spec.get('max_history', 10000):
                component.order_history.pop(0)
        
        component.process_order = process_order
        
        # Add status tracking
        def get_order_status(order_id: str) -> str:
            """Get order status"""
            if order_id in component.orders:
                return component.orders[order_id]['status']
            return 'UNKNOWN'
        
        component.get_order_status = get_order_status
        
        # Add order update method
        def update_order_status(order_id: str, status: str, **kwargs) -> None:
            """Update order status"""
            if order_id in component.orders:
                component.orders[order_id]['status'] = status
                component.orders[order_id].update(kwargs)
                
                if hasattr(component, 'logger'):
                    component.logger.info(f"Order {order_id} status: {status}")
        
        component.update_order_status = update_order_status
        
        return component
```

### 3.2 Market Simulation Capability

```python
class MarketSimulationCapability(Capability):
    """Adds market simulation for backtesting"""
    
    def get_name(self) -> str:
        return "market_simulation"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        # Create slippage model
        slippage_config = spec.get('slippage', {'type': 'fixed', 'value': 0})
        component.slippage_model = self._create_slippage_model(slippage_config)
        
        # Create commission model
        commission_config = spec.get('commission', {'type': 'fixed', 'value': 0})
        component.commission_model = self._create_commission_model(commission_config)
        
        # Add market data tracking
        if not hasattr(component, 'latest_prices'):
            component.latest_prices = {}
            component.pending_orders = {}
        
        # Add order execution method
        def execute_order(order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Execute order with market simulation"""
            symbol = order['symbol']
            
            # Check for market price
            if symbol not in component.latest_prices:
                # Store as pending
                if symbol not in component.pending_orders:
                    component.pending_orders[symbol] = []
                component.pending_orders[symbol].append(order)
                return None
            
            market_price = component.latest_prices[symbol]
            
            # Apply slippage
            execution_price = component.slippage_model.apply_slippage(
                order, market_price
            )
            
            # Calculate commission
            commission = component.commission_model.calculate_commission(
                order, execution_price
            )
            
            # Create fill
            fill = {
                'fill_id': f"FILL_{order['order_id']}",
                'order_id': order['order_id'],
                'symbol': symbol,
                'quantity': order['quantity'],
                'price': execution_price,
                'commission': commission,
                'timestamp': datetime.now(),
                'slippage': execution_price - market_price
            }
            
            return fill
        
        component.execute_order = execute_order
        
        # Add market data update
        def update_market_price(symbol: str, price: float) -> None:
            """Update market price and process pending orders"""
            component.latest_prices[symbol] = price
            
            # Process pending orders for this symbol
            if symbol in component.pending_orders:
                pending = component.pending_orders[symbol]
                component.pending_orders[symbol] = []
                
                for order in pending:
                    fill = component.execute_order(order)
                    if fill and hasattr(component, '_events'):
                        event = Event(EventType.FILL, fill)
                        component._events.event_bus.publish(event)
        
        component.update_market_price = update_market_price
        
        return component
    
    def _create_slippage_model(self, config: Dict[str, Any]) -> Any:
        """Create slippage model from config"""
        model_type = config.get('type', 'fixed')
        
        if model_type == 'fixed':
            return FixedSlippageModel(config.get('value', 0))
        elif model_type == 'percentage':
            return PercentageSlippageModel(config.get('value', 0))
        elif model_type == 'volume':
            return VolumeSlippageModel(config.get('impact', 0))
        else:
            raise ValueError(f"Unknown slippage model: {model_type}")
```

### 3.3 Execution Context Capability

```python
class ExecutionContextCapability(Capability):
    """Adds execution context awareness"""
    
    def get_name(self) -> str:
        return "execution_context"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        # Determine execution mode
        exec_mode = spec.get('execution_mode', 'BACKTEST_SINGLE')
        component.execution_mode = ExecutionMode[exec_mode]
        
        # Set thread model based on mode
        thread_model = spec.get('thread_model')
        if not thread_model:
            thread_model = ExecutionContext.DEFAULT_THREAD_MODELS[component.execution_mode]
        component.thread_model = thread_model
        
        # Add context methods
        def is_live_trading() -> bool:
            """Check if in live trading mode"""
            return component.execution_mode in [
                ExecutionMode.LIVE_TRADING,
                ExecutionMode.PAPER_TRADING
            ]
        
        component.is_live_trading = is_live_trading
        
        def requires_thread_safety() -> bool:
            """Check if thread safety is required"""
            return component.thread_model != ThreadModel.SINGLE_THREADED
        
        component.requires_thread_safety = requires_thread_safety
        
        # Set up appropriate collections based on thread model
        if component.requires_thread_safety():
            # Replace regular dicts with thread-safe versions
            if hasattr(component, 'orders') and not isinstance(component.orders, ThreadSafeDict):
                component.orders = ThreadSafeDict(component.orders)
            if hasattr(component, 'latest_prices'):
                component.latest_prices = ThreadSafeDict(component.latest_prices)
        
        # Add thread pool if needed
        if component.thread_model == ThreadModel.MULTI_THREADED:
            component.thread_pool = ThreadPoolExecutor(
                max_workers=spec.get('max_workers', 4)
            )
        
        return component
```

## 4. Risk Module Capabilities

### 4.1 Risk Management Capability

```python
class RiskManagementCapability(Capability):
    """Adds risk management functionality"""
    
    def get_name(self) -> str:
        return "risk_management"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        # Initialize risk components
        component.risk_limits = []
        component.position_sizers = {}
        component.signal_processors = []
        
        # Create risk limit registry
        component.risk_registry = RiskLimitRegistry()
        
        # Add default limits from spec
        for limit_config in spec.get('risk_limits', []):
            limit = self._create_risk_limit(limit_config)
            component.risk_limits.append(limit)
            component.risk_registry.register(limit_config['name'], limit)
        
        # Add default position sizers
        for sizer_config in spec.get('position_sizers', []):
            sizer = self._create_position_sizer(sizer_config)
            component.position_sizers[sizer_config['name']] = sizer
        
        # Add signal processing
        def process_signal(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Process signal through risk management"""
            # Validate signal
            if not self._validate_signal(signal):
                return None
            
            # Process through signal processors
            processed_signal = signal
            for processor in component.signal_processors:
                processed_signal = processor.process_signal(processed_signal)
                if processed_signal is None:
                    return None  # Filtered out
            
            # Get position sizer
            symbol = processed_signal['symbol']
            sizer = component.position_sizers.get(
                symbol, 
                component.position_sizers.get('default')
            )
            
            if not sizer:
                return None
            
            # Calculate position size
            portfolio = component.resolve('portfolio') if hasattr(component, 'resolve') else None
            current_position = 0
            if portfolio:
                current_position = portfolio.get_position(symbol).quantity
            
            quantity = sizer.calculate_position_size(
                processed_signal, portfolio, current_position
            )
            
            if quantity == 0:
                return None
            
            # Check risk limits
            passed, reason = self._check_risk_limits(
                processed_signal, quantity, portfolio
            )
            
            if not passed:
                if hasattr(component, 'logger'):
                    component.logger.warning(f"Risk limit failed: {reason}")
                return None
            
            # Create order
            order = self._create_order(processed_signal, quantity)
            return order
        
        component.process_signal = process_signal
        
        # Add risk limit checking
        def check_risk_limits(signal: Dict[str, Any], 
                            quantity: int) -> tuple[bool, str]:
            """Check if trade passes risk limits"""
            portfolio = component.resolve('portfolio') if hasattr(component, 'resolve') else None
            
            for limit in component.risk_limits:
                passed, reason = limit.check(signal, quantity, portfolio)
                if not passed:
                    return False, reason
            
            return True, ""
        
        component.check_risk_limits = check_risk_limits
        
        return component
    
    def _create_risk_limit(self, config: Dict[str, Any]) -> Any:
        """Create risk limit from config"""
        limit_type = config['type']
        
        if limit_type == 'position':
            return MaxPositionLimit(config.get('max_position', 1000))
        elif limit_type == 'exposure':
            return MaxExposureLimit(config.get('max_exposure_pct', 10))
        elif limit_type == 'drawdown':
            return MaxDrawdownLimit(config.get('max_drawdown_pct', 20))
        else:
            raise ValueError(f"Unknown risk limit type: {limit_type}")
```

### 4.2 Portfolio Management Capability

```python
class PortfolioManagementCapability(Capability):
    """Adds portfolio management functionality"""
    
    def get_name(self) -> str:
        return "portfolio_management"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        # Initialize portfolio state
        if not hasattr(component, 'positions'):
            component.positions = ThreadSafeDict()
            component.initial_cash = Decimal(str(spec.get('initial_cash', 100000)))
            component.cash = component.initial_cash
            component.equity_curve = []
            component.trades = []
        
        # Add position management
        def update_position(symbol: str, quantity: int, 
                          price: float, commission: float = 0) -> None:
            """Update position from trade"""
            if symbol not in component.positions:
                component.positions[symbol] = Position(symbol)
            
            position = component.positions[symbol]
            
            # Update position
            position.update(
                Decimal(str(quantity)),
                Decimal(str(price)),
                datetime.now(),
                f"TRADE_{len(component.trades)}",
                {},
                Decimal(str(commission))
            )
            
            # Update cash
            trade_value = Decimal(str(quantity * price))
            component.cash -= trade_value + Decimal(str(commission))
            
            # Record trade
            component.trades.append({
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'timestamp': datetime.now(),
                'cash_after': float(component.cash)
            })
            
            # Update equity curve
            component.equity_curve.append({
                'timestamp': datetime.now(),
                'equity': float(component.get_portfolio_value())
            })
        
        component.update_position = update_position
        
        # Add portfolio value calculation
        def get_portfolio_value() -> float:
            """Calculate total portfolio value"""
            holdings_value = Decimal('0')
            
            for position in component.positions.values():
                if position.quantity != 0 and hasattr(position, 'market_value'):
                    holdings_value += position.market_value
            
            return float(component.cash + holdings_value)
        
        component.get_portfolio_value = get_portfolio_value
        
        # Add position getter
        def get_position(symbol: str) -> Any:
            """Get position for symbol"""
            if symbol not in component.positions:
                component.positions[symbol] = Position(symbol)
            return component.positions[symbol]
        
        component.get_position = get_position
        
        # Add close all positions
        def close_all_positions(timestamp: Any, prices: Dict[str, float]) -> None:
            """Close all open positions"""
            for symbol, position in list(component.positions.items()):
                if position.quantity != 0:
                    price = prices.get(symbol, position.average_price)
                    position.close(
                        Decimal(str(price)),
                        timestamp,
                        f"CLOSE_ALL_{symbol}",
                        {'reason': 'close_all_positions'}
                    )
                    
                    # Update cash
                    trade_value = position.quantity * Decimal(str(price))
                    component.cash += trade_value
        
        component.close_all_positions = close_all_positions
        
        return component
```

### 4.3 Signal Processing Capability

```python
class SignalProcessingCapability(Capability):
    """Adds signal processing pipeline"""
    
    def get_name(self) -> str:
        return "signal_processing"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        # Create signal processing pipeline
        component.signal_pipeline = SignalProcessingPipeline()
        
        # Add processors from config
        for processor_config in spec.get('signal_processors', []):
            processor = self._create_signal_processor(processor_config)
            component.signal_pipeline.add_processor(processor)
        
        # Override signal processing if risk management exists
        if hasattr(component, 'process_signal'):
            original_process = component.process_signal
            
            def enhanced_process_signal(signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                """Process signal through pipeline first"""
                # Run through signal pipeline
                processed = component.signal_pipeline.process_signal(signal)
                if processed is None:
                    return None
                
                # Then through normal risk processing
                return original_process(processed)
            
            component.process_signal = enhanced_process_signal
        
        return component
    
    def _create_signal_processor(self, config: Dict[str, Any]) -> Any:
        """Create signal processor from config"""
        processor_type = config['type']
        
        if processor_type == 'regime_filter':
            return RegimeFilter(
                config.get('regime_detector', 'regime_detector'),
                config.get('regime_rules', {})
            )
        elif processor_type == 'confidence_filter':
            return ConfidenceFilter(
                config.get('min_confidence', 0.6),
                config.get('confidence_source', 'meta_labeler')
            )
        elif processor_type == 'signal_smoother':
            return SignalSmoother(
                config.get('window_size', 3),
                config.get('consensus_threshold', 0.6)
            )
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")
```

## 5. Complete Component Examples

### 5.1 Simulated Broker with Capabilities

```python
class SimulatedBroker:
    """Simulated broker for backtesting - no inheritance needed"""
    
    def __init__(self, name: str = "simulated_broker"):
        self.name = name
        # State will be added by capabilities

# Create with capabilities
broker = ComponentFactory().create_component({
    'name': 'broker',
    'class': 'SimulatedBroker',
    'capabilities': [
        'lifecycle',
        'events',
        'order_management',
        'market_simulation',
        'execution_context',
        'logging',
        'monitoring'
    ],
    
    # Market simulation config
    'slippage': {
        'type': 'percentage',
        'value': 0.001  # 0.1%
    },
    'commission': {
        'type': 'fixed',
        'value': 1.0  # $1 per trade
    },
    
    # Execution context
    'execution_mode': 'BACKTEST_SINGLE',
    
    # Logging
    'logger_name': 'execution.broker',
    
    # Monitoring
    'track_performance': ['execute_order', 'process_pending']
})
```

### 5.2 Risk Manager with Full Capabilities

```python
class RiskManager:
    """Risk manager using composition"""
    
    def __init__(self, name: str = "risk_manager"):
        self.name = name
        self.order_id_counter = 0

# Create with all risk capabilities
risk_manager = ComponentFactory().create_component({
    'name': 'risk_manager',
    'class': 'RiskManager',
    'capabilities': [
        'lifecycle',
        'events',
        'risk_management',
        'signal_processing',
        'logging',
        'monitoring',
        'error_handling'
    ],
    
    # Risk limits
    'risk_limits': [
        {'name': 'position_limit', 'type': 'position', 'max_position': 1000},
        {'name': 'exposure_limit', 'type': 'exposure', 'max_exposure_pct': 10},
        {'name': 'drawdown_limit', 'type': 'drawdown', 'max_drawdown_pct': 20}
    ],
    
    # Position sizers
    'position_sizers': [
        {'name': 'default', 'type': 'fixed', 'size': 100},
        {'name': 'percent_equity', 'type': 'percent_equity', 'percentage': 2}
    ],
    
    # Signal processors
    'signal_processors': [
        {
            'type': 'regime_filter',
            'regime_rules': {
                'trending_up': ['BUY'],
                'trending_down': ['SELL'],
                'high_volatility': []  # No trades
            }
        },
        {
            'type': 'confidence_filter',
            'min_confidence': 0.7
        }
    ],
    
    # Error handling
    'error_handling': {
        'critical_methods': ['process_signal', 'check_risk_limits']
    }
})
```

### 5.3 Portfolio with Decimal Precision

```python
class Portfolio:
    """Portfolio manager with precise decimal calculations"""
    
    def __init__(self, name: str = "portfolio", 
                 initial_cash: float = 100000):
        self.name = name
        self.initial_cash = Decimal(str(initial_cash))

# Create with capabilities
portfolio = ComponentFactory().create_component({
    'name': 'portfolio',
    'class': 'Portfolio',
    'capabilities': [
        'lifecycle',
        'events',
        'portfolio_management',
        'logging',
        'monitoring',
        'validation'
    ],
    
    'initial_cash': 100000,
    
    # Monitoring
    'health_check': {
        'checks': ['cash_balance', 'position_consistency'],
        'thresholds': {
            'min_cash': -1000,  # Allow small negative for rounding
            'max_position_age_days': 30
        }
    },
    
    # Validation
    'validation': {
        'rules': [
            {
                'type': 'state',
                'attribute': 'cash',
                'validator': 'non_negative',
                'message': 'Cash cannot be significantly negative'
            }
        ]
    }
})
```

## 6. Container Integration

### 6.1 Execution Container

```python
class ExecutionContainer(UniversalScopedContainer):
    """Container for execution components"""
    
    def create_execution_stack(self, config: Dict[str, Any]) -> None:
        """Create complete execution stack"""
        
        # Create order manager
        self.create_component({
            'name': 'order_manager',
            'class': 'OrderManager',
            'capabilities': ['lifecycle', 'events', 'order_management'],
            'dependencies': []
        })
        
        # Create broker based on mode
        if config.get('execution_mode') == 'LIVE_TRADING':
            broker_class = 'LiveBroker'
            broker_capabilities = ['lifecycle', 'events', 'order_management', 
                                 'execution_context', 'error_handling']
        else:
            broker_class = 'SimulatedBroker'
            broker_capabilities = ['lifecycle', 'events', 'order_management', 
                                 'market_simulation', 'execution_context']
        
        self.create_component({
            'name': 'broker',
            'class': broker_class,
            'capabilities': broker_capabilities,
            'dependencies': ['order_manager'],
            **config.get('broker_config', {})
        })
```

### 6.2 Risk Container

```python
class RiskContainer(UniversalScopedContainer):
    """Container for risk components"""
    
    def create_risk_stack(self, config: Dict[str, Any]) -> None:
        """Create complete risk management stack"""
        
        # Create portfolio first (no dependencies)
        self.create_component({
            'name': 'portfolio',
            'class': 'Portfolio',
            'capabilities': ['lifecycle', 'events', 'portfolio_management'],
            'initial_cash': config.get('initial_cash', 100000)
        })
        
        # Create risk manager with portfolio dependency
        self.create_component({
            'name': 'risk_manager',
            'class': 'RiskManager',
            'capabilities': ['lifecycle', 'events', 'risk_management', 
                           'signal_processing'],
            'dependencies': ['portfolio'],
            **config.get('risk_config', {})
        })
```

## 7. Execution Modes and Thread Safety

### 7.1 Execution Context Integration

```python
def create_execution_environment(mode: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create execution environment based on mode"""
    
    # Set execution context
    exec_mode = ExecutionMode[mode]
    thread_model = ExecutionContext.DEFAULT_THREAD_MODELS[exec_mode]
    
    context = ExecutionContext(exec_mode, thread_model)
    
    # Create components with appropriate thread safety
    with context:
        if exec_mode == ExecutionMode.BACKTEST_SINGLE:
            # Single-threaded: no locks needed
            components = create_backtest_components(config, thread_safe=False)
            
        elif exec_mode == ExecutionMode.LIVE_TRADING:
            # Multi-threaded: full thread safety
            components = create_live_components(config, thread_safe=True)
            
        elif exec_mode == ExecutionMode.OPTIMIZATION:
            # Process-level parallelism
            components = create_optimization_components(config)
    
    return components
```

### 7.2 Thread-Safe Collections

```python
class ThreadSafePortfolioCapability(Capability):
    """Enhanced portfolio capability with thread safety"""
    
    def get_name(self) -> str:
        return "thread_safe_portfolio"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        # First apply base portfolio capability
        PortfolioManagementCapability().apply(component, spec)
        
        # Check if thread safety needed
        if hasattr(component, 'requires_thread_safety') and component.requires_thread_safety():
            # Replace collections with thread-safe versions
            import threading
            
            # Add locks
            component._position_lock = threading.RLock()
            component._cash_lock = threading.RLock()
            
            # Wrap methods with locks
            original_update = component.update_position
            
            def thread_safe_update(symbol: str, quantity: int, 
                                 price: float, commission: float = 0) -> None:
                with component._position_lock:
                    with component._cash_lock:
                        original_update(symbol, quantity, price, commission)
            
            component.update_position = thread_safe_update
        
        return component
```

## 8. Configuration Examples

### 8.1 Backtest Configuration

```yaml
execution:
  mode: "BACKTEST_SINGLE"
  
  broker:
    class: "SimulatedBroker"
    capabilities: 
      - "lifecycle"
      - "events"
      - "order_management"
      - "market_simulation"
      - "execution_context"
    
    slippage:
      type: "percentage"
      value: 0.001
    
    commission:
      type: "tiered"
      tiers:
        - max_size: 1000
          rate: 1.0
        - max_size: 10000
          rate: 0.5
        - max_size: null
          rate: 0.1

risk:
  portfolio:
    class: "Portfolio"
    capabilities: ["lifecycle", "events", "portfolio_management"]
    initial_cash: 100000
  
  risk_manager:
    class: "RiskManager"
    capabilities: 
      - "lifecycle"
      - "events"
      - "risk_management"
      - "signal_processing"
    
    risk_limits:
      - name: "position"
        type: "position"
        max_position: 1000
      
      - name: "exposure"
        type: "exposure"
        max_exposure_pct: 10
      
      - name: "drawdown"
        type: "drawdown"
        max_drawdown_pct: 15
        reduce_at_pct: 10
```

### 8.2 Live Trading Configuration

```yaml
execution:
  mode: "LIVE_TRADING"
  thread_model: "MULTI_THREADED"
  
  broker:
    class: "LiveBroker"
    capabilities: 
      - "lifecycle"
      - "events"
      - "order_management"
      - "execution_context"
      - "error_handling"
      - "monitoring"
    
    api_config:
      endpoint: "wss://api.broker.com"
      auth_token: "${BROKER_TOKEN}"
    
    error_handling:
      retry:
        max_attempts: 3
        backoff: "exponential"
      critical_methods: ["submit_order", "cancel_order"]

risk:
  portfolio:
    class: "LivePortfolio"
    capabilities: 
      - "lifecycle"
      - "events"
      - "portfolio_management"
      - "thread_safe_portfolio"
      - "monitoring"
    
    reconciliation:
      enabled: true
      frequency: "5m"
      broker_sync: true
```

## 9. Benefits of Protocol-Based Execution & Risk

### 9.1 Clean Separation

```python
# Execution focuses on order processing
class MyBroker:
    def execute_order(self, order):
        # Pure execution logic
        pass

# Risk focuses on position management
class MyRiskManager:
    def process_signal(self, signal):
        # Pure risk logic
        pass

# Capabilities added separately
broker = add_capabilities(MyBroker(), ['order_management', 'market_simulation'])
risk = add_capabilities(MyRiskManager(), ['risk_management', 'signal_processing'])
```

### 9.2 Flexible Testing

```python
# Test with minimal setup
def test_risk_limits():
    # Create risk manager with just needed capabilities
    risk = create_component({
        'class': 'RiskManager',
        'capabilities': ['risk_management'],
        'risk_limits': [
            {'type': 'position', 'max_position': 100}
        ]
    })
    
    # Test limit checking
    passed, reason = risk.check_risk_limits(
        {'symbol': 'TEST', 'price': 100}, 
        quantity=150
    )
    
    assert not passed
    assert 'position' in reason
```

### 9.3 Container Isolation

```python
# Each strategy gets isolated execution and risk stack
def create_strategy_container(strategy_id: str, config: Dict[str, Any]):
    container = UniversalScopedContainer(f"strategy_{strategy_id}")
    
    # Create isolated components
    container.create_component({
        'name': 'portfolio',
        'class': 'Portfolio',
        'capabilities': ['portfolio_management'],
        'initial_cash': config['allocated_capital']
    })
    
    container.create_component({
        'name': 'risk_manager',
        'class': 'RiskManager',
        'capabilities': ['risk_management'],
        'dependencies': ['portfolio'],
        'risk_limits': config['risk_limits']
    })
    
    return container
```

## Summary

The Execution and Risk modules with Protocol + Composition provide:

1. **Clean Interfaces** - Components implement protocols without inheritance overhead
2. **Flexible Capabilities** - Add only the features needed (thread safety, monitoring, etc.)
3. **Container Integration** - Natural isolation for parallel execution
4. **Thread Safety Options** - Automatic based on execution mode
5. **Precise Calculations** - Decimal arithmetic for financial accuracy
6. **Comprehensive Testing** - Easy to test components in isolation

This architecture enables building robust, production-ready trading systems while maintaining simplicity and flexibility. Whether running a simple backtest or complex live trading with multiple strategies, the same clean patterns apply.

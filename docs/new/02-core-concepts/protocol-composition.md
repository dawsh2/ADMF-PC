# Protocol + Composition

Protocol + Composition is the foundational architectural principle that enables ADMF-PC's infinite flexibility. This approach eliminates inheritance hierarchies in favor of composable protocols, allowing any components to work together seamlessly.

## 🎯 The Core Problem with Inheritance

Traditional trading systems use inheritance hierarchies:

```python
# Traditional inheritance approach ❌
class BaseStrategy:
    def calculate_signal(self, data): pass
    def get_parameters(self): pass
    def validate_config(self): pass

class MomentumStrategy(BaseStrategy):
    def calculate_signal(self, data):
        # Momentum implementation
        pass

class MLStrategy(BaseStrategy):  # Problem: ML models don't fit this pattern
    def calculate_signal(self, data):
        # How do you fit sklearn here?
        pass
```

**Problems with this approach**:
- **Rigid Structure**: Everything must inherit from base classes
- **Framework Lock-in**: Can't use external libraries easily
- **Multiple Inheritance**: Complex when components need multiple capabilities
- **Tight Coupling**: Changes to base class affect all children
- **Limited Composability**: Hard to mix different component types

## 🚀 The Protocol + Composition Solution

ADMF-PC uses **duck typing** with **protocols** instead of inheritance:

```python
# Protocol-based approach ✅
from typing import Protocol

class SignalGenerator(Protocol):
    """Any component that can generate trading signals"""
    
    def generate_signal(self, market_data: MarketData) -> TradingSignal:
        """Generate trading signal from market data"""
        ...

# Any component implementing this protocol works
class MomentumStrategy:
    def generate_signal(self, market_data):
        # Momentum logic
        return TradingSignal(action="BUY", strength=0.8)

class SklearnModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
    
    def generate_signal(self, market_data):
        # ML prediction logic
        prediction = self.model.predict(market_data.features)
        return TradingSignal(action="BUY" if prediction > 0 else "SELL", 
                           strength=abs(prediction))

class CustomFunction:
    def __init__(self, func):
        self.func = func
    
    def generate_signal(self, market_data):
        # User-defined function
        return self.func(market_data)

class ExternalAPI:
    def __init__(self, api_endpoint):
        self.endpoint = api_endpoint
    
    def generate_signal(self, market_data):
        # External service call
        response = requests.post(self.endpoint, json=market_data.to_dict())
        return TradingSignal.from_dict(response.json())
```

**Key Insight**: If it has a `generate_signal` method that returns a `TradingSignal`, it's a strategy! No inheritance required.

## 📋 Core Protocols in ADMF-PC

### 1. Data Provider Protocol
```python
class DataProvider(Protocol):
    """Provides market data"""
    
    def get_bars(self, symbol: str, start: datetime, end: datetime) -> List[Bar]:
        """Get historical price bars"""
        ...
    
    def subscribe_real_time(self, symbol: str, callback: Callable):
        """Subscribe to real-time data"""
        ...

# Implementations can be anything:
# - CSV files
# - Databases  
# - Live data feeds
# - Synthetic data generators
# - External APIs
```

### 2. Signal Generator Protocol
```python
class SignalGenerator(Protocol):
    """Generates trading signals"""
    
    def generate_signal(self, market_data: MarketData) -> TradingSignal:
        """Generate trading signal"""
        ...
    
    def get_required_indicators(self) -> List[str]:
        """List required indicators"""
        ...

# Can be implemented by:
# - Technical analysis strategies
# - ML models
# - Fundamental analysis
# - Sentiment analysis
# - Custom algorithms
```

### 3. Risk Manager Protocol
```python
class RiskManager(Protocol):
    """Manages trading risk"""
    
    def check_risk(self, signal: TradingSignal, portfolio: Portfolio) -> bool:
        """Check if signal passes risk constraints"""
        ...
    
    def size_position(self, signal: TradingSignal, portfolio: Portfolio) -> int:
        """Determine position size"""
        ...

# Implementations:
# - Fixed position sizing
# - Volatility-based sizing
# - Kelly criterion
# - Portfolio optimization
# - Custom risk models
```

### 4. Execution Engine Protocol
```python
class ExecutionEngine(Protocol):
    """Executes trading orders"""
    
    def submit_order(self, order: Order) -> OrderResult:
        """Submit order for execution"""
        ...
    
    def get_position(self, symbol: str) -> Position:
        """Get current position"""
        ...

# Implementations:
# - Simulated execution (backtesting)
# - Broker APIs
# - Custom execution algorithms
# - Market making engines
```

## 🧩 Composition in Action

### Mixing Any Components

With protocols, you can mix any compatible components:

```yaml
# YAML configuration mixing different component types
strategies:
  - type: "momentum"              # Built-in technical strategy
    weight: 0.3
    
  - type: "sklearn_model"         # Machine learning model
    model_path: "models/random_forest.pkl"
    weight: 0.3
    
  - function: "sentiment_signal"   # Custom Python function
    module: "user_strategies"
    weight: 0.2
    
  - type: "external_api"          # External service
    endpoint: "https://signals.example.com/api"
    weight: 0.2

risk_management:
  - type: "volatility_based"      # Built-in risk manager
  - function: "custom_risk_check" # Custom risk function
    module: "user_risk"

execution:
  type: "smart_router"            # Custom execution engine
  params:
    venues: ["NYSE", "NASDAQ", "BATS"]
```

### Component Enhancement Through Composition

Add capabilities to any component without modifying it:

```python
# Original component
class SimpleStrategy:
    def generate_signal(self, data):
        return TradingSignal(action="BUY", strength=0.5)

# Enhanced with logging (composition, not inheritance)
class LoggingEnhancer:
    def __init__(self, wrapped_component, logger):
        self.wrapped = wrapped_component
        self.logger = logger
    
    def generate_signal(self, data):
        self.logger.info(f"Generating signal for {data.symbol}")
        signal = self.wrapped.generate_signal(data)
        self.logger.info(f"Generated signal: {signal}")
        return signal

# Enhanced with caching
class CachingEnhancer:
    def __init__(self, wrapped_component):
        self.wrapped = wrapped_component
        self.cache = {}
    
    def generate_signal(self, data):
        cache_key = (data.symbol, data.timestamp)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        signal = self.wrapped.generate_signal(data)
        self.cache[cache_key] = signal
        return signal

# Enhanced with error handling
class ErrorHandlingEnhancer:
    def __init__(self, wrapped_component, fallback_signal):
        self.wrapped = wrapped_component
        self.fallback = fallback_signal
    
    def generate_signal(self, data):
        try:
            return self.wrapped.generate_signal(data)
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return self.fallback

# Compose multiple enhancements
enhanced_strategy = LoggingEnhancer(
    CachingEnhancer(
        ErrorHandlingEnhancer(
            SimpleStrategy(),
            TradingSignal(action="HOLD", strength=0.0)
        )
    ),
    logger
)
```

## 🔄 Dynamic Composition

### Runtime Component Swapping

```python
class StrategyContainer:
    def __init__(self):
        self.signal_generator = None
        self.risk_manager = None
        
    def set_signal_generator(self, generator: SignalGenerator):
        """Swap signal generator at runtime"""
        self.signal_generator = generator
        
    def set_risk_manager(self, risk_manager: RiskManager):
        """Swap risk manager at runtime"""
        self.risk_manager = risk_manager
        
    def process_data(self, data):
        """Process with current components"""
        if self.signal_generator:
            signal = self.signal_generator.generate_signal(data)
            if self.risk_manager and self.risk_manager.check_risk(signal):
                return signal
        return None

# Runtime composition
container = StrategyContainer()

# Start with momentum strategy
container.set_signal_generator(MomentumStrategy())
container.set_risk_manager(FixedRiskManager())

# Switch to ML model based on market conditions
if market_regime == "volatile":
    container.set_signal_generator(SklearnModel("models/volatility_model.pkl"))
    container.set_risk_manager(VolatilityRiskManager())
```

### Configuration-Driven Composition

```yaml
# Dynamic strategy composition based on conditions
adaptive_strategy:
  base_strategy:
    type: "momentum"
    params:
      fast_period: 10
      slow_period: 20
      
  enhancements:
    - type: "logging"
      level: "INFO"
      
    - type: "caching" 
      ttl_seconds: 300
      
    - type: "error_handling"
      fallback_action: "HOLD"
      max_retries: 3
      
  regime_adaptations:
    bull_market:
      replace_strategy:
        type: "breakout"
        params:
          period: 20
          
    bear_market:
      add_enhancement:
        type: "defensive_filter"
        max_drawdown: 0.05
```

## 🏗️ Building Custom Protocols

### Defining New Protocols

```python
class RegimeClassifier(Protocol):
    """Classifies market regimes"""
    
    def classify_regime(self, market_data: MarketData) -> RegimeInfo:
        """Classify current market regime"""
        ...
    
    def get_regime_probability(self, regime: str) -> float:
        """Get probability of specific regime"""
        ...

class PortfolioOptimizer(Protocol):
    """Optimizes portfolio allocations"""
    
    def optimize_weights(self, 
                        signals: List[TradingSignal], 
                        constraints: PortfolioConstraints) -> Dict[str, float]:
        """Optimize portfolio weights"""
        ...
    
    def rebalance_schedule(self) -> RebalanceSchedule:
        """Get rebalancing schedule"""
        ...
```

### Protocol Composition

Protocols can be composed to create more complex interfaces:

```python
class AdvancedStrategy(SignalGenerator, RegimeClassifier):
    """Strategy that also classifies market regimes"""
    
    def generate_signal(self, market_data: MarketData) -> TradingSignal:
        # First classify regime
        regime = self.classify_regime(market_data)
        
        # Generate signal based on regime
        if regime.type == "trending":
            return self.momentum_signal(market_data)
        else:
            return self.mean_reversion_signal(market_data)
    
    def classify_regime(self, market_data: MarketData) -> RegimeInfo:
        # Regime classification logic
        pass
```

## 🎛️ Configuration Integration

### Protocol-Based Configuration

```yaml
# Configuration specifies protocols, not implementations
components:
  signal_generators:
    - protocol: "SignalGenerator"
      implementation: "momentum"
      params:
        fast_period: 10
        slow_period: 20
        
    - protocol: "SignalGenerator"  
      implementation: "sklearn_model"
      model_path: "models/rf.pkl"
      
  risk_managers:
    - protocol: "RiskManager"
      implementation: "volatility_based"
      params:
        lookback_days: 20
        max_volatility: 0.3
        
  enhancements:
    - protocol: "ComponentEnhancer"
      implementation: "logging"
      apply_to: ["signal_generators", "risk_managers"]
      
    - protocol: "ComponentEnhancer"
      implementation: "monitoring"
      apply_to: ["*"]  # Apply to all components
```

### Automatic Protocol Detection

```python
class ComponentRegistry:
    """Automatically registers components by protocol"""
    
    def __init__(self):
        self.components_by_protocol = {}
        
    def register_component(self, component_class):
        """Register component and detect its protocols"""
        protocols = self.detect_protocols(component_class)
        for protocol in protocols:
            if protocol not in self.components_by_protocol:
                self.components_by_protocol[protocol] = []
            self.components_by_protocol[protocol].append(component_class)
            
    def detect_protocols(self, component_class) -> List[type]:
        """Detect which protocols a component implements"""
        protocols = []
        for protocol in [SignalGenerator, RiskManager, DataProvider]:
            if self.implements_protocol(component_class, protocol):
                protocols.append(protocol)
        return protocols
        
    def implements_protocol(self, component_class, protocol) -> bool:
        """Check if component implements protocol"""
        protocol_methods = set(dir(protocol))
        component_methods = set(dir(component_class))
        return protocol_methods.issubset(component_methods)
```

## 💡 Benefits of Protocol + Composition

### 1. **Infinite Flexibility**
```yaml
# Mix any components that implement compatible protocols
strategies:
  - type: "momentum"           # Traditional technical analysis
  - type: "lstm_model"         # Deep learning  
  - function: "my_algorithm"   # Custom algorithm
  - type: "crowd_wisdom"       # Social sentiment
  - type: "fundamental"        # Financial metrics
```

### 2. **Zero Framework Lock-in**
```python
# Use any library without modification
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoModel
import your_proprietary_library

# All work if they implement the right protocol
```

### 3. **Universal Enhancement**
```yaml
# Add logging, monitoring, caching to ANY component
enhancements:
  logging: 
    apply_to: "*"  # Everything gets logging
  monitoring:
    apply_to: ["strategies", "risk_managers"]
  caching:
    apply_to: ["data_providers"]
```

### 4. **Easy Testing**
```python
# Mock any protocol for testing
class MockSignalGenerator:
    def generate_signal(self, data):
        return TradingSignal(action="BUY", strength=0.5)

# Works anywhere a SignalGenerator is expected
strategy_container.set_signal_generator(MockSignalGenerator())
```

### 5. **Gradual Migration**
```python
# Gradually replace components without breaking the system
# Start with simple momentum
config = {"type": "momentum", "fast_period": 10}

# Upgrade to ML without changing anything else
config = {"type": "sklearn_model", "model_path": "model.pkl"}

# Add custom logic
config = {"function": "my_strategy", "module": "strategies"}
```

## 🎯 Best Practices

### Protocol Design
1. **Keep Protocols Small**: Single responsibility per protocol
2. **Use Type Hints**: Clear parameter and return types
3. **Document Behavior**: Specify expected behavior, not implementation
4. **Version Protocols**: Handle protocol evolution gracefully

### Component Design
1. **Implement Complete Protocols**: Don't partially implement protocols
2. **Handle Edge Cases**: Graceful error handling
3. **Document Dependencies**: Specify required inputs and outputs
4. **Make Components Stateless**: When possible, avoid internal state

### Composition Strategies
1. **Compose at Configuration Time**: Specify composition in YAML
2. **Use Dependency Injection**: Let the system wire components
3. **Layer Enhancements**: Apply enhancements in logical order
4. **Test Compositions**: Test component combinations thoroughly

## 🤔 Common Questions

**Q: Is this approach slower than inheritance?**
A: No! Protocol dispatch is as fast as method calls. The flexibility benefits far outweigh any minimal overhead.

**Q: How do I know what protocols to implement?**
A: Check the [Component Catalog](../04-reference/component-catalog.md) for standard protocols and examples.

**Q: Can I use existing libraries without modification?**
A: Often yes! Many libraries already implement compatible interfaces. For others, create thin wrapper classes.

**Q: How do I handle protocol versioning?**
A: Use semantic versioning for protocols and provide migration tools for breaking changes.

## 🎯 Key Takeaways

1. **Protocols > Inheritance**: Duck typing enables infinite flexibility
2. **Composition > Modification**: Enhance components without changing them
3. **Configuration > Code**: Specify composition in YAML, not Python
4. **Standards > Custom**: Use standard protocols for maximum compatibility
5. **Testing > Trust**: Mock protocols make testing easy and reliable

Protocol + Composition is what enables ADMF-PC to integrate any component while maintaining the zero-code philosophy. It's the foundation that makes infinite composability possible.

---

Next: [Coordinator Orchestration](coordinator-orchestration.md) - How the central brain manages it all →
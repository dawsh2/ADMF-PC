# ADMF-PC Style Guide

This guide defines coding standards and best practices for ADMF-PC development.

## 🚨 CRITICAL: No Duplicate Files Policy

### The Golden Rule: One Canonical File Per Component

**NEVER create "enhanced", "improved", "better", or versioned variants of existing files.**

❌ **WRONG**:
```
src/execution/
├── backtest_broker.py              # Original
├── backtest_broker_refactored.py   # NO! Don't do this
├── improved_backtest_broker.py     # NO! Don't do this
├── backtest_broker_v2.py          # NO! Don't do this
└── enhanced_backtest_broker.py     # NO! Don't do this
```

✅ **CORRECT**:
```
src/execution/
└── backtest_broker.py              # THE canonical file - modify this directly
```

### Finding Canonical Files

The canonical files are those **actually used by the Coordinator and imported in production code**.

To identify canonical files:
1. Check what the Coordinator imports
2. Look at `src/core/coordinator/` for actual usage
3. Check main.py imports
4. Run: `python scripts/list_canonical_files.py`

### Why This Matters

1. **Confusion**: Multiple versions create uncertainty about which to use
2. **Maintenance**: Changes must be synchronized across duplicates
3. **Bugs**: Fixes applied to wrong version don't help
4. **Testing**: Tests may run against non-production code
5. **Onboarding**: New developers/AI agents get confused

### How to Improve Existing Code

Instead of creating new files:

```python
# 1. Create a feature branch
git checkout -b improve-backtest-broker

# 2. Modify the canonical file directly
# src/execution/backtest_broker.py

# 3. Ensure all tests pass
pytest tests/test_execution/

# 4. Update documentation if needed
# docs/execution/README.md

# 5. Submit for review
```

---

## 📐 Protocol + Composition Philosophy

### Core Principle: Zero Inheritance

**NEVER use class inheritance. Use protocols and composition.**

❌ **WRONG**:
```python
class MomentumStrategy(BaseStrategy):  # NO inheritance!
    def __init__(self):
        super().__init__()  # Framework coupling
```

✅ **CORRECT**:
```python
from typing import Protocol

class StrategyProtocol(Protocol):
    """Define what a strategy must do, not how"""
    def evaluate(self, market_data: MarketData) -> Signal:
        ...

class MomentumStrategy:  # No inheritance!
    """Implements protocol through duck typing"""
    def evaluate(self, market_data: MarketData) -> Signal:
        # Implementation
        return Signal(...)
```

### Composition Over Inheritance

Build complex behavior by combining simple components:

```python
class AdaptiveStrategy:
    """Compose strategies instead of inheriting"""
    def __init__(self):
        self.strategies = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            MLPredictor()  # Can mix any components!
        ]
```

---

## 🏗️ Container Architecture Standards

### Container Isolation

Each container must be completely isolated:

```python
class StrategyContainer:
    def __init__(self, config: Dict):
        # Own event bus
        self.event_bus = EventBus()
        
        # Own state
        self.state = ContainerState()
        
        # No shared globals
        # No cross-container references
```

### Container Communication

Containers communicate ONLY through the Event Router - never direct method calls:

```python
# WRONG: Direct method calls
other_container.do_something()  # NO!
other_container.get_data()      # NO!

# WRONG: Direct event bus access
other_container.event_bus.subscribe(...)  # NO!

# CORRECT: Cross-container via Event Router
class StrategyContainer(ContainerProtocol):
    # Declare what you publish
    publishes = {"SIGNAL"}
    
    # Declare what you subscribe to
    subscribes_to = {
        "INDICATOR": {"indicator_container"},
        "BAR": {"data_container"}
    }
    
    def register_with_router(self, router: EventRouter):
        """Register pub/sub interests with central router"""
        router.register_publisher(self.container_id, self.publishes)
        router.register_subscriber(self.container_id, self.subscribes_to)
    
    def handle_routed_event(self, event: Event, source: str):
        """Handle events routed from other containers"""
        if event.type == "INDICATOR":
            self._process_indicator_update(event)
        elif event.type == "BAR":
            self._process_market_data(event)
```

### Cross-Container Event Flow

The mandatory pattern for cross-container communication:

```
Container A              Event Router              Container B
    │                         │                         │
    │──publish(SIGNAL)────────►│                         │
    │                         │                         │
    │                         │──handle_routed_event───►│
    │                         │                         │
    │                         │                         │
    │◄────response(ACK)───────│◄──publish(ACK)─────────│
```

### Configuration-Driven Routing

Define container communication in YAML:

```yaml
containers:
  strategy_container:
    subscribes_to:
      - source: "indicator_container"
        events: ["INDICATOR"]
        filters: {"symbol": "SPY"}  # Optional filtering
      - source: "data_container" 
        events: ["BAR"]
        delivery: "sync"  # For critical path
        
    publishes:
      - events: ["SIGNAL"]
        visibility: "parent"  # Scope control
        
  risk_container:
    subscribes_to:
      - source: "strategy_container"
        events: ["SIGNAL"]
        delivery: "async"  # Default async
```

---

## 📝 Code Organization

### File Structure

```python
# src/module/filename.py

"""
Module purpose in one line.

Detailed description if needed.
Architecture: References src/core/containers/protocols.py
Events: Publishes SIGNAL, subscribes to BAR
"""

from typing import Protocol, Dict, List
from src.core.events import Event, EventType

# Protocols first
class ComponentProtocol(Protocol):
    ...

# Main implementation
class Component:
    ...

# Helper functions last
def helper_function():
    ...
```

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Protocols**: `SomethingProtocol`

### Import Order

```python
# 1. Standard library
import os
import sys
from typing import Dict, List

# 2. Third-party
import numpy as np
import pandas as pd

# 3. ADMF-PC modules
from src.core.events import Event
from src.core.protocols import Protocol
```

---

## 📊 Logging Standards

### Mandatory: Use Structured Logging

**ALWAYS use the structured logging system - never use print() or basic logging.**

❌ **WRONG**:
```python
import logging
print("Processing order")  # NO!
logging.info("Order complete")  # NO! Basic logging
```

✅ **CORRECT**:
```python
from src.core.logging.structured import StructuredLogger, LogContext

# Initialize with context
context = LogContext(
    container_id="backtest_001",
    component_id="momentum_strategy"
)
logger = StructuredLogger(__name__, context)

# Structured logging with context
logger.info("Processing order", order_id="123", symbol="SPY")
```

### Required Logging Patterns

Every component must implement:

1. **ComponentLogger Pattern** (required by COMPLEXITY_CHECKLIST.md):
```python
def process_signal(self, signal):
    # Log event flow
    logger.log_event_flow(
        "SIGNAL", 
        "StrategyContainer", 
        "RiskContainer", 
        f"{signal.action} {signal.symbol}"
    )
    
    # Log state changes
    logger.log_state_change("IDLE", "PROCESSING", f"signal_{signal.id}")
    
    # Log performance metrics
    logger.log_performance_metric(
        "signal_processing_time_ms", 
        processing_time, 
        {"symbol": signal.symbol, "strategy": "momentum"}
    )
```

2. **Container-Aware Logging**:
```python
from src.core.logging.structured import ContainerLogger

class StrategyContainer:
    def __init__(self, container_id: str):
        self.logger = ContainerLogger(
            name=__name__,
            container_id=container_id,
            component_id="strategy"
        )
```

3. **Correlation Tracking**:
```python
# For tracking across components
with logger.correlation_context("trade_123"):
    logger.info("Starting trade processing")
    process_trade()
    logger.info("Trade completed")
```

### Log Levels and Usage

Use appropriate log levels:

```python
# TRACE: Ultra-detailed debugging (use sparingly)
logger.trace("Entering calculation loop", iteration=i, data_size=len(data))

# DEBUG: Development debugging
logger.debug("Processing signal", signal_strength=0.8, threshold=0.5)

# INFO: Normal operations
logger.info("Strategy initialized", strategy="momentum", period=20)

# WARNING: Concerning but not failing
logger.warning("High memory usage", memory_mb=512, threshold=400)

# ERROR: Error conditions
logger.error("Order failed", order_id="123", reason="insufficient_funds")

# CRITICAL: System failures
logger.critical("Database connection lost", exc_info=True)
```

### Method Tracing

Use the trace decorator for complex methods:

```python
from src.core.logging.structured import trace_method

class TradingStrategy:
    @trace_method(include_args=True, include_result=True)
    def calculate_signal(self, market_data):
        # Automatically logs entry/exit with timing
        return {"action": "buy", "strength": 0.8}
```

### Context and Extra Fields

Always include relevant context:

```python
# Good context examples
logger.info(
    "Order executed",
    order_id="123",
    symbol="SPY", 
    quantity=100,
    price=150.25,
    execution_time_ms=45
)

logger.error(
    "Risk limit exceeded",
    current_exposure=0.85,
    max_exposure=0.80,
    strategy="momentum",
    action="blocked_order"
)
```

### Configuration

Setup logging in main.py:

```python
from src.core.logging.structured import setup_logging

# Production setup
setup_logging(
    level="INFO",
    console=True,
    file_path="logs/trading.log",
    json_format=True
)

# Development setup  
setup_logging(
    level="DEBUG",
    console=True,
    json_format=True
)
```

### Logging Checklist

Every component must:

- [ ] Use StructuredLogger (never print() or basic logging)
- [ ] Include container_id and component_id in context
- [ ] Implement ComponentLogger pattern methods
- [ ] Log event flows between components
- [ ] Log state changes with triggers
- [ ] Log performance metrics with context
- [ ] Use appropriate log levels
- [ ] Include correlation tracking for related operations
- [ ] Use method tracing for complex operations

---

## 🧪 Testing Standards

### Three-Tier Testing

Every component needs:

1. **Unit Tests**: Test in isolation
2. **Integration Tests**: Test with other components
3. **System Tests**: Test in full system context

```python
# tests/unit/test_component.py
def test_component_isolation():
    """Test component without dependencies"""
    
# tests/integration/test_component_integration.py
def test_component_with_dependencies():
    """Test component with real dependencies"""
    
# tests/system/test_component_system.py
def test_component_in_full_system():
    """Test component in complete system"""
```

---

## 📚 Documentation Standards

### Docstring Format

```python
def process_signal(self, signal: Signal) -> Order:
    """
    Convert signal to order with risk management.
    
    Args:
        signal: Trading signal with direction and strength
        
    Returns:
        Order ready for execution
        
    Raises:
        RiskLimitExceeded: If order would breach risk limits
        
    Events:
        Subscribes: SIGNAL
        Publishes: ORDER
    """
```

### Architecture References

Always reference architecture documents:

```python
"""
Risk management implementation.

Architecture: 
- Event-driven (see docs/architecture/01-EVENT-DRIVEN-ARCHITECTURE.md)
- Container-based (see docs/architecture/02-CONTAINER-HIERARCHY.md)
- Protocol composition (see docs/architecture/03-PROTOCOL-COMPOSITION.md)
"""
```

---

## 🚀 Configuration Standards

### YAML Best Practices

```yaml
# Clear hierarchical structure
workflow:
  type: "backtest"
  name: "descriptive_name"  # Use underscores

# Group related settings
risk:
  position_sizing:
    method: "kelly_criterion"
    kelly_fraction: 0.25
  
  limits:
    max_position_pct: 5.0
    max_drawdown_pct: 20.0

# Use meaningful names
strategies:
  - name: "trend_following_spy"  # Descriptive!
    type: "momentum"
    # ... parameters
```

---

## ❌ Anti-Patterns to Avoid

### 1. Creating Duplicate Files
See top of this document - NEVER create enhanced/improved versions.

### 2. Using Inheritance
Always use protocols and composition.

### 3. Sharing State
Containers must be isolated.

### 4. Direct Method Calls
Use events for communication.

### 5. Modifying Core Without Tests
Always maintain three-tier test coverage.

### 6. Skipping Documentation
Every file needs proper docstrings and architecture references.

---

## ✅ Checklist for New Code

Before submitting code:

- [ ] No duplicate files created
- [ ] Using canonical file (check Coordinator imports)
- [ ] No inheritance used
- [ ] Protocols defined for interfaces
- [ ] Container isolation maintained
- [ ] Events used for communication
- [ ] Three-tier tests written
- [ ] Documentation complete
- [ ] Architecture references included
- [ ] Style guide followed

---

## 🔧 Tools and Scripts

### Check Canonical Files
```bash
# List all canonical files used by system
python scripts/list_canonical_files.py

# Check if a file is canonical
python scripts/check_canonical.py src/execution/backtest_broker.py
```

### Style Checking
```bash
# Run style checks
python scripts/check_style.py

# Auto-format code
black src/
isort src/
```

---

## 📚 References

- [SYSTEM_ARCHITECTURE_V5.MD](../SYSTEM_ARCHITECTURE_V5.MD) - Canonical architecture reference
- [Protocol + Composition Benefits](../docs/PC/BENEFITS.MD)
- [Event-Driven Architecture](../architecture/01-EVENT-DRIVEN-ARCHITECTURE.md)
- [Container Architecture](../architecture/02-CONTAINER-HIERARCHY.md)

---

*Remember: When in doubt, check what the Coordinator actually uses!*
# Documentation Standards

Comprehensive documentation standards for ADMF-PC, ensuring clarity, consistency, and architectural alignment.

## File Headers

### Python Module Header

Every Python file must start with:

```python
"""
Module: {module_name}
Location: {file_path}

{Brief description of module purpose}

Architecture:
    - Container: {container_type if applicable}
    - Protocol: {protocol(s) implemented}
    - Events: {event types published/subscribed}

Dependencies:
    - Core: {core modules used}
    - External: {external libraries}
"""
```

### Example Module Header

```python
"""
Module: momentum_strategy
Location: src/strategy/strategies/momentum.py

Momentum-based trading strategy using moving average crossovers.
Implements SignalGenerator protocol with Protocol + Composition design.

Architecture:
    - Container: StrategyContainer
    - Protocol: SignalGenerator
    - Events: 
        - Publishes: SIGNAL
        - Subscribes: BAR_DATA, INDICATOR

Dependencies:
    - Core: events, logging
    - External: numpy, pandas
"""
```

## Class Documentation

### Class Header Format

```python
class ComponentName:
    """
    Brief description of the component.
    
    Detailed explanation of purpose and behavior.
    Emphasize Protocol + Composition usage.
    
    Architecture:
        Container: {where this runs}
        Lifecycle: {initialization → ready → running → stopped}
        Protocol: {protocol(s) implemented}
    
    Configuration:
        param1 (type): Description (default: value)
        param2 (type): Description (default: value)
    
    Events:
        Published:
            - EVENT_TYPE: When and what data
        Subscribed:
            - EVENT_TYPE: What it does with event
    
    Example:
        >>> component = ComponentName({"param1": value})
        >>> component.process(data)
    """
```

### Real Example

```python
class MomentumStrategy:
    """
    Momentum trading strategy using dual moving averages.
    
    Generates BUY signals when fast MA crosses above slow MA with
    sufficient momentum. Uses Protocol + Composition - no inheritance.
    
    Architecture:
        Container: StrategyContainer
        Lifecycle: initialize → ready → process bars → stopped
        Protocol: SignalGenerator
    
    Configuration:
        fast_period (int): Fast MA period (default: 10)
        slow_period (int): Slow MA period (default: 30)
        momentum_threshold (float): Min momentum (default: 0.02)
        
    Events:
        Published:
            - SIGNAL: Trading signal with action, strength, metadata
        Subscribed:
            - BAR_DATA: Updates indicators and checks for signals
            - INDICATOR: Receives pre-calculated indicator values
    
    Example:
        >>> strategy = MomentumStrategy({"fast_period": 5})
        >>> strategy.initialize(context)
        >>> strategy.process_bar(bar_data)
    """
```

## Function/Method Documentation

### Function Header Format

```python
def function_name(
    param1: Type,
    param2: Type,
    param3: Dict[str, Any]
) -> ReturnType:
    """
    Brief description of what function does.
    
    Detailed explanation if needed. Mention any Protocol + Composition
    patterns used. Explain duck typing if applicable.
    
    Args:
        param1: Description of parameter
        param2: Description of parameter
        param3: Configuration dict with expected keys:
            - key1: Description
            - key2: Description
    
    Returns:
        Description of return value and format
        
    Raises:
        ExceptionType: When this exception occurs
        
    Events:
        - EVENT_TYPE: If function publishes events
        
    Example:
        >>> result = function_name(val1, val2, {"key1": "value"})
        >>> assert result.status == "success"
    """
```

### Real Example

```python
def calculate_position_size(
    signal: Dict[str, Any],
    portfolio_value: float,
    risk_params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Calculate position size from signal using risk parameters.
    
    Implements Kelly Criterion with safety factor. Works with any
    signal dict that has 'strength' key (duck typing).
    
    Args:
        signal: Signal dict with at least 'strength' key
        portfolio_value: Current portfolio value
        risk_params: Risk configuration:
            - max_position_pct: Max position as % of portfolio
            - kelly_fraction: Kelly safety factor (0-1)
            - min_position_size: Minimum viable position
    
    Returns:
        Position dict with 'size' and 'direction' or None if no position
        
    Raises:
        ValueError: If signal strength outside [0, 1]
        
    Events:
        - POSITION_SIZED: Published with position details
        
    Example:
        >>> position = calculate_position_size(
        ...     {"strength": 0.8, "action": "BUY"},
        ...     100000,
        ...     {"max_position_pct": 2.0}
        ... )
        >>> assert position['size'] <= 2000
    """
```

## Logging Integration

### Required Logging Setup

Every component must include logging initialization:

```python
class Component:
    def __init__(self, config: Dict[str, Any], container_id: str):
        # Required: Initialize logger with component context
        self.logger = ComponentLogger(
            component_name=self.__class__.__name__,
            container_id=container_id
        )
        
        # Log initialization
        self.logger.log_state_change("created", "initializing", "constructor")
```

### Logging Documentation

Document what gets logged:

```python
def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process signal and generate order.
    
    Logging:
        - Signal receipt (INFO)
        - Risk check results (INFO/WARNING)
        - Order generation (INFO)
        - Errors (ERROR)
    """
```

## Architectural References

### Required Links

Every class and module must reference:

1. **Container Type**: Which container pattern it uses
2. **Protocol**: Which protocols it implements
3. **Events**: Event flow documentation
4. **Complexity Step**: Related complexity guide step

### Example References

```python
class SignalReplayContainer:
    """
    Container for ultra-fast signal replay optimization.
    
    See Also:
        - Architecture: docs/architecture/02-CONTAINER-HIERARCHY.md
        - Pattern: docs/architecture/04-THREE-PATTERN-BACKTEST.md#signal-replay
        - Implementation: docs/complexity-guide/03-signal-capture-replay/step-08-signal-replay.md
        - Events: docs/architecture/01-EVENT-DRIVEN-ARCHITECTURE.md
    """
```

## Configuration Documentation

### YAML Configuration Docs

```yaml
# Document each configuration section
strategy:
  # Strategy configuration for signal generation
  type: "momentum"  # Strategy type (momentum|mean_reversion|breakout)
  
  parameters:
    # Strategy-specific parameters
    fast_period: 10     # Fast MA period (5-50)
    slow_period: 30     # Slow MA period (20-200)
    threshold: 0.02     # Signal threshold (0.01-0.1)
  
  # Optional enhancements
  capabilities:
    - "logging"         # Add structured logging
    - "monitoring"      # Add performance monitoring
```

### Configuration Class Docs

```python
@dataclass
class StrategyConfig:
    """
    Strategy configuration.
    
    Attributes:
        strategy_type: Type of strategy (momentum, mean_reversion, etc.)
        parameters: Strategy-specific parameters
        capabilities: Optional capability enhancements
        
    Configuration File:
        See config/example_strategy.yaml for full example
    """
    strategy_type: str
    parameters: Dict[str, Any]
    capabilities: List[str] = field(default_factory=list)
```

## API Documentation

### Public API Documentation

```python
# module/__init__.py
"""
Public API for strategy module.

Main Classes:
    - SignalGenerator: Protocol for signal generation
    - StrategyContainer: Container for strategy execution
    - MomentumStrategy: Momentum-based strategy
    - MeanReversionStrategy: Mean reversion strategy

Main Functions:
    - create_strategy: Factory for strategy creation
    - validate_signal: Signal validation utility

Example:
    >>> from strategy import create_strategy
    >>> strategy = create_strategy("momentum", fast_period=10)
    >>> signal = strategy.generate_signal(data)
"""

from .protocols import SignalGenerator
from .containers import StrategyContainer
from .strategies import MomentumStrategy, MeanReversionStrategy
from .factory import create_strategy
from .utils import validate_signal

__all__ = [
    'SignalGenerator',
    'StrategyContainer', 
    'MomentumStrategy',
    'MeanReversionStrategy',
    'create_strategy',
    'validate_signal'
]
```

## Testing Documentation

### Test Documentation Format

```python
def test_signal_generation():
    """
    Test signal generation with various inputs.
    
    Tests:
        - Normal market conditions
        - Extreme values
        - Missing data
        - Duck typing compatibility
        
    Validates:
        - Protocol compliance
        - Event generation
        - Error handling
    """
```

### Integration Test Docs

```python
class TestStrategyIntegration:
    """
    Integration tests for strategy module.
    
    Test Flow:
        1. Create isolated container
        2. Initialize strategy with mock data
        3. Process events through full cycle
        4. Validate outputs
        
    Architecture Validation:
        - Container isolation
        - Event flow correctness
        - Protocol compliance
    """
```

## Change Documentation

### Change Log Format

```python
"""
Change Log:
    2024-01-15: Initial implementation (author)
    2024-01-20: Added Protocol + Composition (author)
    2024-01-25: Removed inheritance, pure composition (author)
    
Migration Notes:
    - v1 → v2: Replace inheritance with protocols
    - v2 → v3: Add event-driven communication
"""
```

## README Files

### Module README Format

```markdown
# Module Name

Brief description of module purpose and responsibility.

## Architecture

- **Container**: Which container type
- **Protocols**: Implemented protocols
- **Events**: Published/subscribed events
- **Dependencies**: Core and external

## Quick Start

```python
# Simple usage example
from module import MainClass

instance = MainClass(config)
result = instance.process(data)
```

## Configuration

```yaml
# Example configuration
module:
  setting1: value1
  setting2: value2
```

## API Reference

See docstrings in:
- `protocols.py` - Interface definitions
- `implementations.py` - Concrete classes

## Testing

```bash
pytest tests/module/
```

## See Also

- [Architecture Guide](../../architecture/README.md)
- [Complexity Step X](../../complexity-guide/step-X.md)
```

## Documentation Tools

### Docstring Validation

```python
def validate_docstring(obj):
    """Validate docstring completeness"""
    doc = obj.__doc__
    if not doc:
        return False
        
    required_sections = [
        "Architecture:",
        "Configuration:" if is_configurable(obj) else None,
        "Events:" if uses_events(obj) else None,
        "Example:"
    ]
    
    return all(
        section in doc 
        for section in required_sections 
        if section
    )
```

### Documentation Generation

```bash
# Generate API docs from docstrings
sphinx-apidoc -o docs/api src/

# Generate architecture diagrams from code
pyreverse -o png -p ADMF src/
```

## Standards Checklist

### For Every File

- [ ] Module header with architecture info
- [ ] Proper imports organization
- [ ] Public API defined in `__all__`

### For Every Class

- [ ] Class docstring with architecture section
- [ ] Configuration documented
- [ ] Events documented
- [ ] Example provided

### For Every Function

- [ ] Purpose clearly stated
- [ ] Args and returns documented
- [ ] Duck typing explained if used
- [ ] Events noted if published

### For Every Module

- [ ] README.md exists
- [ ] Architecture references included
- [ ] Configuration examples provided
- [ ] Testing instructions clear

## Summary

Good documentation in ADMF-PC:
- **Explains architecture** and design decisions
- **Shows Protocol + Composition** patterns
- **Documents events** and data flow
- **Includes examples** for everything
- **References related** documentation
- **Integrates logging** requirements
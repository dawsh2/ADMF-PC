# Risk Validators Refactoring Summary

## Changes Made

### 1. Created new `validators.py` file
- Renamed `stateless_validators.py` to `stateless_validators_backup.py` as backup
- Created new `src/risk/validators.py` with pure functions decorated with `@risk_validator`
- Removed all class-based implementations (StatelessMaxPositionValidator, etc.)
- Kept only the pure validation logic as decorated functions
- Removed "stateless" from all function names and docstrings
- Removed all factory functions (create_stateless_*)

### 2. Updated Function Names
The following pure functions are now available:
- `validate_max_position` - Validates order against position size limits
- `calculate_position_size` - Calculates appropriate position size for a signal
- `validate_drawdown` - Validates order against drawdown limits
- `calculate_drawdown_adjusted_size` - Calculates position size considering drawdown
- `validate_composite` - Validates order against all risk limits (combines position and drawdown)
- `calculate_composite_size` - Calculates position size considering all risk factors

### 3. Added Compatibility Helper
- Added `create_validator_wrapper()` function that wraps pure functions to provide the `validate_order` method interface
- This allows components expecting objects with methods to work with pure functions
- Used by RiskServiceAdapter which expects validators with `validate_order` method

### 4. Updated Imports
Updated the following files to use the new validators:
- `src/core/coordinator/topology.py` - Now imports from `validators` and uses wrapper
- `tests/test_unified_architecture.py` - Updated to import pure functions
- `tests/test_unified_simple.py` - Updated to call validator as function
- `test_event_flow_complete.py` - Removed invalid set_risk_validator call
- `test_event_flow_portfolio.py` - Removed invalid set_risk_validator call
- `src/core/coordinator/event_flow_adapter.py` - Removed invalid set_risk_validator call

### 5. Architecture Notes
- In the EVENT_FLOW_ARCHITECTURE, risk validation happens through the RiskServiceAdapter
- Portfolio containers emit ORDER_REQUEST events
- RiskServiceAdapter intercepts these, validates using risk validators, and emits ORDER events
- Portfolio containers do not have direct risk validators - this is handled by the adapter pattern
- The `set_risk_validator` method calls were incorrect and have been removed

## Usage Example

```python
from src.risk.validators import validate_max_position, create_validator_wrapper

# Direct function usage
result = validate_max_position(order, portfolio_state, risk_limits, market_data)

# Wrapped for compatibility with components expecting objects
validator = create_validator_wrapper(validate_max_position)
result = validator.validate_order(order, portfolio_state, risk_limits, market_data)
```

## Benefits
- Cleaner, more functional approach
- Works with the discovery system via `@risk_validator` decorator
- Maintains compatibility with existing architecture through wrapper
- Removes unnecessary class boilerplate
- Makes validators easier to test and compose
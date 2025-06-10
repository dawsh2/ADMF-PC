# ADMF-PC Testing Suite

## Overview

This testing suite is designed to validate the rewritten ADMF-PC architecture progressively, helping identify and fix integration issues systematically.

## Testing Strategy

### 1. Isolation-First Approach

Since the rewrite has import issues, we test individual modules in isolation first:

```
tests/
├── isolated/          # Tests that don't import full module graph
│   ├── test_events_isolated.py
│   ├── test_containers_isolated.py
│   └── test_barriers_isolated.py
├── unit/              # Standard unit tests (when imports work)
│   ├── events/
│   ├── containers/
│   └── coordinator/
├── integration/       # Module integration tests
│   ├── test_events_containers.py
│   ├── test_containers_coordinator.py
│   └── test_full_pipeline.py
├── system/            # End-to-end system tests
│   ├── test_basic_backtest.py
│   └── test_parallel_execution.py
└── utils/             # Test utilities and fixtures
    ├── fixtures.py
    ├── mocks.py
    └── test_data.py
```

### 2. Progressive Integration

1. **Isolated Tests** - Test individual classes without imports
2. **Unit Tests** - Test modules once imports are fixed
3. **Integration Tests** - Test module combinations
4. **System Tests** - Test complete workflows

### 3. Import Issue Detection

Each test level helps identify:
- Missing modules
- Circular imports
- Interface mismatches
- Configuration errors

## Test Categories

### Isolated Tests
- Test individual classes by copying/mocking them
- No imports from main codebase
- Validates core logic works

### Unit Tests  
- Test individual modules once imports work
- Use proper imports
- Mock external dependencies

### Integration Tests
- Test module combinations
- Focus on interfaces and data flow
- Minimal end-to-end logic

### System Tests
- Complete workflow testing
- Real data and configurations
- Performance and reliability testing

## Running Tests

```bash
# Run isolated tests (should always work)
python tests/isolated/run_isolated_tests.py

# Run unit tests (when imports are fixed)
python tests/unit/run_unit_tests.py

# Run integration tests  
python tests/integration/run_integration_tests.py

# Run all tests
python tests/run_all_tests.py
```

## Test Development Guidelines

1. **Start Isolated** - Write isolated tests first to validate logic
2. **Mock Heavy** - Use mocks to isolate dependencies  
3. **Progressive** - Build up from simple to complex
4. **Document Issues** - Record import/integration problems
5. **Fix and Test** - Fix issues and add regression tests
EOF < /dev/null
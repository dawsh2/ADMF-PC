# Validation Framework

This directory contains all validation patterns and requirements for the ADMF-PC system.

## 📋 Contents

1. **[event-bus-isolation.md](event-bus-isolation.md)** - Event bus isolation validation
2. **[synthetic-data-framework.md](synthetic-data-framework.md)** - Deterministic testing with synthetic data
3. **[optimization-reproducibility.md](optimization-reproducibility.md)** - Ensuring optimization results are reproducible
4. **[memory-monitoring.md](memory-monitoring.md)** - Memory usage validation and monitoring
5. **[performance-benchmarks.md](performance-benchmarks.md)** - Performance requirements and benchmarks

## 🎯 Core Validation Principles

### 1. Event Bus Isolation
- Events MUST NOT leak between containers
- Each container has its own isolated event bus
- Validation must be run before ANY development

### 2. Synthetic Data Validation
- Use pre-computed expected results
- Deterministic test scenarios
- Exact result matching

### 3. Optimization Reproducibility
- Test set results must match optimization OOS results exactly
- Configuration preservation
- No training data leakage

### 4. Memory Efficiency
- Monitor memory usage per container
- Implement batch processing for large parameter spaces
- Resource pooling and container recycling

### 5. Performance Requirements
- Event processing latency < 1ms
- Signal processing < 100μs
- Memory scaling must be linear

## 🚀 Quick Start

Before implementing ANY step in the complexity guide:

```python
# Run event bus isolation validation
python -m src.core.events.enhanced_isolation

# Run synthetic data tests
python tests/test_synthetic_validation.py

# Check memory usage
python tools/memory_profiler.py
```

## ✅ Validation Checklist

Every implementation step MUST pass:
- [ ] Event bus isolation tests
- [ ] Synthetic data validation
- [ ] Memory efficiency checks
- [ ] Performance benchmarks
- [ ] Optimization reproducibility

See individual documents for detailed requirements and implementation.
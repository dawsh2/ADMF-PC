# ADMF-PC Project Context

Welcome! This is the **Adaptive Decision Making Framework - Protocol Composition (ADMF-PC)**, a sophisticated zero-code algorithmic trading system built on revolutionary architectural principles.

## 🚀 IMPORTANT: Start with Onboarding

**For new AI assistants in fresh context windows:**

### → First, read the [Onboarding Documentation](docs/onboarding/README.md) ←

The onboarding process is designed to give you comprehensive understanding of the system in under 2 hours. Key documents in order:

1. **[Onboarding Hub](docs/onboarding/README.md)** - START HERE - Central navigation
2. **[Quick Start Guide](docs/onboarding/QUICK_START.md)** - 5-minute system overview  
3. **[Core Concepts](docs/onboarding/CONCEPTS.md)** - Essential architectural principles
4. **[System Architecture V5](docs/SYSTEM_ARCHITECTURE_V5.MD)** - **CANONICAL REFERENCE** (Note: V5, not V4)
5. **[First Task](docs/onboarding/FIRST_TASK.md)** - Hands-on tutorial
6. **[Common Pitfalls](docs/onboarding/COMMON_PITFALLS.md)** - Critical mistakes to avoid

### Key Architectural Principles
This system is built on three foundational concepts:
1. **Protocol + Composition** - No inheritance, pure composition through protocols
2. **Container Architecture** - Complete isolation and modularity
3. **Event-Driven Design** - Everything communicates through events

### Important Notes
- **Zero-Code System**: Everything is YAML configuration, no programming required
- **Protocol-Based**: Components interact through events, not method calls
- **Configuration-Driven**: YAML files fully specify system behavior
- **Production-Ready**: Same configuration works for backtesting and live trading

## 🏗️ Zero-Code Architecture

### Instead of This (Don't Do):
```python
class MyStrategy(BaseStrategy):
    def calculate_signal(self, data):
        # Programming logic...
```

### Do This:
```yaml
strategies:
  - type: momentum
    fast_period: 10
    slow_period: 30
```

**Key Insight**: All logic is pre-implemented and optimized. You configure behavior, not implementation.

### Event Flow Understanding:
```
Data → Indicators → Strategies → Signals → Risk → Orders → Execution
  ↓        ↓           ↓         ↓       ↓       ↓         ↓
[BAR]  [INDICATOR] [STRATEGY] [SIGNAL] [ORDER] [FILL] [POSITION]
```

### Component Composition Examples:
```yaml
# Mix ANY component types seamlessly
strategies:
  - type: momentum           # Built-in strategy
  - type: sklearn_model      # ML model  
  - function: my_custom_func # Your function
  - type: external_library   # Third-party
```

## 📁 Project Structure

```
ADMF-PC/
├── src/                      # Source code (Protocol + Composition architecture)
│   ├── core/                # Core infrastructure (containers, events, protocols)
│   ├── data/                # Data handling and streaming
│   ├── strategy/            # Strategy implementations and optimization
│   ├── risk/                # Risk management and portfolio tracking
│   └── execution/           # Order execution and backtesting
├── docs/                    # Documentation
│   ├── onboarding/          # Start here for learning the system
│   ├── complexity-guide/    # Step-by-step advanced features (18 steps)
│   ├── architecture/        # Core architecture documentation
│   └── SYSTEM_ARCHITECTURE_V5.MD  # Canonical system reference
├── config/                  # Example YAML configurations
├── tests/                   # Three-tier testing framework
└── scripts/                 # Utility scripts
```

## 🎯 Common Tasks

### For Strategy Development
```yaml
# Everything is configuration - no code needed
strategies:
  - type: momentum
    fast_period: 10
    slow_period: 30
```
See: [Strategy Developer Path](docs/onboarding/learning-paths/strategy-developer.md)

### For System Understanding
1. Read [SYSTEM_ARCHITECTURE_V5.MD](docs/SYSTEM_ARCHITECTURE_V5.MD)
2. Study [Event-Driven Architecture](docs/architecture/01-EVENT-DRIVEN-ARCHITECTURE.md)
3. Understand [Cross-Container Communication](docs/architecture/04-CROSS-CONTAINER-COMMUNICATION.md)
4. Explore [Complexity Guide](docs/complexity-guide/README.md)

### For Testing and Validation
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- System tests: `tests/test_integration/`
- See: [Three-Tier Testing Strategy](docs/complexity-guide/testing-framework/three-tier-strategy.md)

## 🚀 Quick Reference

### Essential Commands
```bash
# Run backtest
python main.py config/simple_backtest.yaml

# Run optimization  
python main.py config/optimization_workflow.yaml

# Run with options
python main.py config.yaml --bars 100 --verbose

# Run tests
python -m pytest tests/

# Download sample data
python scripts/download_sample_data.py
```

### Core File Locations
- **Strategies**: `src/strategy/strategies/`
- **Risk Management**: `src/risk/`  
- **Example Configs**: `config/`
- **Architecture Docs**: `docs/SYSTEM_ARCHITECTURE_V5.MD`
- **Onboarding**: `docs/onboarding/README.md`

## 📚 Learning Path

For comprehensive understanding, follow this path:
1. **[5-min Quick Start](docs/onboarding/QUICK_START.md)** - See it work
2. **[15-min Core Concepts](docs/onboarding/CONCEPTS.md)** - Understand the philosophy
3. **[20-min System Architecture](docs/SYSTEM_ARCHITECTURE_V5.MD)** - Technical foundation
4. **[30-min First Task](docs/onboarding/FIRST_TASK.md)** - Build something
5. **[10-min Common Pitfalls](docs/onboarding/COMMON_PITFALLS.md)** - Avoid mistakes
6. **[Complexity Guide](docs/complexity-guide/README.md)** - Master advanced features

## ⚠️ Critical Guidelines - Never Violate These

### Architectural Constraints
- **NEVER write Python code for strategies** - Only YAML configuration
- **NEVER use inheritance** - Only Protocol + Composition  
- **NEVER share state between containers** - Complete isolation required
- **NEVER assume direct component communication** - Everything via Event Router
- **NEVER access other containers directly** - Use cross-container event flow only

### Configuration Requirements  
- **ALWAYS use spaces, never tabs** in YAML files
- **ALWAYS define comprehensive risk management** before returns
- **ALWAYS validate out-of-sample** - coarse grids, avoid overfitting
- **ALWAYS include realistic transaction costs** in backtests

### Best Practices
1. Always use YAML configuration, never modify code
2. Test strategies with walk-forward validation
3. Use three-tier testing (unit, integration, system)
4. Monitor memory usage for large-scale operations
5. **NEVER create duplicate files** - modify canonical files directly (see [Style Guide](docs/standards/STYLE-GUIDE.md))
6. Start simple, add complexity gradually
7. Risk management first, then optimize returns
8. Markets change - make strategies adaptive

### Common Pitfalls
See [Common Pitfalls Guide](docs/onboarding/COMMON_PITFALLS.md) for the complete list of 12 critical mistakes to avoid.

## 🆘 Getting Help

- **Documentation**: Start with [Onboarding Hub](docs/onboarding/README.md)
- **Architecture**: Read [SYSTEM_ARCHITECTURE_V5.MD](docs/SYSTEM_ARCHITECTURE_V5.MD)
- **Examples**: Check `config/` directory
- **Issues**: File at project repository

## 💡 Key Insights

This system represents a paradigm shift in trading system design:
- **Composition > Inheritance**: Mix any components freely
- **Configuration > Code**: Define behavior without programming
- **Isolation > Integration**: Containers prevent interference
- **Events > Methods**: Loose coupling enables flexibility

---

**Remember**: When in doubt, check the [Onboarding Documentation](docs/onboarding/README.md) - it's designed to answer most questions within 2 hours.
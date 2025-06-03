# ADMF-PC Documentation Hub

> **Adaptive Decision Making Framework - Protocol Composition**: The zero-code algorithmic trading system that scales from research to production

## 🚀 Quick Navigation

### New to ADMF-PC?
**Start Here → [Getting Started Guide](01-getting-started/README.md)** (5 minutes)

### Building Something?
**Go to → [User Guide](03-user-guide/README.md)** for task-oriented tutorials

### Need Details?
**Check → [Reference Documentation](04-reference/README.md)** for complete specifications

### Understanding Architecture?
**Read → [Architecture Guide](05-architecture/README.md)** for technical deep-dives

---

## 📚 Documentation Structure

### [01. Getting Started](01-getting-started/README.md)
Start here if you're new. Learn the basics and run your first backtest in under 30 minutes.
- Installation and setup
- Your first backtest
- Understanding workflows
- Common issues

### [02. Core Concepts](02-core-concepts/README.md)
Understand the revolutionary architecture that makes ADMF-PC unique.
- Protocol + Composition (no inheritance)
- Container Architecture (complete isolation)
- Event-Driven Design (universal communication)
- Zero-Code Philosophy (YAML configuration)
- Coordinator Orchestration (the central brain)
- Workflow Composition (building blocks)

### [03. User Guide](03-user-guide/README.md)
Task-oriented guides for common activities.
- Configuring strategies
- Setting up risk management
- Running backtests
- Optimizing parameters
- Building workflows
- Deploying to production

### [04. Reference](04-reference/README.md)
Complete technical specifications.
- Component catalog
- Configuration schemas
- Event specifications
- Coordinator modes
- Container patterns
- Performance benchmarks

### [05. Architecture](05-architecture/README.md)
Deep technical documentation for developers and contributors.
- System design philosophy
- Coordinator architecture
- Container organization patterns
- Event adapter patterns
- Workflow execution engine
- Isolation implementation

### [06. Patterns](06-patterns/README.md)
Common patterns and best practices.
- Container organization patterns
- Workflow composition patterns
- Communication patterns

### [07. Learning Paths](07-learning-paths/README.md)
Role-specific guided learning journeys.
- Strategy Developer Path
- ML Engineer Path
- Workflow Designer Path
- System Administrator Path

### [08. Advanced Topics](08-advanced-topics/README.md)
Master complex features and optimizations.
- Custom workflows
- Container specialization
- Signal replay optimization
- Distributed execution

### [09. Examples](09-examples/README.md)
Complete working examples you can run and modify.
- Simple momentum strategy
- Multi-strategy portfolio
- ML integration
- Complex workflows

### [10. Standards](10-standards/README.md)
Development and contribution guidelines.
- Coding standards
- Documentation standards
- Testing standards
- Workflow design standards

### [11. Diagrams](11-diagrams/README.md)
Visual documentation for better understanding.
- System architecture
- Container hierarchies
- Event flows
- Workflow patterns

---

## 🎯 Common Tasks

| I want to... | Go to... |
|-------------|----------|
| Run my first backtest | [First Backtest Tutorial](01-getting-started/first-backtest.md) |
| Build a trading strategy | [Configuring Strategies](03-user-guide/configuring-strategies.md) |
| Optimize parameters | [Optimization Guide](03-user-guide/optimization.md) |
| Understand the architecture | [Core Concepts](02-core-concepts/README.md) |
| Build complex workflows | [Workflow Patterns](06-patterns/workflow-patterns/README.md) |
| Integrate ML models | [ML Engineer Path](07-learning-paths/ml-engineer.md) |
| Deploy to production | [Live Trading Guide](03-user-guide/live-trading.md) |

---

## 🔑 Key Insights

ADMF-PC represents a paradigm shift in trading system design through:

1. **Zero-Code Operation**: Everything is YAML configuration - no programming required
2. **Infinite Composability**: Mix any components (strategies, ML models, indicators) freely
3. **Complete Isolation**: Containers prevent interference, enabling massive parallelization
4. **Perfect Reproducibility**: Coordinator ensures identical results across runs
5. **Workflow Building Blocks**: Complex operations built from simple, reusable components
6. **Production-Ready**: Same configuration works for backtesting and live trading

---

## 📖 Reading Order

### For Beginners (2 hours)
1. [Getting Started](01-getting-started/README.md) (30 min)
2. [Core Concepts Overview](02-core-concepts/README.md) (30 min)
3. [First Backtest](01-getting-started/first-backtest.md) (30 min)
4. [Basic User Guide](03-user-guide/README.md) (30 min)

### For Developers (4 hours)
1. Complete Beginner track
2. [Architecture Guide](05-architecture/README.md) (1 hour)
3. [Container Patterns](06-patterns/container-organization/README.md) (30 min)
4. [Workflow Patterns](06-patterns/workflow-patterns/README.md) (30 min)

### For Researchers (6 hours)
1. Complete Developer track
2. [Advanced Topics](08-advanced-topics/README.md) (1 hour)
3. [Examples](09-examples/README.md) (1 hour)

---

## 🔍 Quick Reference

### Essential Commands
```bash
# Run a backtest
python main.py config/simple_backtest.yaml

# Run optimization
python main.py config/optimization_workflow.yaml

# Run with options
python main.py config.yaml --bars 100 --verbose
```

### Key File Locations
- Example configs: `config/`
- Source code: `src/`
- Tests: `tests/`
- Legacy docs: `docs/legacy/`

---

## 📝 Documentation Status

This documentation is actively maintained and represents the current state of ADMF-PC. For historical documentation, see the [legacy](../legacy/) directory.

**Last Updated**: January 2025  
**Version**: 5.0  
**Canonical Reference**: [System Architecture V5](../SYSTEM_ARCHITECTURE_v5.MD)
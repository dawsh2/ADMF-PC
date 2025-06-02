# ADMF-PC Onboarding Hub

Welcome to ADMF-PC - the Adaptive Decision Making Framework for Protocol Composition. This hub will guide you through understanding and using this powerful algorithmic trading system.

## 🎯 Quick Links

- **[5-Minute Quick Start](QUICK_START.md)** - Get your first backtest running
- **[Comprehensive Onboarding Guide](ONBOARDING.md)** - Full system introduction
- **[Key Concepts](CONCEPTS.md)** - Understand the core philosophy
- **[Your First Task](FIRST_TASK.md)** - Hands-on strategy creation

## 🚀 Choose Your Path

### I want to...

#### **Run a backtest right now** (5 minutes)
→ Start with [QUICK_START.md](QUICK_START.md)

#### **Understand the system thoroughly** (2 hours)
→ Follow [ONBOARDING.md](ONBOARDING.md)

#### **Create trading strategies** 
→ Check [learning-paths/strategy-developer.md](learning-paths/strategy-developer.md)

#### **Integrate with the system**
→ See [learning-paths/system-integrator.md](learning-paths/system-integrator.md)

#### **Run research experiments**
→ Read [learning-paths/researcher.md](learning-paths/researcher.md)

#### **Deploy ML models**
→ Follow [learning-paths/ml-practitioner.md](learning-paths/ml-practitioner.md)

## 📚 Essential Reading Order

1. **[QUICK_START.md](QUICK_START.md)** (5 min) - See it work
2. **[CONCEPTS.md](CONCEPTS.md)** (15 min) - Understand the philosophy
3. **[SYSTEM_ARCHITECTURE_V5.MD](../SYSTEM_ARCHITECTURE_V5.MD)** (20 min) - Technical foundation
4. **[FIRST_TASK.md](FIRST_TASK.md)** (30 min) - Build something
5. **[COMMON_PITFALLS.md](COMMON_PITFALLS.md)** (10 min) - Avoid mistakes

## 🏗️ System Overview

ADMF-PC is a **zero-code** algorithmic trading system that uses YAML configuration instead of programming. It's built on three core principles:

1. **Protocol + Composition**: No inheritance, pure composition
2. **Container Architecture**: Complete isolation and modularity
3. **Event-Driven Design**: Reactive, scalable processing

## 🎓 What Makes ADMF-PC Different?

### Traditional Trading Systems
```python
# Inheritance hell
class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()  # Framework coupling
        
# Can't mix libraries
# Hard to test
# Rigid structure
```

### ADMF-PC Approach
```yaml
# Just configuration
strategies:
  - type: momentum
    fast_period: 10
    slow_period: 30
    
# Mix anything
# Easy to test
# Completely flexible
```

## 📋 Onboarding Checklist

### First Hour
- [ ] Run your first backtest (5 min)
- [ ] Understand core concepts (15 min)
- [ ] Read technical architecture (20 min)
- [ ] Modify a strategy (10 min)
- [ ] Avoid common mistakes (10 min)

### First Day
- [ ] Explore available components
- [ ] Run parameter optimization
- [ ] Understand event flow
- [ ] Create multi-strategy portfolio

### First Week
- [ ] Master configuration options
- [ ] Implement custom indicators
- [ ] Run walk-forward analysis
- [ ] Deploy to production simulation

## 🗺️ Complete Documentation Map

### Getting Started
- [QUICK_START.md](QUICK_START.md) - Hello World in 5 minutes
- [ONBOARDING.md](ONBOARDING.md) - Complete introduction
- [CONCEPTS.md](CONCEPTS.md) - Core philosophy
- [FIRST_TASK.md](FIRST_TASK.md) - Guided tutorial

### Reference
- [FAQ.md](FAQ.md) - Common questions
- [GLOSSARY.md](GLOSSARY.md) - Terms and definitions
- [COMMON_PITFALLS.md](COMMON_PITFALLS.md) - What to avoid
- [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md) - Why things work this way

### Learning Paths
- [Strategy Developer](learning-paths/strategy-developer.md)
- [System Integrator](learning-paths/system-integrator.md)
- [Researcher](learning-paths/researcher.md)
- [ML Practitioner](learning-paths/ml-practitioner.md)

### Deep Dives
- [System Architecture](../SYSTEM_ARCHITECTURE_V5.MD) - Complete technical overview
- [Complexity Guide](../complexity-guide/README.md) - Step-by-step advanced features
- [Component Catalog](../COMPONENT_CATALOG.md) - All available components
- [Style Guide](../standards/STYLE-GUIDE.md) - Coding standards and best practices

## 🆘 Getting Help

- **Documentation Issues**: Check [RESOURCES.md](RESOURCES.md)
- **Bug Reports**: File at [GitHub Issues](https://github.com/anthropics/claude-code/issues)
- **Community**: Join our Discord/Slack (links in RESOURCES.md)

## ⚡ Ready to Start?

**Recommended**: Begin with [QUICK_START.md](QUICK_START.md) to see the system in action, then return here to explore deeper topics.

---

*Remember: ADMF-PC is designed to make complex trading strategies simple. You don't need to be a programmer - just understand markets and configure your ideas in YAML.*
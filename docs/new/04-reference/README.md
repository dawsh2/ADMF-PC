# Reference Documentation

This section provides complete technical specifications for ADMF-PC components, configurations, and APIs. Use this as your authoritative reference for exact parameter definitions, event schemas, and component interfaces.

## 📚 Reference Sections

### [Component Catalog](component-catalog.md)
Complete catalog of all available components with detailed specifications:
- **Strategy Components**: All built-in trading strategies
- **Risk Management Components**: Position sizing and risk controls
- **Data Components**: Data sources and processors
- **Execution Components**: Order execution and brokers
- **Analysis Components**: Performance analysis and statistics
- **ML Components**: Machine learning model integrations

### [Configuration Schema](configuration-schema.md)
Comprehensive YAML configuration reference:
- **Workflow Configuration**: All workflow types and options
- **Data Configuration**: Data sources and preprocessing
- **Strategy Configuration**: Strategy parameters and settings
- **Risk Management Configuration**: Risk controls and limits
- **Execution Configuration**: Order execution settings
- **Infrastructure Configuration**: Resource and system settings

### [Event Reference](event-reference.md)
Complete event system specifications:
- **Event Types**: All event classes and schemas
- **Event Flow Patterns**: How events route through the system
- **Event Adapters**: Communication pattern configurations
- **Event Correlation**: Tracing and correlation mechanisms
- **Custom Events**: Creating custom event types

### [Coordinator Modes](coordinator-modes.md)
Detailed coordinator configuration reference:
- **Execution Modes**: TRADITIONAL, AUTO, COMPOSABLE, HYBRID
- **Resource Management**: Memory, CPU, and container allocation
- **Workflow Orchestration**: Phase management and data flow
- **Fault Tolerance**: Error handling and recovery options
- **Monitoring**: Performance tracking and alerting

### [Container Patterns](container-patterns.md)
Container organization and communication patterns:
- **Organization Patterns**: Strategy-First, Classifier-First, Risk-First, Portfolio-First
- **Communication Adapters**: Pipeline, Broadcast, Hierarchical, Selective
- **Resource Allocation**: Memory and CPU management per container
- **Lifecycle Management**: Container creation, initialization, and disposal

### [Event Adapters](event-adapters.md)
Event communication adapter specifications:
- **Adapter Types**: Configuration and behavior of each adapter type
- **Routing Rules**: How to configure event routing
- **Performance Characteristics**: Latency and throughput specifications
- **Error Handling**: Failure modes and recovery mechanisms

### [Workflow Blocks](workflow-blocks.md)
Building block reference for workflow composition:
- **Core Blocks**: Backtest, Optimization, Analysis, Validation
- **Block Parameters**: All configuration options for each block
- **Data Flow**: Input and output specifications
- **Composition Rules**: How blocks can be combined

### [Performance Benchmarks](performance-benchmarks.md)
Performance specifications and benchmarks:
- **Execution Speed**: Benchmarks for different operation types
- **Memory Usage**: Typical memory consumption patterns
- **Scaling Characteristics**: Performance vs container count
- **Resource Requirements**: Minimum and recommended specifications

## 🎯 Quick Reference

### Most Common Lookups

| I need to... | Go to... |
|-------------|----------|
| Find strategy parameters | [Component Catalog - Strategies](component-catalog.md#strategies) |
| Configure YAML correctly | [Configuration Schema](configuration-schema.md) |
| Understand event flow | [Event Reference](event-reference.md) |
| Set up container communication | [Event Adapters](event-adapters.md) |
| Choose coordinator mode | [Coordinator Modes](coordinator-modes.md) |
| Organize containers | [Container Patterns](container-patterns.md) |
| Build workflows | [Workflow Blocks](workflow-blocks.md) |
| Check system requirements | [Performance Benchmarks](performance-benchmarks.md) |

### Configuration Templates

**Basic Strategy Configuration**:
```yaml
strategies:
  - type: "momentum"
    params:
      fast_period: 10
      slow_period: 20
      signal_threshold: 0.01
```

**Basic Risk Management**:
```yaml
risk_management:
  type: "fixed"
  params:
    position_size_pct: 0.02
    max_exposure_pct: 0.10
    stop_loss_pct: 0.02
```

**Basic Workflow**:
```yaml
workflow:
  type: "backtest"
  
data:
  symbols: ["SPY"]
  start_date: "2023-01-01"
  end_date: "2023-12-31"
```

## 📖 How to Use This Reference

### For Configuration
1. **Start with [Configuration Schema](configuration-schema.md)** for overall YAML structure
2. **Check [Component Catalog](component-catalog.md)** for specific component parameters
3. **Validate with examples** in each section

### For System Design
1. **Review [Container Patterns](container-patterns.md)** for architecture decisions
2. **Check [Event Adapters](event-adapters.md)** for communication setup
3. **Consult [Coordinator Modes](coordinator-modes.md)** for execution configuration

### For Development
1. **Study [Event Reference](event-reference.md)** for event system integration
2. **Review [Workflow Blocks](workflow-blocks.md)** for custom workflow building
3. **Check [Performance Benchmarks](performance-benchmarks.md)** for optimization

### For Troubleshooting
1. **Verify configuration** against [Configuration Schema](configuration-schema.md)
2. **Check component parameters** in [Component Catalog](component-catalog.md)
3. **Review event flow** in [Event Reference](event-reference.md)

## 🔍 Search Tips

### Finding Information Quickly

**Parameter Lookup**:
- Strategy parameters → [Component Catalog](component-catalog.md)
- Configuration syntax → [Configuration Schema](configuration-schema.md)
- Event specifications → [Event Reference](event-reference.md)

**Architecture Questions**:
- Container setup → [Container Patterns](container-patterns.md)
- Communication patterns → [Event Adapters](event-adapters.md)
- Workflow building → [Workflow Blocks](workflow-blocks.md)

**Performance Questions**:
- Resource requirements → [Performance Benchmarks](performance-benchmarks.md)
- Scaling behavior → [Container Patterns](container-patterns.md)
- Execution modes → [Coordinator Modes](coordinator-modes.md)

### Parameter Validation

When configuring ADMF-PC:

1. **Check parameter types** in component specifications
2. **Verify value ranges** in configuration schema
3. **Validate event schemas** for custom components
4. **Review resource limits** for your use case

## 📋 Reference Standards

### Parameter Documentation Format

Each parameter is documented with:
- **Type**: Data type (int, float, str, bool, list, dict)
- **Range**: Valid value ranges or options
- **Default**: Default value if not specified
- **Description**: What the parameter controls
- **Examples**: Common values and use cases

### Configuration Documentation Format

Each configuration section includes:
- **Required Fields**: Must be specified
- **Optional Fields**: Can be omitted (defaults used)
- **Valid Combinations**: Which options work together
- **Examples**: Complete working configurations
- **Common Errors**: Typical mistakes and solutions

### Event Documentation Format

Each event type includes:
- **Schema**: Complete field specification
- **Triggers**: What causes the event to be emitted
- **Consumers**: What components typically handle the event
- **Flow Patterns**: How the event moves through the system
- **Examples**: Sample event instances

## 🔧 API Reference

### Python API

For advanced users who need to extend ADMF-PC:

```python
# Core imports
from src.core.components import ComponentRegistry
from src.core.events import EventBus, TradingSignal
from src.core.containers import Container
from src.coordinator import Coordinator

# Creating custom components
from src.core.protocols import SignalGenerator, RiskManager
```

### Configuration API

For programmatic configuration:

```python
from src.core.config import ConfigBuilder

# Build configuration programmatically
config = ConfigBuilder() \
    .workflow("backtest") \
    .data(symbols=["SPY"], start_date="2023-01-01") \
    .strategy("momentum", fast_period=10, slow_period=20) \
    .build()
```

## 📝 Contributing to Reference

### Reference Documentation Standards

When contributing to the reference:

1. **Accuracy**: All information must be current and tested
2. **Completeness**: Include all parameters and options
3. **Clarity**: Use clear, precise language
4. **Examples**: Provide working examples for all features
5. **Cross-References**: Link related concepts

### Update Process

1. **Test Changes**: Verify all examples work
2. **Update Cross-References**: Ensure links remain valid
3. **Version Documentation**: Note version compatibility
4. **Review Impact**: Check dependent documentation

---

Start with [Component Catalog](component-catalog.md) for a complete overview of available components →
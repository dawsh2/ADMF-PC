# Architecture Documentation

Comprehensive documentation of the ADMF-PC system architecture, design principles, and implementation patterns.

## 📚 Core Architecture Documents

1. **[01-EVENT-DRIVEN-ARCHITECTURE.md](01-EVENT-DRIVEN-ARCHITECTURE.md)**
   - Event bus design and isolation
   - Event flow patterns
   - Testing event-driven systems
   - Performance considerations

2. **[02-CONTAINER-HIERARCHY.md](02-CONTAINER-HIERARCHY.md)** *(To be created)*
   - Container design philosophy
   - Nesting patterns and lifecycle
   - Resource management
   - Inter-container communication

3. **[03-PROTOCOL-COMPOSITION.md](03-PROTOCOL-COMPOSITION.md)** *(To be created)*
   - Zero inheritance manifesto
   - Duck typing benefits
   - Composition over inheritance
   - Protocol definitions

4. **[04-THREE-PATTERN-BACKTEST.md](04-THREE-PATTERN-BACKTEST.md)** *(To be created)*
   - Full backtest pattern
   - Signal replay pattern
   - Signal generation pattern
   - Pattern selection criteria

5. **[05-MODULE-STRUCTURE.md](05-MODULE-STRUCTURE.md)** *(To be created)*
   - Module organization
   - Dependency management
   - Interface boundaries
   - Module communication patterns

## 🎯 Design Principles

### 1. Protocol + Composition
- No inheritance hierarchies
- Behavior through composition
- Duck typing for flexibility
- Explicit protocols/interfaces

### 2. Event-Driven Communication
- Loose coupling via events
- Isolated event buses
- Asynchronous processing
- Clear event contracts

### 3. Container-Based Architecture
- Encapsulated components
- Lifecycle management
- Resource isolation
- Hierarchical organization

### 4. Testability First
- Dependency injection
- Mock-friendly interfaces
- Isolated testing
- Deterministic behavior

## 🔗 Architecture References

### Implementation Examples
- [Core Container Implementation](../core/containers/)
- [Event Bus Implementation](../core/events/)
- [Strategy Module](../strategy/)
- [Risk Module](../risk/)
- [Execution Module](../execution/)

### Related Documentation
- [BACKTEST_README.md](../BACKTEST_README.md) - Backtest architecture overview
- [MULTIPHASE_OPTIMIZATION.md](../MULTIPHASE_OPTIMIZATION.md) - Optimization workflow
- [PC/BENEFITS.md](../PC/BENEFITS.md) - Protocol + Composition benefits
- [PC/R1.md](../PC/R1.md) - Original architecture proposal

## 🛠️ Implementation Guidelines

### When Creating New Components

1. **Start with Protocols**
   ```python
   class DataProvider(Protocol):
       def get_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
           ...
   ```

2. **Use Composition**
   ```python
   class BacktestEngine:
       def __init__(self, data_provider: DataProvider, 
                    strategy: Strategy, risk_manager: RiskManager):
           self.data_provider = data_provider
           self.strategy = strategy
           self.risk_manager = risk_manager
   ```

3. **Emit Events**
   ```python
   def process_signal(self, signal: Signal) -> None:
       order = self.create_order(signal)
       self.event_bus.publish("ORDER_CREATED", order)
   ```

4. **Isolate Containers**
   ```python
   def create_container(self, container_id: str) -> Container:
       event_bus = self.isolation_manager.create_isolated_bus(container_id)
       return Container(container_id, event_bus)
   ```

## 📊 Architecture Metrics

### Current Statistics
- Zero inheritance chains
- 100% protocol-based interfaces
- Average container size: < 500 LOC
- Event types: ~20 distinct types
- Container types: 6 major types

### Performance Targets
- Event latency: < 1ms
- Container creation: < 10ms
- Memory per container: < 50MB
- Event throughput: > 10k/sec

## 🚀 Evolution Guidelines

### Adding New Features
1. Define protocols first
2. Create isolated containers
3. Use existing event types when possible
4. Follow composition patterns
5. Add comprehensive tests

### Refactoring Existing Code
1. Identify inheritance to remove
2. Extract protocols from classes
3. Convert to composition
4. Add event emissions
5. Create container boundaries

## ❓ Frequently Asked Questions

### Why no inheritance?
Inheritance creates tight coupling and makes testing difficult. Composition provides more flexibility and better testability.

### Why isolated event buses?
Isolation prevents unintended interactions between components and makes the system more predictable and testable.

### How do containers communicate?
Only through events. Parent containers can subscribe to child events, but children never know about parents.

### What about performance?
The event-driven architecture has minimal overhead. The benefits in maintainability and testability far outweigh any performance costs.

## 📝 Next Steps

1. Review the event-driven architecture document
2. Study the container hierarchy patterns
3. Understand protocol + composition benefits
4. Explore the implementation examples
5. Start building with these patterns

---

*"The best architecture is one that makes change easy, testing simple, and reasoning straightforward."*
# README for LLM Interactions

**This file provides specific guidance for LLMs working on the ADMF-PC project.**

## MANDATORY: Read Before Making Changes

### Core Principle: Documentation-First Development

**You MUST read relevant documentation before proposing any code changes.**

**Never create files with adjectives like `enhanced_`, `improved_`, `advanced_`, `better_`, `optimized_`.**

## Required Reading Checklist

Before making ANY changes, confirm you have read:

- [ ] **This file** - LLM interaction guidelines
- [ ] **docs/architecture/STANDARD_PATTERN_ARCHITECTURE.md** - MANDATORY architecture standard
- [ ] **STYLE.md** - Naming conventions and canonical implementations
- [ ] **README.md** - Project overview and architecture  
- [ ] **src/docs/arch-101.md** - Core architecture patterns


## LLM Response Protocol

**ALWAYS structure your responses using this template:**

```markdown
## Documentation Review ✅
I have read and understood:
- [file1]: [key concepts learned]
- [file2]: [patterns identified]  
- [file3]: [constraints noted]

## Canonical Implementation Analysis 🎯
The canonical implementation for [functionality] is: **[specific_file.py]**

Evidence:
- [reason 1 - e.g., "Most recent, comprehensive implementation"]
- [reason 2 - e.g., "Referenced in documentation as authoritative"]
- [reason 3 - e.g., "Other files appear to be test/experimental versions"]

## Proposed Approach 🛠️
Instead of creating [adjective]_[component].py, I will:

1. **Enhance canonical file**: [specific_file.py]
2. **Method**: [composition/configuration/feature flags]
3. **Compatibility**: [how existing code remains unchanged]

## Implementation Plan 📋
1. [specific step 1]
2. [specific step 2]  
3. [specific step 3]

## Uncertainties ❓
I need clarification on:
- [specific question 1]
- [specific question 2]

[Only proceed with implementation AFTER getting clarification]
```

## Canonical File Identification

**If you're unsure which file is canonical, look for:**

1. **File without adjective prefixes** (container.py vs enhanced_container.py)
2. **Referenced in documentation** as "THE implementation"
3. **More recent timestamps** and comprehensive functionality
4. **Imported by other modules** more frequently
5. **Matches naming patterns** described in STYLE_GUIDE.md

**When in doubt, ASK:** "Which file should I treat as canonical for [functionality]?"

## Prohibited Actions

**NEVER do these without explicit permission:**

❌ Create files named: `enhanced_*`, `improved_*`, `advanced_*`, `better_*`, `optimized_*`, `superior_*`, `premium_*`

❌ Duplicate existing functionality in new files

❌ Assume what needs to be "enhanced" without reading existing code

❌ Make changes without understanding existing architecture patterns

❌ Create inheritance hierarchies when composition would work

❌ Add features through new classes when configuration would work

❌ **Mix container creation with communication setup** (violates standard architecture)

❌ **Define workflow patterns in multiple places** (must be single source of truth)

❌ **Create communication config in container patterns** (wrong responsibility)

## Required Actions

**ALWAYS do these:**

✅ Read documentation first

✅ Identify canonical implementations explicitly

✅ Ask for clarification when uncertain

✅ Use composition and configuration to add features

✅ Enhance existing files rather than creating new ones

✅ Update documentation when making changes

✅ Follow existing naming patterns

✅ **Follow pattern-based architecture standard** (containers → container factory, communication → communication factory, orchestration → workflow manager)

✅ **Define each workflow pattern exactly once** in WorkflowManager

✅ **Use delegation** to appropriate factories rather than mixing responsibilities

## Architecture Patterns to Follow

### 1. Pattern-Based Architecture (MANDATORY)
```python
# ✅ Good: Single pattern definition in WorkflowManager
self._workflow_patterns = {
    'simple_backtest': {
        'container_pattern': 'simple_backtest',    # → Container Factory
        'communication_config': [...]              # → Communication Factory
    }
}

# ❌ Bad: Mixing responsibilities
class ContainerFactoryWithCommunication:  # NO!
    pass
```

### 2. Factory Separation
```python
# ✅ Good: Each factory has single responsibility
containers = container_factory.compose_pattern('simple_backtest')
adapters = comm_factory.create_adapters_from_config(comm_config)

# ❌ Bad: Mixed responsibilities
everything = unified_factory.create_everything()  # NO!
```

### 3. Protocol + Composition
```python
# ✅ Good: Use protocols and composition
class SignalProcessor:
    def __init__(self, enhancers: List[SignalEnhancer] = None):
        self.enhancers = enhancers or []

# ❌ Bad: Create enhanced versions
class EnhancedSignalProcessor(SignalProcessor):
    pass
```

### 4. Configuration-Driven Features
```python
# ✅ Good: Feature flags in config
class Container:
    def __init__(self, config: ContainerConfig):
        if 'subcontainers' in config.features:
            self.subcontainer_manager = SubcontainerManager()

# ❌ Bad: New files for features  
class EnhancedContainer:
    pass
```

### 3. Single Responsibility Files
```python
# ✅ Good: One authoritative implementation
# momentum_strategy.py - THE momentum strategy implementation

# ❌ Bad: Multiple versions
# momentum_strategy.py
# enhanced_momentum_strategy.py  
# improved_momentum_strategy.py
```

## Quick Decision Tree

```
Need to add functionality?
│
├─ Is this a workflow pattern?
│  ├─ YES → Define ONCE in WorkflowManager (delegates to factories)
│  └─ NO → Continue below
│
├─ Does canonical implementation exist?
│  ├─ YES → Enhance the canonical file
│  └─ NO → Create with role-based name (not adjective)
│
├─ Can this be configuration-driven?
│  ├─ YES → Add feature flags to canonical file
│  └─ NO → Use composition in canonical file
│
└─ Uncertain which file is canonical?
   └─ STOP and ASK for clarification
```

**For workflow-related functionality:**
- Container creation → `core/containers/factory.py`
- Communication setup → `core/communication/factory.py`
- Pattern orchestration → `coordinator/workflows/workflow_manager.py`

## Common Scenarios & Responses

### Scenario 1: "Add ML capabilities to signal processing"

✅ **Correct Response:**
```
## Documentation Review ✅
I have read:
- src/strategy/README.md: Signal processing architecture  
- src/strategy/signal_processor.py: Current implementation
- STYLE_GUIDE.md: Composition patterns

## Canonical Implementation Analysis 🎯
The canonical implementation is: **signal_processor.py**
- Only signal processing file without adjective prefix
- Referenced in documentation as main implementation
- Imported by other strategy components

## Proposed Approach 🛠️
I will enhance signal_processor.py using composition:
- Add MLEnhancer as a composable component
- Use configuration to enable ML features
- Maintain backward compatibility

Would you like me to proceed with this approach?
```

❌ **Incorrect Response:**
```
I'll create an ml_enhanced_signal_processor.py with machine learning capabilities...
```

### Scenario 2: "Improve container performance"

✅ **Correct Response:**
```
## Documentation Review ✅
I have read:
- src/core/containers/README.md: Container architecture
- src/core/containers/container.py: Current implementation  
- src/core/containers/protocols.py: Container interfaces

## Canonical Implementation Analysis 🎯
The canonical implementation is: **container.py**
- Matches protocol interfaces exactly
- Referenced as "canonical container implementation" in docs
- No adjective prefix

## Uncertainties ❓
I see there's also enhanced_container.py. Should I:
1. Merge enhanced_container.py capabilities into container.py?
2. Treat enhanced_container.py as the canonical version?
3. Ignore enhanced_container.py as deprecated?

Please clarify which approach aligns with your architecture goals.
```

## Testing Your Understanding

**Before implementing, ask yourself:**

1. Have I read the relevant documentation?
2. Can I clearly identify the canonical file?
3. Am I enhancing existing code rather than duplicating?
4. Will my changes follow existing patterns?
5. Have I asked for clarification on uncertainties?

**If you answer "no" to any question, STOP and seek clarification.**

## Documentation Update Requirements

**When making changes, you must also:**

1. **Update README.md** if architecture changes
2. **Update module README.md** if module behavior changes  
3. **Update inline docstrings** for modified functions
4. **Add configuration examples** for new features
5. **Update STYLE_GUIDE.md** if new patterns emerge

## Emergency Protocol

**If you find yourself about to create a file with an adjective name:**

1. **STOP immediately**
2. **Re-read this document**  
3. **Ask:** "How can I achieve this by enhancing the canonical implementation?"
4. **Propose composition/configuration approach instead**

## Success Criteria

**Your response is ready when:**

- [ ] You've explicitly stated which documentation you read
- [ ] You've identified the canonical file with evidence
- [ ] You've proposed enhancing existing files, not creating new ones
- [ ] You've asked for clarification on any uncertainties
- [ ] Your approach uses composition/configuration patterns
- [ ] You've planned documentation updates

---

**Remember: When in doubt, ASK. Never guess at canonical implementations.**

**MANDATORY: All workflow patterns must follow the standard architecture defined in `docs/architecture/STANDARD_PATTERN_ARCHITECTURE.md`**

# ADMF-PC Style Guide: Preventing "Enhanced" Bloat and Maintaining Canonical Code

## The Problem: "Enhanced" Death Spiral

**Anti-Pattern**: Creating files like `enhanced_container.py`, `advanced_strategy.py`, `improved_xyz.py` leads to:
- Multiple implementations of the same concept
- Unclear which version is canonical
- Code duplication and inconsistency  
- LLMs creating more bloat when unclear what's authoritative

**Solution**: This style guide establishes **ONE canonical implementation per concept** with clear naming patterns.

---

## Core Principle: Single Source of Truth

### ✅ Canonical Implementation Pattern

```
src/core/containers/
├── container.py          # THE container implementation
├── protocols.py         # THE container protocols  
├── factory.py           # THE container factory
└── lifecycle.py         # THE container lifecycle manager
```

### ❌ Anti-Pattern to Avoid

```
src/core/containers/
├── container.py
├── enhanced_container.py     # ❌ Delete this
├── improved_container.py     # ❌ Don't create this
├── advanced_container.py     # ❌ Avoid this
├── container_v2.py          # ❌ No versions
└── better_container.py      # ❌ No adjectives
```

---

## Naming Conventions

### Files and Classes

**Use descriptive, specific names based on role/responsibility:**

✅ **Good Names** (Role-Based):
```python
# Container types by role
BacktestContainer
SignalReplayContainer  
AnalysisContainer
DataContainer

# Strategy types by approach
MomentumStrategy
MeanReversionStrategy
BreakoutStrategy

# Risk types by method
PortfolioRiskManager
PositionSizer
DrawdownController
```

❌ **Bad Names** (Adjective-Based):
```python
# Avoid adjectives that don't specify role
EnhancedContainer      # Enhanced how?
ImprovedStrategy       # Improved from what?
AdvancedRiskManager    # Advanced compared to what?
BetterExecutor         # Better than what?
OptimizedDataHandler   # Optimized how?
```

### Component Organization

**Organize by capability, not quality level:**

✅ **Good Organization**:
```
src/strategy/
├── momentum/
│   ├── simple_ma.py         # Simple moving average
│   ├── dual_ma.py          # Dual moving average  
│   └── adaptive_ma.py      # Adaptive moving average
├── mean_reversion/
│   ├── rsi_strategy.py     # RSI-based
│   ├── bollinger_strategy.py # Bollinger bands
│   └── pairs_trading.py   # Pairs trading
└── breakout/
    ├── donchian_strategy.py # Donchian channels
    └── volume_breakout.py  # Volume-based breakout
```

❌ **Bad Organization**:
```
src/strategy/
├── basic_strategies.py      # ❌ Quality levels
├── intermediate_strategies.py # ❌ Skill levels  
├── advanced_strategies.py   # ❌ Difficulty levels
├── enhanced_strategies.py   # ❌ Adjectives
└── optimized_strategies.py  # ❌ Marketing speak
```

---

## Evolution Patterns

### When Components Need Enhancement

**Use composition and versioning through configuration, not new files:**

✅ **Good Enhancement Pattern**:
```python
# momentum_strategy.py - THE momentum strategy implementation
class MomentumStrategy:
    def __init__(self, config: MomentumConfig):
        self.fast_period = config.fast_period
        self.slow_period = config.slow_period
        self.signal_threshold = config.signal_threshold
        self.use_volume_filter = config.use_volume_filter  # New feature
        self.adaptive_periods = config.adaptive_periods   # New feature
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        # Single implementation that handles all cases
        signal = self._basic_momentum_signal(data)
        
        if self.use_volume_filter:
            signal = self._apply_volume_filter(signal, data)
            
        if self.adaptive_periods:
            signal = self._apply_adaptive_logic(signal, data)
            
        return signal
```

✅ **Configuration-Driven Variants**:
```yaml
# Simple momentum
momentum_strategy:
  type: momentum
  fast_period: 10
  slow_period: 20

# Enhanced momentum (same code, different config)
enhanced_momentum:
  type: momentum
  fast_period: 10
  slow_period: 20
  use_volume_filter: true
  adaptive_periods: true
  signal_threshold: 0.02
```

❌ **Bad Enhancement Pattern**:
```python
# momentum_strategy.py
class MomentumStrategy:
    # Basic implementation

# enhanced_momentum_strategy.py  ❌ DON'T DO THIS
class EnhancedMomentumStrategy:
    # Copy-paste + modifications

# advanced_momentum_strategy.py  ❌ DON'T DO THIS  
class AdvancedMomentumStrategy:
    # More copy-paste + modifications
```

---

## Protocol + Composition Guidelines

### Extending Capabilities

**Use composition and protocols, not inheritance or "enhanced" versions:**

✅ **Good Extension Pattern**:
```python
# signal_processor.py - THE signal processing implementation
class SignalProcessor:
    def __init__(self, filters: List[SignalFilter] = None):
        self.filters = filters or []
    
    def process_signal(self, signal: Signal) -> Signal:
        processed = signal
        for filter in self.filters:
            processed = filter.apply(processed)
        return processed

# volume_filter.py - composable filter
class VolumeFilter:
    def apply(self, signal: Signal) -> Signal:
        # Volume filtering logic
        pass

# ml_filter.py - composable filter  
class MLFilter:
    def apply(self, signal: Signal) -> Signal:
        # ML filtering logic
        pass

# Usage: Compose capabilities as needed
processor = SignalProcessor([
    VolumeFilter(min_volume=1000000),
    MLFilter(model_path="signal_filter.pkl")
])
```

❌ **Bad Extension Pattern**:
```python
# signal_processor.py
class SignalProcessor:
    # Basic implementation

# enhanced_signal_processor.py  ❌ DON'T DO THIS
class EnhancedSignalProcessor(SignalProcessor):
    # Inheritance + modifications

# ml_enhanced_signal_processor.py  ❌ DON'T DO THIS
class MLEnhancedSignalProcessor(EnhancedSignalProcessor):
    # More inheritance
```

---

## File Organization Rules

### Directory Structure

**Organize by domain/responsibility, not quality:**

✅ **Good Structure**:
```
src/
├── core/
│   ├── containers/
│   │   ├── container.py           # THE container
│   │   ├── protocols.py          # THE protocols
│   │   └── factory.py            # THE factory
│   ├── events/
│   │   ├── event.py              # THE event system
│   │   └── event_bus.py          # THE event bus
│   └── coordinator/
│       ├── coordinator.py        # THE coordinator
│       └── workflow_manager.py   # THE workflow manager
├── strategy/
│   ├── strategies/               # Strategy implementations
│   ├── components/              # Strategy building blocks
│   └── optimization/            # Strategy optimization
├── risk/
│   ├── position_sizing.py       # Position sizing
│   ├── portfolio_state.py       # Portfolio tracking
│   └── risk_limits.py           # Risk constraints
└── execution/
    ├── backtest_engine.py       # Backtest execution
    ├── market_simulation.py     # Market simulation
    └── order_manager.py         # Order management
```

### File Naming Rules

1. **Use descriptive nouns, not adjectives**
   - ✅ `momentum_strategy.py`
   - ❌ `enhanced_strategy.py`

2. **Use specific roles, not quality levels**
   - ✅ `portfolio_risk_manager.py`
   - ❌ `advanced_risk_manager.py`

3. **Use implementation method, not marketing terms**
   - ✅ `dual_moving_average.py`
   - ❌ `optimized_moving_average.py`

4. **Use protocol names consistently**
   - ✅ `signal_generator.py` (implements SignalGenerator protocol)
   - ❌ `signal_creator.py` or `signal_producer.py`

---

## Configuration-Driven Development

### The Golden Rule: Add Features Through Config, Not New Files

When LLMs suggest adding capabilities:

✅ **Do This** - Enhance through configuration:
```yaml
# Original configuration
strategy:
  type: momentum
  fast_period: 10
  slow_period: 20

# Enhanced configuration (same code!)
strategy:
  type: momentum
  fast_period: 10
  slow_period: 20
  features:
    - volume_filter
    - ml_enhancement
    - adaptive_periods
  volume_filter:
    min_volume: 1000000
  ml_enhancement:
    model_path: "enhancement.pkl"
  adaptive_periods:
    lookback: 50
```

❌ **Don't Do This** - Create new files:
```python
# enhanced_momentum_strategy.py  ❌ NO!
# advanced_momentum_strategy.py  ❌ NO!
# ml_momentum_strategy.py        ❌ NO!
```

### Feature Flags in Code

**Use feature flags and composition, not inheritance:**

✅ **Good Feature Addition**:
```python
class MomentumStrategy:
    def __init__(self, config: MomentumConfig):
        self.config = config
        self.enhancers = self._build_enhancers(config.features)
    
    def _build_enhancers(self, features: List[str]) -> List[SignalEnhancer]:
        enhancers = []
        if 'volume_filter' in features:
            enhancers.append(VolumeFilter(self.config.volume_filter))
        if 'ml_enhancement' in features:
            enhancers.append(MLEnhancer(self.config.ml_enhancement))
        return enhancers
    
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        signal = self._base_momentum_signal(data)
        
        for enhancer in self.enhancers:
            signal = enhancer.enhance(signal)
            
        return signal
```

## Refactoring Guidelines

### Cleaning Up Existing "Enhanced" Files

**Step 1: Identify canonical implementation**
```bash
# Find all "enhanced" files
find src/ -name "*enhanced*" -o -name "*improved*" -o -name "*advanced*"
```

**Step 2: Merge capabilities into canonical file**
```python
# Take EnhancedContainer features and merge into Container
# Delete EnhancedContainer
# Update all imports to use Container
```

**Step 3: Use configuration for variants**
```yaml
# Replace multiple implementations with configuration
container:
  type: universal  # One container type
  features:
    - subcontainer_support
    - scoped_event_bus
    - hierarchical_resolution
```

### Migration Pattern

✅ **Good Migration**:
```python
# Before: Multiple files
# container.py
# enhanced_container.py  
# advanced_container.py

# After: Single file with composition
# container.py
class Container:
    def __init__(self, config: ContainerConfig):
        self.capabilities = self._build_capabilities(config.features)
    
    def _build_capabilities(self, features: List[str]) -> Dict[str, Any]:
        caps = {}
        if 'subcontainers' in features:
            caps['subcontainer_manager'] = SubcontainerManager()
        if 'scoped_events' in features:
            caps['scoped_bus'] = ScopedEventBus()
        return caps
```

---

## Testing Strategy

### Test Names Follow Same Rules

✅ **Good Test Names**:
```python
def test_momentum_strategy_generates_buy_signal():
def test_portfolio_risk_manager_limits_exposure():
def test_backtest_container_processes_events():
```

❌ **Bad Test Names**:
```python
def test_enhanced_strategy_works_better():
def test_improved_risk_manager_features():  
def test_advanced_container_capabilities():
```

### Test Organization

**Mirror source structure without adjectives:**

```
tests/
├── unit/
│   ├── core/
│   │   ├── test_container.py      # Tests container.py
│   │   └── test_coordinator.py    # Tests coordinator.py
│   ├── strategy/
│   │   └── test_momentum.py       # Tests momentum strategies
│   └── risk/
│       └── test_portfolio_risk.py # Tests portfolio risk
└── integration/
    ├── test_backtest_workflow.py
    └── test_signal_generation.py
```

---

## Documentation Standards

### README Files

**Each module gets ONE README explaining the canonical approach:**

```markdown
# Core Containers

This module contains THE container implementation for ADMF-PC.

## Files

- `container.py` - The canonical container implementation
- `protocols.py` - Container protocols and interfaces  
- `factory.py` - Container creation and lifecycle
- `lifecycle.py` - Container state management

## No "Enhanced" Versions

Do not create enhanced_container.py, improved_container.py, etc.
Use composition and configuration to add capabilities.
```

### Code Comments

**Indicate canonical status in docstrings:**

```python
class Container:
    """
    THE canonical container implementation for ADMF-PC.
    
    This is the single source of truth for container behavior.
    Do not create enhanced/improved/advanced versions.
    Use composition and configuration for additional capabilities.
    """
```

---

## Enforcement

### Pre-commit Hooks

**Add pre-commit hook to catch "enhanced" files:**

```bash
#!/bin/bash
# check-naming.sh

# Check for files with problematic adjectives
problematic_files=$(find src/ -name "*enhanced*" -o -name "*improved*" -o -name "*advanced*" -o -name "*better*" -o -name "*optimized*")

if [ ! -z "$problematic_files" ]; then
    echo "❌ Found files with adjective-based names:"
    echo "$problematic_files"
    echo ""
    echo "Use descriptive, role-based names instead:"
    echo "  enhanced_strategy.py → momentum_strategy.py"
    echo "  improved_container.py → use composition in container.py"
    echo "  advanced_risk.py → portfolio_risk_manager.py"
    exit 1
fi

echo "✅ No problematic file names found"
```

### Code Review Template

```markdown
## Naming Review Checklist

- [ ] No files with adjective names (enhanced_, improved_, advanced_)
- [ ] Uses role-based naming (momentum_, portfolio_, backtest_)  
- [ ] Follows single responsibility principle
- [ ] Uses composition over inheritance
- [ ] Capabilities configurable, not hard-coded in new classes
- [ ] Updates canonical implementation instead of creating duplicates
```

---

## Summary

### The Golden Rules

1. **One canonical implementation per concept**
   - `container.py` not `enhanced_container.py`

2. **Role-based naming, not adjective-based**
   - `MomentumStrategy` not `EnhancedStrategy`

3. **Configuration-driven features, not new files**
   - Add features through config, not new classes

4. **Composition over inheritance**
   - Use protocols and composition to add capabilities

5. **Explicit canonical status**
   - Mark THE implementation clearly in docs and comments

### Quick Reference

**When LLM suggests creating:**
- `enhanced_xyz.py` → Enhance existing `xyz.py` instead
- `improved_abc.py` → Use composition in existing `abc.py`
- `advanced_def.py` → Add features via configuration to `def.py`
- `better_ghi.py` → Improve existing `ghi.py` directly
- `optimized_jkl.py` → Use feature flags in existing `jkl.py`

**Always ask: "Can this be achieved by enhancing the existing canonical implementation?"**

The answer is almost always **YES** with Protocol + Composition architecture.


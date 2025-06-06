# ADMF-PC Style Guide: Preventing "Enhanced" Bloat and Maintaining Canonical Code

## The Problem: "Enhanced" Death Spiral

**Anti-Pattern**: Creating files like `enhanced_container.py`, `advanced_strategy.py`, `improved_xyz.py` leads to:
- Multiple implementations of the same concept
- Unclear which version is canonical
- Code duplication and inconsistency  
- LLMs creating more bloat when unclear what's authoritative

**Solution**: This style guide establishes **ONE canonical implementation per concept** with clear naming patterns.

---

## Anti-Pattern: Backward Compatibility as Technical Debt

**ADMF-PC Principle**: Backward compatibility is an anti-pattern that prevents architectural evolution.

### Why Backward Compatibility Hurts

❌ **The Backward Compatibility Trap**:
```python
# This thinking creates technical debt:
"We can't delete old_container.py because something might depend on it"
"Let's keep both implementations to avoid breaking changes"
"We'll deprecate it gradually over several releases"
```

**Problems with this approach:**
- Creates multiple implementations of the same concept
- Leads to "enhanced" file proliferation 
- Prevents adopting better architectural patterns
- Increases cognitive load for developers
- Makes the system harder to understand and maintain

### ADMF-PC's Approach: Bold Architectural Evolution

✅ **Break Things to Make Them Better**:
```python
# This thinking enables architectural evolution:
"Delete the old implementation completely"
"Force all code to use the new canonical approach"
"Fix breaking changes immediately, don't paper over them"
"Architectural purity is more valuable than temporary convenience"
```

**Benefits of this approach:**
- Eliminates duplicate implementations
- Forces adoption of better patterns
- Keeps the codebase clean and understandable
- Enables rapid architectural iteration
- Prevents technical debt accumulation

### Practical Implementation

When improving architecture:

1. **Delete, don't deprecate**: Remove old implementations completely
2. **Fix breaks immediately**: Update all dependent code in the same change
3. **No compatibility layers**: Don't create wrappers or adapters for old code
4. **Wrappers are a code smell**: This is OUR codebase - we shouldn't need wrappers to interface our own code
5. **Document breaking changes**: Clearly explain what changed and why
6. **Trust the process**: Protocol + Composition makes changes easier, not harder

**Note**: Backup files (*.bak, *.old) for safety during refactoring are fine, but should not be imported or referenced by active code. Clean them up after successful migration.

### Examples

❌ **Backward Compatibility Anti-Pattern**:
```python
# Keep old implementation around "just in case"
class OldContainer:  # deprecated
    pass

class NewContainer:  # preferred
    pass

# Create compatibility wrapper (CODE SMELL!)
class Container(NewContainer):  # for backward compatibility
    def old_method(self):  # deprecated method
        return self.new_method()

# This is OUR code - why do we need wrappers to use it?
# Wrappers indicate architectural confusion and technical debt
```

✅ **ADMF-PC Bold Evolution**:
```python
# Delete old implementation completely
# Update ALL imports to use new implementation
# Fix ALL breaking changes in one atomic commit

class Container:  # THE implementation
    def method(self):  # THE interface
        pass
```

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

### Module Organization Pattern

**Each module should follow this lean structure:**

✅ **Ideal Module Structure**:
```
module/
├── protocols.py         # THE interfaces this module provides
├── [specific].py        # THE canonical implementations
├── __init__.py          # Clean exports
└── README.md            # Documentation
```

**That's it!** No extra abstraction layers, no enhancement files, no utilities.

❌ **Anti-Pattern Files to Avoid**:
```
module/
├── capabilities.py      # ❌ Enhancement anti-pattern
├── enhanced_*.py        # ❌ Adjective prefixes
├── base_*.py           # ❌ Inheritance hierarchies
├── abstract_*.py       # ❌ Abstract base classes
├── manager.py          # ❌ Often a god object smell
├── utils.py            # ❌ Usually a dumping ground
└── helpers.py          # ❌ Another dumping ground
```

### Why This Works

1. **protocols.py** defines WHAT the module can do (interfaces)
2. **Implementation files** define HOW it does it (concrete classes)
3. **No capabilities.py** - components ARE capabilities themselves
4. **No enhanced versions** - one canonical implementation per concept

### Example: Clean Module Organization

✅ **Risk Module (Correct)**:
```
risk/
├── protocols.py          # RiskLimitProtocol, PositionSizerProtocol
├── limits.py            # MaxPositionLimit, MaxDrawdownLimit
├── position_sizing.py   # FixedPositionSizer, PercentagePositionSizer
├── portfolio_state.py   # PortfolioState
└── models.py           # Risk data models
```

❌ **Risk Module (Wrong)**:
```
risk/
├── protocols.py
├── base_risk.py         # ❌ Inheritance base
├── risk_manager.py      # ❌ God object
├── capabilities.py      # ❌ Enhancement layer
├── enhanced_limits.py   # ❌ Adjective prefix
├── utils.py            # ❌ Dumping ground
└── helpers.py          # ❌ More dumping
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

## Working with LLMs: Documentation-First Development

### The Documentation Imperative

**LLMs must read and understand existing documentation before making changes.**

✅ **Required LLM Behavior**:
```
Before making ANY code changes, I will:
1. Read the relevant documentation files
2. Understand the existing architecture patterns
3. Ask for clarification if uncertain about canonical implementations
4. Propose changes that align with established patterns
5. Explicitly state which files I'm treating as canonical
6. Delete old implementations completely (no backward compatibility)
7. Fix ALL breaking changes in the same atomic commit
```

❌ **Prohibited LLM Behavior**:
```
- Creating files without reading documentation
- Assuming what's "enhanced" or "improved" 
- Making changes without understanding existing patterns
- Duplicating functionality instead of extending canonical implementations
- Keeping old implementations for "backward compatibility"
- Creating compatibility wrappers or deprecation layers
- Gradual migration plans that leave multiple implementations
- Creating wrappers around our own code (this is a code smell indicating architectural confusion)
- Creating temporary files outside tmp/ directory
- Creating status reports, debug scripts, or analysis files in root or src/
```

### LLM Interaction Protocol

When working with LLMs on ADMF-PC, **always include this instruction**:

```
MANDATORY: Before proposing any changes, read these documentation files:
1. README.md - Project overview and architecture
2. STYLE_GUIDE.md - Naming conventions and canonical implementations  
3. src/core/README.md - Core architecture documentation
4. Relevant module README.md files

If you're uncertain about:
- Which file is the canonical implementation
- Whether a capability already exists
- How to extend existing functionality
- Naming conventions for new components

STOP and ASK FOR CLARIFICATION. Do not guess or assume.

State explicitly: "I have read [list files] and understand that [canonical file] is the authoritative implementation."
```

### LLM Verification Checklist

Before accepting LLM suggestions, verify the LLM has:

- [ ] **Read documentation**: LLM explicitly lists documentation files it read
- [ ] **Identified canonical files**: LLM states which files are authoritative
- [ ] **Justified approach**: LLM explains why it chose this approach vs alternatives
- [ ] **Checked for existing functionality**: LLM verified the feature doesn't already exist
- [ ] **Asked for clarification**: LLM asked about uncertainties instead of guessing

### Documentation-Driven Prompts

✅ **Good LLM Prompts** (Documentation-First):
```
"Please read src/core/containers/README.md and container.py to understand 
the canonical container implementation. Then enhance the canonical container.py 
file to support the new capability, following the existing patterns."

"After reading the strategy module documentation, extend the existing 
MomentumStrategy class to support the new feature. Do not create 
EnhancedMomentumStrategy."

"Please review STYLE_GUIDE.md and tell me which file is canonical for 
risk management, then propose how to add this capability."
```

❌ **Bad LLM Prompts** (Lead to Bloat):
```
"Create an enhanced version of the container"
"Make an improved signal processor"  
"Build an advanced risk manager"
"Add this feature" (without specifying canonical file)
```

### Required LLM Response Format

LLMs must structure responses like this:

```
## Documentation Review
I have read:
- [file1]: Understanding of key concepts
- [file2]: Understanding of patterns
- [file3]: Understanding of constraints

## Canonical Implementation Analysis  
The canonical implementation for [functionality] is [file.py] because:
- [reason 1]
- [reason 2]

## Proposed Approach
Instead of creating enhanced_[file].py, I will:
1. [specific approach]
2. [following existing patterns]
3. [maintaining compatibility]

## Uncertainties
I need clarification on:
- [specific question 1]
- [specific question 2]
```

### Enforcing Documentation Reading

**Include this in every prompt to LLMs:**

```
CRITICAL: This project follows strict canonical implementation patterns. 
Read the documentation first. If you create any file with 'enhanced', 
'improved', 'advanced', 'better', or similar adjectives, the change 
will be rejected. Use composition and configuration instead.

BACKWARD COMPATIBILITY IS AN ANTI-PATTERN: Delete old implementations 
completely. Do not create compatibility layers, deprecation warnings, 
or gradual migration plans. Fix all breaking changes immediately.

TEMPORARY FILES GO IN tmp/: All debug scripts, status reports, analysis 
files, and experimental code must go in tmp/ directory. Never create 
temporary files in root or src/.

Confirm you've read the relevant documentation by stating:
"I have read [specific files] and will enhance the canonical [file.py] 
implementation using [specific approach]. I will delete any competing 
implementations and fix all breaking changes atomically."
```

### Documentation Update Requirement

**When LLMs make changes, they must also update documentation:**

✅ **Required Documentation Updates**:
```
When modifying canonical files, LLMs must:
1. Update the relevant README.md if behavior changes
2. Update inline documentation/docstrings
3. Add configuration examples if new features are configurable
4. Update any architectural diagrams if structure changes
```

### LLM Training Examples

**Show LLMs the correct pattern:**

```
Human: "Add ML capabilities to signal processing"

✅ Correct LLM Response:
"I have read src/strategy/README.md and signal_processor.py. The canonical 
implementation is signal_processor.py. I will add ML capabilities using 
composition:

class SignalProcessor:
    def __init__(self, config):
        self.enhancers = []
        if 'ml_enhancement' in config.features:
            self.enhancers.append(MLEnhancer(config.ml_config))

This follows the existing pattern and avoids creating MLSignalProcessor."

❌ Incorrect LLM Response:
"I'll create an enhanced_signal_processor.py with ML capabilities..."
```

---

## File Organization and Whitelist Rules

### The tmp/ Directory Strategy

**ADMF-PC Principle**: Use `tmp/` for all disposable work to prevent root directory chaos.

#### ✅ What Goes in tmp/
```
tmp/
├── debug/           # Debug scripts, test outputs
├── analysis/        # Ad-hoc analysis scripts  
├── reports/         # Status reports, summaries
├── prototypes/      # Experimental code
├── logs/           # Temporary log files
└── scratch/        # Any other disposable work
```

**Examples of tmp/ usage:**
- `tmp/debug/test_new_feature.py` instead of root `debug_new_feature.py`
- `tmp/reports/migration_status.md` instead of root `MIGRATION_STATUS_SUMMARY.md`
- `tmp/analysis/performance_check.py` instead of root `analyze_performance.py`
- `tmp/prototypes/enhanced_container.py` instead of src `enhanced_container.py`

#### ❌ What NEVER Goes in tmp/
- Canonical implementations
- Core documentation 
- Production configuration
- Permanent test suites
- Anything that other code imports

### File Whitelist Strategy (Post-Cleanup)

**Phase 1**: Complete codebase cleanup (Step 10.0)
**Phase 2**: Establish file whitelist for each directory
**Phase 3**: Require explicit approval for new files outside tmp/

#### Planned Root Directory Whitelist (~10-15 files)
```
/
├── README.md                 # Project overview
├── CLAUDE.md                # LLM interaction guidelines  
├── STYLE.md                 # Style guide
├── requirements.txt         # Dependencies
├── main.py                  # Entry point
├── pyproject.toml          # Project configuration
├── .gitignore              # Git ignore rules
├── LICENSE                 # License file
└── tmp/                    # Everything else goes here
```

#### Planned src/ Directory Whitelist (Post Step 10.0)
```
src/
├── core/                    # Core ADMF-PC infrastructure
├── data/                    # Data handling (1 canonical implementation)
├── execution/               # Execution engines (1 canonical implementation)  
├── risk/                    # Risk management (1 canonical implementation)
├── strategy/                # Strategy components (1 canonical implementation)
└── reporting/               # Reporting (1 canonical implementation)
```

### Enforcement Mechanism

#### Pre-commit Hook (Future)
```bash
#!/bin/bash
# check-file-whitelist.sh

# After cleanup, maintain approved file list
WHITELIST_FILE=".approved-files"

# Check if new files are being added outside tmp/
NEW_FILES=$(git diff --cached --name-only --diff-filter=A | grep -v "^tmp/")

if [ ! -z "$NEW_FILES" ]; then
    echo "❌ New files detected outside tmp/ directory:"
    echo "$NEW_FILES"
    echo ""
    echo "📋 Options:"
    echo "1. Move to tmp/ if temporary: git mv file.py tmp/category/file.py"
    echo "2. Add to whitelist if permanent: Add to $WHITELIST_FILE"
    echo "3. Get explicit approval for new canonical files"
    exit 1
fi
```

#### LLM Instructions Update
```
CRITICAL: After codebase cleanup, new files require approval.

✅ For temporary work: Always use tmp/ directory
✅ For permanent changes: Enhance existing canonical files
❌ NEVER create new files outside tmp/ without explicit approval

When proposing new files, always ask:
"Should this be temporary (tmp/) or enhance existing canonical implementation?"
```

## Stateless Services and Parameter Search Patterns

### Core Principle: Stateless Functions with Runtime Parameters

**For parameter search and dynamic generation, use stateless functions that receive parameters at runtime, not classes that store parameters.**

### ✅ Correct Pattern:
```python
# Stateless service - parameters passed at runtime
def momentum_strategy(features: Dict, bar: Dict, params: Dict) -> Dict:
    """Pure function - no internal state"""
    return calculate_signal(features, params)

# Usage in parameter search:
# 1000 portfolios can share 1 function instance
for portfolio in portfolios:
    signal = momentum_strategy(features, bar, portfolio.strategy_params)
```

### ❌ Anti-Pattern:
```python
# Don't create instances per parameter combination
class MomentumStrategy:
    def __init__(self, fast_period, slow_period):
        self.fast_period = fast_period  # Wasteful state storage

# This would create 1000 instances for 1000 portfolios!
strategies = {}
for combo in parameter_combinations:
    strategies[combo.id] = MomentumStrategy(combo.fast, combo.slow)
```

### When to Use Factories

Use factories ONLY for complex orchestration:

| Use Factory For                    | Don't Use Factory For        |
|------------------------------------|------------------------------|
| Container orchestration            | Stateless functions          |
| Adapter wiring                     | Simple parameter objects     |
| Topology composition               | Pure calculations            |
| Complex dependency injection       | Decorator-registered items   |

**Decision Tree**:
```
┌─────────────────────────────────────┐
│ Does creation require orchestration? │
│ (wiring, lifecycle, dependencies)    │
└─────────────┬───────────────────────┘
              │
        ┌─────┴─────┐
        │    YES    │ → Use Factory
        │           │   (Containers, Adapters, Topologies)
        └───────────┘
              │
        ┌─────┴─────┐
        │    NO     │ → Direct Creation or Decorator
        │           │   (Stateless functions, Simple objects)
        └───────────┘
```

### Discovery Patterns and Decorators

ADMF-PC uses a decorator-based discovery system for strategies, classifiers, and features:

#### ✅ Use Decorators For:
```python
# Strategies
@strategy(
    feature_config={
        'sma': {'params': ['sma_period'], 'default': 20},
        'rsi': {'params': ['rsi_period'], 'default': 14}
    }
)
def momentum_strategy(features: Dict, bar: Dict, params: Dict) -> Dict:
    # Pure function implementation
    pass

# Classifiers
@classifier(
    regime_types=['trending_up', 'trending_down', 'ranging'],
    features=['sma_fast', 'sma_slow']
)
def trend_classifier(features: Dict, params: Dict) -> Dict:
    # Pure function implementation
    pass

# Features/Indicators
@feature(
    params=['period'],
    min_history=20
)
def sma(data: pd.Series, period: int) -> float:
    return data.rolling(period).mean().iloc[-1]
```

#### Benefits of Decorator Discovery:
1. **Auto-discovery**: Components found automatically at startup
2. **Feature Requirements**: Strategies declare needed features
3. **Parameter Metadata**: Default values and constraints
4. **Type Safety**: Registry provides typed resolution
5. **Plugin Architecture**: Easy to add new components

### Parameter Search Architecture

```
Parameter Search Flow:
┌─────────────────────────────────────────────────────────────┐
│                     Configuration Phase                      │
├─────────────────────────────────────────────────────────────┤
│ strategies:                                                  │
│   - type: momentum                                          │
│     parameters:                                             │
│       fast: [10, 20]  ─┐                                   │
│       slow: [30, 40]   ├─→ Cartesian Product              │
│                        │   Creates 4 combinations          │
│ risk:                  │                                   │
│   - type: conservative │                                   │
│     max_position: [0.1, 0.2] → 2 combinations            │
└────────────────────────┴───────────────────────────────────┘
                                 ↓
┌─────────────────────────────────────────────────────────────┐
│                      Runtime Phase                           │
├─────────────────────────────────────────────────────────────┤
│ 8 Portfolio Containers Created (4 strategy × 2 risk combos) │
│                                                             │
│ Portfolio_001: {strategy: momentum(10,30), risk: cons(0.1)}│
│ Portfolio_002: {strategy: momentum(10,30), risk: cons(0.2)}│
│ Portfolio_003: {strategy: momentum(10,40), risk: cons(0.1)}│
│ ... etc                                                     │
│                                                             │
│ BUT: Only 1 momentum function + 1 conservative validator!   │
└─────────────────────────────────────────────────────────────┘
```

### Event-Driven Parameter Routing

How parameters flow through events:
```
┌────────────────────┐
│   Portfolio_001    │
│ params: {fast: 10} │
└─────────┬──────────┘
          │ ORDER_REQUEST
          │ {order: ...,
          │  risk_params: {max_pos: 0.1}}
          ↓
┌────────────────────┐
│  Risk Service      │
│  Adapter           │ → validator.validate(order, params)
└─────────┬──────────┘   Uses 0.1 from event
          │ ORDER
          ↓
┌────────────────────┐
│    Execution       │
└────────────────────┘
```

### Architecture Benefits

This pattern enables:
- **Massive parallelization**: 1000 portfolios, 3 strategy functions
- **Zero state contamination**: Parameters in events, not objects
- **Efficient resource usage**: Minimal object creation
- **Clear parameter flow**: Track via event payloads
- **Dynamic discovery**: Components found via decorators

### The Pattern is Consistent:

| Component       | Instance Count | Parameter Combinations | Where Parameters Live   | Discovery Method |
|-----------------|----------------|------------------------|-------------------------|------------------|
| Risk Validators | One per type   | Unlimited              | In ORDER_REQUEST event  | Direct creation  |
| Strategies      | One per type   | Unlimited              | Passed to function call | @strategy        |
| Classifiers     | One per type   | Unlimited              | Passed to function call | @classifier      |
| Features        | One per type   | Unlimited              | Passed to function call | @feature         |
| Portfolios      | One per combo  | N/A                    | In container state      | Factory pattern  |

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

6. **Documentation-first development**
   - LLMs must read docs before making changes
   - Ask for clarification when uncertain
   - Never guess at canonical implementations

7. **Backward compatibility is an anti-pattern**
   - Delete old implementations completely
   - Fix breaking changes immediately
   - No compatibility layers or gradual migrations

8. **Use tmp/ for all temporary work**
   - Disposable scripts, debug files, status reports
   - Prevents root directory chaos
   - Everything in tmp/ can be deleted without consequence

9. **Stateless services with runtime parameters**
   - Pure functions for strategies, classifiers, risk
   - Parameters passed at runtime, not stored
   - Enables massive parallelization

10. **Decorator-based discovery**
    - Use @strategy, @classifier, @feature decorators
    - Auto-discovery at startup
    - Feature requirements declared in metadata

### Quick Reference

**When LLM suggests creating:**
- `enhanced_xyz.py` → Enhance existing `xyz.py` instead
- `improved_abc.py` → Use composition in existing `abc.py`
- `advanced_def.py` → Add features via configuration to `def.py`
- `better_ghi.py` → Improve existing `ghi.py` directly
- `optimized_jkl.py` → Use feature flags in existing `jkl.py`

**Always ask: "Can this be achieved by enhancing the existing canonical implementation?"**

The answer is almost always **YES** with Protocol + Composition architecture.

**LLM Interaction Mantra:**
> "Read documentation first, identify canonical files, ask for clarification when uncertain, enhance existing implementations through composition and configuration."

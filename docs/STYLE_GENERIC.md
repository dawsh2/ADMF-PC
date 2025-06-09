# Style Guide: Maintaining Clean, Canonical Code

## The Problem: "Enhanced" Death Spiral

**Anti-Pattern**: Creating files like `enhanced_component.py`, `advanced_feature.py`, `improved_xyz.py` leads to:
- Multiple implementations of the same concept
- Unclear which version is canonical
- Code duplication and inconsistency  
- Confusion about which implementation to use or extend

**Solution**: This style guide establishes **ONE canonical implementation per concept** with clear naming patterns.

## Core Principles

### 1. Single Source of Truth

Every concept should have exactly ONE canonical implementation. No duplicates, no "enhanced" versions, no quality-based variants.

**✅ Good Structure:**
```
module/
├── component.py          # THE component implementation
├── protocols.py         # THE component protocols  
├── factory.py           # THE component factory
└── models.py            # THE data models
```

**❌ Bad Structure:**
```
module/
├── component.py
├── enhanced_component.py     # ❌ Delete this
├── improved_component.py     # ❌ Don't create this
├── advanced_component.py     # ❌ Avoid this
├── component_v2.py          # ❌ No versions
└── better_component.py      # ❌ No adjectives
```

### 2. Bold Architectural Evolution

**Backward compatibility at the expense of clarity is technical debt.**

When improving architecture:

1. **Delete, don't deprecate**: Remove old implementations completely
2. **Fix breaks immediately**: Update all dependent code in the same change
3. **No compatibility layers**: Don't create wrappers for old interfaces
4. **Document breaking changes**: Clearly explain what changed and why

**❌ Anti-Pattern:**
```python
# Keeping old implementation for compatibility
class OldComponent:  # deprecated
    pass

class NewComponent:  # preferred
    pass

# Compatibility wrapper (CODE SMELL!)
class Component(NewComponent):
    def old_method(self):
        return self.new_method()
```

**✅ Good Pattern:**
```python
# Single, clean implementation
class Component:  # THE implementation
    def method(self):  # THE interface
        pass
```

## Naming Conventions

### Use Role-Based Names, Not Quality Adjectives

**✅ Good Names (Role-Based):**
```python
# Components named by what they do
DataProcessor
RequestHandler
CacheManager
EventDispatcher

# Strategies named by approach
RecursiveParser
IterativeOptimizer
StreamProcessor

# Variants named by specific behavior
InMemoryCache
RedisCache
FileCache
```

**❌ Bad Names (Quality-Based):**
```python
# Avoid adjectives that don't specify role
EnhancedProcessor      # Enhanced how?
ImprovedHandler        # Improved from what?
AdvancedManager        # Advanced compared to what?
BetterDispatcher       # Better than what?
OptimizedCache         # Optimized how?
```

### Component Organization

**Organize by capability, not quality level:**

**✅ Good Organization:**
```
src/
├── parsing/
│   ├── json_parser.py      # Parses JSON
│   ├── xml_parser.py       # Parses XML
│   └── csv_parser.py       # Parses CSV
├── caching/
│   ├── memory_cache.py     # In-memory caching
│   ├── redis_cache.py      # Redis-based caching
│   └── file_cache.py       # File-based caching
└── processing/
    ├── batch_processor.py   # Batch processing
    └── stream_processor.py  # Stream processing
```

**❌ Bad Organization:**
```
src/
├── basic_components.py      # ❌ Quality levels
├── intermediate_features.py # ❌ Skill levels  
├── advanced_modules.py      # ❌ Difficulty levels
├── enhanced_functions.py    # ❌ Adjectives
└── optimized_code.py        # ❌ Marketing speak
```

## Evolution Through Composition

### Configuration-Driven Features

**Add capabilities through configuration, not new files:**

**✅ Good Pattern:**
```python
# processor.py - THE processor implementation
class Processor:
    def __init__(self, config: ProcessorConfig):
        self.batch_size = config.batch_size
        self.enable_caching = config.enable_caching
        self.parallel_workers = config.parallel_workers
        
        # New features via config
        if config.enable_caching:
            self.cache = CacheManager(config.cache_config)
        if config.parallel_workers > 1:
            self.executor = ParallelExecutor(config.parallel_workers)
    
    def process(self, data):
        # Single implementation handles all configurations
        if self.enable_caching and self.cache.has(data.id):
            return self.cache.get(data.id)
            
        result = self._process_data(data)
        
        if self.enable_caching:
            self.cache.set(data.id, result)
            
        return result
```

**❌ Bad Pattern:**
```python
# processor.py
class Processor:
    # Basic implementation

# cached_processor.py  ❌ DON'T DO THIS
class CachedProcessor(Processor):
    # Adds caching

# parallel_processor.py  ❌ DON'T DO THIS  
class ParallelProcessor(CachedProcessor):
    # Adds parallelization
```

### Composition Over Inheritance

**Use protocols and composition to extend functionality:**

**✅ Good Extension Pattern:**
```python
# Define protocols/interfaces
class Filter(Protocol):
    def apply(self, data: Data) -> Data:
        ...

# Implement composable filters
class ValidationFilter:
    def apply(self, data: Data) -> Data:
        # Validation logic
        return validated_data

class TransformFilter:
    def apply(self, data: Data) -> Data:
        # Transformation logic
        return transformed_data

# Use composition in main class
class Pipeline:
    def __init__(self, filters: List[Filter] = None):
        self.filters = filters or []
    
    def process(self, data: Data) -> Data:
        result = data
        for filter in self.filters:
            result = filter.apply(result)
        return result

# Usage: Compose as needed
pipeline = Pipeline([
    ValidationFilter(strict=True),
    TransformFilter(format="json")
])
```

**❌ Bad Extension Pattern:**
```python
# Base class
class Pipeline:
    # Basic implementation

# Inheritance chain ❌
class ValidatingPipeline(Pipeline):
    # Adds validation

# More inheritance ❌
class TransformingValidatingPipeline(ValidatingPipeline):
    # Adds transformation
```

## File Organization Rules

### Lean Module Structure

**Each module should be minimal and focused:**

**✅ Ideal Module Structure:**
```
module/
├── __init__.py          # Clean exports
├── protocols.py         # Interfaces/protocols
├── models.py           # Data models
├── [specific].py       # Concrete implementations
└── README.md           # Documentation
```

**❌ Files to Avoid:**
```
module/
├── base_*.py           # ❌ Inheritance base classes
├── abstract_*.py       # ❌ Abstract classes
├── enhanced_*.py       # ❌ Quality adjectives
├── utils.py            # ❌ Dumping ground
├── helpers.py          # ❌ Another dumping ground
├── common.py           # ❌ Vague shared code
└── misc.py            # ❌ Even more vague
```

### File Naming Rules

1. **Use descriptive nouns, not adjectives**
   - ✅ `json_parser.py`
   - ❌ `enhanced_parser.py`

2. **Use specific roles, not quality levels**
   - ✅ `request_validator.py`
   - ❌ `advanced_validator.py`

3. **Use implementation method, not marketing**
   - ✅ `recursive_finder.py`
   - ❌ `optimized_finder.py`

## Refactoring Guidelines

### Consolidating Multiple Implementations

When you find multiple versions of the same component:

**Step 1: Identify the canonical version**
```bash
# Find variants
find . -name "*processor*" -type f
```

**Step 2: Analyze capabilities**
- List unique features in each variant
- Identify configuration points
- Plan consolidation approach

**Step 3: Merge into canonical implementation**
```python
# Before: Multiple files
processor.py
enhanced_processor.py
cached_processor.py

# After: Single configurable file
processor.py  # With all capabilities via config/composition
```

**Step 4: Update all imports**
```python
# Update imports across codebase
# from enhanced_processor import EnhancedProcessor
from processor import Processor  # Now supports all features
```

## Testing Patterns

### Test Structure Mirrors Source

**✅ Good Test Organization:**
```
tests/
├── unit/
│   ├── test_processor.py    # Tests processor.py
│   ├── test_validator.py    # Tests validator.py
│   └── test_cache.py        # Tests cache.py
├── integration/
│   ├── test_pipeline.py     # Tests component integration
│   └── test_workflow.py     # Tests full workflows
└── fixtures/
    └── test_data.py         # Shared test data
```

### Test Naming

**✅ Good Test Names:**
```python
def test_processor_handles_empty_input():
def test_validator_rejects_invalid_data():
def test_cache_expires_after_timeout():
```

**❌ Bad Test Names:**
```python
def test_enhanced_processor_is_better():
def test_improved_validator_works():
def test_optimized_cache_is_faster():
```

## Documentation Standards

### Clear Canonical Status

**Mark canonical implementations clearly:**

```python
class Processor:
    """
    THE canonical processor implementation.
    
    This is the single source of truth for data processing.
    Use configuration and composition to extend capabilities.
    Do not create enhanced/improved/advanced versions.
    """
```

### Module Documentation

**Each module README should state:**

```markdown
# Module Name

This module provides THE implementation for [functionality].

## Canonical Files

- `component.py` - The canonical component implementation
- `protocols.py` - Component interfaces
- `models.py` - Data models

## Extension Pattern

Use configuration and composition to add features.
Do not create enhanced_*.py or similar variants.
```

## Common Pitfalls and Solutions

### Pitfall 1: Feature Envy

**Problem**: "This class needs one more feature, I'll create EnhancedClass"

**Solution**: Add the feature to the existing class via:
- Configuration flag
- Optional dependency injection
- Composed behavior

### Pitfall 2: Versioning Implementations

**Problem**: "The old version might be needed, I'll keep both"

**Solution**: 
- Make a decisive change
- Update all usages immediately
- Delete the old version

### Pitfall 3: Abstraction Addiction

**Problem**: "I'll create an abstract base class for future flexibility"

**Solution**:
- Use protocols/interfaces instead
- Implement concrete classes directly
- Add abstraction only when you have 3+ implementations

## Quick Reference

### Decision Tree for Adding Features

```
Need new capability?
│
├─ Can existing code be configured?
│  └─ YES → Add configuration option
│
├─ Can behavior be composed?
│  └─ YES → Create composable component
│
├─ Must modify core logic?
│  └─ YES → Enhance canonical implementation
│
└─ Completely new concept?
   └─ YES → Create new file with role-based name
```

### The Golden Rules

1. **One implementation per concept**
2. **Role-based names, not quality adjectives**
3. **Configuration over duplication**
4. **Composition over inheritance**
5. **Delete, don't deprecate**
6. **Clear canonical status**

### Red Flags in Code Review

- Files with adjectives: `enhanced_`, `improved_`, `advanced_`
- Multiple implementations of same concept
- Deep inheritance hierarchies
- Backward compatibility wrappers
- Vague files: `utils.py`, `helpers.py`, `common.py`

---

**Remember**: Clean architecture with single sources of truth leads to maintainable, understandable code. When tempted to create an "enhanced" version, enhance the original instead.
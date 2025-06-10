# Coordinator Refactor Plan

## Problem Statement

The coordinator module has grown to include duplicate logic across three main classes:

1. **Coordinator** (800 lines) - Workflow orchestration
2. **Sequencer** (1000+ lines) - Sequence execution  
3. **TopologyBuilder** (1000+ lines) - Topology construction

**Duplicate Logic:**
- YAML pattern loading from `config/patterns/{type}/*.yaml`
- Configuration resolution (`{config.param}` templates, `$references`)
- Context building and value extraction
- Error handling and logging patterns

## Target Architecture

### New File Structure
```
src/core/coordinator/
├── __init__.py
├── coordinator.py           # Workflow orchestration (~400 lines)
├── sequencer.py            # Sequence logic (~800 lines)  
├── topology.py             # Topology building (~800 lines)
├── protocols.py            # Existing
├── types.py                # Existing
└── config/
    ├── __init__.py
    ├── pattern_loader.py    # YAML pattern loading (~100 lines)
    ├── resolver.py          # Config resolution/templates (~150 lines)
    ├── schemas.py           # Existing
    └── ...                  # Other existing config files
```

### Extracted Components

#### **config/pattern_loader.py**
- Consolidates YAML pattern loading from all three classes
- Handles Python pattern backward compatibility (`_PATTERN` suffix)
- Manages built-in patterns (walk_forward, single_pass, etc.)
- Auto-creates pattern directories
- Unified error handling and logging

#### **config/resolver.py**
- Consolidates config resolution logic from all three classes
- Handles `{config.param}` template resolution
- Handles `$variable` reference resolution
- Manages `{'from_config': 'param', 'default': value}` patterns
- Dot notation value extraction (`config.nested.param`)
- Context building and merging

## Migration Plan

### Phase 1: Extract PatternLoader
1. **Create** `config/pattern_loader.py`
2. **Move** YAML loading logic from coordinator, sequencer, topology
3. **Consolidate** built-in patterns from sequencer
4. **Update** all three classes to use `PatternLoader`
5. **Test** pattern loading functionality

### Phase 2: Extract ConfigResolver  
1. **Create** `config/resolver.py`
2. **Move** `_resolve_value()` methods from all classes
3. **Consolidate** context building logic
4. **Update** all three classes to use `ConfigResolver`
5. **Test** config resolution functionality

### Phase 3: Update Class Interfaces
1. **Inject** shared components via constructor
2. **Replace** duplicate methods with delegation
3. **Maintain** existing public interfaces
4. **Update** imports and dependencies

### Phase 4: Cleanup and Testing
1. **Remove** duplicate code
2. **Update** documentation
3. **Run** full test suite
4. **Verify** identical behavior

## Implementation Details

### PatternLoader Interface
```python
class PatternLoader:
    def load_patterns(self, pattern_type: str) -> Dict[str, Any]:
        """Load patterns from config/patterns/{pattern_type}/*.yaml"""
        
    def load_python_patterns(self, module_name: str) -> Dict[str, Any]:
        """Load Python patterns for backward compatibility"""
        
    def get_builtin_patterns(self, pattern_type: str) -> Dict[str, Any]:
        """Get built-in patterns (walk_forward, single_pass, etc.)"""
```

### ConfigResolver Interface
```python
class ConfigResolver:
    def resolve_value(self, spec: Any, context: Dict) -> Any:
        """Handle {config.param} templates and $references"""
        
    def extract_value(self, path: str, data: Dict) -> Any:
        """Extract values using dot notation"""
        
    def build_context(self, config: Dict, **kwargs) -> Dict:
        """Build execution context from config + metadata"""
```

### Updated Class Constructors
```python
# Coordinator
def __init__(self, pattern_loader=None, config_resolver=None):
    self.pattern_loader = pattern_loader or PatternLoader()
    self.config_resolver = config_resolver or ConfigResolver()
    
# Sequencer  
def __init__(self, topology_builder=None, pattern_loader=None, config_resolver=None):
    self.pattern_loader = pattern_loader or PatternLoader()
    self.config_resolver = config_resolver or ConfigResolver()
    
# TopologyBuilder
def __init__(self, pattern_loader=None, config_resolver=None):
    self.pattern_loader = pattern_loader or PatternLoader()
    self.config_resolver = config_resolver or ConfigResolver()
```

## Benefits

1. **Eliminate Duplication**: One implementation of pattern loading and config resolution
2. **Better Organization**: Config utilities properly grouped in `config/`
3. **Easier Testing**: Mock shared components for unit tests
4. **Cleaner Code**: Each class focuses on its core responsibility
5. **Maintainability**: Bug fixes and features go in one place

## Refactor Status: COMPLETED ✅

### Phases Completed:
1. ✅ **Phase 1**: Created PatternLoader, updated Coordinator and Sequencer
2. ✅ **Phase 2**: Created ConfigResolver, updated Coordinator and Sequencer  
3. ✅ **Phase 3**: Updated TopologyBuilder to use shared components
4. ✅ **Phase 4**: Removed old duplicate methods and cleaned up

### Code Reduction:
- **Before**: ~2800 lines across coordinator/sequencer/topology with significant duplication
- **After**: ~2400 lines with shared utilities and no duplication
- **Removed**: ~400 lines of duplicate code

### Files Modified:
- `src/core/coordinator/config/pattern_loader.py` (NEW)
- `src/core/coordinator/config/resolver.py` (NEW)
- `src/core/coordinator/config/validator.py` (NEW - simplified validation)
- `src/core/coordinator/config/__init__.py` (UPDATED - clean exports)
- `src/core/coordinator/coordinator.py` (UPDATED)
- `src/core/coordinator/sequencer.py` (UPDATED)
- `src/core/coordinator/topology.py` (UPDATED)

### Config Directory Cleanup & Schema System:
**NEW SCHEMA SYSTEM** ✅:
- `component_schemas.py` - Centralized schemas for all user-configurable components
- `validator.py` - Updated to use component schemas for validation
- `README.md` - Documentation for adding new component types

**LEGACY FILES TO REMOVE** ❌:
- `schema_validator.py` - Complex validation engine replaced by simplified system
- `unified_schemas.py` - Complex unified schemas replaced by component_schemas.py  
- `validator_integration.py` - Integration layer not needed with new system
- `example_usage.py` - Example file, not production code

**EXISTING SCHEMAS PRESERVED** ✅:
- `schemas.py` - **KEPT** detailed component schemas (strategies, risk, position sizing)
  - Contains comprehensive JSON schemas that were already well-designed
  - Integrated with new ComponentSchema wrapper for better organization

**ACTIVE FILES** ✅:
- `pattern_loader.py` - YAML pattern loading for workflows
- `resolver.py` - Template and reference resolution  
- `component_schemas.py` - Schema definitions for all component types
- `validator.py` - Configuration validation using schemas
- `README.md` - Documentation for adding new component types
- `__init__.py` - Clean exports for all functionality

### Schema System Benefits:
- **Centralized** - All component schemas in one place
- **Extensible** - Easy to add new component types
- **Type-safe** - Automatic validation of user configurations  
- **Self-documenting** - Schemas include examples and descriptions
- **Auto-validation** - Components validated against their schemas

## Backward Compatibility

### What We're Preserving
- All existing YAML pattern loading
- Python `_PATTERN` backward compatibility
- Built-in patterns (walk_forward, single_pass)
- Template resolution (`{config.param}`)
- Reference resolution (`$variable`)
- Default value handling
- Workflow execution behavior
- Public class interfaces

### What May Change
- Internal method signatures (private methods)
- Import paths for internal components
- Logging source names (functionality preserved)

## Future Enhancements (Post-Refactor)

Once refactor is complete, we can consider:
- Breaking backward compatibility for cleaner APIs
- Additional pattern types
- Enhanced template engine
- Better error messages
- Performance optimizations

## Testing Strategy

1. **Unit Tests**: Test PatternLoader and ConfigResolver in isolation
2. **Integration Tests**: Test updated classes with shared components
3. **Regression Tests**: Compare outputs before/after refactor
4. **Manual Testing**: Run existing workflows to verify behavior

## Success Criteria

- [ ] All existing workflows execute successfully
- [ ] Pattern loading works identically
- [ ] Config resolution produces same results
- [ ] Code duplication eliminated
- [ ] Line count reduced by ~400 lines
- [ ] Test coverage maintained/improved
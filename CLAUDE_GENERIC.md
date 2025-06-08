# LLM Interaction Guidelines

**This file provides guidance for LLMs working on software projects using documentation-first development.**

## Core Principle: Documentation-First Development

**You MUST read relevant documentation before proposing any code changes.**

**Never create files with quality adjectives like `enhanced_`, `improved_`, `advanced_`, `better_`, `optimized_`, `superior_`, `premium_`.**

## Required Actions Before Making Changes

Before making ANY changes, you must:

1. **Read all relevant documentation**
   - Project README
   - Module-specific documentation
   - Style guides
   - Architecture documents

2. **Identify canonical implementations**
   - Look for files without adjective prefixes
   - Check documentation references
   - Review import patterns
   - Examine timestamps and comprehensiveness

3. **Ask for clarification when uncertain**
   - Which file is canonical?
   - Does this capability already exist?
   - What's the preferred pattern here?
   - Should this be temporary or permanent?

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
3. **Compatibility**: [how existing code remains functional]

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

**When identifying which file is canonical, look for:**

1. **Files without adjective prefixes** (component.py vs enhanced_component.py)
2. **Referenced in documentation** as "THE implementation"
3. **More recent timestamps** and comprehensive functionality
4. **Imported by other modules** more frequently
5. **Matches naming patterns** described in style guides

**When in doubt, ASK:** "Which file should I treat as canonical for [functionality]?"

## Prohibited Actions

**NEVER do these without explicit permission:**

❌ Create files named: `enhanced_*`, `improved_*`, `advanced_*`, `better_*`, `optimized_*`, `superior_*`, `premium_*`

❌ Duplicate existing functionality in new files

❌ Assume what needs to be "enhanced" without reading existing code

❌ Make changes without understanding existing architecture patterns

❌ Create inheritance hierarchies when composition would work

❌ Add features through new classes when configuration would work

❌ Create temporary files outside designated temporary directories

❌ Generate status reports or debug scripts in source directories

## Required Actions

**ALWAYS do these:**

✅ Read documentation first

✅ Identify canonical implementations explicitly

✅ Ask for clarification when uncertain

✅ Use composition and configuration to add features

✅ Enhance existing files rather than creating new ones

✅ Update documentation when making changes

✅ Follow existing naming patterns

✅ Place temporary/debug files in designated directories

## Enhancement Patterns

### Use Configuration, Not New Files

```python
# ✅ Good: Feature flags in config
class Component:
    def __init__(self, config: ComponentConfig):
        if 'new_feature' in config.features:
            self.feature_handler = FeatureHandler()

# ❌ Bad: New files for features  
class EnhancedComponent(Component):
    pass
```

### Use Composition, Not Inheritance

```python
# ✅ Good: Composition
class Processor:
    def __init__(self, enhancers: List[Enhancer] = None):
        self.enhancers = enhancers or []

# ❌ Bad: Inheritance chain
class EnhancedProcessor(Processor):
    pass
```

## Quick Decision Tree

```
Need to add functionality?
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

## Common Scenarios & Correct Responses

### Scenario 1: "Add new capabilities to existing component"

✅ **Correct Response:**
```
## Documentation Review ✅
I have read:
- README.md: Project structure and patterns
- component.py: Current implementation
- STYLE.md: Composition patterns

## Canonical Implementation Analysis 🎯
The canonical implementation is: **component.py**
- Only implementation without adjective prefix
- Referenced in documentation
- Imported by other modules

## Proposed Approach 🛠️
I will enhance component.py using composition:
- Add capability as a composable element
- Use configuration to enable features
- Maintain backward compatibility

Would you like me to proceed with this approach?
```

❌ **Incorrect Response:**
```
I'll create an enhanced_component.py with the new capabilities...
```

### Scenario 2: "Improve performance of existing code"

✅ **Correct Response:**
```
## Documentation Review ✅
I have read the relevant files and documentation.

## Canonical Implementation Analysis 🎯
The canonical implementation is: **processor.py**

## Uncertainties ❓
I see there's also optimized_processor.py. Should I:
1. Merge optimized_processor.py improvements into processor.py?
2. Treat optimized_processor.py as the canonical version?
3. Ignore optimized_processor.py as experimental?

Please clarify which approach aligns with your architecture.
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

1. **Update README** if architecture changes
2. **Update module documentation** if behavior changes  
3. **Update inline docstrings** for modified functions
4. **Add configuration examples** for new features
5. **Update style guides** if new patterns emerge

## Working with Multiple Implementations

**If you find multiple implementations of the same concept:**

1. **STOP and identify** which is canonical
2. **Ask for confirmation** before proceeding
3. **Propose consolidation** if appropriate
4. **Never add another variant** without explicit approval

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

**The goal is clean, maintainable code with single sources of truth, not a proliferation of similar implementations.**
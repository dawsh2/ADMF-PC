# ADMF-PC Architecture Validation System

## Overview

The ADMF-PC Architecture Validation System ensures that the codebase follows the mandatory pattern-based architecture defined in [CLAUDE.md](../../CLAUDE.md) and [STYLE.md](../../STYLE.md).

This system prevents architectural violations that could compromise the consistency, maintainability, and evolution of the ADMF-PC framework.

## Components

### 1. Architecture Compliance Validator (`scripts/validate_architecture_compliance.py`)

**Purpose**: Comprehensive validation of architecture compliance across the entire codebase.

**Key Validation Rules**:

#### R001: No Enhanced/Improved Files
- **What**: Prevents creation of adjective-based file variants
- **Why**: Maintains single canonical implementation per concept
- **Examples**: 
  - ❌ `enhanced_container.py`, `improved_strategy.py`, `advanced_risk.py`
  - ✅ Use canonical files with composition and configuration

#### R002: Factory Separation  
- **What**: Ensures factories have single responsibilities
- **Why**: Prevents mixing container creation, communication setup, and orchestration
- **Checks**:
  - Container factory doesn't handle communication
  - Communication factory doesn't create containers
  - Workflow files delegate to factories instead of creating containers directly

#### R003: Single Source of Truth
- **What**: Each workflow pattern defined exactly once
- **Why**: Prevents duplicate pattern definitions across files
- **Authority**: WorkflowManager is the single source for pattern definitions

#### R004: Protocol + Composition
- **What**: Enforces composition over inheritance
- **Why**: Maintains flexibility and prevents inheritance hierarchies
- **Allows**: Protocol inheritance, specific exceptions (Enum, Exception, etc.)

#### R005: Canonical File Usage
- **What**: Ensures imports point to canonical implementations
- **Why**: Prevents usage of deprecated or non-canonical files
- **Detects**: Imports of known non-canonical modules

#### R006: Workflow Pattern Architecture
- **What**: Validates workflow patterns follow standard architecture
- **Why**: Ensures patterns specify both container_pattern and communication_config
- **Standard**: Each pattern must delegate to appropriate factories

### 2. Pre-commit Hook (`scripts/pre-commit-hook.sh`)

**Purpose**: Prevents architecture violations from being committed.

**Installation**:
```bash
# Option 1: Copy to git hooks
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Option 2: Symbolic link (recommended)
ln -sf ../../scripts/pre-commit-hook.sh .git/hooks/pre-commit
```

**Checks Performed**:
1. **Prohibited File Patterns**: Blocks enhanced/improved file names
2. **Temporary File Placement**: Warns about temp files outside `tmp/`
3. **Factory Separation**: Prevents container creation in workflow files
4. **Canonical Imports**: Blocks imports of non-canonical modules
5. **Full Validation**: Runs complete architecture validation

### 3. CI/CD Integration (`.github/workflows/architecture-compliance.yml`)

**Purpose**: Automated validation in continuous integration pipeline.

**Workflows**:

#### Architecture Compliance
- Runs full validation on push/PR
- Generates detailed compliance report
- Comments on PRs with violations
- Uploads artifacts for review

#### Pre-commit Check
- Tests pre-commit hook functionality
- Ensures hook works in CI environment

#### Documentation Check
- Verifies required documentation files exist
- Validates core architecture files are present

#### Workflow Patterns Check
- Ensures WorkflowManager is single source of truth
- Detects duplicate pattern definitions

## Usage

### Running Validation Locally

#### Full Validation
```bash
# Basic validation
python scripts/validate_architecture_compliance.py

# Verbose output
python scripts/validate_architecture_compliance.py --verbose

# Generate detailed report
python scripts/validate_architecture_compliance.py --report compliance-report.json

# Specify different root directory
python scripts/validate_architecture_compliance.py --root /path/to/project
```

#### Pre-commit Testing
```bash
# Test pre-commit hook
./scripts/pre-commit-hook.sh

# Test with specific files
git add file.py
./scripts/pre-commit-hook.sh
```

### Understanding Validation Output

#### Validation Results Format
```
🔍 ADMF-PC Architecture Compliance Validation
==================================================

🚨 R001: No Enhanced/Improved Files - ❌ FAIL
  • src/risk/enhanced_portfolio.py - Use canonical file instead
  • src/strategy/improved_momentum.py - Use canonical file instead

✅ R002: Factory Separation - ✅ PASS

⚠️ R004: Protocol + Composition - ⚠️ WARNING  
  • src/strategy/base_strategy.py:15 - Class Strategy inherits from BaseStrategy (use composition)
```

#### Result Types
- **✅ PASS**: Rule compliant, no violations
- **❌ FAIL**: Critical violations that must be fixed
- **⚠️ WARNING**: Non-critical issues to address

#### Severity Levels
- **Error**: Blocks commits/PRs, must be fixed
- **Warning**: Should be addressed but doesn't block
- **Info**: Informational, no action required

## Fixing Common Violations

### R001: Enhanced/Improved Files

**Problem**: Files with adjective-based names
```
❌ src/risk/enhanced_portfolio.py
❌ src/strategy/improved_momentum.py
❌ src/execution/advanced_broker.py
```

**Solution**: Use canonical files with composition
```python
# Instead of enhanced_portfolio.py, enhance portfolio.py
class Portfolio:
    def __init__(self, config: PortfolioConfig):
        self.enhancers = self._build_enhancers(config.features)
    
    def _build_enhancers(self, features: List[str]) -> List[Enhancer]:
        enhancers = []
        if 'advanced_analytics' in features:
            enhancers.append(AdvancedAnalytics(config.analytics))
        return enhancers
```

### R002: Factory Separation

**Problem**: Mixed factory responsibilities
```python
# ❌ In communication/factory.py
class CommunicationFactory:
    def create_adapters(self):
        # This is correct
        pass
        
    def create_containers(self):  # ❌ WRONG!
        # Communication factory shouldn't create containers
        pass
```

**Solution**: Use appropriate factories
```python
# ✅ Correct approach
from ...containers.factory import get_global_factory
from ...communication.factory import AdapterFactory

container_factory = get_global_factory()
containers = container_factory.compose_pattern('simple_backtest', config)

comm_factory = AdapterFactory() 
adapters = comm_factory.create_adapters_from_config(comm_config, containers)
```

### R003: Single Source of Truth

**Problem**: Multiple pattern definitions
```python
# ❌ In multiple files
# workflow_manager.py
self._patterns = {'simple_backtest': {...}}

# other_manager.py  
self._workflow_patterns = {'simple_backtest': {...}}  # DUPLICATE!
```

**Solution**: Define patterns only in WorkflowManager
```python
# ✅ Only in workflow_manager.py
class WorkflowManager:
    def __init__(self):
        self._workflow_patterns = {  # Single source of truth
            'simple_backtest': {
                'container_pattern': 'simple_backtest',
                'communication_config': [...]
            }
        }
```

### R004: Protocol + Composition

**Problem**: Inheritance hierarchies
```python
# ❌ Inheritance anti-pattern
class MomentumStrategy(BaseStrategy):  # Don't inherit!
    def __init__(self):
        super().__init__()  # Framework coupling
```

**Solution**: Use protocols and composition
```python
# ✅ Protocol-based approach
from typing import Protocol

class StrategyProtocol(Protocol):
    def evaluate(self, data: MarketData) -> Signal: ...

class MomentumStrategy:  # No inheritance!
    def evaluate(self, data: MarketData) -> Signal:
        # Implementation
        return Signal(...)
```

### R005: Canonical File Usage

**Problem**: Importing non-canonical files
```python
# ❌ Non-canonical imports
from .container_factories import create_backtest_container
from .enhanced_container import EnhancedContainer
```

**Solution**: Use canonical imports
```python
# ✅ Canonical imports
from ...containers.factory import get_global_factory
from ...containers.container import Container

factory = get_global_factory()
container = factory.compose_pattern('simple_backtest', config)
```

### R006: Workflow Pattern Architecture

**Problem**: Patterns without proper delegation
```python
# ❌ Pattern missing delegation info
self._patterns = {
    'simple_backtest': {
        'description': 'Simple backtest'
        # Missing container_pattern and communication_config!
    }
}
```

**Solution**: Follow standard pattern structure
```python
# ✅ Complete pattern definition
self._workflow_patterns = {
    'simple_backtest': {
        'description': 'Simple backtest workflow',
        'container_pattern': 'simple_backtest',    # → Container Factory
        'communication_config': [                 # → Communication Factory
            {'type': 'pipeline', 'containers': ['data', 'strategy', 'risk']}
        ]
    }
}
```

## Enforcement Levels

### Development (Local)
- **Pre-commit Hook**: Blocks commits with violations
- **Manual Validation**: Developer runs validation before push
- **IDE Integration**: Future integration with editors/IDEs

### CI/CD (Automated)
- **PR Validation**: Runs on all pull requests
- **Branch Protection**: Blocks merges with violations
- **Automated Comments**: Provides violation details in PRs
- **Report Generation**: Creates detailed compliance reports

### Release (Gated)
- **Release Validation**: Must pass before release creation
- **Compliance Certification**: Generates compliance certificate
- **Archive Reports**: Stores validation reports with releases

## Integration with Development Workflow

### For Developers

1. **Before Making Changes**:
   ```bash
   # Check current compliance
   python scripts/validate_architecture_compliance.py --verbose
   ```

2. **During Development**:
   - Follow patterns in CLAUDE.md and STYLE.md
   - Use canonical files instead of creating new ones
   - Apply composition over inheritance

3. **Before Committing**:
   ```bash
   # Pre-commit hook runs automatically
   git commit -m "Your message"
   ```

4. **If Violations Found**:
   - Fix violations using guidance above
   - Re-run validation to confirm fixes
   - Commit clean changes

### For Code Reviewers

1. **Check PR Status**: Ensure architecture compliance workflow passes
2. **Review Violations**: Check automated PR comments for issues
3. **Validate Approach**: Ensure changes follow pattern-based architecture
4. **Request Changes**: If violations exist, request fixes before approval

### For CI/CD Integration

1. **Workflow Triggers**: Runs on push to main/develop and all PRs
2. **Failure Handling**: Blocks merges if critical violations exist
3. **Report Storage**: Uploads compliance reports as artifacts
4. **Notification**: Posts detailed violation info in PR comments

## Customization and Extension

### Adding New Validation Rules

1. **Create Rule Method**:
   ```python
   def _validate_new_rule(self):
       """Rule N: Description of what this rule validates"""
       violations = []
       
       # Rule implementation
       
       result = ValidationResult(
           rule_id="R00N",
           rule_name="New Rule Name",
           passed=len(violations) == 0,
           violations=violations,
           severity="error"
       )
       self.results.append(result)
   ```

2. **Add to Validation Pipeline**:
   ```python
   def validate_all(self) -> bool:
       # Existing validations...
       self._validate_new_rule()  # Add here
       return self._generate_summary()
   ```

### Modifying Prohibited Patterns

Update the patterns in `ArchitectureValidator.__init__()`:
```python
self.prohibited_patterns = [
    r".*enhanced_.*\.py$",
    r".*improved_.*\.py$",
    # Add new patterns here
    r".*your_pattern_.*\.py$"
]
```

### Customizing CI/CD Workflows

Modify `.github/workflows/architecture-compliance.yml`:
- Add new jobs for specific checks
- Customize notification behavior
- Integrate with other tools

## Future Enhancements

### Planned Features
1. **IDE Integration**: Real-time validation in editors
2. **Auto-fix Capabilities**: Automatic fixing of certain violations
3. **Metrics Dashboard**: Track compliance trends over time
4. **Rule Configuration**: Configurable validation rules via YAML
5. **Custom Rule Framework**: Easy addition of project-specific rules

### Integration Opportunities
1. **Documentation Generation**: Auto-generate architecture docs
2. **Code Quality Tools**: Integration with linters and formatters
3. **Dependency Analysis**: Validate dependency patterns
4. **Performance Monitoring**: Track architectural impact on performance

## Troubleshooting

### Common Issues

#### False Positives
```bash
# Skip certain files/directories
python scripts/validate_architecture_compliance.py --exclude "tmp/**,venv/**"
```

#### Performance Issues
```bash
# Run specific rules only
python scripts/validate_architecture_compliance.py --rules "R001,R002"
```

#### CI/CD Failures
1. Check workflow logs for specific errors
2. Run validation locally to reproduce
3. Verify required files exist in repository
4. Check Python environment compatibility

### Getting Help

- **Documentation**: [CLAUDE.md](../../CLAUDE.md), [STYLE.md](../../STYLE.md)
- **Examples**: Look at existing canonical implementations
- **Community**: Create GitHub issue for questions
- **Debugging**: Use `--verbose` flag for detailed output

---

**Remember**: The goal is architectural consistency and evolution, not rigid constraints. When in doubt, follow the principle: "Can this be achieved by enhancing the existing canonical implementation?"
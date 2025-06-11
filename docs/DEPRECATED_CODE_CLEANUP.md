# Deprecated Code Cleanup Summary

## Overview
Cleaned up deprecated code from the ADMF-PC codebase to improve maintainability and reduce confusion.

## Changes Made

### 1. CLI Parser (`src/core/cli/parser.py`)
- **Removed deprecated flags**:
  - `--topology` flag (replaced by action flags like `--backtest`)
  - `--mode` flag (old execution mode override)
  - `--signal-log`, `--signal-output`, `--weights` (legacy mode arguments)
- **Cleaned up CLIArgs dataclass**:
  - Removed `topology`, `mode`, `signal_log`, `signal_output`, `weights` fields
  - Kept only the clean action flags and essential arguments

### 2. Main Entry Point (`main.py`)
- **Removed broken import**: Deleted TODO comment and broken `utils.logging` import
- **Simplified routing logic**: Removed handling for deprecated `args.topology`
- **Cleaner execution flow**: Now uses action flags or workflow/config-driven execution

### 3. Risk Protocols (`src/risk/protocols.py`)
- **Removed deprecated protocols**:
  - `RiskManager` class (replaced by `RiskPortfolioProtocol`)
  - `PortfolioManager` class (replaced by `PortfolioStateProtocol`)
- These were marked with TODO comments to migrate to newer protocols

### 4. Topology Builder (`src/core/coordinator/topology.py`)
- **Removed special case handling**: Eliminated hardcoded 'optimization' → 'backtest' fallback
- **Simplified pattern lookup**: Now just checks if pattern exists, no special cases

## Benefits

1. **Cleaner API**: Users now use intuitive action flags instead of generic `--topology`
2. **Less confusion**: No more deprecated flags showing up in help text
3. **Simpler codebase**: Removed special cases and backward compatibility cruft
4. **Better maintainability**: Less code paths to test and maintain

## Migration Guide for Users

### Old Commands → New Commands
```bash
# Old (deprecated)
python main.py --topology signal_generation --config config.yaml
python main.py --mode backtest --config config.yaml

# New (clean)
python main.py --signal-generation config.yaml
python main.py --backtest config.yaml
```

### Removed Options
- `--topology`: Use specific action flags (`--backtest`, `--signal-generation`, etc.)
- `--mode`: No longer needed, use action flags or config-driven execution
- `--signal-log`, `--signal-output`, `--weights`: Configure these in YAML files

## Next Steps

1. Update all example configs to use pure business logic (no topology/workflow fields)
2. Update documentation to reflect the new CLI interface
3. Consider adding shell completion support for the new action flags
4. Remove any remaining references to deprecated patterns in the codebase
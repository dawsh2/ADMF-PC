#!/bin/bash
#
# ADMF-PC Architecture Compliance Pre-commit Hook
#
# This hook prevents commits that violate the mandatory pattern-based
# architecture defined in CLAUDE.md and STYLE.md.
#
# Installation:
#   cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
#
# Or use symbolic link:
#   ln -sf ../../scripts/pre-commit-hook.sh .git/hooks/pre-commit

set -e

echo "🔍 ADMF-PC Architecture Compliance Check"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "CLAUDE.md" ] || [ ! -f "STYLE.md" ]; then
    echo "❌ Error: Not in ADMF-PC root directory"
    echo "This hook should be run from the project root"
    exit 1
fi

# Get list of files being committed
staged_files=$(git diff --cached --name-only --diff-filter=ACM)

if [ -z "$staged_files" ]; then
    echo "ℹ️  No files staged for commit"
    exit 0
fi

echo "📋 Checking staged files:"
echo "$staged_files" | sed 's/^/  • /'
echo

# 1. Check for prohibited file patterns
echo "🚨 Rule R001: No Enhanced/Improved Files"
prohibited_patterns=(
    ".*enhanced_.*\.py"
    ".*improved_.*\.py" 
    ".*advanced_.*\.py"
    ".*better_.*\.py"
    ".*optimized_.*\.py"
    ".*superior_.*\.py"
    ".*premium_.*\.py"
    ".*_v[0-9]+\.py"
    ".*_refactored\.py"
    ".*_new\.py"
)

violations_found=false

for pattern in "${prohibited_patterns[@]}"; do
    if echo "$staged_files" | grep -E "$pattern" | grep -v "^tmp/" | grep -v "^venv/"; then
        echo "❌ Found prohibited file pattern: $pattern"
        violations_found=true
    fi
done

if [ "$violations_found" = true ]; then
    echo
    echo "💡 Fix these violations:"
    echo "  • Use canonical files instead of enhanced/improved variants"
    echo "  • Move temporary files to tmp/ directory"
    echo "  • See STYLE.md for naming conventions"
    echo
    echo "❌ Commit rejected due to architecture violations"
    exit 1
fi

echo "✅ No prohibited file patterns found"

# 2. Check for files outside tmp/ that might be temporary
echo
echo "📁 Rule R002: Temporary Files in tmp/"

temp_file_patterns=(
    "debug.*\.py"
    "test.*\.py"
    "analyze.*\.py"
    ".*analysis.*\.py"
    ".*report.*\.py"
    ".*summary.*\.py"
    ".*status.*\.py"
    ".*audit.*\.py"
)

temp_violations=false

for pattern in "${temp_file_patterns[@]}"; do
    temp_files=$(echo "$staged_files" | grep -E "$pattern" | grep -v "^tmp/" | grep -v "^tests/" | grep -v "^scripts/")
    if [ ! -z "$temp_files" ]; then
        echo "⚠️  Potential temporary files outside tmp/:"
        echo "$temp_files" | sed 's/^/    • /'
        temp_violations=true
    fi
done

if [ "$temp_violations" = true ]; then
    echo
    echo "💡 Consider moving these files to tmp/ if they are temporary:"
    echo "  git mv filename.py tmp/category/filename.py"
    echo
    echo "⚠️  This is a warning, not blocking commit"
fi

# 3. Check for specific anti-patterns in code content
echo
echo "🏗️  Rule R003: Factory Separation"

factory_violations=false

# Check if any non-factory files create containers directly
for file in $staged_files; do
    if [[ "$file" == *.py ]] && [[ "$file" != *factory* ]] && [[ "$file" == src/core/coordinator/workflows/* ]]; then
        if git show :"$file" | grep -q "class.*Container.*:" 2>/dev/null; then
            echo "❌ $file: Workflow files should delegate to factories, not create containers"
            factory_violations=true
        fi
    fi
done

if [ "$factory_violations" = true ]; then
    echo
    echo "💡 Fix factory separation violations:"
    echo "  • Use WorkflowManager delegation to factories"
    echo "  • Don't create containers directly in workflow files"
    echo "  • See CLAUDE.md for pattern-based architecture"
    echo
    echo "❌ Commit rejected due to factory separation violations"
    exit 1
fi

echo "✅ Factory separation looks good"

# 4. Check for import of non-canonical files
echo
echo "📦 Rule R004: Canonical File Usage"

import_violations=false

# Check for imports of known non-canonical modules
non_canonical_imports=(
    "container_factories"
    "containers_pipeline"
    "enhanced_container"
    "improved_backtest"
)

for file in $staged_files; do
    if [[ "$file" == *.py ]]; then
        for bad_import in "${non_canonical_imports[@]}"; do
            if git show :"$file" | grep -q "import $bad_import\|from.*$bad_import" 2>/dev/null; then
                echo "❌ $file: Imports non-canonical module: $bad_import"
                import_violations=true
            fi
        done
    fi
done

if [ "$import_violations" = true ]; then
    echo
    echo "💡 Fix import violations:"
    echo "  • Update imports to use canonical files"
    echo "  • See workflow consolidation documentation"
    echo
    echo "❌ Commit rejected due to non-canonical imports"
    exit 1
fi

echo "✅ Canonical file usage looks good"

# 5. Run full validation if available
echo
echo "🔍 Running full architecture validation..."

if [ -x "scripts/validate_architecture_compliance.py" ]; then
    if ! python scripts/validate_architecture_compliance.py --root . >/dev/null 2>&1; then
        echo "⚠️  Full validation found additional issues"
        echo "💡 Run: python scripts/validate_architecture_compliance.py --verbose"
        echo "   This is a warning, not blocking commit"
    else
        echo "✅ Full validation passed"
    fi
else
    echo "ℹ️  Full validator not available"
fi

echo
echo "🎉 Architecture compliance check PASSED!"
echo "Your commit follows the mandatory pattern-based architecture."
echo

exit 0
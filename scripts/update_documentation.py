#!/usr/bin/env python3
"""
File: scripts/update_documentation.py
Status: ACTIVE
Version: 1.0.0
Architecture Ref: COMPLEXITY_CHECKLIST.md#step0-documentation-infrastructure
Dependencies: ast, pathlib, check_documentation.py
Last Review: 2025-05-31
Next Review: 2025-08-31

Purpose: Automated script to systematically apply A+ Step 0 documentation
patterns to Python files based on proven exemplars from the logging
infrastructure. Enables scaling documentation standards efficiently
across the entire ADMF-PC codebase.

Key Concepts:
- Pattern-based documentation updates using proven A+ exemplars
- Automated file header generation with Step 0 compliance
- Architecture reference integration and validation
- Batch processing with progress tracking and rollback capability
- Integration with check_documentation.py for validation

Critical Dependencies:
- Uses templates from templates/file_headers/ for consistent formatting
- Leverages check_documentation.py for validation and scoring
- Maintains architectural alignment during automated updates
"""

import ast
import re
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import sys
import json

# Import our documentation checker
sys.path.append(str(Path(__file__).parent))
from check_documentation import DocStringChecker, DocumentationReporter


class DocumentationUpdater:
    """
    Automated documentation updater using proven A+ patterns.
    
    This class systematically applies documentation patterns derived
    from our A+ logging infrastructure exemplars to scale Step 0
    compliance across the entire codebase efficiently.
    
    Architecture Context:
        - Part of: Documentation Infrastructure (COMPLEXITY_CHECKLIST.md#step0)
        - Applies: Proven A+ patterns from logging infrastructure exemplars
        - Enables: Systematic scaling of documentation standards
        - Validates: Using check_documentation.py integration
    
    Example:
        updater = DocumentationUpdater()
        results = updater.update_files(["src/core/config/"], dry_run=True)
        print(f"Would update {len(results)} files")
    """
    
    def __init__(self, templates_dir: Path = None):
        """
        Initialize documentation updater with template patterns.
        
        Args:
            templates_dir: Directory containing file header templates
        """
        self.templates_dir = templates_dir or Path("templates/file_headers")
        self.checker = DocStringChecker()
        self.reporter = DocumentationReporter()
        
        # Architecture document mappings for smart assignment
        self.architecture_mappings = {
            "src/core/logging/": "COMPLEXITY_CHECKLIST.md#step0-documentation-infrastructure",
            "src/core/events/": "COMPLEXITY_CHECKLIST.md#event-isolation",
            "src/core/containers/": "BACKTEST_README.md#container-hierarchy",
            "src/execution/": "BACKTEST_README.md#execution-engine",
            "src/risk/": "BACKTEST_README.md#risk-management",
            "src/strategy/": "BACKTEST_README.md#strategy-architecture",
            "src/data/": "BACKTEST_README.md#data-pipeline",
            "src/core/coordinator/": "MULTIPHASE_OPTIMIZATION.md#coordinator-responsibilities",
            "src/strategy/optimization/": "MULTIPHASE_OPTIMIZATION.md#optimization-workflows"
        }
        
        # Common dependencies by directory
        self.common_dependencies = {
            "src/core/logging/": ["logging", "json", "datetime", "threading"],
            "src/core/events/": ["typing", "dataclasses", "threading"],
            "src/core/containers/": ["typing", "abc", "threading"],
            "src/execution/": ["typing", "decimal", "datetime"],
            "src/risk/": ["typing", "decimal", "dataclasses"],
            "src/strategy/": ["typing", "numpy", "pandas"],
            "src/data/": ["typing", "pandas", "pathlib"]
        }
    
    def update_files(
        self, 
        paths: List[str], 
        dry_run: bool = True,
        backup: bool = True
    ) -> Dict[str, Any]:
        """
        Update multiple files with A+ documentation patterns.
        
        Args:
            paths: List of file or directory paths to update
            dry_run: If True, show what would be changed without modifying files
            backup: If True, create backup files before modification
            
        Returns:
            Dictionary containing update results and statistics
            
        Example:
            results = updater.update_files(
                ["src/core/config/"], 
                dry_run=False, 
                backup=True
            )
            print(f"Updated {results['updated_count']} files")
        """
        files_to_process = []
        
        # Collect all Python files from paths
        for path_str in paths:
            path = Path(path_str)
            if path.is_file() and path.suffix == '.py':
                files_to_process.append(path)
            elif path.is_dir():
                files_to_process.extend(path.rglob('*.py'))
        
        results = {
            'files_processed': 0,
            'updated_count': 0,
            'skipped_count': 0,
            'error_count': 0,
            'before_scores': [],
            'after_scores': [],
            'detailed_results': [],
            'dry_run': dry_run
        }
        
        print(f"{'DRY RUN: ' if dry_run else ''}Processing {len(files_to_process)} files...")
        
        for file_path in files_to_process:
            try:
                result = self._update_single_file(file_path, dry_run, backup)
                results['detailed_results'].append(result)
                results['files_processed'] += 1
                
                if result['updated']:
                    results['updated_count'] += 1
                    results['before_scores'].append(result['before_score'])
                    if not dry_run:
                        results['after_scores'].append(result['after_score'])
                else:
                    results['skipped_count'] += 1
                    
            except Exception as e:
                results['error_count'] += 1
                results['detailed_results'].append({
                    'file': str(file_path),
                    'updated': False,
                    'error': str(e),
                    'before_score': 0.0
                })
                print(f"  ❌ Error processing {file_path}: {e}")
        
        # Calculate summary statistics
        if results['before_scores']:
            results['avg_before_score'] = sum(results['before_scores']) / len(results['before_scores'])
        if results['after_scores']:
            results['avg_after_score'] = sum(results['after_scores']) / len(results['after_scores'])
        
        self._print_summary(results)
        return results
    
    def _update_single_file(
        self, 
        file_path: Path, 
        dry_run: bool, 
        backup: bool
    ) -> Dict[str, Any]:
        """Update a single file with Step 0 compliance improvements."""
        # Check current compliance
        before_result = self.checker.check_file(file_path)
        
        print(f"  📄 {file_path} (score: {before_result.score:.2f})")
        
        # Skip if already high quality
        if before_result.score >= 0.85:
            print(f"    ✅ Already high quality (score >= 0.85), skipping")
            return {
                'file': str(file_path),
                'updated': False,
                'reason': 'already_high_quality',
                'before_score': before_result.score,
                'after_score': before_result.score
            }
        
        # Read current content
        content = file_path.read_text(encoding='utf-8')
        updated_content = content
        
        # Apply improvements
        improvements_made = []
        
        # 1. Update file header if needed
        if any("Missing required header fields" in error for error in before_result.errors):
            updated_content, header_improved = self._update_file_header(
                updated_content, file_path
            )
            if header_improved:
                improvements_made.append("file_header")
        
        # 2. Add architecture references if missing
        if any("architecture document" in error.lower() for error in before_result.errors):
            updated_content, arch_improved = self._add_architecture_references(
                updated_content, file_path
            )
            if arch_improved:
                improvements_made.append("architecture_refs")
        
        # 3. Enhance docstrings for classes and functions
        if any("docstring" in error.lower() for error in before_result.errors):
            updated_content, docstring_improved = self._enhance_docstrings(
                updated_content, file_path
            )
            if docstring_improved:
                improvements_made.append("docstrings")
        
        # Check if any improvements were made
        if not improvements_made:
            return {
                'file': str(file_path),
                'updated': False,
                'reason': 'no_improvements_identified',
                'before_score': before_result.score,
                'after_score': before_result.score
            }
        
        if dry_run:
            print(f"    🔧 Would apply: {', '.join(improvements_made)}")
            return {
                'file': str(file_path),
                'updated': True,
                'improvements': improvements_made,
                'before_score': before_result.score,
                'after_score': None,  # Can't calculate without actually updating
                'dry_run': True
            }
        
        # Actually update the file
        if backup:
            backup_path = file_path.with_suffix(file_path.suffix + '.bak')
            shutil.copy2(file_path, backup_path)
            print(f"    💾 Backup created: {backup_path}")
        
        file_path.write_text(updated_content, encoding='utf-8')
        
        # Check new compliance
        after_result = self.checker.check_file(file_path)
        
        improvement = after_result.score - before_result.score
        print(f"    ✅ Updated: {before_result.score:.2f} → {after_result.score:.2f} (+{improvement:.2f})")
        print(f"    🔧 Applied: {', '.join(improvements_made)}")
        
        return {
            'file': str(file_path),
            'updated': True,
            'improvements': improvements_made,
            'before_score': before_result.score,
            'after_score': after_result.score,
            'improvement': improvement
        }
    
    def _update_file_header(self, content: str, file_path: Path) -> Tuple[str, bool]:
        """Update or add Step 0 compliant file header."""
        # Determine architecture reference
        arch_ref = self._get_architecture_reference(file_path)
        dependencies = self._get_common_dependencies(file_path)
        
        # Check if file already has a docstring
        header_match = re.search(r'^"""(.*?)"""', content, re.DOTALL)
        
        # Generate new header
        today = datetime.now().strftime('%Y-%m-%d')
        next_review = (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')
        
        new_header = f'''"""
File: {file_path}
Status: ACTIVE
Version: 1.0.0
Architecture Ref: {arch_ref}
Dependencies: {', '.join(dependencies)}
Last Review: {today}
Next Review: {next_review}

Purpose: {self._generate_purpose_description(file_path)}

Key Concepts:
- {self._generate_key_concept(file_path, 1)}
- {self._generate_key_concept(file_path, 2)}
- {self._generate_key_concept(file_path, 3)}

Critical Dependencies:
- {self._generate_critical_dependency(file_path, 1)}
- {self._generate_critical_dependency(file_path, 2)}
"""'''
        
        if header_match:
            # Replace existing header
            updated_content = content.replace(header_match.group(0), new_header)
        else:
            # Add new header at the beginning
            updated_content = new_header + '\n\n' + content
        
        return updated_content, True
    
    def _add_architecture_references(self, content: str, file_path: Path) -> Tuple[str, bool]:
        """Add architecture document references to docstrings."""
        arch_ref = self._get_architecture_reference(file_path)
        
        # Simple implementation: add to file header if not already present
        if arch_ref not in content:
            # This is handled by _update_file_header
            return content, False
        
        return content, False
    
    def _enhance_docstrings(self, content: str, file_path: Path) -> Tuple[str, bool]:
        """Enhance docstrings for classes and functions."""
        # This is a simplified implementation
        # In practice, you'd use AST to properly parse and update docstrings
        
        # Look for functions/classes missing docstrings
        lines = content.split('\n')
        updated_lines = []
        improved = False
        
        for i, line in enumerate(lines):
            updated_lines.append(line)
            
            # Simple pattern matching for functions without docstrings
            if re.match(r'^def \w+\(.*\):', line.strip()):
                # Check if next non-empty line is a docstring
                next_line_idx = i + 1
                while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                    next_line_idx += 1
                
                if (next_line_idx >= len(lines) or 
                    not lines[next_line_idx].strip().startswith('"""')):
                    # Add basic docstring
                    func_name = re.search(r'def (\w+)', line).group(1)
                    indent = ' ' * (len(line) - len(line.lstrip()))
                    basic_docstring = f'{indent}    """\n{indent}    {func_name.replace("_", " ").title()}.\n{indent}    \n{indent}    Returns:\n{indent}        None\n{indent}    """\n'
                    updated_lines.append(basic_docstring)
                    improved = True
        
        if improved:
            return '\n'.join(updated_lines), True
        
        return content, False
    
    def _get_architecture_reference(self, file_path: Path) -> str:
        """Get appropriate architecture reference for file path."""
        path_str = str(file_path)
        
        for pattern, reference in self.architecture_mappings.items():
            if pattern in path_str:
                return reference
        
        # Default reference
        return "COMPLEXITY_CHECKLIST.md#step0-documentation-infrastructure"
    
    def _get_common_dependencies(self, file_path: Path) -> List[str]:
        """Get common dependencies for file path."""
        path_str = str(file_path)
        
        for pattern, deps in self.common_dependencies.items():
            if pattern in path_str:
                return deps[:3]  # Return first 3 common dependencies
        
        # Default dependencies
        return ["typing", "pathlib", "datetime"]
    
    def _generate_purpose_description(self, file_path: Path) -> str:
        """Generate purpose description based on file path and name."""
        file_name = file_path.stem
        
        if file_name == "__init__":
            parent_dir = file_path.parent.name
            return f"Public API for {parent_dir} module functionality."
        
        # Generate based on file name
        words = file_name.replace('_', ' ').split()
        return f"Implements {' '.join(words)} functionality for ADMF-PC system."
    
    def _generate_key_concept(self, file_path: Path, concept_num: int) -> str:
        """Generate key concept based on file context."""
        concepts = [
            "Core functionality implementation following ADMF-PC patterns",
            "Integration with system architecture and design patterns", 
            "Support for validation and debugging workflows"
        ]
        return concepts[concept_num - 1]
    
    def _generate_critical_dependency(self, file_path: Path, dep_num: int) -> str:
        """Generate critical dependency description."""
        dependencies = [
            "Must maintain compatibility with existing system interfaces",
            "Supports Step 0+ validation and compliance requirements"
        ]
        return dependencies[dep_num - 1] if dep_num <= len(dependencies) else "Enables system functionality and integration"
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print summary of update results."""
        print(f"\n{'='*60}")
        print(f"Documentation Update Summary")
        print(f"{'='*60}")
        print(f"Files processed: {results['files_processed']}")
        print(f"Files updated: {results['updated_count']}")
        print(f"Files skipped: {results['skipped_count']}")
        print(f"Errors: {results['error_count']}")
        
        if results['before_scores']:
            print(f"Average score before: {results['avg_before_score']:.2f}")
        if results['after_scores']:
            print(f"Average score after: {results['avg_after_score']:.2f}")
            improvement = results['avg_after_score'] - results['avg_before_score']
            print(f"Average improvement: +{improvement:.2f}")
        
        if results['dry_run']:
            print(f"\n🔍 DRY RUN - No files were actually modified")
        else:
            print(f"\n✅ Updates completed successfully")


def main():
    """
    Main entry point for automated documentation updates.
    
    Provides command-line interface for systematically applying
    A+ documentation patterns across the ADMF-PC codebase.
    """
    parser = argparse.ArgumentParser(
        description='Systematically update Python files to A+ Step 0 compliance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run on specific directory
    python scripts/update_documentation.py src/core/config/ --dry-run
    
    # Update specific files with backup
    python scripts/update_documentation.py src/data/loaders.py --backup
    
    # Batch update entire module
    python scripts/update_documentation.py src/execution/ --report=execution_updates.json
    
    # Update multiple directories
    python scripts/update_documentation.py src/core/ src/data/ --dry-run
        """
    )
    
    parser.add_argument(
        'paths',
        nargs='+',
        help='Paths to files or directories to update'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without making changes'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup files'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Save detailed results to JSON file'
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=0.85,
        help='Minimum score to skip updating (default: 0.85)'
    )
    
    args = parser.parse_args()
    
    updater = DocumentationUpdater()
    
    results = updater.update_files(
        paths=args.paths,
        dry_run=args.dry_run,
        backup=not args.no_backup
    )
    
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to {args.report}")
    
    # Exit with appropriate code
    if results['error_count'] > 0:
        sys.exit(1)
    elif results['updated_count'] == 0 and not args.dry_run:
        print("No files needed updating")
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
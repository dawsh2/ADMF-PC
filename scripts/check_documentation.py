#!/usr/bin/env python3
"""
File: scripts/check_documentation.py
Status: ACTIVE
Version: 1.0.0
Architecture Ref: COMPLEXITY_CHECKLIST.md#step0-documentation-infrastructure
Dependencies: ast, pathlib, yaml
Last Review: 2025-01-31
Next Review: 2025-03-31

Purpose: Automated documentation validation tool that enforces Step 0
documentation standards from COMPLEXITY_CHECKLIST.md. Validates file
headers, docstrings, architecture references, and logging setup.

Key Concepts:
- File header validation with status tracking
- Architecture document reference validation
- Complete docstring enforcement with quality metrics
- Logging infrastructure verification
- Pre-commit integration support

Critical Dependencies:
- Must validate against COMPLEXITY_CHECKLIST.md Step 0 requirements
- Enforces architecture document references in all files
- Validates structured logging setup patterns
"""

import ast
import re
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import sys


class FileStatus(Enum):
    """Valid file status values for header validation."""
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED" 
    EXPERIMENTAL = "EXPERIMENTAL"
    REFACTORING = "REFACTORING"


@dataclass
class DocumentationResult:
    """
    Result of documentation validation check.
    
    This represents the outcome of validating a single file against
    Step 0 documentation requirements from COMPLEXITY_CHECKLIST.md.
    """
    filepath: str
    passed: bool
    errors: List[str]
    warnings: List[str]
    score: float  # 0.0 to 1.0
    checks_performed: Dict[str, bool]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class DocStringChecker:
    """
    Validates documentation completeness and architectural alignment.
    
    This checker enforces the documentation standards specified in
    COMPLEXITY_CHECKLIST.md Step 0, ensuring all code maintains
    references to architecture documents and proper logging setup.
    
    Architecture Context:
        - Validates against: COMPLEXITY_CHECKLIST.md#step0-documentation-infrastructure
        - Enforces: File headers with architecture references
        - Requires: Structured logging setup in all components
        - Validates: Docstring quality and completeness
    
    Example:
        checker = DocStringChecker()
        result = checker.check_file(Path("src/core/logging/structured.py"))
        print(f"File passed: {result.passed}, Score: {result.score}")
    """
    
    def __init__(self):
        """Initialize documentation checker with validation rules."""
        # Architecture documents that should be referenced
        self.required_architecture_refs = {
            'BACKTEST_README.md': [
                'container-hierarchy', 'event-flow', 'three-pattern-architecture',
                'event-isolation', 'container-lifecycle', 'backtest-container'
            ],
            'MULTIPHASE_OPTIMIZATION.md': [
                'phase-transitions', 'coordinator-responsibilities', 'data-flow',
                'file-based-communication'
            ],
            'WORKFLOW_COMPOSITION.md': [
                'workflow-patterns', 'phase-dependencies', 'composable-workflows',
                'building-blocks'
            ],
            'COMPLEXITY_CHECKLIST.md': [
                'logging', 'event-isolation', 'validation-framework',
                'step0-documentation-infrastructure', 'event-flow-validation'
            ]
        }
        
        # Required fields in file headers
        self.required_file_header_fields = [
            'File:', 'Status:', 'Version:', 'Architecture Ref:', 
            'Dependencies:', 'Last Review:', 'Purpose:'
        ]
        
        # Logging patterns that indicate proper setup
        self.logging_patterns = [
            r'ComponentLogger',
            r'StructuredLogger',
            r'ContainerLogger',
            r'logging\.getLogger',
            r'from.*logging.*import',
            r'log_event_flow',
            r'log_state_change',
            r'log_performance_metric',
            r'log_validation_result'
        ]
    
    def check_file(self, filepath: Path) -> DocumentationResult:
        """
        Comprehensive documentation validation for a single file.
        
        Validates according to COMPLEXITY_CHECKLIST.md Step 0 requirements:
        1. File header with status and architecture references
        2. Complete docstrings for all classes/functions
        3. Proper logging setup
        4. Architecture document references
        5. Code quality and consistency
        
        Args:
            filepath: Path to file being validated
            
        Returns:
            DocumentationResult with comprehensive validation details
            
        Raises:
            FileNotFoundError: If filepath does not exist
        """
        if not filepath.exists():
            return DocumentationResult(
                filepath=str(filepath),
                passed=False,
                errors=["File not found"],
                warnings=[],
                score=0.0,
                checks_performed={}
            )
        
        errors = []
        warnings = []
        checks = {}
        
        try:
            content = filepath.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return DocumentationResult(
                filepath=str(filepath),
                passed=False,
                errors=["Cannot read file - encoding issue"],
                warnings=[],
                score=0.0,
                checks_performed={}
            )
        
        # Check file header
        header_result = self._check_file_header(content)
        errors.extend(header_result['errors'])
        warnings.extend(header_result['warnings'])
        checks['file_header'] = len(header_result['errors']) == 0
        
        # Check Python-specific requirements
        if filepath.suffix == '.py':
            python_result = self._check_python_file(content, filepath)
            errors.extend(python_result['errors'])
            warnings.extend(python_result['warnings'])
            checks.update(python_result['checks'])
        
        # Check architecture references
        arch_result = self._check_architecture_references(content)
        errors.extend(arch_result['errors'])
        warnings.extend(arch_result['warnings'])
        checks['architecture_refs'] = len(arch_result['errors']) == 0
        
        # Check logging setup
        logging_result = self._check_logging_setup(content)
        errors.extend(logging_result['errors'])
        warnings.extend(logging_result['warnings'])
        checks['logging_setup'] = len(logging_result['errors']) == 0
        
        # Calculate score based on passed checks
        total_checks = len(checks)
        passed_checks = sum(1 for passed in checks.values() if passed)
        score = passed_checks / total_checks if total_checks > 0 else 0.0
        
        # Apply penalty for errors vs warnings
        error_penalty = min(0.5, len(errors) * 0.1)
        warning_penalty = min(0.2, len(warnings) * 0.02)
        score = max(0.0, score - error_penalty - warning_penalty)
        
        return DocumentationResult(
            filepath=str(filepath),
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            score=score,
            checks_performed=checks
        )
    
    def _check_file_header(self, content: str) -> Dict[str, List[str]]:
        """
        Validate file header documentation.
        
        Enforces COMPLEXITY_CHECKLIST.md Step 0 file header requirements
        including status tracking, architecture references, and metadata.
        """
        errors = []
        warnings = []
        
        # Extract file header (first docstring)
        header_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if not header_match:
            errors.append("Missing file header docstring")
            return {'errors': errors, 'warnings': warnings}
        
        header = header_match.group(1)
        
        # Check required fields
        missing_fields = []
        for field in self.required_file_header_fields:
            if field not in header:
                missing_fields.append(field)
        
        if missing_fields:
            errors.append(f"Missing required header fields: {', '.join(missing_fields)}")
        
        # Check status field specifically
        status_match = re.search(r'Status:\s*(\w+)', header)
        if status_match:
            status = status_match.group(1)
            try:
                FileStatus(status)
            except ValueError:
                errors.append(f"Invalid status '{status}'. Must be one of: {[s.value for s in FileStatus]}")
        else:
            errors.append("Missing Status field in header")
        
        # Check version format
        version_match = re.search(r'Version:\s*([\d.]+)', header)
        if not version_match:
            warnings.append("Version should follow semantic versioning (e.g., 1.0.0)")
        
        # Check review date format
        review_match = re.search(r'Last Review:\s*(\d{4}-\d{2}-\d{2})', header)
        if not review_match:
            warnings.append("Last Review date should be in YYYY-MM-DD format")
        else:
            # Check if review date is recent (within 3 months)
            try:
                review_date = datetime.strptime(review_match.group(1), '%Y-%m-%d')
                days_since_review = (datetime.now() - review_date).days
                if days_since_review > 90:
                    warnings.append(f"File not reviewed in {days_since_review} days - consider updating")
            except ValueError:
                warnings.append("Invalid date format in Last Review")
        
        # Check for next review date
        if 'Next Review:' not in header:
            warnings.append("Consider adding Next Review date for maintenance tracking")
        
        # Check for Purpose section
        if 'Purpose:' not in header:
            errors.append("Missing Purpose section in header")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _check_python_file(self, content: str, filepath: Path) -> Dict[str, Any]:
        """
        Validate Python-specific documentation requirements.
        
        Checks for proper docstrings, logging setup, and code quality
        according to COMPLEXITY_CHECKLIST.md standards.
        """
        errors = []
        warnings = []
        checks = {}
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            errors.append(f"Syntax error in Python file: {e}")
            return {'errors': errors, 'warnings': warnings, 'checks': {'syntax': False}}
        
        checks['syntax'] = True
        
        # Check class and function docstrings
        docstring_issues = self._check_docstrings(tree)
        errors.extend(docstring_issues['errors'])
        warnings.extend(docstring_issues['warnings'])
        checks['docstrings'] = len(docstring_issues['errors']) == 0
        
        # Check for imports and structure
        imports_result = self._check_imports_and_structure(tree, content)
        errors.extend(imports_result['errors'])
        warnings.extend(imports_result['warnings'])
        checks.update(imports_result['checks'])
        
        return {'errors': errors, 'warnings': warnings, 'checks': checks}
    
    def _check_docstrings(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Check quality and completeness of docstrings."""
        errors = []
        warnings = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                
                # Skip private methods and special methods
                if node.name.startswith('_') and not node.name.startswith('__'):
                    continue
                
                if not docstring:
                    if not node.name.startswith('_'):  # Public methods must have docstrings
                        errors.append(f"Missing docstring for {type(node).__name__[3:].lower()} '{node.name}'")
                else:
                    # Validate docstring quality
                    docstring_result = self._validate_docstring_quality(node, docstring)
                    errors.extend(docstring_result['errors'])
                    warnings.extend(docstring_result['warnings'])
        
        return {'errors': errors, 'warnings': warnings}
    
    def _check_imports_and_structure(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        """Check imports and overall file structure."""
        errors = []
        warnings = []
        checks = {}
        
        # Check for typing imports for better documentation
        has_typing = any(
            isinstance(node, ast.Import) and any(alias.name == 'typing' for alias in node.names)
            or isinstance(node, ast.ImportFrom) and node.module == 'typing'
            for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        )
        
        # Look for type hints in functions
        has_type_hints = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                node.returns is not None or 
                any(arg.annotation is not None for arg in node.args.args)
            )
            for node in ast.walk(tree)
        )
        
        if has_type_hints and not has_typing:
            warnings.append("Using type hints but missing typing imports")
        
        checks['type_annotations'] = has_type_hints
        
        # Check for proper class structure
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        for cls in classes:
            # Check if class has __init__ method
            init_methods = [
                node for node in cls.body 
                if isinstance(node, ast.FunctionDef) and node.name == '__init__'
            ]
            
            if not init_methods and len(cls.body) > 1:  # More than just docstring
                warnings.append(f"Class '{cls.name}' should have __init__ method")
        
        checks['class_structure'] = len(errors) == 0
        
        return {'errors': errors, 'warnings': warnings, 'checks': checks}
    
    def _check_architecture_references(self, content: str) -> Dict[str, List[str]]:
        """
        Validate references to architecture documents.
        
        Ensures files reference appropriate architecture documents
        as required by COMPLEXITY_CHECKLIST.md Step 0.
        """
        errors = []
        warnings = []
        
        # Check for at least one architecture document reference
        found_refs = []
        for doc in self.required_architecture_refs.keys():
            if doc in content:
                found_refs.append(doc)
        
        if not found_refs:
            errors.append(
                f"Must reference at least one architecture document. "
                f"Available: {list(self.required_architecture_refs.keys())}"
            )
        
        # Check for specific section references (e.g., BACKTEST_README.md#event-flow)
        section_pattern = r'(\w+\.md)#([\w-]+)'
        section_refs = re.findall(section_pattern, content)
        
        if section_refs:
            for doc, section in section_refs:
                if doc in self.required_architecture_refs:
                    valid_sections = self.required_architecture_refs[doc]
                    if section not in valid_sections:
                        warnings.append(
                            f"Section '{section}' not found in {doc}. "
                            f"Valid sections: {valid_sections}"
                        )
        else:
            if found_refs:  # Has doc refs but no section refs
                warnings.append(
                    "Consider using specific section references (e.g., BACKTEST_README.md#event-flow) "
                    "for better architectural alignment"
                )
        
        return {'errors': errors, 'warnings': warnings}
    
    def _check_logging_setup(self, content: str) -> Dict[str, List[str]]:
        """
        Validate proper logging setup.
        
        Checks for structured logging patterns required by
        COMPLEXITY_CHECKLIST.md ComponentLogger standards.
        """
        errors = []
        warnings = []
        
        # Check for logging imports/setup
        has_logging = any(re.search(pattern, content) for pattern in self.logging_patterns)
        
        if not has_logging:
            # Check if this is a simple utility file that might not need logging
            has_classes = 'class ' in content
            has_functions = 'def ' in content and not content.count('def ') == 1  # More than just one function
            
            if has_classes or has_functions:
                errors.append(
                    "Missing logging setup. Must import and configure ComponentLogger, "
                    "StructuredLogger, or similar for components with classes or multiple functions"
                )
        
        # Check for structured logging methods if it's a component file
        if 'class ' in content:
            structured_methods = [
                'log_event_flow', 'log_state_change', 
                'log_performance_metric', 'log_validation_result'
            ]
            
            missing_methods = [method for method in structured_methods if method not in content]
            
            if missing_methods and has_logging:
                warnings.append(
                    f"Consider adding structured logging methods for better observability: "
                    f"{missing_methods}"
                )
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_docstring_quality(self, node: ast.AST, docstring: str) -> Dict[str, List[str]]:
        """
        Validate docstring completeness and quality.
        
        Enforces Step 0 docstring standards including architecture
        references, parameter documentation, and usage examples.
        """
        errors = []
        warnings = []
        
        # Check minimum length
        if len(docstring.strip()) < 20:
            warnings.append(f"{type(node).__name__[3:].lower()} '{node.name}' has very short docstring")
        
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Check for Args/Parameters section
            if len(node.args.args) > 1:  # More than just 'self'
                if not any(keyword in docstring for keyword in ['Args:', 'Parameters:', ':param']):
                    errors.append(f"Function '{node.name}' has parameters but no Args/Parameters documentation")
            
            # Check for Returns section for non-private functions
            if not node.name.startswith('_'):
                if not any(keyword in docstring for keyword in ['Returns:', 'Return:', ':returns:', ':return:']):
                    warnings.append(f"Function '{node.name}' missing Returns documentation")
            
            # Check for example for public functions
            if not node.name.startswith('_') and len(docstring) > 100:
                if not any(keyword in docstring for keyword in ['Example:', 'Examples:', '>>>']):
                    warnings.append(f"Function '{node.name}' missing usage example")
        
        elif isinstance(node, ast.ClassDef):
            # Check for architecture references in class docstrings
            has_arch_ref = any(doc in docstring for doc in self.required_architecture_refs.keys())
            if not has_arch_ref:
                warnings.append(
                    f"Class '{node.name}' docstring should reference relevant architecture documents"
                )
            
            # Check for Architecture Context section
            if 'Architecture Context:' not in docstring and len(docstring) > 100:
                warnings.append(
                    f"Class '{node.name}' should include Architecture Context section "
                    "linking to design documents"
                )
        
        return {'errors': errors, 'warnings': warnings}


class DocumentationReporter:
    """
    Generates comprehensive reports on documentation compliance.
    
    This reporter creates detailed compliance reports suitable for
    CI/CD integration and team dashboards, following the validation
    framework from COMPLEXITY_CHECKLIST.md Step 0.
    
    Architecture Context:
        - Part of: Documentation Infrastructure (COMPLEXITY_CHECKLIST.md#step0)
        - Generates: Compliance reports for automation
        - Supports: CI/CD integration and team dashboards
    
    Example:
        reporter = DocumentationReporter()
        report = reporter.generate_report(validation_results)
        reporter.save_report(report, "compliance_report.json")
    """
    
    def generate_report(self, results: List[DocumentationResult]) -> Dict[str, Any]:
        """
        Generate comprehensive documentation compliance report.
        
        Args:
            results: List of validation results from DocStringChecker
            
        Returns:
            Comprehensive report dictionary suitable for JSON serialization
        """
        total_files = len(results)
        passed_files = sum(1 for r in results if r.passed)
        
        # Calculate overall score
        if total_files > 0:
            avg_score = sum(r.score for r in results) / total_files
        else:
            avg_score = 0.0
        
        # Group errors and warnings by type
        error_types = {}
        warning_types = {}
        
        for result in results:
            for error in result.errors:
                error_type = self._categorize_issue(error)
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for warning in result.warnings:
                warning_type = self._categorize_issue(warning)
                warning_types[warning_type] = warning_types.get(warning_type, 0) + 1
        
        # Calculate check-specific pass rates
        check_stats = {}
        if results:
            all_checks = set()
            for result in results:
                all_checks.update(result.checks_performed.keys())
            
            for check in all_checks:
                passed = sum(1 for r in results if r.checks_performed.get(check, False))
                check_stats[check] = {
                    'passed': passed,
                    'total': total_files,
                    'pass_rate': passed / total_files if total_files > 0 else 0.0
                }
        
        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'tool_version': '1.0.0',
                'validation_standard': 'COMPLEXITY_CHECKLIST.md Step 0'
            },
            'summary': {
                'total_files': total_files,
                'passed_files': passed_files,
                'failed_files': total_files - passed_files,
                'pass_rate': passed_files / total_files if total_files > 0 else 0.0,
                'average_score': avg_score,
                'grade': self._calculate_grade(avg_score)
            },
            'check_statistics': check_stats,
            'issue_breakdown': {
                'error_types': error_types,
                'warning_types': warning_types,
                'total_errors': sum(len(r.errors) for r in results),
                'total_warnings': sum(len(r.warnings) for r in results)
            },
            'detailed_results': [r.to_dict() for r in results],
            'recommendations': self._generate_recommendations(results, error_types, warning_types)
        }
    
    def _categorize_issue(self, issue: str) -> str:
        """Categorize issues for reporting."""
        if 'docstring' in issue.lower():
            return 'Docstring Issues'
        elif 'header' in issue.lower():
            return 'File Header Issues'
        elif 'architecture' in issue.lower() or 'reference' in issue.lower():
            return 'Architecture Reference Issues'
        elif 'logging' in issue.lower():
            return 'Logging Setup Issues'
        elif 'status' in issue.lower():
            return 'File Status Issues'
        else:
            return 'Other Issues'
    
    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from score."""
        if score >= 0.95:
            return 'A+'
        elif score >= 0.90:
            return 'A'
        elif score >= 0.85:
            return 'B+'
        elif score >= 0.80:
            return 'B'
        elif score >= 0.70:
            return 'C'
        else:
            return 'F'
    
    def _generate_recommendations(
        self, 
        results: List[DocumentationResult], 
        error_types: Dict[str, int],
        warning_types: Dict[str, int]
    ) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        total_files = len(results)
        failed_files = sum(1 for r in results if not r.passed)
        
        if failed_files > 0:
            fail_rate = failed_files / total_files
            if fail_rate > 0.5:
                recommendations.append(
                    "HIGH PRIORITY: Over 50% of files failing validation. "
                    "Consider running documentation update sprint before proceeding."
                )
        
        # Specific recommendations based on error types
        if error_types.get('File Header Issues', 0) > 0:
            recommendations.append(
                "Create file header templates and update existing files. "
                "See templates/file_headers/ for standardized formats."
            )
        
        if error_types.get('Docstring Issues', 0) > 0:
            recommendations.append(
                "Implement docstring templates for classes and functions. "
                "Focus on Args/Returns documentation for public APIs."
            )
        
        if error_types.get('Architecture Reference Issues', 0) > 0:
            recommendations.append(
                "Add architecture document references to file headers and docstrings. "
                "Link specific sections (e.g., BACKTEST_README.md#event-flow)."
            )
        
        if error_types.get('Logging Setup Issues', 0) > 0:
            recommendations.append(
                "Integrate ComponentLogger pattern throughout codebase. "
                "See src/core/logging/structured.py for implementation examples."
            )
        
        # Warning-based recommendations
        if warning_types.get('Architecture Reference Issues', 0) > 5:
            recommendations.append(
                "Consider creating architecture reference guide for developers. "
                "Document which files should reference which architecture documents."
            )
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filepath: Path) -> None:
        """Save report to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)


def main():
    """
    Main documentation checker entry point.
    
    Provides command-line interface for validating documentation
    compliance according to COMPLEXITY_CHECKLIST.md Step 0 standards.
    """
    parser = argparse.ArgumentParser(
        description='Check documentation compliance against Step 0 standards',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Check single file
    python scripts/check_documentation.py src/core/logging/structured.py
    
    # Check directory recursively
    python scripts/check_documentation.py src/core/
    
    # Strict mode (warnings as errors)
    python scripts/check_documentation.py src/ --strict
    
    # Generate detailed report
    python scripts/check_documentation.py src/ --report compliance_report.json
    
    # Check specific file types
    python scripts/check_documentation.py src/ --include="*.py"
        """
    )
    
    parser.add_argument(
        'paths', 
        nargs='+', 
        help='Paths to check (files or directories)'
    )
    parser.add_argument(
        '--strict', 
        action='store_true', 
        help='Treat warnings as errors'
    )
    parser.add_argument(
        '--report', 
        type=str, 
        help='Output detailed report to JSON file'
    )
    parser.add_argument(
        '--include', 
        type=str, 
        default='*.py',
        help='File pattern to include (default: *.py)'
    )
    parser.add_argument(
        '--min-score', 
        type=float, 
        default=0.8,
        help='Minimum average score required (default: 0.8)'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Show detailed output for each file'
    )
    
    args = parser.parse_args()
    
    # Initialize checker and reporter
    checker = DocStringChecker()
    reporter = DocumentationReporter()
    results = []
    
    # Collect files to check
    files_to_check = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file():
            files_to_check.append(path)
        elif path.is_dir():
            files_to_check.extend(path.rglob(args.include))
        else:
            print(f"Warning: Path not found: {path}")
    
    if not files_to_check:
        print("No files found to check")
        return 1
    
    # Check each file
    print(f"Checking {len(files_to_check)} files against Step 0 standards...")
    
    for file_path in sorted(files_to_check):
        result = checker.check_file(file_path)
        results.append(result)
        
        if args.verbose:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} {file_path} (score: {result.score:.2f})")
            
            if result.errors:
                for error in result.errors:
                    print(f"    ❌ {error}")
            
            if result.warnings and (args.verbose or not result.passed):
                for warning in result.warnings:
                    print(f"    ⚠️  {warning}")
    
    # Generate report
    report = reporter.generate_report(results)
    
    # Print summary
    print(f"\nDocumentation Compliance Report")
    print(f"{'='*50}")
    print(f"Files checked: {report['summary']['total_files']}")
    print(f"Passed: {report['summary']['passed_files']}")
    print(f"Failed: {report['summary']['failed_files']}")
    print(f"Pass rate: {report['summary']['pass_rate']:.1%}")
    print(f"Average score: {report['summary']['average_score']:.2f}")
    print(f"Grade: {report['summary']['grade']}")
    
    # Print top issues
    if report['issue_breakdown']['error_types']:
        print(f"\nTop Error Types:")
        for issue_type, count in sorted(
            report['issue_breakdown']['error_types'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]:
            print(f"  ❌ {issue_type}: {count}")
    
    if report['issue_breakdown']['warning_types']:
        print(f"\nTop Warning Types:")
        for issue_type, count in sorted(
            report['issue_breakdown']['warning_types'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]:
            print(f"  ⚠️  {issue_type}: {count}")
    
    # Print recommendations
    if report['recommendations']:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
    
    # Save detailed report if requested
    if args.report:
        report_path = Path(args.report)
        reporter.save_report(report, report_path)
        print(f"\nDetailed report saved to {report_path}")
    
    # Determine exit code
    exit_code = 0
    
    # Check minimum score requirement
    if report['summary']['average_score'] < args.min_score:
        print(f"\n❌ Average score {report['summary']['average_score']:.2f} below minimum {args.min_score}")
        exit_code = 1
    
    # Check for failures
    if report['summary']['failed_files'] > 0:
        print(f"\n❌ {report['summary']['failed_files']} files failed validation")
        exit_code = 1
    
    # Check strict mode
    if args.strict and report['issue_breakdown']['total_warnings'] > 0:
        print(f"\n❌ Strict mode: {report['issue_breakdown']['total_warnings']} warnings treated as errors")
        exit_code = 1
    
    if exit_code == 0:
        print(f"\n✅ All checks passed! Documentation meets Step 0 standards.")
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
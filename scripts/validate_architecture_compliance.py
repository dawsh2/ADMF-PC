#!/usr/bin/env python3
"""
Architecture Compliance Validator for ADMF-PC

This script validates that the codebase follows the mandatory pattern-based 
architecture defined in CLAUDE.md and docs/architecture/STANDARD_PATTERN_ARCHITECTURE.md.

Key validations:
1. No "enhanced", "improved", "advanced" etc. files
2. Single source of truth for workflow patterns
3. Factory separation (containers vs communication vs orchestration)
4. Protocol + Composition over inheritance
5. Canonical file usage

Usage:
    python scripts/validate_architecture_compliance.py [--fix] [--verbose]
    
    --fix:     Automatically fix certain violations where possible
    --verbose: Show detailed output for all checks
"""

import os
import sys
import re
import ast
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ValidationResult:
    """Result of a validation check"""
    rule_id: str
    rule_name: str
    passed: bool
    violations: List[str]
    fixable: bool = False
    severity: str = "error"  # error, warning, info


@dataclass
class ArchitectureViolation:
    """A specific architecture violation"""
    file_path: str
    line_number: int
    violation_type: str
    message: str
    suggestion: str = ""


class ArchitectureValidator:
    """Validates ADMF-PC architecture compliance"""
    
    def __init__(self, root_path: str, verbose: bool = False):
        self.root_path = Path(root_path)
        self.verbose = verbose
        self.violations: List[ArchitectureViolation] = []
        self.results: List[ValidationResult] = []
        
        # Prohibited file patterns from CLAUDE.md
        self.prohibited_patterns = [
            r".*enhanced_.*\.py$",
            r".*improved_.*\.py$", 
            r".*advanced_.*\.py$",
            r".*better_.*\.py$",
            r".*optimized_.*\.py$",
            r".*superior_.*\.py$",
            r".*premium_.*\.py$",
            r".*_v\d+\.py$",  # versioned files
            r".*_refactored\.py$",
            r".*_new\.py$",
            r".*_backup\.py$"
        ]
        
        # Canonical file mappings (from architecture analysis)
        self.canonical_files = {
            "container_factory": "src/core/containers/factory.py",
            "communication_factory": "src/core/communication/factory.py", 
            "workflow_manager": "src/core/coordinator/workflows/workflow_manager.py",
            "container": "src/core/containers/container.py",
            "event_bus": "src/core/events/event_bus.py"
        }
    
    def validate_all(self) -> bool:
        """Run all validation checks"""
        print("🔍 ADMF-PC Architecture Compliance Validation")
        print("=" * 50)
        
        # 1. Check for prohibited file names
        self._validate_no_enhanced_files()
        
        # 2. Check factory separation
        self._validate_factory_separation()
        
        # 3. Check single source of truth
        self._validate_single_source_of_truth()
        
        # 4. Check protocol usage
        self._validate_protocol_composition()
        
        # 5. Check canonical file usage
        self._validate_canonical_usage()
        
        # 6. Check workflow pattern definitions
        self._validate_workflow_patterns()
        
        # Generate summary
        return self._generate_summary()
    
    def _validate_no_enhanced_files(self):
        """Rule 1: No enhanced/improved/advanced files allowed"""
        violations = []
        
        for pattern in self.prohibited_patterns:
            for file_path in self.root_path.rglob("*.py"):
                rel_path = file_path.relative_to(self.root_path)
                # Skip venv, emacs temp files, and hidden files
                if ("venv" in str(rel_path) or rel_path.name.startswith('.#') or 
                    str(rel_path).startswith('.')):
                    continue
                if re.match(pattern, str(rel_path)):
                    violations.append(f"{rel_path} - Use canonical file instead")
        
        result = ValidationResult(
            rule_id="R001",
            rule_name="No Enhanced/Improved Files",
            passed=len(violations) == 0,
            violations=violations,
            fixable=False,  # Requires manual review
            severity="error"
        )
        self.results.append(result)
        
        if self.verbose or not result.passed:
            self._print_result(result)
    
    def _validate_factory_separation(self):
        """Rule 2: Factory responsibilities must be separated"""
        violations = []
        
        # Check that container factory doesn't do communication
        container_factory = self.root_path / "src/core/containers/factory.py"
        if container_factory.exists():
            content = container_factory.read_text()
            if "AdapterFactory" in content or "adapter" in content.lower():
                violations.append(f"{container_factory.relative_to(self.root_path)} - Container factory should not handle communication")
        
        # Check that communication factory doesn't create containers
        comm_factory = self.root_path / "src/core/communication/factory.py"
        if comm_factory.exists():
            content = comm_factory.read_text()
            if "ContainerFactory" in content or re.search(r"create.*container", content.lower()):
                violations.append(f"{comm_factory.relative_to(self.root_path)} - Communication factory should not create containers")
        
        # Check workflow patterns aren't mixed with container creation
        workflow_files = list((self.root_path / "src/core/coordinator/workflows").glob("*.py"))
        for workflow_file in workflow_files:
            if workflow_file.name in ["__init__.py"]:
                continue
                
            content = workflow_file.read_text()
            # Check for direct container instantiation instead of factory delegation
            if re.search(r"class.*Container.*:", content) and "factory" not in workflow_file.name:
                violations.append(f"{workflow_file.relative_to(self.root_path)} - Workflow files should delegate to factories, not create containers directly")
        
        result = ValidationResult(
            rule_id="R002", 
            rule_name="Factory Separation",
            passed=len(violations) == 0,
            violations=violations,
            severity="error"
        )
        self.results.append(result)
        
        if self.verbose or not result.passed:
            self._print_result(result)
    
    def _validate_single_source_of_truth(self):
        """Rule 3: Each workflow pattern defined exactly once"""
        violations = []
        pattern_definitions = {}
        
        # Find all workflow pattern definitions
        workflow_dir = self.root_path / "src/core/coordinator/workflows"
        if workflow_dir.exists():
            for py_file in workflow_dir.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                    
                content = py_file.read_text()
                
                # Look for pattern dictionaries
                if "_workflow_patterns" in content or "_patterns" in content:
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Assign):
                                for target in node.targets:
                                    if isinstance(target, ast.Name) and "pattern" in target.id.lower():
                                        if isinstance(node.value, ast.Dict):
                                            # Found pattern definition
                                            file_rel = py_file.relative_to(self.root_path)
                                            if target.id not in pattern_definitions:
                                                pattern_definitions[target.id] = []
                                            pattern_definitions[target.id].append(str(file_rel))
                    except SyntaxError:
                        # Skip files with syntax errors
                        continue
        
        # Check for duplicates
        for pattern_name, files in pattern_definitions.items():
            if len(files) > 1:
                violations.append(f"Pattern '{pattern_name}' defined in multiple files: {', '.join(files)}")
        
        result = ValidationResult(
            rule_id="R003",
            rule_name="Single Source of Truth",
            passed=len(violations) == 0,
            violations=violations,
            severity="error"
        )
        self.results.append(result)
        
        if self.verbose or not result.passed:
            self._print_result(result)
    
    def _validate_protocol_composition(self):
        """Rule 4: Protocol + Composition over inheritance"""
        violations = []
        
        for py_file in self.root_path.rglob("*.py"):
            if ("test" in str(py_file) or "__pycache__" in str(py_file) or 
                "venv" in str(py_file) or str(py_file).startswith('.') or
                py_file.name.startswith('.#')):  # Skip emacs temp files
                continue
                
            try:
                content = py_file.read_text()
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check for inheritance (bases should be empty or only protocols)
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                base_name = base.id
                                # Allow Protocol inheritance and specific exceptions
                                if (base_name not in ["Protocol", "ABC"] and 
                                    not base_name.endswith("Protocol") and
                                    base_name not in ["Enum", "IntEnum", "Exception", "BaseException"]):
                                    rel_path = py_file.relative_to(self.root_path)
                                    violations.append(f"{rel_path}:{node.lineno} - Class {node.name} inherits from {base_name} (use composition)")
                            elif isinstance(base, ast.Attribute):
                                # Handle module.ClassName inheritance
                                if hasattr(base, 'attr') and not base.attr.endswith("Protocol"):
                                    rel_path = py_file.relative_to(self.root_path)
                                    violations.append(f"{rel_path}:{node.lineno} - Class {node.name} uses inheritance (use composition)")
                
            except (SyntaxError, UnicodeDecodeError):
                # Skip files with syntax errors or encoding issues
                continue
        
        result = ValidationResult(
            rule_id="R004",
            rule_name="Protocol + Composition",
            passed=len(violations) == 0,
            violations=violations[:10],  # Limit output for readability
            severity="warning" if len(violations) < 5 else "error"
        )
        self.results.append(result)
        
        if self.verbose or not result.passed:
            self._print_result(result)
    
    def _validate_canonical_usage(self):
        """Rule 5: Code should use canonical implementations"""
        violations = []
        
        # Check imports point to canonical files
        for py_file in self.root_path.rglob("*.py"):
            if ("test" in str(py_file) or "__pycache__" in str(py_file) or 
                "venv" in str(py_file) or str(py_file).startswith('.') or
                py_file.name.startswith('.#')):  # Skip emacs temp files
                continue
                
            try:
                content = py_file.read_text()
                
                # Check for imports of non-canonical files
                for pattern in self.prohibited_patterns:
                    if re.search(rf"from.*{pattern[:-4]}.*import", content):  # Remove .py$ from pattern
                        rel_path = py_file.relative_to(self.root_path)
                        violations.append(f"{rel_path} - Imports non-canonical file matching {pattern}")
                        
                # Check for specific non-canonical imports we know about
                problematic_imports = [
                    "container_factories",
                    "containers_pipeline", 
                    "enhanced_container",
                    "improved_backtest"
                ]
                
                for imp in problematic_imports:
                    if f"import {imp}" in content or f"from .{imp}" in content:
                        rel_path = py_file.relative_to(self.root_path)
                        violations.append(f"{rel_path} - Imports non-canonical module: {imp}")
                
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        
        result = ValidationResult(
            rule_id="R005",
            rule_name="Canonical File Usage", 
            passed=len(violations) == 0,
            violations=violations,
            severity="error"
        )
        self.results.append(result)
        
        if self.verbose or not result.passed:
            self._print_result(result)
    
    def _validate_workflow_patterns(self):
        """Rule 6: Workflow patterns must follow standard architecture"""
        violations = []
        
        workflow_manager = self.root_path / "src/core/coordinator/workflows/workflow_manager.py"
        if workflow_manager.exists():
            content = workflow_manager.read_text()
            
            # Check that WorkflowManager is the authority for patterns
            if "_workflow_patterns" not in content:
                violations.append(f"{workflow_manager.relative_to(self.root_path)} - WorkflowManager should define _workflow_patterns")
                
            # Check pattern structure follows standard
            if "container_pattern" not in content:
                violations.append(f"{workflow_manager.relative_to(self.root_path)} - Patterns should specify 'container_pattern' for delegation")
                
            if "communication_config" not in content:
                violations.append(f"{workflow_manager.relative_to(self.root_path)} - Patterns should specify 'communication_config' for adapters")
        else:
            violations.append("Missing canonical WorkflowManager at src/core/coordinator/workflows/workflow_manager.py")
        
        result = ValidationResult(
            rule_id="R006",
            rule_name="Workflow Pattern Architecture",
            passed=len(violations) == 0,
            violations=violations,
            severity="error"
        )
        self.results.append(result)
        
        if self.verbose or not result.passed:
            self._print_result(result)
    
    def _print_result(self, result: ValidationResult):
        """Print validation result"""
        status = "✅ PASS" if result.passed else "❌ FAIL"
        severity_symbol = {"error": "🚨", "warning": "⚠️", "info": "ℹ️"}[result.severity]
        
        print(f"\n{severity_symbol} {result.rule_id}: {result.rule_name} - {status}")
        
        if not result.passed:
            for violation in result.violations:
                print(f"  • {violation}")
                
        if result.fixable and not result.passed:
            print(f"  💡 This violation can be auto-fixed with --fix")
    
    def _generate_summary(self) -> bool:
        """Generate validation summary"""
        total_rules = len(self.results)
        passed_rules = sum(1 for r in self.results if r.passed)
        failed_rules = total_rules - passed_rules
        
        errors = [r for r in self.results if not r.passed and r.severity == "error"]
        warnings = [r for r in self.results if not r.passed and r.severity == "warning"]
        
        print("\n" + "=" * 50)
        print("📋 VALIDATION SUMMARY")
        print("=" * 50)
        print(f"Rules Passed: {passed_rules}/{total_rules}")
        print(f"Errors: {len(errors)}")
        print(f"Warnings: {len(warnings)}")
        
        if errors:
            print("\n🚨 CRITICAL ERRORS:")
            for error in errors:
                print(f"  {error.rule_id}: {error.rule_name}")
                
        if warnings:
            print("\n⚠️  WARNINGS:")
            for warning in warnings:
                print(f"  {warning.rule_id}: {warning.rule_name}")
        
        if failed_rules == 0:
            print("\n🎉 Architecture compliance validation PASSED!")
            print("Your codebase follows the mandatory pattern-based architecture.")
        else:
            print("\n❌ Architecture compliance validation FAILED!")
            print("Please fix the violations above before proceeding.")
            print("\nFor guidance on fixing violations:")
            print("📖 See: STYLE.md")
            print("📖 See: CLAUDE.md")
        
        return failed_rules == 0
    
    def generate_report(self, output_path: str):
        """Generate detailed JSON report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_rules": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
                "errors": len([r for r in self.results if not r.passed and r.severity == "error"]),
                "warnings": len([r for r in self.results if not r.passed and r.severity == "warning"])
            },
            "results": [
                {
                    "rule_id": r.rule_id,
                    "rule_name": r.rule_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "violations": r.violations,
                    "fixable": r.fixable
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate ADMF-PC architecture compliance")
    parser.add_argument("--fix", action="store_true", help="Automatically fix certain violations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--report", help="Generate detailed JSON report to file")
    parser.add_argument("--root", default=".", help="Root path of the project")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.root):
        print(f"❌ Error: Root path {args.root} does not exist")
        sys.exit(1)
    
    validator = ArchitectureValidator(args.root, args.verbose)
    
    try:
        success = validator.validate_all()
        
        if args.report:
            validator.generate_report(args.report)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
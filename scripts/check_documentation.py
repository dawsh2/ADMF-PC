#!/usr/bin/env python3
"""
Documentation Checker for ADMF-PC

Validates that all Python files have proper documentation according to standards.
Checks for:
- Module headers with architecture info
- Class docstrings with required sections
- Function documentation
- Logging setup
"""

import os
import re
import sys
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class DocumentationChecker:
    """Check documentation compliance across codebase"""
    
    def __init__(self, root_path: str = "src"):
        self.root_path = Path(root_path)
        self.errors = []
        self.warnings = []
        self.stats = {
            'files_checked': 0,
            'modules_compliant': 0,
            'classes_compliant': 0,
            'functions_compliant': 0,
            'logging_setup': 0
        }
    
    def check_module_header(self, file_path: Path, content: str) -> bool:
        """Check if module has proper header documentation"""
        # Check for module docstring
        if not content.strip().startswith('"""'):
            self.errors.append(f"{file_path}: Missing module docstring")
            return False
        
        # Extract docstring
        docstring_match = re.match(r'"""(.*?)"""', content, re.DOTALL)
        if not docstring_match:
            self.errors.append(f"{file_path}: Invalid module docstring format")
            return False
        
        docstring = docstring_match.group(1)
        
        # Check required sections
        required_sections = ['Module:', 'Location:', 'Architecture:']
        missing = []
        for section in required_sections:
            if section not in docstring:
                missing.append(section)
        
        if missing:
            self.errors.append(f"{file_path}: Missing sections in module docstring: {missing}")
            return False
        
        # Check architecture subsections
        arch_subsections = ['Container:', 'Protocol:', 'Events:']
        if 'Architecture:' in docstring:
            arch_part = docstring.split('Architecture:')[1].split('Dependencies:')[0]
            missing_arch = [s for s in arch_subsections if s not in arch_part]
            if missing_arch:
                self.warnings.append(f"{file_path}: Missing architecture details: {missing_arch}")
        
        return True
    
    def check_class_documentation(self, class_node: ast.ClassDef, file_path: Path) -> bool:
        """Check if class has proper documentation"""
        docstring = ast.get_docstring(class_node)
        
        if not docstring:
            self.errors.append(f"{file_path}:{class_node.lineno} Class '{class_node.name}' missing docstring")
            return False
        
        # Check for required sections in class docstring
        required = ['Architecture:', 'Configuration:', 'Events:']
        missing = [s for s in required if s not in docstring]
        
        if missing:
            self.warnings.append(
                f"{file_path}:{class_node.lineno} Class '{class_node.name}' "
                f"missing documentation sections: {missing}"
            )
        
        # Check for example
        if 'Example:' not in docstring and '>>>' not in docstring:
            self.warnings.append(
                f"{file_path}:{class_node.lineno} Class '{class_node.name}' "
                "missing usage example"
            )
        
        return len(missing) == 0
    
    def check_function_documentation(self, func_node: ast.FunctionDef, file_path: Path) -> bool:
        """Check if function has proper documentation"""
        # Skip private methods and special methods
        if func_node.name.startswith('_') and not func_node.name.startswith('__'):
            return True
        
        docstring = ast.get_docstring(func_node)
        
        if not docstring:
            self.warnings.append(
                f"{file_path}:{func_node.lineno} Function '{func_node.name}' missing docstring"
            )
            return False
        
        # Check for Args section if function has parameters
        if func_node.args.args and 'Args:' not in docstring:
            self.warnings.append(
                f"{file_path}:{func_node.lineno} Function '{func_node.name}' "
                "has parameters but missing Args section"
            )
        
        # Check for Returns section if not __init__
        if func_node.name != '__init__' and 'Returns:' not in docstring:
            # Check if function has return annotation or return statements
            has_return = any(isinstance(node, ast.Return) for node in ast.walk(func_node))
            if has_return:
                self.warnings.append(
                    f"{file_path}:{func_node.lineno} Function '{func_node.name}' "
                    "returns value but missing Returns section"
                )
        
        return True
    
    def check_logging_setup(self, tree: ast.AST, file_path: Path) -> bool:
        """Check if file has proper logging setup"""
        # Look for ComponentLogger import and usage
        has_import = False
        has_logger_init = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and 'logging' in node.module:
                    has_import = True
            
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and target.attr == 'logger':
                        has_logger_init = True
        
        # Classes should have logging
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        if classes and not has_logger_init:
            self.warnings.append(f"{file_path}: Has classes but no logger initialization")
            return False
        
        return True
    
    def check_file(self, file_path: Path) -> None:
        """Check a single Python file"""
        self.stats['files_checked'] += 1
        
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
        except Exception as e:
            self.errors.append(f"{file_path}: Failed to parse: {e}")
            return
        
        # Check module header
        if self.check_module_header(file_path, content):
            self.stats['modules_compliant'] += 1
        
        # Check all classes
        classes_compliant = True
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not self.check_class_documentation(node, file_path):
                    classes_compliant = False
        
        if classes_compliant:
            self.stats['classes_compliant'] += 1
        
        # Check all functions
        functions_compliant = True
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not self.check_function_documentation(node, file_path):
                    functions_compliant = False
        
        if functions_compliant:
            self.stats['functions_compliant'] += 1
        
        # Check logging setup
        if self.check_logging_setup(tree, file_path):
            self.stats['logging_setup'] += 1
    
    def check_directory(self, directory: Path) -> None:
        """Recursively check all Python files in directory"""
        for file_path in directory.rglob("*.py"):
            # Skip test files and __pycache__
            if '__pycache__' in str(file_path) or 'test_' in file_path.name:
                continue
            
            self.check_file(file_path)
    
    def print_report(self) -> None:
        """Print documentation compliance report"""
        print("\n" + "="*60)
        print("ADMF-PC Documentation Compliance Report")
        print("="*60)
        
        print(f"\nFiles checked: {self.stats['files_checked']}")
        print(f"Modules with compliant headers: {self.stats['modules_compliant']}")
        print(f"Files with all classes documented: {self.stats['classes_compliant']}")
        print(f"Files with all functions documented: {self.stats['functions_compliant']}")
        print(f"Files with logging setup: {self.stats['logging_setup']}")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors[:10]:  # Show first 10
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:10]:  # Show first 10
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")
        
        # Calculate compliance score
        if self.stats['files_checked'] > 0:
            module_score = self.stats['modules_compliant'] / self.stats['files_checked']
            overall_score = (
                module_score * 0.4 +  # Module headers are important
                (self.stats['classes_compliant'] / self.stats['files_checked']) * 0.3 +
                (self.stats['functions_compliant'] / self.stats['files_checked']) * 0.2 +
                (self.stats['logging_setup'] / self.stats['files_checked']) * 0.1
            )
            
            print(f"\n📊 Overall Compliance Score: {overall_score:.1%}")
            
            if overall_score >= 0.9:
                print("✅ Excellent documentation compliance!")
            elif overall_score >= 0.7:
                print("👍 Good documentation, some improvements needed")
            elif overall_score >= 0.5:
                print("⚠️  Documentation needs significant improvement")
            else:
                print("❌ Poor documentation compliance, major work needed")
    
    def run(self) -> int:
        """Run the documentation check"""
        if not self.root_path.exists():
            print(f"Error: Path {self.root_path} does not exist")
            return 1
        
        print(f"Checking documentation in {self.root_path}...")
        self.check_directory(self.root_path)
        self.print_report()
        
        # Return non-zero if there are errors
        return 1 if self.errors else 0


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check ADMF-PC documentation compliance")
    parser.add_argument(
        "path",
        nargs="?",
        default="src",
        help="Path to check (default: src)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    
    args = parser.parse_args()
    
    checker = DocumentationChecker(args.path)
    exit_code = checker.run()
    
    if args.strict and checker.warnings:
        exit_code = 1
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
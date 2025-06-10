#!/usr/bin/env python3
"""
Import checker utility for ADMF-PC.

This tool helps identify and fix import issues in the rewritten codebase
by testing imports systematically and reporting specific problems.
"""

import sys
import os
import importlib
import traceback
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class ImportChecker:
    """Check imports systematically to identify issues."""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or "src")
        self.results = {}
        self.failed_imports = []
        self.successful_imports = []
    
    def check_module(self, module_path: str) -> Dict[str, any]:
        """Check if a module can be imported."""
        result = {
            'module': module_path,
            'success': False,
            'error': None,
            'error_type': None,
            'traceback': None
        }
        
        try:
            # Clear module from cache if it exists
            if module_path in sys.modules:
                del sys.modules[module_path]
            
            # Try to import
            module = importlib.import_module(module_path)
            result['success'] = True
            result['module_object'] = module
            self.successful_imports.append(module_path)
            
        except Exception as e:
            result['error'] = str(e)
            result['error_type'] = type(e).__name__
            result['traceback'] = traceback.format_exc()
            self.failed_imports.append(module_path)
        
        self.results[module_path] = result
        return result
    
    def check_core_modules(self) -> Dict[str, Dict]:
        """Check all core ADMF-PC modules."""
        core_modules = [
            'src.core.events',
            'src.core.events.bus',
            'src.core.events.types',
            'src.core.events.protocols',
            'src.core.events.barriers',
            'src.core.containers',
            'src.core.containers.container',
            'src.core.containers.protocols',
            'src.core.containers.factory',
            'src.core.containers.types',
            'src.core.coordinator',
            'src.core.coordinator.coordinator',
            'src.core.coordinator.topology',
        ]
        
        print("Checking core modules...")
        for module in core_modules:
            print(f"  Testing {module}...", end=" ")
            result = self.check_module(module)
            if result['success']:
                print("✅ OK")
            else:
                print(f"❌ FAILED: {result['error_type']}")
        
        return self.results
    
    def check_specific_imports(self, imports: List[str]) -> Dict[str, Dict]:
        """Check specific import statements."""
        print("\nChecking specific imports...")
        
        for import_stmt in imports:
            print(f"  Testing: {import_stmt}...", end=" ")
            
            try:
                # Execute the import statement
                exec(import_stmt)
                print("✅ OK")
                self.successful_imports.append(import_stmt)
                
            except Exception as e:
                print(f"❌ FAILED: {type(e).__name__}: {e}")
                self.failed_imports.append(import_stmt)
                self.results[import_stmt] = {
                    'module': import_stmt,
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc()
                }
        
        return self.results
    
    def analyze_import_error(self, module_path: str) -> Dict[str, any]:
        """Analyze a specific import error in detail."""
        if module_path not in self.results:
            self.check_module(module_path)
        
        result = self.results[module_path]
        
        if result['success']:
            return {'analysis': 'Module imports successfully'}
        
        error_analysis = {
            'module': module_path,
            'error_type': result['error_type'],
            'error_message': result['error'],
            'suggestions': []
        }
        
        error_msg = result['error'].lower()
        
        # Analyze common error types
        if 'no module named' in error_msg:
            error_analysis['problem'] = 'Missing module'
            error_analysis['suggestions'].extend([
                'Check if file exists at expected path',
                'Verify __init__.py files exist in package directories',
                'Check for typos in module name'
            ])
        
        elif 'cannot import name' in error_msg:
            error_analysis['problem'] = 'Missing symbol in module'
            error_analysis['suggestions'].extend([
                'Check if the imported symbol exists in the target module',
                'Verify the symbol is exported in __all__',
                'Check for typos in symbol name'
            ])
        
        elif 'circular import' in error_msg or 'partially initialized module' in error_msg:
            error_analysis['problem'] = 'Circular import'
            error_analysis['suggestions'].extend([
                'Move imports inside functions',
                'Use TYPE_CHECKING imports',
                'Reorganize module dependencies'
            ])
        
        else:
            error_analysis['problem'] = 'Other import error'
            error_analysis['suggestions'].append('Check the full traceback for details')
        
        return error_analysis
    
    def generate_report(self) -> str:
        """Generate a comprehensive import report."""
        report = []
        report.append("=" * 60)
        report.append("ADMF-PC IMPORT ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Summary
        total = len(self.results)
        successful = len(self.successful_imports)
        failed = len(self.failed_imports)
        
        report.append(f"\nSUMMARY:")
        report.append(f"  Total modules tested: {total}")
        report.append(f"  Successful imports: {successful}")
        report.append(f"  Failed imports: {failed}")
        report.append(f"  Success rate: {(successful/total*100):.1f}%" if total > 0 else "  Success rate: N/A")
        
        # Successful imports
        if self.successful_imports:
            report.append(f"\n✅ SUCCESSFUL IMPORTS ({len(self.successful_imports)}):")
            for module in self.successful_imports:
                report.append(f"  - {module}")
        
        # Failed imports with analysis
        if self.failed_imports:
            report.append(f"\n❌ FAILED IMPORTS ({len(self.failed_imports)}):")
            for module in self.failed_imports:
                if module in self.results:
                    result = self.results[module]
                    report.append(f"\n  {module}:")
                    report.append(f"    Error: {result['error_type']}: {result['error']}")
                    
                    # Add analysis
                    analysis = self.analyze_import_error(module)
                    if 'problem' in analysis:
                        report.append(f"    Problem: {analysis['problem']}")
                    if analysis['suggestions']:
                        report.append(f"    Suggestions:")
                        for suggestion in analysis['suggestions']:
                            report.append(f"      - {suggestion}")
        
        # Next steps
        report.append(f"\n📋 NEXT STEPS:")
        if self.failed_imports:
            report.append("  1. Fix the failed imports starting with core modules")
            report.append("  2. Check file existence and __init__.py files")
            report.append("  3. Resolve circular import issues")
            report.append("  4. Re-run this check after fixes")
        else:
            report.append("  ✅ All imports working! Ready for integration testing.")
        
        return "\n".join(report)
    
    def save_report(self, filename: str = "import_analysis.txt"):
        """Save report to file."""
        report = self.generate_report()
        with open(filename, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {filename}")


def main():
    """Main function to run import checks."""
    print("ADMF-PC Import Checker")
    print("=" * 30)
    
    # Change to project directory
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    
    checker = ImportChecker()
    
    # Check core modules
    checker.check_core_modules()
    
    # Check specific import patterns used in tests
    test_imports = [
        "from src.core.events import EventBus, Event, EventType",
        "from src.core.containers import Container, ContainerConfig, ContainerRole",
        "from src.core.events import create_market_event, create_signal_event",
        "from src.core.containers import ContainerProtocol, ContainerComponent",
        "from src.core.events.barriers import BarrierProtocol, create_standard_barriers",
    ]
    
    checker.check_specific_imports(test_imports)
    
    # Generate and print report
    print("\n" + checker.generate_report())
    
    # Save report
    checker.save_report("tests/import_analysis.txt")
    
    # Return exit code
    return 0 if not checker.failed_imports else 1


if __name__ == "__main__":
    sys.exit(main())
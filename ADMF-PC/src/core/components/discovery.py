"""
Component discovery system for ADMF-PC.

This module provides automatic discovery of components in the codebase,
scanning for classes that implement the required protocols.
"""

from __future__ import annotations
from typing import List, Set, Type, Any, Optional, Dict, Callable
import importlib
import inspect
import pkgutil
from pathlib import Path
import logging
import ast

from .protocols import Component, detect_capabilities
from .registry import ComponentRegistry, get_registry


logger = logging.getLogger(__name__)


class ComponentScanner:
    """
    Scans modules and packages for protocol-based components.
    
    This scanner can automatically discover components that implement
    the Component protocol and register them with the registry.
    """
    
    def __init__(self, registry: Optional[ComponentRegistry] = None):
        """
        Initialize the scanner.
        
        Args:
            registry: Registry to register components with
        """
        self.registry = registry or get_registry()
        self._scanned_modules: Set[str] = set()
        
        # Filters to determine which classes to register
        self._filters: List[Callable[[Type[Any]], bool]] = []
        
        # Add default filter - must implement Component protocol
        self.add_filter(lambda cls: hasattr(cls, 'component_id'))
        
        logger.debug("ComponentScanner initialized")
    
    def scan_package(
        self,
        package_path: Union[str, Path],
        recursive: bool = True,
        prefix: str = ""
    ) -> List[str]:
        """
        Scan a package for components.
        
        Args:
            package_path: Path to the package to scan
            recursive: Whether to scan subpackages
            prefix: Module prefix for imports
            
        Returns:
            List of registered component names
        """
        package_path = Path(package_path)
        if not package_path.exists():
            raise ValueError(f"Package path does not exist: {package_path}")
        
        registered = []
        
        # Scan Python files in the package
        pattern = "**/*.py" if recursive else "*.py"
        for py_file in package_path.glob(pattern):
            if py_file.name.startswith('_'):
                continue
                
            # Convert file path to module name
            relative_path = py_file.relative_to(package_path)
            module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
            module_name = '.'.join(module_parts)
            
            if prefix:
                module_name = f"{prefix}.{module_name}"
            
            # Scan the module
            try:
                names = self.scan_module(module_name)
                registered.extend(names)
            except Exception as e:
                logger.warning(f"Failed to scan module {module_name}: {e}")
        
        return registered
    
    def scan_module(self, module_name: str) -> List[str]:
        """
        Scan a module for components.
        
        Args:
            module_name: Fully qualified module name
            
        Returns:
            List of registered component names
        """
        if module_name in self._scanned_modules:
            logger.debug(f"Module {module_name} already scanned")
            return []
        
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            logger.error(f"Failed to import module {module_name}: {e}")
            return []
        
        self._scanned_modules.add(module_name)
        registered = []
        
        # Scan module members
        for name, obj in inspect.getmembers(module):
            if name.startswith('_'):
                continue
                
            # Check if it's a class defined in this module
            if not inspect.isclass(obj):
                continue
                
            if obj.__module__ != module_name:
                continue
            
            # Check if it passes filters
            if not self._should_register(obj):
                continue
            
            # Register the component
            try:
                component_name = self._get_component_name(obj)
                self.registry.register(
                    obj,
                    name=component_name,
                    override=False
                )
                registered.append(component_name)
                logger.info(f"Discovered component: {component_name}")
            except Exception as e:
                logger.warning(f"Failed to register {name}: {e}")
        
        return registered
    
    def scan_file(self, file_path: Union[str, Path]) -> List[str]:
        """
        Scan a single Python file for components.
        
        This uses AST parsing to find components without importing,
        useful for scanning files that might have import issues.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            List of component class names found (not registered)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        
        finder = ComponentFinder()
        finder.visit(tree)
        
        return finder.components
    
    def add_filter(self, filter_func: Callable[[Type[Any]], bool]) -> None:
        """
        Add a filter function for component discovery.
        
        Args:
            filter_func: Function that returns True if component should be registered
        """
        self._filters.append(filter_func)
    
    def clear_filters(self) -> None:
        """Clear all filters except the default."""
        self._filters = [self._filters[0]]  # Keep default filter
    
    # Private methods
    
    def _should_register(self, cls: Type[Any]) -> bool:
        """Check if a class should be registered."""
        for filter_func in self._filters:
            if not filter_func(cls):
                return False
        return True
    
    def _get_component_name(self, cls: Type[Any]) -> str:
        """Get the name to register a component under."""
        # Check for explicit name attribute
        if hasattr(cls, '__component_name__'):
            return cls.__component_name__
        
        # Use class name
        return cls.__name__


class ComponentFinder(ast.NodeVisitor):
    """AST visitor to find component classes without importing."""
    
    def __init__(self):
        self.components: List[str] = []
        self._current_class: Optional[str] = None
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        old_class = self._current_class
        self._current_class = node.name
        
        # Check if class might be a component
        has_component_id = False
        
        # Look for component_id property
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name == 'component_id':
                    # Check if it's a property
                    for decorator in item.decorator_list:
                        if isinstance(decorator, ast.Name) and decorator.id == 'property':
                            has_component_id = True
                            break
            
            # Also check for direct assignment
            elif isinstance(item, ast.AnnAssign) or isinstance(item, ast.Assign):
                if isinstance(item.target, ast.Name) and item.target.id == 'component_id':
                    has_component_id = True
        
        if has_component_id:
            self.components.append(node.name)
        
        # Continue visiting
        self.generic_visit(node)
        self._current_class = old_class


def discover_components(
    paths: Union[str, Path, List[Union[str, Path]]],
    recursive: bool = True,
    register: bool = True
) -> Dict[str, List[str]]:
    """
    Discover components in the specified paths.
    
    Args:
        paths: Path(s) to scan for components
        recursive: Whether to scan recursively
        register: Whether to register found components
        
    Returns:
        Dictionary mapping paths to lists of discovered component names
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    
    scanner = ComponentScanner()
    results = {}
    
    for path in paths:
        path = Path(path)
        
        if path.is_file():
            # Scan single file
            if register:
                logger.warning(f"Cannot auto-register from file scan: {path}")
            components = scanner.scan_file(path)
        elif path.is_dir():
            # Scan package
            if register:
                components = scanner.scan_package(path, recursive=recursive)
            else:
                # Just find components without registering
                components = []
                for py_file in path.glob("**/*.py" if recursive else "*.py"):
                    components.extend(scanner.scan_file(py_file))
        else:
            logger.warning(f"Path does not exist: {path}")
            continue
        
        results[str(path)] = components
    
    return results


# Auto-discovery on import (optional)
def auto_discover_components(package_name: str = "admf_pc.components") -> None:
    """
    Automatically discover and register components from a package.
    
    This is typically called once during application startup.
    
    Args:
        package_name: The package to scan for components
    """
    try:
        package = importlib.import_module(package_name)
        package_path = Path(package.__file__).parent
        
        scanner = ComponentScanner()
        components = scanner.scan_package(package_path, recursive=True)
        
        logger.info(f"Auto-discovered {len(components)} components from {package_name}")
    except Exception as e:
        logger.warning(f"Auto-discovery failed for {package_name}: {e}")
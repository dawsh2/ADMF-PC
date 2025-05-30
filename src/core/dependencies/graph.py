"""
Dependency graph for protocol-based components in ADMF-PC.

This module provides dependency tracking and resolution for components,
ensuring correct initialization order and detecting circular dependencies.
"""

from __future__ import annotations
from typing import Dict, Set, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

from ..components.protocols import Component, detect_capabilities


logger = logging.getLogger(__name__)


@dataclass
class ComponentNode:
    """Represents a component in the dependency graph."""
    
    name: str
    component_type: Optional[type] = None
    instance: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    
    @property
    def is_resolved(self) -> bool:
        """Check if component has been instantiated."""
        return self.instance is not None
    
    @property
    def capabilities(self) -> List[str]:
        """Get component capabilities."""
        if self.instance:
            return detect_capabilities(self.instance)
        elif self.component_type:
            return detect_capabilities(self.component_type)
        return []


class DependencyGraph:
    """
    Manages component dependencies for containerized execution.
    
    This graph tracks dependencies between components, detects cycles,
    and provides initialization order for proper component startup.
    """
    
    def __init__(self, container_id: Optional[str] = None):
        """
        Initialize the dependency graph.
        
        Args:
            container_id: Optional container identifier for scoping
        """
        self.container_id = container_id
        self._nodes: Dict[str, ComponentNode] = {}
        self._edges: Dict[str, Set[str]] = defaultdict(set)  # component -> dependencies
        self._reverse_edges: Dict[str, Set[str]] = defaultdict(set)  # component -> dependents
        
        logger.debug(f"DependencyGraph created for container: {container_id}")
    
    def add_component(
        self,
        name: str,
        component_type: Optional[type] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ComponentNode:
        """
        Add a component to the graph.
        
        Args:
            name: Component name
            component_type: Optional component class
            metadata: Optional metadata about the component
            
        Returns:
            The created or existing ComponentNode
        """
        if name not in self._nodes:
            node = ComponentNode(
                name=name,
                component_type=component_type,
                metadata=metadata or {}
            )
            self._nodes[name] = node
            logger.debug(f"Added component '{name}' to dependency graph")
        else:
            # Update existing node if needed
            node = self._nodes[name]
            if component_type and not node.component_type:
                node.component_type = component_type
            if metadata:
                node.metadata.update(metadata)
        
        return node
    
    def add_dependency(
        self,
        component: str,
        dependency: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a dependency relationship.
        
        Args:
            component: The component that has the dependency
            dependency: The component it depends on
            metadata: Optional metadata about the relationship
        """
        # Ensure both components exist
        self.add_component(component)
        self.add_component(dependency)
        
        # Add the edge
        self._edges[component].add(dependency)
        self._reverse_edges[dependency].add(component)
        
        # Update node relationships
        self._nodes[component].dependencies.add(dependency)
        self._nodes[dependency].dependents.add(component)
        
        logger.debug(f"Added dependency: {component} -> {dependency}")
    
    def add_dependencies(
        self,
        component: str,
        dependencies: List[str]
    ) -> None:
        """Add multiple dependencies for a component."""
        for dep in dependencies:
            self.add_dependency(component, dep)
    
    def get_dependencies(self, component: str) -> Set[str]:
        """Get direct dependencies of a component."""
        return self._edges.get(component, set()).copy()
    
    def get_dependents(self, component: str) -> Set[str]:
        """Get components that depend on this component."""
        return self._reverse_edges.get(component, set()).copy()
    
    def get_all_dependencies(self, component: str) -> Set[str]:
        """Get all transitive dependencies of a component."""
        visited = set()
        to_visit = deque([component])
        
        while to_visit:
            current = to_visit.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            
            # Add direct dependencies to visit
            for dep in self._edges.get(current, set()):
                if dep not in visited:
                    to_visit.append(dep)
        
        # Remove the component itself
        visited.discard(component)
        return visited
    
    def detect_cycles(self) -> List[List[str]]:
        """
        Detect circular dependencies in the graph.
        
        Returns:
            List of cycles, where each cycle is a list of component names
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node: str) -> None:
            """Depth-first search to find cycles."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self._edges.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
            
            path.pop()
            rec_stack.remove(node)
        
        # Check all components
        for node in self._nodes:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    def get_initialization_order(self) -> List[str]:
        """
        Get the order in which components should be initialized.
        
        Uses topological sort to ensure dependencies are initialized first.
        
        Returns:
            List of component names in initialization order
            
        Raises:
            ValueError: If cycles are detected
        """
        # Check for cycles first
        cycles = self.detect_cycles()
        if cycles:
            raise ValueError(f"Cannot determine initialization order due to cycles: {cycles}")
        
        # Kahn's algorithm for topological sort
        in_degree = defaultdict(int)
        
        # Initialize all nodes with 0 in-degree
        for node in self._nodes:
            in_degree[node] = 0
        
        # Calculate in-degrees (number of dependencies each node has)
        for node, deps in self._edges.items():
            in_degree[node] = len(deps)
        
        # Find all nodes with no incoming edges
        queue = deque([node for node in self._nodes if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            # Process nodes that depend on this node
            for dependent in self._reverse_edges.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Return all processed nodes
        # Some nodes might not have dependencies (isolated components)
        # Add any remaining nodes without dependencies
        for node in self._nodes:
            if node not in result:
                result.append(node)
        
        return result
    
    def get_teardown_order(self) -> List[str]:
        """
        Get the order in which components should be torn down.
        
        This is the reverse of initialization order.
        
        Returns:
            List of component names in teardown order
        """
        return list(reversed(self.get_initialization_order()))
    
    def validate(self) -> Dict[str, List[str]]:
        """
        Validate the dependency graph.
        
        Returns:
            Dictionary of validation errors by type
        """
        errors = defaultdict(list)
        
        # Check for cycles
        cycles = self.detect_cycles()
        if cycles:
            for cycle in cycles:
                errors["circular_dependencies"].append(
                    f"Circular dependency: {' -> '.join(cycle)}"
                )
        
        # Check for missing dependencies
        for component, deps in self._edges.items():
            for dep in deps:
                if dep not in self._nodes:
                    errors["missing_dependencies"].append(
                        f"Component '{component}' depends on missing component '{dep}'"
                    )
        
        # Check for orphaned components (no dependencies and no dependents)
        for name, node in self._nodes.items():
            if not node.dependencies and not node.dependents:
                # This might be OK for some components, so just warn
                errors["warnings"].append(
                    f"Component '{name}' has no dependencies or dependents"
                )
        
        return dict(errors)
    
    def set_instance(self, name: str, instance: Any) -> None:
        """
        Set the resolved instance for a component.
        
        Args:
            name: Component name
            instance: The instantiated component
        """
        if name in self._nodes:
            self._nodes[name].instance = instance
            logger.debug(f"Set instance for component '{name}'")
        else:
            raise ValueError(f"Component '{name}' not found in graph")
    
    def get_instance(self, name: str) -> Optional[Any]:
        """Get the resolved instance for a component."""
        node = self._nodes.get(name)
        return node.instance if node else None
    
    def is_resolved(self, name: str) -> bool:
        """Check if a component has been resolved."""
        node = self._nodes.get(name)
        return node.is_resolved if node else False
    
    def get_unresolved_dependencies(self, component: str) -> Set[str]:
        """Get dependencies that haven't been resolved yet."""
        unresolved = set()
        
        for dep in self.get_dependencies(component):
            if not self.is_resolved(dep):
                unresolved.add(dep)
        
        return unresolved
    
    def can_resolve(self, component: str) -> bool:
        """Check if a component can be resolved (all dependencies met)."""
        return len(self.get_unresolved_dependencies(component)) == 0
    
    def get_resolution_order(self, components: Optional[List[str]] = None) -> List[str]:
        """
        Get the order to resolve specific components.
        
        Args:
            components: Specific components to resolve (None for all)
            
        Returns:
            Ordered list of components to resolve
        """
        if components is None:
            return self.get_initialization_order()
        
        # Build subgraph of required components
        required = set(components)
        
        # Add all transitive dependencies
        for comp in components:
            required.update(self.get_all_dependencies(comp))
        
        # Filter initialization order to only required components
        full_order = self.get_initialization_order()
        return [comp for comp in full_order if comp in required]
    
    def export_graph(self) -> Dict[str, Any]:
        """
        Export graph data for visualization.
        
        Returns:
            Dictionary with nodes and edges suitable for visualization
        """
        nodes = []
        edges = []
        
        for name, node in self._nodes.items():
            nodes.append({
                "id": name,
                "type": node.component_type.__name__ if node.component_type else "Unknown",
                "resolved": node.is_resolved,
                "capabilities": node.capabilities,
                "metadata": node.metadata
            })
        
        for source, targets in self._edges.items():
            for target in targets:
                edges.append({
                    "source": source,
                    "target": target
                })
        
        return {
            "container_id": self.container_id,
            "nodes": nodes,
            "edges": edges
        }
    
    def clear(self) -> None:
        """Clear all components and dependencies."""
        self._nodes.clear()
        self._edges.clear()
        self._reverse_edges.clear()
        logger.debug(f"DependencyGraph cleared for container: {self.container_id}")


class DependencyValidator:
    """Validates dependencies according to architectural rules."""
    
    def __init__(self, module_hierarchy: Optional[Dict[str, int]] = None):
        """
        Initialize validator.
        
        Args:
            module_hierarchy: Module levels (lower can depend on higher)
        """
        self.module_hierarchy = module_hierarchy or {
            "core": 0,
            "events": 1,
            "components": 1,
            "dependencies": 1,
            "data": 2,
            "strategy": 3,
            "risk": 4,
            "execution": 5,
            "analytics": 6
        }
    
    def validate_direction(
        self,
        component: str,
        dependency: str,
        component_module: str,
        dependency_module: str
    ) -> Optional[str]:
        """
        Validate dependency direction according to module hierarchy.
        
        Returns:
            Error message if invalid, None if valid
        """
        comp_level = self.module_hierarchy.get(component_module, 999)
        dep_level = self.module_hierarchy.get(dependency_module, 999)
        
        if comp_level < dep_level:
            return (
                f"Invalid dependency direction: '{component}' (module: {component_module}, "
                f"level: {comp_level}) cannot depend on '{dependency}' "
                f"(module: {dependency_module}, level: {dep_level}). "
                f"Lower-level modules cannot depend on higher-level modules."
            )
        
        return None
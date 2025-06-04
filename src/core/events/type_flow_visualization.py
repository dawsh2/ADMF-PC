"""
Type flow visualization tools for ADMF-PC.

This module provides visualization utilities for type flow analysis,
helping debug adapter configurations and understand event flows.
"""

from typing import Dict, List, Set, Any, Optional, Type
import logging
from collections import defaultdict

from .semantic import SemanticEvent, MarketDataEvent, FeatureEvent, TradingSignal, OrderEvent, FillEvent, PortfolioUpdateEvent
from .type_flow_analysis import FlowNode, EventTypeRegistry, TypeFlowAnalyzer, ValidationResult
from ..types.events import EventType
from ..containers.protocols import Container


class TypeFlowVisualizer:
    """Visualize type flow through adapter configurations."""
    
    def __init__(self, analyzer: Optional[TypeFlowAnalyzer] = None):
        """Initialize visualizer.
        
        Args:
            analyzer: Type flow analyzer (creates default if None)
        """
        self.analyzer = analyzer or TypeFlowAnalyzer()
        self.logger = logging.getLogger(__name__)
        
    def generate_text_visualization(self, 
                                  flow_map: Dict[str, FlowNode]) -> str:
        """Generate human-readable flow visualization.
        
        Args:
            flow_map: Flow analysis results
            
        Returns:
            Formatted text visualization
        """
        lines = ["Type Flow Analysis", "=" * 50, ""]
        
        # Sort containers by typical flow order
        ordered_containers = self._order_by_flow(flow_map)
        
        for container_name in ordered_containers:
            node = flow_map[container_name]
            lines.append(f"{container_name}:")
            
            if node.will_receive:
                lines.append(f"  receives: {self._format_types(node.will_receive)}")
            else:
                lines.append("  receives: [none - source]")
                
            produced = self.analyzer._compute_produced_types(node)
            if produced:
                lines.append(f"  produces: {self._format_types(produced)}")
            else:
                lines.append("  produces: [none]")
                
            # Show semantic event types if available
            if node.semantic_inputs:
                lines.append(f"  semantic inputs: {[t.__name__ for t in node.semantic_inputs]}")
            if node.semantic_outputs:
                lines.append(f"  semantic outputs: {[t.__name__ for t in node.semantic_outputs]}")
                
            lines.append("")
        
        return "\n".join(lines)
    
    def generate_mermaid_diagram(self, 
                                flow_map: Dict[str, FlowNode],
                                adapters: List[Any]) -> str:
        """Generate Mermaid diagram for visualization.
        
        Args:
            flow_map: Flow analysis results
            adapters: List of adapters
            
        Returns:
            Mermaid diagram as string
        """
        lines = ["graph TD", "    %% Type Flow Diagram"]
        
        # Add nodes with their event types
        for name, node in flow_map.items():
            produced = self.analyzer._compute_produced_types(node)
            if produced:
                label = f"{name}<br/>[{self._format_types(produced)}]"
            else:
                label = name
            lines.append(f"    {self._sanitize_name(name)}[\"{label}\"]")
        
        # Add edges with event types
        connections = self.analyzer._build_connections(adapters)
        for source, targets in connections.items():
            source_node = flow_map.get(source)
            if not source_node:
                continue
                
            produced = self.analyzer._compute_produced_types(source_node)
            
            for target in targets:
                target_node = flow_map.get(target)
                if not target_node:
                    continue
                    
                # Find what types actually flow
                flowing_types = produced & target_node.can_receive
                if flowing_types:
                    label = self._format_types(flowing_types)
                    lines.append(
                        f"    {self._sanitize_name(source)} -->|{label}| {self._sanitize_name(target)}"
                    )
                else:
                    lines.append(
                        f"    {self._sanitize_name(source)} -.->|no match| {self._sanitize_name(target)}"
                    )
        
        return "\n".join(lines)
    
    def generate_validation_report(self, 
                                 validation_result: ValidationResult) -> str:
        """Generate detailed validation report.
        
        Args:
            validation_result: Validation results
            
        Returns:
            Formatted validation report
        """
        lines = ["Type Flow Validation Report", "=" * 50, ""]
        
        if validation_result.valid:
            lines.append("✓ Type flow validation PASSED")
        else:
            lines.append("✗ Type flow validation FAILED")
        
        if validation_result.errors:
            lines.extend(["", "Errors:", "-" * 20])
            for error in validation_result.errors:
                lines.append(f"  • {error}")
        
        if validation_result.warnings:
            lines.extend(["", "Warnings:", "-" * 20])
            for warning in validation_result.warnings:
                lines.append(f"  • {warning}")
        
        if validation_result.flow_map:
            lines.extend(["", "", self.generate_text_visualization(
                validation_result.flow_map
            )])
        
        return "\n".join(lines)
    
    def generate_adapter_analysis(self, adapters: List[Dict[str, Any]], 
                                 containers: Dict[str, Container]) -> str:
        """Generate analysis of adapter configurations.
        
        Args:
            adapters: List of adapter configurations
            containers: Available containers
            
        Returns:
            Formatted adapter analysis
        """
        lines = ["Adapter Configuration Analysis", "=" * 50, ""]
        
        for i, adapter_config in enumerate(adapters):
            adapter_type = adapter_config.get('type', 'unknown')
            adapter_name = adapter_config.get('name', f'adapter_{i}')
            
            lines.append(f"Adapter: {adapter_name} ({adapter_type})")
            lines.append("-" * 30)
            
            if adapter_type == 'pipeline':
                lines.extend(self._analyze_pipeline_adapter(adapter_config, containers))
            elif adapter_type == 'broadcast':
                lines.extend(self._analyze_broadcast_adapter(adapter_config, containers))
            elif adapter_type == 'hierarchical':
                lines.extend(self._analyze_hierarchical_adapter(adapter_config, containers))
            elif adapter_type == 'selective':
                lines.extend(self._analyze_selective_adapter(adapter_config, containers))
            else:
                lines.append(f"  Unknown adapter type: {adapter_type}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _analyze_pipeline_adapter(self, config: Dict[str, Any], 
                                 containers: Dict[str, Container]) -> List[str]:
        """Analyze pipeline adapter configuration."""
        lines = []
        container_names = config.get('containers', [])
        
        lines.append(f"  Pipeline: {' → '.join(container_names)}")
        
        # Analyze each connection
        for i in range(len(container_names) - 1):
            source_name = container_names[i]
            target_name = container_names[i + 1]
            
            if source_name in containers and target_name in containers:
                source = containers[source_name]
                target = containers[target_name]
                
                # Try to infer container types
                source_type = self._infer_container_type(source)
                target_type = self._infer_container_type(target)
                
                lines.append(f"    {source_name} ({source_type}) → {target_name} ({target_type})")
            else:
                missing = []
                if source_name not in containers:
                    missing.append(source_name)
                if target_name not in containers:
                    missing.append(target_name)
                lines.append(f"    Missing containers: {missing}")
        
        return lines
    
    def _analyze_broadcast_adapter(self, config: Dict[str, Any], 
                                  containers: Dict[str, Container]) -> List[str]:
        """Analyze broadcast adapter configuration."""
        lines = []
        source_name = config.get('source', 'unknown')
        target_names = config.get('targets', [])
        
        lines.append(f"  Source: {source_name}")
        lines.append(f"  Targets: {', '.join(target_names)}")
        
        # Check if containers exist
        missing = []
        if source_name not in containers:
            missing.append(source_name)
        for target in target_names:
            if target not in containers:
                missing.append(target)
        
        if missing:
            lines.append(f"  Missing containers: {missing}")
        
        return lines
    
    def _analyze_hierarchical_adapter(self, config: Dict[str, Any], 
                                     containers: Dict[str, Container]) -> List[str]:
        """Analyze hierarchical adapter configuration."""
        lines = []
        parent_name = config.get('parent', 'unknown')
        children = config.get('children', [])
        
        lines.append(f"  Parent: {parent_name}")
        lines.append(f"  Children: {[c['name'] if isinstance(c, dict) else c for c in children]}")
        
        return lines
    
    def _analyze_selective_adapter(self, config: Dict[str, Any], 
                                  containers: Dict[str, Container]) -> List[str]:
        """Analyze selective adapter configuration."""
        lines = []
        source_name = config.get('source', 'unknown')
        rules = config.get('rules', [])
        
        lines.append(f"  Source: {source_name}")
        lines.append(f"  Rules: {len(rules)}")
        
        for i, rule in enumerate(rules):
            condition = rule.get('condition', 'unknown')
            target = rule.get('target', 'unknown')
            lines.append(f"    Rule {i+1}: {condition} → {target}")
        
        return lines
    
    def generate_semantic_flow_diagram(self, 
                                     flow_map: Dict[str, FlowNode]) -> str:
        """Generate diagram showing semantic event types.
        
        Args:
            flow_map: Flow analysis results
            
        Returns:
            Mermaid diagram with semantic events
        """
        lines = ["graph TD", "    %% Semantic Event Flow"]
        
        # Add nodes with semantic event types
        for name, node in flow_map.items():
            semantic_types = node.semantic_outputs
            
            if semantic_types:
                # Show actual event class names
                type_names = [t.__name__ for t in semantic_types]
                label = f"{name}<br/>[{', '.join(type_names)}]"
            else:
                label = name
                
            lines.append(f"    {self._sanitize_name(name)}[\"{label}\"]")
        
        # Add semantic event flow connections
        # This would need actual adapter information to show flows
        lines.append("    %% Event flows would be added here with adapter data")
        
        return "\n".join(lines)
    
    def create_comprehensive_report(self, 
                                  adapters: List[Dict[str, Any]],
                                  containers: Dict[str, Container]) -> str:
        """Create comprehensive type flow report.
        
        Args:
            adapters: Adapter configurations
            containers: Available containers
            
        Returns:
            Complete analysis report
        """
        lines = ["ADMF-PC Type Flow Analysis Report", "=" * 60, ""]
        
        # System overview
        lines.extend([
            "System Overview:",
            "-" * 20,
            f"Containers: {len(containers)}",
            f"Adapters: {len(adapters)}",
            ""
        ])
        
        # Container analysis
        lines.extend([
            "Container Analysis:",
            "-" * 20
        ])
        
        for name, container in containers.items():
            container_type = self._infer_container_type(container)
            lines.append(f"  {name}: {container_type}")
        
        lines.append("")
        
        # Flow analysis
        try:
            flow_map = self.analyzer.analyze_flow(containers, adapters)
            lines.extend([
                "Type Flow Analysis:",
                "-" * 20,
                self.generate_text_visualization(flow_map),
                ""
            ])
        except Exception as e:
            lines.extend([
                "Type Flow Analysis:",
                "-" * 20,
                f"Error: {e}",
                ""
            ])
        
        # Adapter analysis
        lines.extend([
            "Adapter Configuration Analysis:",
            "-" * 35,
            self.generate_adapter_analysis(adapters, containers)
        ])
        
        return "\n".join(lines)
    
    def _format_types(self, types: Set[EventType]) -> str:
        """Format event types for display."""
        return ", ".join(sorted(t.value for t in types))
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize container name for Mermaid."""
        # Replace characters that might cause issues in Mermaid
        return name.replace('-', '_').replace(' ', '_').replace('.', '_')
    
    def _order_by_flow(self, flow_map: Dict[str, FlowNode]) -> List[str]:
        """Order containers by typical flow sequence."""
        # Simple heuristic ordering
        order_priority = {
            'data': 0, 'market': 0, 
            'feature': 1, 'technical': 1,
            'strategy': 2, 'signal': 2,
            'risk': 3, 'limit': 3,
            'execution': 4, 'broker': 4, 'order': 4,
            'portfolio': 5, 'position': 5
        }
        
        def priority(name: str) -> int:
            name_lower = name.lower()
            for key, pri in order_priority.items():
                if key in name_lower:
                    return pri
            return 999
        
        return sorted(flow_map.keys(), key=priority)
    
    def _infer_container_type(self, container: Container) -> str:
        """Infer container type from container object."""
        # Use the type inferencer from analyzer
        return self.analyzer.type_inferencer.infer_container_type(container)


def create_flow_visualization(containers: Dict[str, Container],
                            adapters: List[Dict[str, Any]],
                            output_format: str = "text") -> str:
    """Create type flow visualization.
    
    Args:
        containers: Available containers
        adapters: Adapter configurations  
        output_format: "text", "mermaid", or "report"
        
    Returns:
        Formatted visualization
    """
    visualizer = TypeFlowVisualizer()
    
    try:
        flow_map = visualizer.analyzer.analyze_flow(containers, adapters)
        
        if output_format == "text":
            return visualizer.generate_text_visualization(flow_map)
        elif output_format == "mermaid":
            return visualizer.generate_mermaid_diagram(flow_map, adapters)
        elif output_format == "report":
            return visualizer.create_comprehensive_report(adapters, containers)
        else:
            return f"Unknown output format: {output_format}"
            
    except Exception as e:
        return f"Error generating visualization: {e}"


def validate_and_visualize(adapters: List[Dict[str, Any]],
                          containers: Dict[str, Container],
                          execution_mode: str = "full_backtest") -> str:
    """Validate configuration and create visualization.
    
    Args:
        adapters: Adapter configurations
        containers: Available containers
        execution_mode: Execution mode to validate against
        
    Returns:
        Combined validation and visualization report
    """
    analyzer = TypeFlowAnalyzer()
    visualizer = TypeFlowVisualizer(analyzer)
    
    lines = ["Type Flow Validation and Visualization", "=" * 50, ""]
    
    try:
        # Analyze flow
        flow_map = analyzer.analyze_flow(containers, adapters)
        
        # Validate
        validation = analyzer.validate_mode(flow_map, execution_mode)
        
        # Add validation results
        lines.append(visualizer.generate_validation_report(validation))
        lines.append("")
        
        # Add visualization if validation passed or in non-strict mode
        if validation.valid or not validation.errors:
            lines.extend([
                "Flow Visualization:",
                "-" * 20,
                visualizer.generate_text_visualization(flow_map)
            ])
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error in validation and visualization: {e}"


def export_mermaid_diagram(containers: Dict[str, Container],
                          adapters: List[Dict[str, Any]],
                          filename: str) -> bool:
    """Export Mermaid diagram to file.
    
    Args:
        containers: Available containers
        adapters: Adapter configurations
        filename: Output filename
        
    Returns:
        True if successful
    """
    try:
        diagram = create_flow_visualization(containers, adapters, "mermaid")
        
        with open(filename, 'w') as f:
            f.write(diagram)
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to export Mermaid diagram: {e}")
        return False
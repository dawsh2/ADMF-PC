"""
Type flow integration utilities for ADMF-PC.

This module provides convenience functions for integrating type flow analysis
with adapters and containers throughout the system.
"""

from typing import Dict, List, Any, Optional, Set, Type
import logging

from .semantic import SemanticEvent, MarketDataEvent, FeatureEvent, TradingSignal, OrderEvent, FillEvent, PortfolioUpdateEvent
from .type_flow_analysis import EventTypeRegistry, TypeFlowAnalyzer, ContainerTypeInferencer, ValidationResult
from ..types.events import EventType
from ..containers.protocols import Container


class TypeFlowValidator:
    """Convenience class for type flow validation throughout the system."""
    
    def __init__(self, strict_mode: bool = False):
        """Initialize validator.
        
        Args:
            strict_mode: If True, raise exceptions on validation failures
        """
        self.strict_mode = strict_mode
        self.registry = EventTypeRegistry()
        self.analyzer = TypeFlowAnalyzer(self.registry)
        self.inferencer = ContainerTypeInferencer(self.registry)
        self.logger = logging.getLogger(__name__)
        
    def validate_adapter_config(self, config: Dict[str, Any], 
                               containers: Dict[str, Container]) -> ValidationResult:
        """Validate an adapter configuration for type flow compatibility.
        
        Args:
            config: Adapter configuration
            containers: Available containers
            
        Returns:
            ValidationResult with validation details
        """
        adapter_type = config.get('type', 'unknown')
        
        try:
            if adapter_type == 'pipeline':
                return self._validate_pipeline_config(config, containers)
            elif adapter_type == 'broadcast':
                return self._validate_broadcast_config(config, containers)
            elif adapter_type == 'hierarchical':
                return self._validate_hierarchical_config(config, containers)
            elif adapter_type == 'selective':
                return self._validate_selective_config(config, containers)
            else:
                self.logger.warning(f"Unknown adapter type for validation: {adapter_type}")
                return ValidationResult(valid=True, warnings=[f"Skipped validation for unknown adapter type: {adapter_type}"])
                
        except Exception as e:
            error_msg = f"Error validating {adapter_type} adapter: {e}"
            self.logger.error(error_msg)
            return ValidationResult(valid=False, errors=[error_msg])
    
    def _validate_pipeline_config(self, config: Dict[str, Any], 
                                 containers: Dict[str, Container]) -> ValidationResult:
        """Validate pipeline adapter configuration."""
        container_names = config.get('containers', [])
        if len(container_names) < 2:
            return ValidationResult(valid=False, errors=["Pipeline needs at least 2 containers"])
        
        errors = []
        warnings = []
        
        # Check each connection in the pipeline
        for i in range(len(container_names) - 1):
            source_name = container_names[i]
            target_name = container_names[i + 1]
            
            if source_name not in containers:
                errors.append(f"Container '{source_name}' not found")
                continue
            if target_name not in containers:
                errors.append(f"Container '{target_name}' not found")
                continue
                
            source = containers[source_name]
            target = containers[target_name]
            
            # Check type compatibility
            source_outputs = self.inferencer.get_expected_outputs(source)
            target_inputs = self.inferencer.get_expected_inputs(target)
            
            if source_outputs and target_inputs:
                compatible = source_outputs & target_inputs
                if not compatible:
                    warnings.append(
                        f"No type compatibility between {source_name} and {target_name}: "
                        f"{[t.__name__ for t in source_outputs]} → {[t.__name__ for t in target_inputs]}"
                    )
        
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    def _validate_broadcast_config(self, config: Dict[str, Any], 
                                  containers: Dict[str, Container]) -> ValidationResult:
        """Validate broadcast adapter configuration."""
        source_name = config.get('source')
        target_names = config.get('targets', [])
        
        if not source_name:
            return ValidationResult(valid=False, errors=["Broadcast adapter needs a source"])
        if not target_names:
            return ValidationResult(valid=False, errors=["Broadcast adapter needs targets"])
        
        errors = []
        warnings = []
        
        if source_name not in containers:
            errors.append(f"Source container '{source_name}' not found")
            return ValidationResult(valid=False, errors=errors)
        
        source = containers[source_name]
        source_outputs = self.inferencer.get_expected_outputs(source)
        
        for target_name in target_names:
            if target_name not in containers:
                errors.append(f"Target container '{target_name}' not found")
                continue
                
            target = containers[target_name]
            target_inputs = self.inferencer.get_expected_inputs(target)
            
            if source_outputs and target_inputs:
                compatible = source_outputs & target_inputs
                if not compatible:
                    warnings.append(
                        f"No type compatibility between {source_name} and {target_name}: "
                        f"{[t.__name__ for t in source_outputs]} → {[t.__name__ for t in target_inputs]}"
                    )
        
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    def _validate_hierarchical_config(self, config: Dict[str, Any], 
                                     containers: Dict[str, Container]) -> ValidationResult:
        """Validate hierarchical adapter configuration."""
        parent_name = config.get('parent')
        children = config.get('children', [])
        
        if not parent_name:
            return ValidationResult(valid=False, errors=["Hierarchical adapter needs a parent"])
        if not children:
            return ValidationResult(valid=False, errors=["Hierarchical adapter needs children"])
        
        errors = []
        warnings = []
        
        if parent_name not in containers:
            errors.append(f"Parent container '{parent_name}' not found")
        
        for child in children:
            child_name = child['name'] if isinstance(child, dict) else child
            if child_name not in containers:
                errors.append(f"Child container '{child_name}' not found")
        
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    def _validate_selective_config(self, config: Dict[str, Any], 
                                  containers: Dict[str, Container]) -> ValidationResult:
        """Validate selective adapter configuration."""
        source_name = config.get('source')
        rules = config.get('rules', [])
        
        if not source_name:
            return ValidationResult(valid=False, errors=["Selective adapter needs a source"])
        if not rules:
            return ValidationResult(valid=False, errors=["Selective adapter needs routing rules"])
        
        errors = []
        warnings = []
        
        if source_name not in containers:
            errors.append(f"Source container '{source_name}' not found")
            return ValidationResult(valid=False, errors=errors)
        
        # Check that all target containers exist
        for rule in rules:
            target_name = rule.get('target')
            if target_name and target_name not in containers:
                errors.append(f"Target container '{target_name}' not found")
        
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    def validate_semantic_event_flow(self, event: SemanticEvent, 
                                   source: Container, 
                                   target: Container) -> ValidationResult:
        """Validate that a semantic event can flow from source to target."""
        errors = []
        warnings = []
        
        # Validate the event itself
        from .semantic import validate_semantic_event
        if not validate_semantic_event(event):
            errors.append(f"Invalid semantic event: {event}")
        
        # Check type compatibility
        target_inputs = self.inferencer.get_expected_inputs(target)
        event_type = type(event)
        
        if target_inputs and event_type not in target_inputs:
            if self.strict_mode:
                errors.append(
                    f"Event type mismatch: {target.name} expects "
                    f"{[t.__name__ for t in target_inputs]} but got {event_type.__name__}"
                )
            else:
                warnings.append(
                    f"Event type mismatch: {target.name} expects "
                    f"{[t.__name__ for t in target_inputs]} but got {event_type.__name__}"
                )
        
        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    def get_container_recommendations(self, container: Container) -> Dict[str, Any]:
        """Get recommendations for improving container type flow."""
        container_type = self.inferencer.infer_container_type(container)
        expected_inputs = self.inferencer.get_expected_inputs(container)
        expected_outputs = self.inferencer.get_expected_outputs(container)
        
        recommendations = {
            'container_name': container.name,
            'inferred_type': container_type,
            'expected_inputs': [t.__name__ for t in expected_inputs],
            'expected_outputs': [t.__name__ for t in expected_outputs],
            'suggestions': []
        }
        
        # Add specific suggestions based on container type
        if container_type == 'unknown':
            recommendations['suggestions'].append(
                "Consider adding explicit role metadata or renaming container to indicate its purpose"
            )
        
        if not expected_inputs and not expected_outputs:
            recommendations['suggestions'].append(
                "Container has no expected event types - consider implementing semantic event interfaces"
            )
        
        return recommendations


def create_default_validator(strict_mode: bool = False) -> TypeFlowValidator:
    """Create a default type flow validator with standard configuration."""
    return TypeFlowValidator(strict_mode=strict_mode)


def validate_adapter_network(adapters_config: List[Dict[str, Any]], 
                            containers: Dict[str, Container],
                            strict_mode: bool = False) -> ValidationResult:
    """Validate an entire adapter network configuration.
    
    Args:
        adapters_config: List of adapter configurations
        containers: Available containers
        strict_mode: If True, treat warnings as errors
        
    Returns:
        ValidationResult for the entire network
    """
    validator = create_default_validator(strict_mode)
    
    all_errors = []
    all_warnings = []
    
    for i, adapter_config in enumerate(adapters_config):
        adapter_name = adapter_config.get('name', f'adapter_{i}')
        result = validator.validate_adapter_config(adapter_config, containers)
        
        # Prefix errors/warnings with adapter name
        for error in result.errors:
            all_errors.append(f"{adapter_name}: {error}")
        for warning in result.warnings:
            all_warnings.append(f"{adapter_name}: {warning}")
    
    # In strict mode, treat warnings as errors
    if strict_mode:
        all_errors.extend(all_warnings)
        all_warnings = []
    
    return ValidationResult(
        valid=len(all_errors) == 0,
        errors=all_errors,
        warnings=all_warnings
    )


def get_semantic_event_suggestions(container_type: str) -> List[Type[SemanticEvent]]:
    """Get suggested semantic event types for a container type.
    
    Args:
        container_type: Type of container (e.g., 'strategy', 'risk_manager')
        
    Returns:
        List of recommended semantic event types
    """
    suggestions = {
        'data_source': [MarketDataEvent],
        'feature_engine': [FeatureEvent],
        'strategy': [TradingSignal],
        'risk_manager': [OrderEvent],
        'execution_engine': [FillEvent],
        'portfolio_manager': [PortfolioUpdateEvent],
    }
    
    return suggestions.get(container_type, [])


def create_type_flow_report(containers: Dict[str, Container], 
                           adapters_config: List[Dict[str, Any]]) -> str:
    """Create a comprehensive type flow analysis report.
    
    Args:
        containers: Available containers
        adapters_config: Adapter configurations
        
    Returns:
        Formatted report string
    """
    validator = create_default_validator()
    lines = ["Type Flow Analysis Report", "=" * 50, ""]
    
    # Container analysis
    lines.append("Container Analysis:")
    lines.append("-" * 20)
    
    for name, container in containers.items():
        recommendations = validator.get_container_recommendations(container)
        lines.append(f"{name}:")
        lines.append(f"  Type: {recommendations['inferred_type']}")
        lines.append(f"  Expected Inputs: {recommendations['expected_inputs']}")
        lines.append(f"  Expected Outputs: {recommendations['expected_outputs']}")
        
        if recommendations['suggestions']:
            lines.append("  Suggestions:")
            for suggestion in recommendations['suggestions']:
                lines.append(f"    • {suggestion}")
        lines.append("")
    
    # Adapter validation
    lines.append("Adapter Validation:")
    lines.append("-" * 20)
    
    network_result = validate_adapter_network(adapters_config, containers)
    
    if network_result.valid:
        lines.append("✓ All adapters passed validation")
    else:
        lines.append("✗ Adapter validation failed")
        for error in network_result.errors:
            lines.append(f"  ERROR: {error}")
    
    for warning in network_result.warnings:
        lines.append(f"  WARNING: {warning}")
    
    return "\n".join(lines)
"""
Configuration Resolution Utilities

Consolidates configuration resolution logic used by 
Coordinator, Sequencer, and TopologyBuilder.
"""

import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigResolver:
    """
    Unified configuration resolver for workflow execution.
    
    Handles:
    - Template resolution: {config.param} → actual values
    - Reference resolution: $variable → referenced values  
    - Default value handling: {'from_config': 'param', 'default': value}
    - Nested dict traversal: config.nested.param dot notation
    - Context building and merging
    """
    
    def __init__(self):
        """Initialize config resolver."""
        pass
    
    def resolve_value(self, spec: Any, context: Dict[str, Any]) -> Any:
        """
        Resolve a value specification against a context.
        
        Args:
            spec: Value specification (string template, dict with from_config, etc.)
            context: Context dictionary with config, metadata, runtime values
            
        Returns:
            Resolved value
        """
        if isinstance(spec, str):
            # Check for template pattern {config.param}
            if '{' in spec and '}' in spec:
                return self._resolve_template(spec, context)
            
            # Check for reference pattern $variable
            elif spec.startswith('$'):
                return self.extract_value(spec[1:], context)
        
        elif isinstance(spec, dict):
            if 'from_config' in spec:
                # Config reference with optional default
                value = self.extract_value(f"config.{spec['from_config']}", context)
                if value is None:
                    value = spec.get('default')
                return value
            
            # Recursively resolve dict values
            resolved = {}
            for k, v in spec.items():
                resolved[k] = self.resolve_value(v, context)
            return resolved
        
        elif isinstance(spec, list):
            # Recursively resolve list items
            return [self.resolve_value(item, context) for item in spec]
        
        return spec
    
    def _resolve_template(self, template: str, context: Dict[str, Any]) -> str:
        """
        Resolve template string with {variable} placeholders.
        
        Supports both simple {variable} and complex {config.nested.param} syntax.
        """
        # Try coordinator-style resolution first (more sophisticated)
        if self._has_complex_template(template):
            return self._resolve_complex_template(template, context)
        else:
            # Try sequencer-style resolution (simpler)
            return self._resolve_simple_template(template, context)
    
    def _has_complex_template(self, template: str) -> bool:
        """Check if template has complex variable references."""
        pattern = r'\{([^}]+)\}'
        matches = re.findall(pattern, template)
        return any('.' in match for match in matches)
    
    def _resolve_complex_template(self, template: str, context: Dict[str, Any]) -> str:
        """Resolve complex template with dot notation (coordinator style)."""
        pattern = r'\{([^}]+)\}'
        matches = re.findall(pattern, template)
        
        result = template
        for match in matches:
            value = self.extract_value(match, context)
            if value is not None:
                result = result.replace(f"{{{match}}}", str(value))
        
        return result
    
    def _resolve_simple_template(self, template: str, context: Dict[str, Any]) -> str:
        """Resolve simple template with format() (sequencer style)."""
        try:
            return template.format(**context)
        except KeyError:
            # Try flattening context
            flat_context = self._flatten_context(context)
            try:
                return template.format(**flat_context)
            except KeyError:
                logger.warning(f"Could not resolve template: {template}")
                return template
    
    def extract_value(self, path: str, data: Dict[str, Any]) -> Any:
        """
        Extract value from nested dict using dot notation.
        
        Args:
            path: Dot-separated path (e.g., 'config.nested.param')
            data: Dictionary to extract from
            
        Returns:
            Extracted value or None if path not found
        """
        parts = path.split('.')
        value = data
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value
    
    def _flatten_context(self, context: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        """
        Flatten nested context for template resolution.
        
        Converts {'config': {'param': 'value'}} → {'config.param': 'value'}
        """
        flat = {}
        
        for key, value in context.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flat.update(self._flatten_context(value, full_key))
            else:
                flat[full_key] = value
        
        return flat
    
    def build_context(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Build execution context from config and additional parameters.
        
        Args:
            config: Base configuration
            **kwargs: Additional context items (metadata, runtime values, etc.)
            
        Returns:
            Complete execution context
        """
        context = {
            'config': config.copy()
        }
        
        # Add all additional context items
        context.update(kwargs)
        
        return context
    
    def merge_configs(self, base_config: Dict[str, Any], 
                     overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration with overrides.
        
        Args:
            base_config: Base configuration
            overrides: Configuration overrides
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        merged.update(overrides)
        return merged
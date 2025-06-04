"""
Analysis pattern definitions.

Defines container patterns for analysis workflows.
"""

from typing import Dict, Any


def get_analysis_patterns() -> Dict[str, Any]:
    """Get available analysis patterns."""
    
    return {
        'signal_generation': {
            'description': 'Signal generation and capture workflow',
            'container_roles': ['data', 'indicator', 'strategy', 'analysis'],
            'communication_pattern': 'pipeline',
            'supports_multi_parameter': False
        },
        'signal_replay': {
            'description': 'Signal replay for ensemble optimization',
            'container_roles': ['signal_log', 'ensemble', 'risk', 'execution'],
            'communication_pattern': 'pipeline',
            'supports_multi_parameter': False
        }
    }
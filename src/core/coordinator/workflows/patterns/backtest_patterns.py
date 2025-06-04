"""
Backtest pattern definitions.

Defines container patterns for backtest workflows.
"""

from typing import Dict, Any


def get_backtest_patterns() -> Dict[str, Any]:
    """Get available backtest patterns."""
    
    return {
        'simple_backtest': {
            'description': 'Simple backtest with basic container pipeline',
            'container_roles': ['data', 'indicator', 'strategy', 'risk', 'execution'],
            'communication_pattern': 'pipeline',
            'supports_multi_parameter': False
        },
        'full_backtest': {
            'description': 'Full backtest with classifier and advanced features',
            'container_roles': ['data', 'indicator', 'classifier', 'strategy', 'risk', 'execution', 'portfolio'],
            'communication_pattern': 'hierarchical',
            'supports_multi_parameter': False
        }
    }
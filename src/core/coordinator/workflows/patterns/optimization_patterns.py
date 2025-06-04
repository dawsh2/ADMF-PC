"""
Optimization pattern definitions.

Defines container patterns for optimization workflows.
"""

from typing import Dict, Any


def get_optimization_patterns() -> Dict[str, Any]:
    """Get available optimization patterns."""
    
    return {
        'multi_parameter_backtest': {
            'description': 'Multi-parameter backtest with separate portfolios per combination',
            'container_roles': ['hub', 'portfolio', 'strategy', 'risk', 'execution'],
            'communication_pattern': 'multi_portfolio_hub',
            'supports_multi_parameter': True
        },
        'optimization_grid': {
            'description': 'Parameter grid optimization with isolated execution',
            'container_roles': ['hub', 'optimization', 'portfolio', 'strategy', 'risk', 'execution'],
            'communication_pattern': 'optimization_hub',
            'supports_multi_parameter': True
        }
    }
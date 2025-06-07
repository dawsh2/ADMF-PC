"""
Signal Generation Workflow

Workflow for generating and saving trading signals without execution.
Useful for signal analysis, feature engineering, and as input to other workflows.
"""

from typing import Dict, Any


def signal_generation_workflow() -> Dict[str, Any]:
    """
    Define signal generation workflow.
    
    Generates signals and saves them for later analysis or replay.
    """
    return {
        'name': 'signal_generation',
        'description': 'Generate and save trading signals',
        'phases': [
            {
                'name': 'generate_signals',
                'topology': 'signal_generation',
                'description': 'Generate signals from strategies',
                'config_override': {
                    'save_signals': True,
                    'signal_output_dir': './results/signals/generated/',
                    'include_features': True,  # Save features along with signals
                    'compression': 'gzip'      # Compress output files
                }
            }
        ],
        'outputs': {
            'signals': './results/signals/generated/',
            'signal_metadata': './results/signals/metadata.json'
        },
        'metadata': {
            'estimated_duration': '1 hour',
            'complexity': 'low',
            'recommended_for': ['signal analysis', 'strategy development', 'workflow input']
        }
    }

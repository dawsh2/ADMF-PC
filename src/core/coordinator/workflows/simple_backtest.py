"""
Simple backtest workflow implementation.

This is the most basic workflow - just runs a single backtest phase.
It demonstrates the workflow pattern without complexity.
"""

from typing import Dict, Any, List, Optional
from ..protocols import WorkflowProtocol, PhaseConfig, PhaseEnhancerProtocol


class SimpleBacktestWorkflow:
    """
    Simple backtest workflow using Protocol + Composition.
    
    This workflow:
    1. Takes user configuration
    2. Creates a single backtest phase
    3. Executes using single_pass sequence
    4. Returns backtest results
    """
    
    def __init__(self, enhancers: Optional[List[PhaseEnhancerProtocol]] = None):
        """
        Initialize with optional phase enhancers.
        
        Args:
            enhancers: List of components that can enhance phase configs
        """
        self.enhancers = enhancers or []
        
        # Workflow defaults
        self.defaults = {
            'trace_level': 'minimal',
            'objective_function': {'name': 'sharpe_ratio'},
            'results': {
                'retention_policy': 'trade_complete',
                'streaming_metrics': True,
                'store_trades': True,
                'store_equity_curve': False
            }
        }
    
    def get_phases(self, config: Dict[str, Any]) -> Dict[str, PhaseConfig]:
        """
        Convert user config to phase definitions.
        
        For simple backtest, we just have one phase.
        
        Args:
            config: User configuration
            
        Returns:
            Dict with single backtest phase
        """
        # Create single backtest phase
        phases = {
            "backtest": PhaseConfig(
                name="backtest",
                sequence="single_pass",  # Use single_pass sequence
                topology="backtest",     # Use backtest topology
                description="Run simple backtest",
                config=config,           # Pass entire config to phase
                output={
                    'metrics': True,     # Collect performance metrics
                    'trades': True       # Collect trade records
                }
            )
        }
        
        # Apply enhancers
        for enhancer in self.enhancers:
            phases = enhancer.enhance(phases)
        
        return phases

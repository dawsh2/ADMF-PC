"""
Execution mode handlers for different workflow patterns.

This module belongs in the execution package, not coordinator.
The coordinator should only orchestrate, not know about specific execution details.
"""
from typing import Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from .signal_generation_engine import (
    SignalGenerationContainer,
    SignalGenerationContainerFactory
)
from .signal_replay_engine import (
    SignalReplayContainer,
    SignalReplayContainerFactory
)


class ExecutionModeHandler:
    """Handles different execution modes for the system."""
    
    @staticmethod
    async def run_signal_generation(
        base_config: Dict[str, Any],
        signal_output: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run signal generation mode for analysis.
        
        Args:
            base_config: Base configuration
            signal_output: Path to save generated signals
            **kwargs: Additional arguments
            
        Returns:
            Results dictionary
        """
        print("Running in Signal Generation mode...")
        
        # Extract configuration
        data_config = base_config.get('data', {})
        strategies = base_config.get('strategies', [])
        indicators = base_config.get('indicators', [])
        classifiers = base_config.get('classifiers', [])
        
        # Create signal generation configuration
        signal_gen_config = {
            'data_config': data_config,
            'strategies': strategies,
            'indicators': indicators,
            'classifiers': classifiers,
            'output_path': signal_output
        }
        
        # Create and run container
        factory = SignalGenerationContainerFactory()
        container = await factory.create_container("signal_gen", signal_gen_config)
        
        # Execute signal generation
        results = await container.generate_signals()
        
        # Save signals if output path specified
        if signal_output:
            output_path = Path(signal_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump({
                    'metadata': {
                        'generated_at': datetime.now().isoformat(),
                        'config': base_config
                    },
                    'signals': results.get('signals', [])
                }, f, indent=2)
            
            print(f"Signals saved to: {signal_output}")
        
        return {
            'success': True,
            'mode': 'signal_generation',
            'signal_count': len(results.get('signals', [])),
            'results': results
        }
    
    @staticmethod
    async def run_signal_replay(
        base_config: Dict[str, Any],
        signal_log: str,
        weights: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run signal replay mode for ensemble optimization.
        
        Args:
            base_config: Base configuration
            signal_log: Path to signal log file
            weights: JSON string or file with strategy weights
            **kwargs: Additional arguments
            
        Returns:
            Results dictionary
        """
        print("Running in Signal Replay mode...")
        
        # Load signal log
        signal_path = Path(signal_log)
        if not signal_path.exists():
            raise FileNotFoundError(f"Signal log not found: {signal_log}")
        
        with open(signal_path, 'r') as f:
            signal_data = json.load(f)
        
        # Parse weights
        strategy_weights = {}
        if weights:
            if weights.startswith('{'):
                # JSON string
                strategy_weights = json.loads(weights)
            else:
                # File path
                with open(weights, 'r') as f:
                    strategy_weights = json.load(f)
        
        # Create replay configuration
        replay_config = {
            'signal_log': signal_data,
            'strategy_weights': strategy_weights,
            'portfolio_config': base_config.get('portfolio', {}),
            'risk_config': base_config.get('risk', {})
        }
        
        # Create and run container
        factory = SignalReplayContainerFactory()
        container = await factory.create_container("signal_replay", replay_config)
        
        # Execute replay
        results = await container.replay_signals()
        
        return {
            'success': True,
            'mode': 'signal_replay',
            'results': results
        }
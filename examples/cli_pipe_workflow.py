"""
Example implementation of CLI pipe workflow support.

This shows how we could implement piping between phases for the
regime-adaptive optimization workflow.
"""

import json
import sys
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PhaseOutput:
    """Structured output from a workflow phase."""
    metadata: Dict[str, Any]
    artifacts: Dict[str, str]
    config: Dict[str, Any]
    results: Dict[str, Any]
    
    def to_json(self) -> str:
        """Convert to JSON for piping."""
        return json.dumps({
            'metadata': self.metadata,
            'artifacts': self.artifacts,
            'config': self.config,
            'results': self.results
        }, indent=2, default=str)
    
    @classmethod
    def from_json(cls, data: str) -> 'PhaseOutput':
        """Parse from JSON input."""
        parsed = json.loads(data)
        return cls(**parsed)


class PipelinePhase:
    """Base class for pipeline phases."""
    
    def __init__(self, args, config: Dict[str, Any]):
        self.args = args
        self.config = config
        self.previous_output: Optional[PhaseOutput] = None
        
    def read_from_pipe(self) -> Optional[PhaseOutput]:
        """Read previous phase output from stdin."""
        if self.args.from_pipe or self.config == "-":
            try:
                input_data = sys.stdin.read()
                return PhaseOutput.from_json(input_data)
            except Exception as e:
                logger.error(f"Failed to read from pipe: {e}")
                return None
        return None
    
    def inherit_config(self, prev: PhaseOutput) -> Dict[str, Any]:
        """Inherit configuration from previous phase."""
        inherited = self.config.copy() if isinstance(self.config, dict) else {}
        
        # Inherit data settings unless overridden
        if 'data' not in inherited and 'data' in prev.config:
            inherited['data'] = prev.config['data']
            
        # Inherit strategies unless overridden
        if 'strategies' not in inherited and 'strategies' in prev.config:
            inherited['strategies'] = prev.config['strategies']
            
        # Inherit features
        if 'features' not in inherited and 'features' in prev.config:
            inherited['features'] = prev.config['features']
            
        # Add artifact paths
        inherited['artifacts'] = prev.artifacts
        
        return inherited
    
    def output_results(self, output: PhaseOutput):
        """Output results based on format flag."""
        if self.args.output_format == 'json' or self.args.output_format == 'pipe':
            print(output.to_json())
        else:
            # Human readable format
            print(f"\n{'='*60}")
            print(f"Phase: {output.metadata['phase']}")
            print(f"Completed: {output.metadata['timestamp']}")
            print(f"\nResults:")
            for key, value in output.results.items():
                print(f"  {key}: {value}")
            print(f"\nArtifacts:")
            for key, path in output.artifacts.items():
                print(f"  {key}: {path}")
            print(f"{'='*60}\n")


class SignalGenerationPhase(PipelinePhase):
    """Phase 1: Grid search signal generation."""
    
    def execute(self) -> PhaseOutput:
        """Run signal generation with grid search."""
        # Check for piped input
        prev = self.read_from_pipe()
        if prev:
            self.config = self.inherit_config(prev)
        
        # Simulate signal generation
        logger.info("Running signal generation grid search...")
        
        # Extract parameters for grid search
        strategies = self.config.get('strategies', [])
        param_combinations = []
        
        for strategy in strategies:
            if 'param_grid' in strategy:
                # Generate all combinations
                # In real implementation, use itertools.product
                param_combinations.extend(self._expand_grid(strategy))
        
        # Simulate walk-forward windows
        walk_forward = self.config.get('walk_forward', {})
        windows = self._generate_windows(walk_forward)
        
        # Output
        execution_id = f"signal_gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return PhaseOutput(
            metadata={
                'phase': 'signal_generation',
                'timestamp': datetime.now().isoformat(),
                'execution_id': execution_id,
                'strategies': [s['type'] for s in strategies],
                'param_combinations': len(param_combinations),
                'windows': len(windows)
            },
            artifacts={
                'signals': f"workspaces/{execution_id}/signals/",
                'summary': f"workspaces/{execution_id}/summary.json",
                'params': f"workspaces/{execution_id}/parameters.json"
            },
            config={
                'strategies': strategies,
                'features': self.config.get('features', []),
                'data': self.config.get('data', {}),
                'walk_forward': walk_forward
            },
            results={
                'total_signals': len(param_combinations) * len(windows) * 100,  # Simulated
                'windows_processed': len(windows),
                'parameter_sets': len(param_combinations),
                'execution_time': 45.2  # Simulated
            }
        )
    
    def _expand_grid(self, strategy: Dict[str, Any]) -> list:
        """Expand parameter grid (simplified)."""
        # In real implementation, use itertools.product
        return [{'type': strategy['type'], 'params': {}}]  # Placeholder
    
    def _generate_windows(self, walk_forward: Dict[str, Any]) -> list:
        """Generate walk-forward windows (simplified)."""
        # In real implementation, generate actual date windows
        return [f"window_{i}" for i in range(4)]  # Placeholder


class SignalReplayPhase(PipelinePhase):
    """Phase 2: Ensemble weight optimization using saved signals."""
    
    def execute(self) -> PhaseOutput:
        """Run ensemble optimization on saved signals."""
        # Must have previous phase output
        prev = self.read_from_pipe()
        if not prev:
            raise ValueError("Signal replay requires piped input from signal generation")
        
        self.config = self.inherit_config(prev)
        
        logger.info(f"Optimizing ensemble weights using signals from {prev.artifacts['signals']}")
        
        # Use signals from previous phase
        signal_path = prev.artifacts['signals']
        param_sets = prev.results['parameter_sets']
        
        # Simulate ensemble optimization
        regime_weights = {
            'bull': {'momentum': 0.7, 'mean_reversion': 0.3},
            'bear': {'momentum': 0.3, 'mean_reversion': 0.7},
            'sideways': {'momentum': 0.5, 'mean_reversion': 0.5}
        }
        
        execution_id = f"ensemble_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return PhaseOutput(
            metadata={
                'phase': 'signal_replay',
                'timestamp': datetime.now().isoformat(),
                'execution_id': execution_id,
                'signal_source': signal_path,
                'optimization': 'ensemble_weights'
            },
            artifacts={
                'weights': f"workspaces/{execution_id}/regime_weights.json",
                'performance': f"workspaces/{execution_id}/performance.json",
                'signals': signal_path  # Pass through
            },
            config={
                'strategies': self.config['strategies'],
                'ensemble_weights': regime_weights,
                'data': self.config['data']
            },
            results={
                'regimes_identified': 3,
                'optimal_sharpe': 1.85,
                'weight_stability': 0.92,
                'execution_time': 12.3
            }
        )


class BacktestPhase(PipelinePhase):
    """Phase 3: Final validation backtest."""
    
    def execute(self) -> PhaseOutput:
        """Run final validation backtest."""
        # Must have previous phase output
        prev = self.read_from_pipe()
        if not prev:
            raise ValueError("Backtest validation requires piped input")
        
        self.config = self.inherit_config(prev)
        
        logger.info("Running final validation backtest with regime-adaptive ensemble...")
        
        # Use ensemble weights from previous phase
        ensemble_weights = self.config.get('ensemble_weights', {})
        
        # Simulate backtest results
        execution_id = f"backtest_val_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return PhaseOutput(
            metadata={
                'phase': 'backtest',
                'timestamp': datetime.now().isoformat(),
                'execution_id': execution_id,
                'validation_type': 'out_of_sample'
            },
            artifacts={
                'results': f"workspaces/{execution_id}/results.json",
                'trades': f"workspaces/{execution_id}/trades.csv",
                'report': f"workspaces/{execution_id}/report.html"
            },
            config=self.config,
            results={
                'total_return': 0.342,
                'sharpe_ratio': 1.76,
                'max_drawdown': -0.087,
                'win_rate': 0.623,
                'total_trades': 847,
                'regime_switches': 28
            }
        )


# Example usage functions

def example_signal_generation(args, config):
    """Example: Run signal generation phase."""
    phase = SignalGenerationPhase(args, config)
    output = phase.execute()
    phase.output_results(output)
    return output


def example_signal_replay(args):
    """Example: Run signal replay phase reading from pipe."""
    # When called with --from-pipe, config is inherited
    phase = SignalReplayPhase(args, {})
    output = phase.execute()
    phase.output_results(output)
    return output


def example_backtest_validation(args):
    """Example: Run final backtest validation."""
    phase = BacktestPhase(args, {})
    output = phase.execute()
    phase.output_results(output)
    return output


# Example of how main.py would integrate this

def handle_piped_workflow(args, coordinator):
    """Handle piped workflow execution."""
    
    # Detect if we're in a pipe
    if args.from_pipe or (not sys.stdin.isatty() and args.config == "-"):
        # Read previous phase output
        try:
            prev_output = PhaseOutput.from_json(sys.stdin.read())
            
            # Inherit configuration
            config = prev_output.config.copy()
            
            # Add artifact references
            config['artifacts'] = prev_output.artifacts
            
            # Route to appropriate phase
            if args.signal_replay:
                logger.info(f"Signal replay using signals from {prev_output.artifacts.get('signals')}")
                result = coordinator.run_topology('signal_replay', config)
            elif args.backtest:
                logger.info("Running backtest with inherited configuration")
                result = coordinator.run_topology('backtest', config)
            else:
                raise ValueError("No valid action specified for piped input")
                
            # Format output for next phase
            if args.output_format in ['json', 'pipe']:
                output = create_phase_output(result, args)
                print(output.to_json())
            else:
                return result
                
        except Exception as e:
            logger.error(f"Failed to process piped input: {e}")
            raise
    
    # Normal execution continues...


if __name__ == "__main__":
    # Example of complete pipeline
    print("""
    Example Pipeline:
    
    # Generate signals with grid search
    python main.py --signal-generation grid_search.yaml --output-format pipe | \\
    
    # Optimize ensemble weights using those signals  
    python main.py --signal-replay --from-pipe --optimize-ensemble --output-format pipe | \\
    
    # Final validation
    python main.py --backtest --from-pipe --validate
    
    This creates a complete regime-adaptive trading system!
    """)
#!/usr/bin/env python3
"""
Example of main.py enhanced with piping support.

This shows the key changes needed to support Unix-style piping
for multi-phase workflows.
"""

import json
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Assume these imports exist
from src.core.coordinator.coordinator import Coordinator
from src.core.cli import parse_arguments
from examples.enhanced_cli_parser import EnhancedCLIArgs, enhanced_parse_arguments


logger = logging.getLogger(__name__)


class PhaseOutput:
    """Structured output for piping between phases."""
    
    def __init__(self, phase: str, result: Dict[str, Any], config: Dict[str, Any]):
        self.metadata = {
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'execution_id': result.get('execution_id', 'unknown')
        }
        
        # Extract artifacts from result
        self.artifacts = self._extract_artifacts(result)
        
        # Forward config for next phase
        self.config = config
        
        # Phase results
        self.results = self._extract_results(phase, result)
    
    def _extract_artifacts(self, result: Dict[str, Any]) -> Dict[str, str]:
        """Extract artifact paths from execution result."""
        artifacts = {}
        
        # Common artifact patterns
        if 'workspace_path' in result:
            base = result['workspace_path']
            artifacts['workspace'] = base
            artifacts['signals'] = f"{base}/signals/"
            artifacts['events'] = f"{base}/events/"
            artifacts['summary'] = f"{base}/summary.json"
            
        # Phase-specific artifacts
        if 'outputs' in result:
            artifacts.update(result['outputs'])
            
        return artifacts
    
    def _extract_results(self, phase: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key results for phase."""
        results = {
            'success': result.get('success', False),
            'execution_time': result.get('duration_seconds', 0)
        }
        
        # Phase-specific results
        if phase == 'signal_generation':
            results.update({
                'total_signals': result.get('metrics', {}).get('total_signals', 0),
                'windows_processed': result.get('metrics', {}).get('windows', 0),
                'strategies_evaluated': result.get('metrics', {}).get('strategies', 0)
            })
        elif phase == 'signal_replay':
            results.update({
                'sharpe_ratio': result.get('metrics', {}).get('sharpe_ratio', 0),
                'regimes_identified': result.get('metrics', {}).get('regimes', 0)
            })
        elif phase == 'backtest':
            results.update({
                'total_return': result.get('metrics', {}).get('total_return', 0),
                'max_drawdown': result.get('metrics', {}).get('max_drawdown', 0),
                'win_rate': result.get('metrics', {}).get('win_rate', 0)
            })
            
        return results
    
    def to_json(self) -> str:
        """Convert to JSON for piping."""
        return json.dumps({
            'metadata': self.metadata,
            'artifacts': self.artifacts,
            'config': self.config,
            'results': self.results
        }, indent=2 if sys.stdout.isatty() else None, default=str)


def read_piped_input() -> Optional[Dict[str, Any]]:
    """Read and parse piped input from previous phase."""
    if sys.stdin.isatty():
        return None
        
    try:
        input_data = sys.stdin.read()
        parsed = json.loads(input_data)
        
        # Check if it's our phase output format
        if all(key in parsed for key in ['metadata', 'artifacts', 'config', 'results']):
            logger.info(f"Received piped input from phase: {parsed['metadata']['phase']}")
            return parsed
        else:
            # Might be raw config
            return {'config': parsed}
            
    except Exception as e:
        logger.error(f"Failed to parse piped input: {e}")
        return None


def load_config_with_piping(args: EnhancedCLIArgs) -> Dict[str, Any]:
    """Load configuration with piping support."""
    config = {}
    
    # Determine which action flag has a config
    config_path = None
    phase = None
    
    if args.signal_generation:
        config_path = args.signal_generation
        phase = 'signal_generation'
    elif args.backtest:
        config_path = args.backtest
        phase = 'backtest'
    elif args.signal_replay:
        config_path = args.signal_replay
        phase = 'signal_replay'
        
    # Handle different config sources
    if args.from_pipe or config_path == "-":
        # Read from pipe
        piped = read_piped_input()
        if piped:
            config = piped.get('config', {})
            # Add artifact references
            if 'artifacts' in piped:
                config['artifacts'] = piped['artifacts']
                
            # Phase-specific handling
            if phase == 'signal_replay' and 'artifacts' in piped:
                config['signal_path'] = piped['artifacts'].get('signals')
                logger.info(f"Using signals from: {config['signal_path']}")
                
    elif config_path:
        # Load from file
        from src.core.cli.args import load_yaml_config
        config = load_yaml_config(config_path)
        
    # Apply phase-specific options
    if args.optimize_ensemble:
        config['optimization'] = {'type': 'ensemble_weights'}
    if args.validate:
        config['validation'] = {'type': 'out_of_sample'}
        
    return config, phase


def output_results(result: Dict[str, Any], args: EnhancedCLIArgs, 
                  phase: str, config: Dict[str, Any]):
    """Output results in requested format."""
    
    if args.output_format in ['json', 'pipe']:
        # Structured output for piping
        output = PhaseOutput(phase, result, config)
        print(output.to_json())
        
    else:
        # Human-readable output
        print(f"\n{'='*60}")
        print(f"Phase: {phase}")
        print(f"Status: {'SUCCESS' if result.get('success') else 'FAILED'}")
        
        if 'metrics' in result:
            print("\nKey Metrics:")
            for key, value in result['metrics'].items():
                print(f"  {key}: {value}")
                
        if 'outputs' in result:
            print("\nGenerated Artifacts:")
            for key, path in result['outputs'].items():
                print(f"  {key}: {path}")
                
        print(f"{'='*60}\n")


def main():
    """Enhanced main with piping support."""
    # Parse arguments (enhanced version)
    args = enhanced_parse_arguments()
    
    # Setup logging
    level = 'DEBUG' if args.verbose else 'INFO'
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration with piping support
    try:
        config, phase = load_config_with_piping(args)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
        
    # Create coordinator
    coordinator = Coordinator()
    
    # Execute phase
    try:
        logger.info(f"Executing {phase} phase...")
        
        # Add phase-specific handling
        if phase == 'signal_generation':
            result = coordinator.run_topology('signal_generation', config)
            
        elif phase == 'signal_replay':
            # Check if we have signals from previous phase
            if 'signal_path' in config:
                logger.info(f"Replaying signals from: {config['signal_path']}")
            result = coordinator.run_topology('signal_replay', config)
            
        elif phase == 'backtest':
            # May use ensemble weights from previous phase
            if 'artifacts' in config and 'weights' in config['artifacts']:
                config['ensemble_weights'] = config['artifacts']['weights']
            result = coordinator.run_topology('backtest', config)
            
        else:
            # Workflow execution
            result = coordinator.run_workflow(config)
            
        # Output results
        output_results(result, args, phase, config)
        
        return 0 if result.get('success') else 1
        
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        
        # Output error in structured format if piping
        if args.output_format in ['json', 'pipe']:
            error_output = {
                'metadata': {'phase': phase, 'error': True},
                'error': str(e)
            }
            print(json.dumps(error_output))
            
        return 1


if __name__ == '__main__':
    sys.exit(main())


"""
Example Usage:

1. Simple pipeline:
   python main.py --signal-generation grid.yaml --output-format pipe | \\
   python main.py --signal-replay --from-pipe --optimize-ensemble | \\
   python main.py --backtest --from-pipe --validate

2. With intermediate files:
   python main.py --signal-generation grid.yaml --output-format json > phase1.json
   cat phase1.json | python main.py --signal-replay - --optimize-ensemble

3. Parallel processing:
   parallel -j 4 'python main.py --signal-generation {} --output-format pipe' \\
     ::: configs/*.yaml | \\
   python main.py --signal-replay --from-pipe --merge --optimize

4. Conditional execution:
   python main.py --signal-generation grid.yaml --output-format pipe | \\
   jq -e '.results.total_signals > 1000' && \\
   python main.py --signal-replay --from-pipe || echo "Not enough signals"
"""
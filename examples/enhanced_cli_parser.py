"""
Enhanced CLI parser with piping support.

This shows the minimal changes needed to support Unix-style piping
for multi-phase workflows.
"""

import argparse
import sys
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass 
class EnhancedCLIArgs:
    """Enhanced CLI args with piping support."""
    # Core arguments
    config: Optional[str] = None
    
    # Clean topology action flags
    signal_generation: Optional[str] = None
    backtest: Optional[str] = None
    signal_replay: Optional[str] = None
    optimize: Optional[str] = None
    
    # Workflow and sequence arguments  
    workflow: Optional[str] = None
    sequence: Optional[str] = None
    
    # Piping support
    from_pipe: bool = False
    output_format: str = 'human'  # human, json, pipe
    
    # Ensemble optimization
    optimize_ensemble: bool = False
    validate: bool = False
    
    # Data arguments
    dataset: Optional[str] = None
    bars: Optional[int] = None
    split_ratio: Optional[float] = None
    
    # Other arguments...
    verbose: bool = False
    dry_run: bool = False


def create_enhanced_parser() -> argparse.ArgumentParser:
    """Create parser with piping support."""
    parser = argparse.ArgumentParser(
        description='ADMF-PC with Unix-style piping support',
        epilog="""
Examples:
  # Single phase
  python main.py --backtest config.yaml
  
  # Piped workflow  
  python main.py --signal-generation grid.yaml --output-format pipe | \\
  python main.py --signal-replay --from-pipe --optimize-ensemble | \\
  python main.py --backtest --from-pipe --validate
  
  # Read from stdin
  cat config.yaml | python main.py --signal-generation -
        """
    )
    
    # Action flags group
    action_group = parser.add_mutually_exclusive_group()
    
    action_group.add_argument(
        '--signal-generation', '-sg',
        metavar='CONFIG',
        type=str,
        help='Generate signals (use "-" for stdin)'
    )
    
    action_group.add_argument(
        '--backtest', '-bt',
        metavar='CONFIG', 
        type=str,
        help='Run backtest (use "-" for stdin)'
    )
    
    action_group.add_argument(
        '--signal-replay', '-sr',
        metavar='CONFIG',
        type=str,
        help='Replay signals (use "-" for stdin)'
    )
    
    # Piping support
    pipe_group = parser.add_argument_group('Piping Options')
    
    pipe_group.add_argument(
        '--from-pipe',
        action='store_true',
        help='Read previous phase output from stdin'
    )
    
    pipe_group.add_argument(
        '--output-format', '-of',
        choices=['human', 'json', 'pipe'],
        default='human',
        help='Output format (pipe is alias for json with no pretty-print)'
    )
    
    # Phase-specific options
    phase_group = parser.add_argument_group('Phase Options')
    
    phase_group.add_argument(
        '--optimize-ensemble',
        action='store_true',
        help='Optimize ensemble weights (signal-replay phase)'
    )
    
    phase_group.add_argument(
        '--validate',
        action='store_true',
        help='Run out-of-sample validation (backtest phase)'
    )
    
    phase_group.add_argument(
        '--merge',
        action='store_true',
        help='Merge multiple piped inputs (for parallel processing)'
    )
    
    # Standard options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file (YAML)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be executed without running'
    )
    
    return parser


def detect_piped_input() -> bool:
    """Check if we're receiving piped input."""
    return not sys.stdin.isatty()


def read_piped_config() -> Dict[str, Any]:
    """Read configuration from piped input."""
    try:
        input_data = sys.stdin.read()
        
        # Try to parse as JSON first (from previous phase)
        try:
            data = json.loads(input_data)
            # If it has our phase output structure, extract config
            if 'metadata' in data and 'config' in data:
                return {
                    'inherited_from': data['metadata'],
                    'artifacts': data['artifacts'],
                    **data['config']
                }
            return data
        except json.JSONDecodeError:
            # Maybe it's YAML config piped directly
            import yaml
            return yaml.safe_load(input_data)
            
    except Exception as e:
        raise ValueError(f"Failed to read piped input: {e}")


def enhanced_parse_arguments():
    """Parse arguments with piping support."""
    parser = create_enhanced_parser()
    args = parser.parse_args()
    
    # Convert to enhanced dataclass
    cli_args = EnhancedCLIArgs(
        config=args.config,
        signal_generation=getattr(args, 'signal_generation', None),
        backtest=getattr(args, 'backtest', None),
        signal_replay=getattr(args, 'signal_replay', None),
        from_pipe=args.from_pipe,
        output_format=args.output_format,
        optimize_ensemble=getattr(args, 'optimize_ensemble', False),
        validate=getattr(args, 'validate', False),
        verbose=args.verbose,
        dry_run=args.dry_run
    )
    
    # Auto-detect piped input
    if detect_piped_input() and not cli_args.config:
        cli_args.from_pipe = True
    
    # Handle stdin config
    config_args = [
        cli_args.signal_generation,
        cli_args.backtest, 
        cli_args.signal_replay
    ]
    
    for conf in config_args:
        if conf == "-":
            cli_args.from_pipe = True
            break
    
    return cli_args


# Usage example
if __name__ == "__main__":
    # Example 1: Parse normal arguments
    print("Example 1: Normal execution")
    sys.argv = ['main.py', '--backtest', 'config.yaml']
    args = enhanced_parse_arguments()
    print(f"Action: backtest, Config: {args.config}")
    
    # Example 2: Piped workflow
    print("\nExample 2: Piped workflow")
    sys.argv = ['main.py', '--signal-replay', '--from-pipe', '--optimize-ensemble']
    args = enhanced_parse_arguments()
    print(f"From pipe: {args.from_pipe}, Optimize ensemble: {args.optimize_ensemble}")
    
    # Example 3: Output formatting
    print("\nExample 3: JSON output")
    sys.argv = ['main.py', '--signal-generation', 'grid.yaml', '--output-format', 'json']
    args = enhanced_parse_arguments()
    print(f"Output format: {args.output_format}")
    
    # Show complete pipeline
    print("\n" + "="*60)
    print("Complete Regime-Adaptive Pipeline:")
    print("="*60)
    print("""
# Phase 1: Generate signals with parameter grid
python main.py --signal-generation grid_search.yaml --output-format pipe | \\

# Phase 2: Optimize ensemble weights per regime  
python main.py --signal-replay --from-pipe --optimize-ensemble --output-format pipe | \\

# Phase 3: Validate on out-of-sample data
python main.py --backtest --from-pipe --validate --output-format human

# Or save intermediate results:
python main.py --signal-generation grid.yaml --output-format json > signals.json
python main.py --signal-replay signals.json --optimize-ensemble --output-format json > weights.json  
python main.py --backtest final.yaml --ensemble-weights weights.json
    """)
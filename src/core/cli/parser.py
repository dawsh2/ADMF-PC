"""
Command-line argument parser for ADMF-PC.

Provides structured argument parsing with proper type annotations
and separation of concerns from main application logic.
"""

import argparse
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class CLIArgs:
    """Structured representation of CLI arguments."""
    # Core arguments
    config: Optional[str] = None
    
    # Clean topology action flags (mutually exclusive)
    signal_generation: bool = False
    backtest: bool = False
    signal_replay: bool = False
    optimize: bool = False
    
    # Workflow and sequence arguments  
    workflow: Optional[str] = None
    sequence: Optional[str] = None
    
    # Data arguments
    dataset: Optional[str] = None
    bars: Optional[int] = None
    split_ratio: Optional[float] = None
    
    # Execution arguments
    parallel: Optional[int] = None
    checkpoint: Optional[str] = None
    output_dir: Optional[str] = None
    
    # Logging arguments
    log_level: str = 'INFO'
    log_events: List[str] = None
    log_file: Optional[str] = None
    log_json: bool = False
    verbose: bool = False
    
    # Development arguments
    dry_run: bool = False
    profile: bool = False
    schema_docs: bool = False

    def __post_init__(self):
        if self.log_events is None:
            self.log_events = []


def parse_arguments() -> CLIArgs:
    """Parse command line arguments and return structured CLIArgs object."""
    parser = argparse.ArgumentParser(
        description='ADMF-PC: Adaptive Decision Making Framework - Protocol Components'
    )
    
    # Core arguments
    parser.add_argument(
        '--config',
        type=str,
        required=False,  # Made optional to allow --schema-docs without config
        help='Path to configuration file (YAML)'
    )
    
    # Clean topology action flags - each implies a specific topology
    action_group = parser.add_mutually_exclusive_group()
    
    action_group.add_argument(
        '--signal-generation', '-sg',
        action='store_true',
        help='Generate trading signals from strategies'
    )
    
    action_group.add_argument(
        '--backtest', '-bt',
        action='store_true',
        help='Run backtest simulation'
    )
    
    action_group.add_argument(
        '--signal-replay', '-sr',
        action='store_true',
        help='Replay previously generated signals'
    )
    
    action_group.add_argument(
        '--optimize', '-opt',
        action='store_true',
        help='Run parameter optimization'
    )
    
    # Workflow for complex multi-phase executions
    action_group.add_argument(
        '--workflow', '-w',
        type=str,
        help='Execute a workflow pattern (e.g., research_pipeline)'
    )
    
    # Sequence patterns
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        help='Apply sequence pattern (e.g., walk_forward, parameter_sweep)'
    )
    
    # Data arguments
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['train', 'test', 'full'],
        default=None,
        help='Dataset to use (enables reproducible train/test splits)'
    )
    
    parser.add_argument(
        '--bars',
        type=int,
        default=None,
        help='Limit data to first N bars (useful for testing)'
    )
    
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=None,
        help='Train/test split ratio when dataset is "full"'
    )
    
    # Execution arguments
    parser.add_argument(
        '--parallel',
        type=int,
        default=None,
        help='Number of parallel workers for optimization'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Resume from checkpoint file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for output files'
    )
    
    # Logging arguments
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['TRACE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-events',
        type=str,
        nargs='*',
        choices=['BAR', 'INDICATOR', 'SIGNAL', 'ORDER', 'FILL', 'PORTFOLIO', 'TRADE_LOOP'],
        default=[],
        help='Enable detailed logging for specific event types'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Log to file instead of console'
    )
    
    parser.add_argument(
        '--log-json',
        action='store_true',
        help='Use structured JSON logging format'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (equivalent to --log-level DEBUG)'
    )
    
    # Development arguments
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without executing'
    )
    
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )
    
    parser.add_argument(
        '--schema-docs',
        action='store_true',
        help='Print configuration schema documentation and exit'
    )
    
    args = parser.parse_args()
    
    # Convert to structured CLIArgs
    return CLIArgs(
        config=args.config,
        # Clean topology action flags
        signal_generation=getattr(args, 'signal_generation', False),
        backtest=getattr(args, 'backtest', False),
        signal_replay=getattr(args, 'signal_replay', False),
        optimize=getattr(args, 'optimize', False),
        # Workflow and sequence arguments
        workflow=getattr(args, 'workflow', None),
        sequence=getattr(args, 'sequence', None),
        # Data arguments
        dataset=args.dataset,
        bars=args.bars,
        split_ratio=getattr(args, 'split_ratio', None),
        # Execution arguments
        parallel=args.parallel,
        checkpoint=args.checkpoint,
        output_dir=getattr(args, 'output_dir', None),
        # Logging arguments
        log_level=getattr(args, 'log_level', 'INFO'),
        log_events=getattr(args, 'log_events', []),
        log_file=getattr(args, 'log_file', None),
        log_json=getattr(args, 'log_json', False),
        verbose=args.verbose,
        # Development arguments
        dry_run=getattr(args, 'dry_run', False),
        profile=args.profile,
        schema_docs=getattr(args, 'schema_docs', False)
    )
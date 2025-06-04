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
    config: str
    
    # Execution mode arguments
    mode: Optional[str] = None
    signal_log: Optional[str] = None
    signal_output: Optional[str] = None
    weights: Optional[str] = None
    
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
        required=True,
        help='Path to configuration file (YAML)'
    )
    
    # Execution mode arguments
    parser.add_argument(
        '--mode',
        type=str,
        choices=['backtest', 'optimization', 'signal-generation', 'signal-replay', 'live'],
        default=None,
        help='Override execution mode from config'
    )
    
    parser.add_argument(
        '--signal-log',
        type=str,
        default=None,
        help='Path to signal log file for replay mode'
    )
    
    parser.add_argument(
        '--signal-output',
        type=str,
        default=None,
        help='Path to save generated signals (signal-generation mode)'
    )
    
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='JSON string or file with strategy weights for signal-replay mode'
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
    
    args = parser.parse_args()
    
    # Convert to structured CLIArgs
    return CLIArgs(
        config=args.config,
        mode=args.mode,
        signal_log=getattr(args, 'signal_log', None),
        signal_output=getattr(args, 'signal_output', None),
        weights=args.weights,
        dataset=args.dataset,
        bars=args.bars,
        split_ratio=getattr(args, 'split_ratio', None),
        parallel=args.parallel,
        checkpoint=args.checkpoint,
        output_dir=getattr(args, 'output_dir', None),
        log_level=getattr(args, 'log_level', 'INFO'),
        log_events=getattr(args, 'log_events', []),
        log_file=getattr(args, 'log_file', None),
        log_json=getattr(args, 'log_json', False),
        verbose=args.verbose,
        dry_run=getattr(args, 'dry_run', False),
        profile=args.profile
    )
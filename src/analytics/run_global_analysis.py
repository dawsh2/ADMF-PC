#!/usr/bin/env python3
"""
Run global signal analysis directly from the traces store.

Usage:
    python -m src.analytics.run_global_analysis --strategy-type bollinger_bands
    python -m src.analytics.run_global_analysis --symbol SPY --timeframe 5m
    python -m src.analytics.run_global_analysis --launch
"""

import argparse
import logging
from pathlib import Path
from papermill_runner import PapermillNotebookRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run global signal analysis from traces store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all bollinger_bands strategies
  python -m src.analytics.run_global_analysis --strategy-type bollinger_bands
  
  # Analyze all strategies for SPY 5m
  python -m src.analytics.run_global_analysis --symbol SPY --timeframe 5m
  
  # Analyze specific strategy type and launch notebook
  python -m src.analytics.run_global_analysis --strategy-type ma_crossover --launch
  
  # Just create notebook without executing
  python -m src.analytics.run_global_analysis --no-execute --launch
        """
    )
    
    # Filter arguments
    parser.add_argument('--strategy-type', type=str, 
                       help='Filter by strategy type (e.g., bollinger_bands, ma_crossover)')
    parser.add_argument('--symbol', type=str, default='SPY',
                       help='Filter by symbol (default: SPY)')
    parser.add_argument('--timeframe', type=str, default='5m',
                       help='Filter by timeframe (default: 5m)')
    
    # Execution arguments
    parser.add_argument('--no-execute', action='store_true',
                       help='Just create parameterized notebook without executing')
    parser.add_argument('--launch', action='store_true',
                       help='Launch Jupyter after creating notebook')
    parser.add_argument('--html', action='store_true',
                       help='Generate HTML report')
    parser.add_argument('--output-dir', type=Path,
                       help='Directory to save analysis (default: current directory)')
    
    # Analysis parameters
    parser.add_argument('--performance-limit', type=int, default=100,
                       help='Limit number of strategies to analyze (default: 100)')
    parser.add_argument('--execution-cost', type=float, default=1.0,
                       help='Execution cost in basis points (default: 1.0)')
    
    args = parser.parse_args()
    
    # Initialize runner
    try:
        runner = PapermillNotebookRunner()
    except ImportError:
        logger.error("Papermill not installed. Run: pip install papermill")
        return 1
    
    # Run global analysis
    logger.info("🚀 Starting global signal analysis...")
    
    notebook_path = runner.run_global_analysis(
        strategy_type=args.strategy_type,
        symbol=args.symbol,
        timeframe=args.timeframe,
        execute=not args.no_execute,
        launch=args.launch,
        generate_html=args.html,
        output_dir=args.output_dir
    )
    
    if notebook_path:
        logger.info(f"✅ Analysis complete: {notebook_path}")
        return 0
    else:
        logger.error("❌ Analysis failed")
        return 1


if __name__ == '__main__':
    exit(main())
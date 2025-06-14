#!/usr/bin/env python3
"""
Analytics CLI - Simplified version without external dependencies.
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.analytics.strategy_filter import StrategyFilter, analyze_grid_search
from src.analytics.mining.pattern_miner import PatternMiner


def main():
    parser = argparse.ArgumentParser(description='ADMF-PC Analytics Toolkit')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Filter strategies by performance')
    filter_parser.add_argument('workspace', help='Workspace path')
    filter_parser.add_argument('data_path', help='Market data path')
    filter_parser.add_argument('--min-trades', type=int, default=10, help='Minimum trades')
    filter_parser.add_argument('--min-sharpe', type=float, default=0.5, help='Minimum Sharpe')
    filter_parser.add_argument('--classifier', help='Classifier signals path')
    
    # Mine command
    mine_parser = subparsers.add_parser('mine', help='Mine for patterns')
    mine_parser.add_argument('workspace', help='Workspace path')
    mine_parser.add_argument('data_path', help='Market data path')
    mine_parser.add_argument('--output', help='Output directory')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Execute SQL query')
    query_parser.add_argument('workspace', help='Workspace path')
    query_parser.add_argument('-q', '--query', help='SQL query string')
    query_parser.add_argument('-f', '--file', help='SQL query file')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Show workspace summary')
    summary_parser.add_argument('workspace', help='Workspace path')
    
    args = parser.parse_args()
    
    if args.command == 'filter':
        print(f"Analyzing workspace: {args.workspace}")
        # Import and use StrategyFilter directly with CLI args
        from src.analytics.strategy_filter import StrategyFilter
        
        filter = StrategyFilter(args.workspace)
        try:
            results = filter.comprehensive_filter(
                data_path=args.data_path,
                classifier_path=args.classifier,
                min_trades=args.min_trades,
                min_sharpe=args.min_sharpe,
                commission_pct=0.01,
                slippage_pct=0.01
            )
        finally:
            filter.close()
        print("\n=== Analysis Summary ===")
        for name, df in results.items():
            print(f"{name}: {len(df)} entries")
            
    elif args.command == 'mine':
        print(f"Mining patterns in workspace: {args.workspace}")
        miner = PatternMiner(args.workspace)
        try:
            output_dir = Path(args.output) if args.output else None
            results = miner.mine_all_patterns(args.data_path, output_dir)
            
            print("\n=== Pattern Mining Results ===")
            for name, df in results.items():
                if not df.empty:
                    print(f"{name}: {len(df)} patterns found")
                    
            # Show top signal combinations
            if not results['signal_combinations'].empty:
                print("\nTop Signal Combinations:")
                top = results['signal_combinations'].nlargest(3, 'avg_return')
                print(top[['entry_type', 'exit_type', 'avg_return', 'occurrences']].to_string(index=False))
        finally:
            miner.close()
            
    elif args.command == 'query':
        import duckdb
        
        db_path = Path(args.workspace) / "analytics.duckdb"
        con = duckdb.connect(str(db_path))
        
        try:
            if args.file:
                with open(args.file, 'r') as f:
                    sql = f.read()
            elif args.query:
                sql = args.query
            else:
                print("Provide either --query or --file")
                return
                
            result = con.execute(sql).df()
            print(result.to_string())
        finally:
            con.close()
            
    elif args.command == 'summary':
        import json
        
        workspace_path = Path(args.workspace)
        
        # Check metadata
        metadata_file = workspace_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"Workspace: {metadata.get('id', 'Unknown')}")
            print(f"Config: {metadata.get('config_path', 'Unknown')}")
            
        # Count files
        signal_files = list(workspace_path.glob("traces/*/signals/*/*.parquet"))
        print(f"\nSignal files: {len(signal_files)}")
        
        # Get strategy types
        strategies = set()
        for f in signal_files:
            strategies.add(f.parent.name)
        print(f"Strategy types: {', '.join(sorted(strategies))}")
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
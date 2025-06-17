#!/usr/bin/env python3
"""
Analyze classifier regime distributions using DuckDB analytics database
to check if distributions are more balanced after code improvements.
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict

def analyze_classifier_regimes_from_db(db_path: str):
    """Analyze classifier regime distributions from DuckDB analytics database."""
    
    try:
        # Connect to DuckDB
        conn = duckdb.connect(db_path)
        
        # First, let's see what tables are available
        tables_query = "SHOW TABLES"
        tables = conn.execute(tables_query).fetchall()
        print("Available tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
        print("\nLooking for classifier-related tables...")
        
        # Look for classifier tables
        classifier_tables = []
        signal_tables = []
        
        for table in tables:
            table_name = table[0]
            if 'classifier' in table_name.lower():
                classifier_tables.append(table_name)
            elif any(keyword in table_name.lower() for keyword in ['signal', 'strategy']):
                signal_tables.append(table_name)
        
        print(f"Found {len(classifier_tables)} classifier tables")
        print(f"Found {len(signal_tables)} signal tables")
        
        # If we have classifier tables, analyze them
        if classifier_tables:
            return analyze_classifier_tables(conn, classifier_tables)
        
        # Otherwise, try to find regime data in signal tables
        elif signal_tables:
            return analyze_signal_tables_for_regimes(conn, signal_tables)
        
        else:
            print("No suitable tables found for regime analysis")
            return None
            
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def analyze_classifier_tables(conn, classifier_tables):
    """Analyze regime distributions from classifier tables."""
    results = defaultdict(list)
    
    for table_name in classifier_tables:
        print(f"\nAnalyzing table: {table_name}")
        
        try:
            # Get table schema
            schema_query = f"DESCRIBE {table_name}"
            schema = conn.execute(schema_query).fetchall()
            
            # Look for regime/state columns
            regime_columns = []
            for col_info in schema:
                col_name = col_info[0]
                if any(keyword in col_name.lower() for keyword in ['regime', 'state', 'classification']):
                    regime_columns.append(col_name)
            
            if not regime_columns:
                print(f"  No regime columns found in {table_name}")
                continue
            
            print(f"  Found regime columns: {regime_columns}")
            
            # Get row count
            count_query = f"SELECT COUNT(*) FROM {table_name}"
            total_rows = conn.execute(count_query).fetchone()[0]
            print(f"  Total rows: {total_rows:,}")
            
            # Analyze each regime column
            for regime_col in regime_columns:
                print(f"    Analyzing column: {regime_col}")
                
                # Get regime distribution
                dist_query = f"""
                SELECT {regime_col}, COUNT(*) as count
                FROM {table_name}
                WHERE {regime_col} IS NOT NULL
                GROUP BY {regime_col}
                ORDER BY count DESC
                """
                
                distribution = conn.execute(dist_query).fetchall()
                
                if not distribution:
                    print(f"      No data found for {regime_col}")
                    continue
                
                # Calculate statistics
                total_valid = sum(row[1] for row in distribution)
                regime_stats = {}
                
                for regime_value, count in distribution:
                    percentage = (count / total_valid * 100) if total_valid > 0 else 0
                    regime_stats[str(regime_value)] = {
                        'count': count,
                        'percentage': round(percentage, 2)
                    }
                
                # Calculate balance metrics
                percentages = [stats['percentage'] for stats in regime_stats.values()]
                if percentages:
                    entropy = -sum(p/100 * np.log2(p/100) for p in percentages if p > 0)
                    max_regime_pct = max(percentages)
                    min_regime_pct = min(percentages)
                    balance_ratio = min_regime_pct / max_regime_pct if max_regime_pct > 0 else 0
                else:
                    entropy = 0
                    max_regime_pct = 0
                    min_regime_pct = 0
                    balance_ratio = 0
                
                result = {
                    'table_name': table_name,
                    'regime_column': regime_col,
                    'total_rows': total_rows,
                    'valid_regime_rows': total_valid,
                    'unique_regimes': len(regime_stats),
                    'regime_distribution': regime_stats,
                    'entropy': entropy,
                    'max_regime_pct': max_regime_pct,
                    'min_regime_pct': min_regime_pct,
                    'balance_ratio': balance_ratio
                }
                
                # Extract classifier type from table name
                classifier_type = extract_classifier_type(table_name)
                results[classifier_type].append(result)
                
                print(f"      Distribution: {regime_stats}")
                print(f"      Entropy: {entropy:.3f}")
                print(f"      Balance ratio: {balance_ratio:.3f}")
                
        except Exception as e:
            print(f"  Error analyzing {table_name}: {e}")
            continue
    
    return dict(results)

def analyze_signal_tables_for_regimes(conn, signal_tables):
    """Look for regime data in signal tables."""
    results = defaultdict(list)
    
    for table_name in signal_tables[:10]:  # Limit to first 10 to avoid too much output
        print(f"\nChecking table: {table_name}")
        
        try:
            # Get table schema
            schema_query = f"DESCRIBE {table_name}"
            schema = conn.execute(schema_query).fetchall()
            
            # Look for any columns that might contain regime info
            interesting_columns = []
            for col_info in schema:
                col_name = col_info[0]
                if any(keyword in col_name.lower() for keyword in 
                      ['regime', 'state', 'classification', 'cluster', 'category']):
                    interesting_columns.append(col_name)
            
            if interesting_columns:
                print(f"  Found interesting columns: {interesting_columns}")
                # Could analyze these if needed
        
        except Exception as e:
            print(f"  Error checking {table_name}: {e}")
            continue
    
    return dict(results)

def extract_classifier_type(table_name):
    """Extract classifier type from table name."""
    # Remove common prefixes/suffixes
    name = table_name.lower()
    
    # Common classifier patterns
    if 'hidden_markov' in name or 'hmm' in name:
        return 'hidden_markov'
    elif 'market_regime' in name:
        return 'market_regime'
    elif 'microstructure' in name:
        return 'microstructure'
    elif 'multi_timeframe_trend' in name:
        return 'multi_timeframe_trend'
    elif 'volatility_momentum' in name:
        return 'volatility_momentum'
    else:
        # Try to extract from table name pattern
        parts = name.split('_')
        if len(parts) >= 2:
            return '_'.join(parts[:2])
        else:
            return table_name

def create_summary_report(results):
    """Create summary report of regime distributions."""
    print("\n" + "="*80)
    print("CLASSIFIER REGIME DISTRIBUTION ANALYSIS (from DuckDB)")
    print("="*80)
    
    if not results:
        print("No results to display")
        return []
    
    overall_stats = []
    
    for classifier_type, analyses in results.items():
        print(f"\n{classifier_type.upper().replace('_', ' ')}")
        print("-" * 60)
        
        type_stats = {
            'classifier_type': classifier_type,
            'num_analyses': len(analyses),
            'avg_entropy': 0,
            'avg_balance_ratio': 0,
            'avg_max_regime_pct': 0,
            'regimes_found': set()
        }
        
        entropies = []
        balance_ratios = []
        max_regime_pcts = []
        
        for analysis in analyses:
            entropies.append(analysis['entropy'])
            balance_ratios.append(analysis['balance_ratio'])
            max_regime_pcts.append(analysis['max_regime_pct'])
            
            # Track unique regimes
            for regime in analysis['regime_distribution'].keys():
                type_stats['regimes_found'].add(regime)
            
            print(f"  Table: {analysis['table_name']}")
            print(f"    Column: {analysis['regime_column']}")
            print(f"    Total rows: {analysis['total_rows']:,}")
            print(f"    Valid regime rows: {analysis['valid_regime_rows']:,}")
            print(f"    Unique regimes: {analysis['unique_regimes']}")
            
            # Show distribution
            dist_str = ", ".join([f"{regime}: {stats['percentage']:.1f}%" 
                                for regime, stats in analysis['regime_distribution'].items()])
            print(f"    Distribution: {dist_str}")
            print(f"    Entropy: {analysis['entropy']:.3f}")
            print(f"    Balance ratio: {analysis['balance_ratio']:.3f}")
            print()
        
        # Calculate averages
        if entropies:
            type_stats['avg_entropy'] = np.mean(entropies)
            type_stats['avg_balance_ratio'] = np.mean(balance_ratios)
            type_stats['avg_max_regime_pct'] = np.mean(max_regime_pcts)
            type_stats['regimes_found'] = sorted(list(type_stats['regimes_found']))
        
        overall_stats.append(type_stats)
        
        print(f"  SUMMARY for {classifier_type}:")
        print(f"    Analyses: {type_stats['num_analyses']}")
        print(f"    Average entropy: {type_stats['avg_entropy']:.3f}")
        print(f"    Average balance ratio: {type_stats['avg_balance_ratio']:.3f}")
        print(f"    Average max regime %: {type_stats['avg_max_regime_pct']:.1f}%")
        print(f"    Regimes found: {type_stats['regimes_found']}")
    
    return overall_stats

def create_balance_assessment(overall_stats):
    """Create balance assessment."""
    if not overall_stats:
        print("\nNo data for balance assessment")
        return
    
    print("\n" + "="*80)
    print("BALANCE ASSESSMENT")
    print("="*80)
    
    # Sort by balance metrics
    by_entropy = sorted(overall_stats, key=lambda x: x['avg_entropy'], reverse=True)
    by_balance_ratio = sorted(overall_stats, key=lambda x: x['avg_balance_ratio'], reverse=True)
    
    print("\nMOST BALANCED (by entropy - higher is better):")
    print("-" * 50)
    for i, stats in enumerate(by_entropy):
        print(f"{i+1}. {stats['classifier_type']}")
        print(f"   Entropy: {stats['avg_entropy']:.3f}")
        print(f"   Balance ratio: {stats['avg_balance_ratio']:.3f}")
        print(f"   Max regime %: {stats['avg_max_regime_pct']:.1f}%")
        print()
    
    print("\nMOST BALANCED (by balance ratio):")
    print("-" * 50)
    for i, stats in enumerate(by_balance_ratio):
        print(f"{i+1}. {stats['classifier_type']}")
        print(f"   Balance ratio: {stats['avg_balance_ratio']:.3f}")
        print(f"   Entropy: {stats['avg_entropy']:.3f}")
        print(f"   Max regime %: {stats['avg_max_regime_pct']:.1f}%")
        print()
    
    # Balance quality assessment
    print("\nBALANCE QUALITY ASSESSMENT:")
    print("-" * 50)
    
    excellent = []
    good = []
    fair = []
    poor = []
    
    for stats in overall_stats:
        if stats['avg_balance_ratio'] >= 0.8 and stats['avg_entropy'] >= 1.5:
            excellent.append(stats['classifier_type'])
        elif stats['avg_balance_ratio'] >= 0.6 and stats['avg_entropy'] >= 1.2:
            good.append(stats['classifier_type'])
        elif stats['avg_balance_ratio'] >= 0.4 and stats['avg_entropy'] >= 0.8:
            fair.append(stats['classifier_type'])
        else:
            poor.append(stats['classifier_type'])
    
    print(f"EXCELLENT balance (ratio ≥ 0.8, entropy ≥ 1.5): {len(excellent)}")
    for clf in excellent:
        print(f"  - {clf}")
    
    print(f"\nGOOD balance (ratio ≥ 0.6, entropy ≥ 1.2): {len(good)}")
    for clf in good:
        print(f"  - {clf}")
    
    print(f"\nFAIR balance (ratio ≥ 0.4, entropy ≥ 0.8): {len(fair)}")
    for clf in fair:
        print(f"  - {clf}")
    
    print(f"\nPOOR balance (below fair thresholds): {len(poor)}")
    for clf in poor:
        print(f"  - {clf}")

def main():
    workspace_path = "/Users/daws/ADMF-PC/workspaces/complete_strategy_grid_v1_acf6c935"
    db_path = f"{workspace_path}/analytics.duckdb"
    
    print("Analyzing classifier regime distributions from DuckDB...")
    print(f"Database: {db_path}")
    
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return
    
    # Analyze regimes from database
    results = analyze_classifier_regimes_from_db(db_path)
    
    if not results:
        print("No regime analysis results found!")
        return
    
    # Create summary report
    overall_stats = create_summary_report(results)
    
    # Create balance assessment
    create_balance_assessment(overall_stats)
    
    # Save results
    output_file = "/Users/daws/ADMF-PC/classifier_regime_analysis_duckdb.json"
    with open(output_file, 'w') as f:
        # Convert sets to lists for JSON serialization
        json_stats = []
        for stats in overall_stats:
            stats_copy = stats.copy()
            if isinstance(stats_copy.get('regimes_found'), set):
                stats_copy['regimes_found'] = list(stats_copy['regimes_found'])
            json_stats.append(stats_copy)
        
        json.dump({
            'workspace': workspace_path,
            'database': db_path,
            'analysis_summary': json_stats,
            'detailed_results': results
        }, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
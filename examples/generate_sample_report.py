#!/usr/bin/env python3
"""
Sample script demonstrating the ADMF-PC reporting system

This script shows how to:
1. Create a sample workspace with results
2. Generate an HTML report
3. View the generated report

Usage:
    python examples/generate_sample_report.py
"""

import sys
from pathlib import Path
import json
import tempfile
import webbrowser
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from reporting.basic_report import BacktestReportGenerator


def create_sample_workspace():
    """Create a sample workspace with realistic backtest results"""
    
    # Create temporary workspace
    workspace = Path(tempfile.mkdtemp(prefix='admf_sample_'))
    print(f"Creating sample workspace: {workspace}")
    
    # Create directory structure
    (workspace / 'performance').mkdir()
    (workspace / 'signals').mkdir() 
    (workspace / 'metadata').mkdir()
    (workspace / 'visualizations').mkdir()
    
    # Create sample performance metrics
    performance_data = {
        "trial_id": "sample_momentum_strategy",
        "parameters": {
            "lookback": 20,
            "threshold": 0.02,
            "timeframe": "1D"
        },
        "metrics": {
            "total_return": 23.4,
            "sharpe_ratio": 1.67,
            "max_drawdown": -12.3,
            "win_rate": 62.1,
            "profit_factor": 2.15,
            "total_trades": 178,
            "avg_trade_return": 0.42,
            "volatility": 15.8,
            "calmar_ratio": 1.90,
            "sortino_ratio": 2.45
        },
        "period": {
            "start": "2023-01-01",
            "end": "2023-12-31"
        },
        "symbols": ["SPY", "QQQ", "IWM"],
        "strategy_type": "momentum"
    }
    
    # Save performance data
    with open(workspace / 'performance' / 'trial_0.json', 'w') as f:
        json.dump(performance_data, f, indent=2)
    
    # Create sample signals
    signals = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(50):  # 50 sample signals
        signal_date = base_date + timedelta(days=i*7)  # Weekly signals
        
        signal = {
            "timestamp": signal_date.isoformat(),
            "symbol": ["SPY", "QQQ", "IWM"][i % 3],
            "action": "BUY" if i % 3 == 0 else "SELL" if i % 3 == 1 else "HOLD",
            "strength": round(0.3 + (i % 10) * 0.07, 2),  # 0.3 to 0.93
            "strategy": "momentum",
            "price": 100 + i * 0.5,  # Sample prices
            "quantity": 100 * (1 + i % 5)  # Sample quantities
        }
        signals.append(signal)
    
    # Save signals
    with open(workspace / 'signals' / 'trial_0.jsonl', 'w') as f:
        for signal in signals:
            f.write(json.dumps(signal) + '\n')
    
    # Create sample metadata
    metadata = {
        "workflow_id": "sample_workflow_001",
        "strategy_name": "SPY Momentum Strategy",
        "created_at": datetime.now().isoformat(),
        "parameters": {
            "data_source": "yahoo_finance",
            "universe": ["SPY", "QQQ", "IWM"],
            "strategy_type": "momentum",
            "optimization_method": "grid_search"
        }
    }
    
    with open(workspace / 'metadata' / 'workflow_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return workspace


def demonstrate_reporting():
    """Demonstrate the reporting functionality"""
    
    print("ADMF-PC Reporting System Demo")
    print("=" * 40)
    
    # Step 1: Create sample workspace
    print("\n1. Creating sample workspace with backtest results...")
    workspace = create_sample_workspace()
    
    # Step 2: Generate report
    print("\n2. Generating HTML report...")
    try:
        generator = BacktestReportGenerator(workspace)
        report_path = generator.generate_report()
        print(f"   ✓ Report generated successfully!")
        print(f"   📄 Location: {report_path}")
        
        # Step 3: Show file size and contents
        file_size = report_path.stat().st_size / 1024  # KB
        print(f"   📊 File size: {file_size:.1f} KB")
        
        # Step 4: Open in browser
        print("\n3. Opening report in browser...")
        webbrowser.open(f'file://{report_path.absolute()}')
        
        print("\n✨ Demo completed successfully!")
        print(f"\nThe report includes:")
        print("   • Key performance metrics")
        print("   • Interactive Plotly charts")
        print("   • Professional styling")
        print("   • Mobile-responsive design")
        
        print(f"\n💡 Pro tip: The report is self-contained HTML")
        print("   You can share it, email it, or host it anywhere!")
        
        return workspace, report_path
        
    except Exception as e:
        print(f"   ❌ Error generating report: {e}")
        return workspace, None


def show_integration_example():
    """Show how this integrates with the main ADMF-PC system"""
    
    print("\n" + "=" * 50)
    print("INTEGRATION WITH ADMF-PC")
    print("=" * 50)
    
    print("""
How this fits into ADMF-PC workflows:

1. COORDINATOR completes a backtest workflow
   └── Saves results to workspace (performance/, signals/, metadata/)

2. CONFIGURATION enables reporting
   └── output:
         generate_report: true
         report_type: "comprehensive"

3. COORDINATOR calls report generator
   └── generator = BacktestReportGenerator(workspace_path)
   └── report_path = generator.generate_report()

4. USER receives professional HTML report
   └── Can view locally, share, or publish

Key Benefits:
✓ Integrates with existing workspace management
✓ Uses calculated metrics from signal analysis
✓ No additional dependencies (Plotly already used)
✓ Self-contained HTML files (easy sharing)
✓ Professional, publication-ready output
""")


if __name__ == "__main__":
    try:
        # Run the demonstration
        workspace, report_path = demonstrate_reporting()
        
        # Show integration information
        show_integration_example()
        
        # Cleanup option
        if report_path:
            print(f"\n🗂️  Sample workspace created at: {workspace}")
            print("   (You can delete this after viewing the report)")
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
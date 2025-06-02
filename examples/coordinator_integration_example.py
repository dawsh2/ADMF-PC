"""
Example showing how reporting integrates with the ADMF-PC Coordinator

This demonstrates how the reporting system would be called from the main
workflow coordinator after backtest completion.
"""

from pathlib import Path
from typing import Dict, Any
import logging

# This would be the actual import in the coordinator
# from src.reporting.basic_report import BacktestReportGenerator


class CoordinatorReportingIntegration:
    """Example of how reporting integrates with the Coordinator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def execute_workflow_with_reporting(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Example workflow execution that includes report generation
        
        This shows where reporting would fit in the coordinator flow:
        1. Execute backtest workflow
        2. Save results to workspace 
        3. Generate report (if configured)
        4. Return results with report path
        """
        
        workflow_results = {}
        
        try:
            # Step 1: Execute the main workflow (existing functionality)
            self.logger.info("Executing backtest workflow...")
            workspace_path = self._execute_backtest_workflow(config)
            workflow_results['workspace_path'] = workspace_path
            
            # Step 2: Generate report if configured
            if config.get('output', {}).get('generate_report', False):
                self.logger.info("Generating report...")
                report_path = self._generate_report(workspace_path, config)
                workflow_results['report_path'] = report_path
                workflow_results['report_url'] = f'file://{report_path.absolute()}'
                
                self.logger.info(f"Report generated: {report_path}")
            
            # Step 3: Return comprehensive results
            workflow_results['status'] = 'completed'
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            workflow_results['status'] = 'failed'
            workflow_results['error'] = str(e)
            return workflow_results
    
    def _execute_backtest_workflow(self, config: Dict[str, Any]) -> Path:
        """
        Placeholder for actual backtest execution
        In real implementation, this would be the existing coordinator logic
        """
        # This represents the existing backtest execution
        # which saves results to workspace directories
        
        workspace_path = Path(f"./results/workflow_{config.get('name', 'default')}")
        
        # Simulate workspace creation (this is already done by existing code)
        for dir_name in ['performance', 'signals', 'metadata', 'visualizations']:
            (workspace_path / dir_name).mkdir(parents=True, exist_ok=True)
        
        return workspace_path
    
    def _generate_report(self, workspace_path: Path, config: Dict[str, Any]) -> Path:
        """Generate report based on configuration"""
        
        # Import here to avoid circular dependencies
        # from src.reporting.basic_report import BacktestReportGenerator
        
        # For now, we'll simulate this
        report_generator = MockReportGenerator(workspace_path)
        
        # Get report configuration
        report_config = config.get('output', {})
        report_type = report_config.get('report_type', 'basic')
        
        if report_type == 'basic':
            return report_generator.generate_basic_report()
        elif report_type == 'comprehensive':
            return report_generator.generate_comprehensive_report()
        else:
            return report_generator.generate_basic_report()


class MockReportGenerator:
    """Mock report generator for example"""
    
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
    
    def generate_basic_report(self) -> Path:
        """Generate basic HTML report"""
        report_path = self.workspace_path / 'visualizations' / 'report.html'
        report_path.write_text("<html><body><h1>Sample Report</h1></body></html>")
        return report_path
    
    def generate_comprehensive_report(self) -> Path:
        """Generate comprehensive report with additional charts"""
        return self.generate_basic_report()  # Same for demo


# Example configuration that would enable reporting
EXAMPLE_CONFIG_WITH_REPORTING = {
    "name": "spy_momentum_strategy",
    "data": {
        "source": "csv",
        "symbols": ["SPY"],
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    },
    "strategy": {
        "type": "momentum",
        "parameters": {"lookback": 20, "threshold": 0.02}
    },
    "output": {
        "results_dir": "./results",
        "save_trades": True,
        "save_signals": True, 
        "save_metrics": True,
        
        # Reporting configuration
        "generate_report": True,
        "report_type": "comprehensive",  # basic, comprehensive, custom
        "include_charts": [
            "equity_curve",
            "drawdown", 
            "trade_analysis",
            "returns_distribution"
        ],
        
        # Optional: Dashboard configuration
        "dashboard": {
            "enabled": False,
            "port": 8050,
            "auto_open": True
        }
    }
}


def demonstrate_integration():
    """Demonstrate the integration"""
    
    print("ADMF-PC Coordinator Integration Demo")
    print("=" * 40)
    
    # Create coordinator with reporting
    coordinator = CoordinatorReportingIntegration()
    
    # Execute workflow with reporting enabled
    print("\n1. Executing workflow with reporting enabled...")
    results = coordinator.execute_workflow_with_reporting(EXAMPLE_CONFIG_WITH_REPORTING)
    
    print(f"\n2. Workflow Results:")
    print(f"   Status: {results.get('status')}")
    print(f"   Workspace: {results.get('workspace_path')}")
    print(f"   Report: {results.get('report_path')}")
    print(f"   Report URL: {results.get('report_url')}")
    
    # Show how this would look in main.py
    print("\n3. Integration in main.py would look like:")
    print("""
def main():
    config = load_config(args.config)
    
    # Add command line options for reporting
    if args.generate_report:
        config.setdefault('output', {})['generate_report'] = True
    
    # Execute workflow (with reporting if configured)
    coordinator = WorkflowCoordinator()
    results = coordinator.execute_workflow(config)
    
    # Show report if generated
    if 'report_path' in results:
        print(f"Report generated: {results['report_path']}")
        if args.open_report:
            webbrowser.open(results['report_url'])
""")

    # Show command line usage
    print("\n4. Command line usage would be:")
    print("""
# Generate report after backtest
python main.py --config config/spy_momentum.yaml --generate-report

# Generate report and open in browser
python main.py --config config/spy_momentum.yaml --generate-report --open-report

# Use configuration file to control reporting
python main.py --config config/spy_momentum_with_reporting.yaml
""")


if __name__ == "__main__":
    demonstrate_integration()
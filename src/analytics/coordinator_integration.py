"""
Reporting integration with ADMF-PC Coordinator

This module provides coordinator-managed reporting following ADMF-PC architectural patterns:
- Configuration-driven behavior
- Event-based integration
- Workspace management compliance
- Container pattern adherence
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import json

from ..core.containers.protocols import Container
from ..core.events.types import Event, EventType
from .basic_report import BacktestReportGenerator


class ReportingContainer:
    """Container for managing report generation within ADMF-PC workflows"""
    
    def __init__(self, container_id: str, config: Dict[str, Any], workspace_path: Path):
        self.container_id = container_id
        self.config = config
        self.workspace_path = Path(workspace_path)
        self.logger = logging.getLogger(f"ReportingContainer.{container_id}")
        
        # Reporting configuration
        self.reporting_config = config.get('reporting', {})
        self.enabled = self.reporting_config.get('enabled', False)
        
        if self.enabled:
            self.logger.info(f"ReportingContainer initialized - enabled: {self.enabled}")
            self.logger.info(f"Report type: {self.reporting_config.get('report_type', 'basic')}")
            self.logger.info(f"Workspace: {workspace_path}")
        
    def is_enabled(self) -> bool:
        """Check if reporting is enabled in configuration"""
        return self.enabled
    
    def process_workflow_completion(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process workflow completion and generate reports if configured
        
        This method is called by the coordinator after workflow completion
        """
        if not self.enabled:
            self.logger.debug("Reporting disabled, skipping report generation")
            return workflow_results
        
        try:
            self.logger.info("🔄 Starting report generation...")
            
            # Generate reports based on configuration
            report_paths = self._generate_reports()
            
            # Update workflow results with report information
            workflow_results['reporting'] = {
                'enabled': True,
                'reports_generated': len(report_paths),
                'report_paths': [str(p) for p in report_paths],
                'workspace_path': str(self.workspace_path),
                'generation_time': datetime.now().isoformat()
            }
            
            # Auto-open report if configured
            if self.reporting_config.get('auto_open', False) and report_paths:
                self._auto_open_report(report_paths[0])
            
            self.logger.info(f"✅ Report generation completed - {len(report_paths)} reports created")
            
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
            workflow_results['reporting'] = {
                'enabled': True,
                'error': str(e),
                'generation_time': datetime.now().isoformat()
            }
            return workflow_results
    
    def _generate_reports(self) -> List[Path]:
        """Generate reports based on configuration"""
        report_paths = []
        
        # Get report configuration
        report_type = self.reporting_config.get('report_type', 'basic')
        formats = self.reporting_config.get('formats', ['html'])
        
        if 'html' in formats:
            html_report = self._generate_html_report(report_type)
            if html_report:
                report_paths.append(html_report)
        
        if 'pdf' in formats:
            pdf_report = self._generate_pdf_report(report_type)
            if pdf_report:
                report_paths.append(pdf_report)
        
        return report_paths
    
    def _generate_html_report(self, report_type: str) -> Optional[Path]:
        """Generate HTML report"""
        try:
            # Create report generator with workspace
            generator = BacktestReportGenerator(self.workspace_path)
            
            # Generate report (this uses the existing implementation)
            report_path = generator.generate_report()
            
            self.logger.info(f"📄 HTML report generated: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Failed to generate HTML report: {e}")
            return None
    
    def _generate_pdf_report(self, report_type: str) -> Optional[Path]:
        """Generate PDF report (placeholder for future implementation)"""
        # TODO: Implement PDF generation using weasyprint or similar
        self.logger.info("📄 PDF report generation not yet implemented")
        return None
    
    def _auto_open_report(self, report_path: Path) -> None:
        """Auto-open report in browser if configured"""
        try:
            import webbrowser
            webbrowser.open(f'file://{report_path.absolute()}')
            self.logger.info(f"🌐 Opened report in browser: {report_path}")
        except Exception as e:
            self.logger.warning(f"Failed to auto-open report: {e}")


class CoordinatorReportingIntegration:
    """
    Integration point for adding reporting to the ADMF-PC Coordinator
    
    This class provides methods that the coordinator can call to handle reporting
    """
    
    def __init__(self):
        self.logger = logging.getLogger("CoordinatorReportingIntegration")
    
    def create_reporting_container(self, container_id: str, config: Dict[str, Any], 
                                 workspace_path: Path) -> ReportingContainer:
        """Create reporting container for workflow"""
        return ReportingContainer(container_id, config, workspace_path)
    
    def should_generate_reports(self, config: Dict[str, Any]) -> bool:
        """Check if reports should be generated based on configuration"""
        return config.get('reporting', {}).get('enabled', False)
    
    def integrate_with_workflow_completion(self, coordinator_instance):
        """
        Integration hook for coordinator workflow completion
        
        This would be called during coordinator initialization to add reporting
        """
        # This is a placeholder for coordinator integration
        # The actual integration would depend on the coordinator's event system
        pass


def add_reporting_to_coordinator_workflow(config: Dict[str, Any], workspace_path: Path, 
                                        workflow_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to add reporting to any workflow
    
    This can be called by the coordinator after workflow completion
    """
    # Create reporting container
    reporting_container = ReportingContainer(
        container_id="reporting_main",
        config=config,
        workspace_path=workspace_path
    )
    
    # Process workflow completion and generate reports
    return reporting_container.process_workflow_completion(workflow_results)


# Example integration with coordinator patterns
class ReportingWorkflowExtension:
    """
    Example of how reporting could be integrated as a workflow extension
    following ADMF-PC patterns
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("ReportingWorkflowExtension")
    
    def on_workflow_completion(self, event: Event) -> None:
        """Handle workflow completion event"""
        if event.event_type == EventType.WORKFLOW_COMPLETED:
            workspace_path = event.data.get('workspace_path')
            workflow_results = event.data.get('results', {})
            
            # Generate reports if configured
            updated_results = add_reporting_to_coordinator_workflow(
                self.config, workspace_path, workflow_results
            )
            
            # Publish updated results
            self._publish_updated_results(updated_results)
    
    def _publish_updated_results(self, results: Dict[str, Any]) -> None:
        """Publish updated results with reporting information"""
        # This would use the ADMF-PC event system
        pass
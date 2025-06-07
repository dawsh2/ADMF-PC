"""
Extractor Registry for ADMF-PC Result Extraction Framework

Central registry for managing result extractors and their configurations.
Provides workflow-specific extractor selection and SQL table mappings.
"""

from typing import Dict, List, Type, Optional, Any
import logging

from .result_extraction import (
    ResultExtractor,
    PortfolioMetricsExtractor,
    SignalExtractor,
    FillExtractor,
    OrderExtractor,
    RiskDecisionExtractor,
    RegimeChangeExtractor,
    PerformanceSnapshotExtractor
)

logger = logging.getLogger(__name__)


class ExtractorRegistry:
    """
    Central registry for all result extractors.
    
    Manages:
    - Registration of extractors
    - SQL table mappings for each extractor type
    - Workflow-specific extractor selection
    - Default extractor configurations
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self.extractors: Dict[str, ResultExtractor] = {}
        self.sql_mappings: Dict[str, str] = {}
        self.workflow_extractors: Dict[str, List[str]] = {}
        
        # Register default extractors
        self._register_default_extractors()
        
        logger.info(f"ExtractorRegistry initialized with {len(self.extractors)} extractors")
    
    def register(
        self, 
        name: str, 
        extractor: ResultExtractor, 
        table: Optional[str] = None
    ) -> None:
        """
        Register a result extractor.
        
        Args:
            name: Unique name for the extractor
            extractor: The extractor instance
            table: SQL table name (defaults to extractor.table_name)
        """
        if name in self.extractors:
            logger.warning(f"Overwriting existing extractor: {name}")
        
        self.extractors[name] = extractor
        self.sql_mappings[name] = table or extractor.table_name
        
        logger.debug(f"Registered extractor '{name}' -> table '{self.sql_mappings[name]}'")
    
    def unregister(self, name: str) -> None:
        """Remove an extractor from the registry."""
        if name in self.extractors:
            del self.extractors[name]
            del self.sql_mappings[name]
            logger.debug(f"Unregistered extractor: {name}")
    
    def get_extractor(self, name: str) -> Optional[ResultExtractor]:
        """Get a specific extractor by name."""
        return self.extractors.get(name)
    
    def get_extractors_for_workflow(self, workflow_type: str) -> List[ResultExtractor]:
        """
        Get relevant extractors based on workflow type.
        
        Args:
            workflow_type: Type of workflow (e.g., 'backtest', 'optimization', 'analysis')
            
        Returns:
            List of extractors appropriate for the workflow
        """
        # Check if we have specific configuration for this workflow
        if workflow_type in self.workflow_extractors:
            extractor_names = self.workflow_extractors[workflow_type]
            return [self.extractors[name] for name in extractor_names if name in self.extractors]
        
        # Otherwise, return default extractors based on workflow type
        return self._get_default_extractors_for_workflow(workflow_type)
    
    def configure_workflow_extractors(
        self, 
        workflow_type: str, 
        extractor_names: List[str]
    ) -> None:
        """
        Configure which extractors to use for a specific workflow type.
        
        Args:
            workflow_type: Type of workflow
            extractor_names: List of extractor names to use
        """
        # Validate extractor names
        invalid_names = [name for name in extractor_names if name not in self.extractors]
        if invalid_names:
            raise ValueError(f"Unknown extractors: {invalid_names}")
        
        self.workflow_extractors[workflow_type] = extractor_names
        logger.info(f"Configured {len(extractor_names)} extractors for workflow '{workflow_type}'")
    
    def get_sql_mapping(self, extractor_name: str) -> Optional[str]:
        """Get SQL table name for an extractor."""
        return self.sql_mappings.get(extractor_name)
    
    def get_all_extractors(self) -> Dict[str, ResultExtractor]:
        """Get all registered extractors."""
        return self.extractors.copy()
    
    def _register_default_extractors(self) -> None:
        """Register the default set of extractors."""
        # Core extractors - always useful
        self.register('portfolio_metrics', PortfolioMetricsExtractor())
        self.register('signals', SignalExtractor())
        self.register('fills', FillExtractor())
        self.register('orders', OrderExtractor())
        
        # Analysis extractors
        self.register('risk_decisions', RiskDecisionExtractor())
        self.register('regime_changes', RegimeChangeExtractor())
        
        # Performance tracking
        self.register('performance_snapshots_1h', PerformanceSnapshotExtractor(interval_seconds=3600))
        self.register('performance_snapshots_1d', PerformanceSnapshotExtractor(interval_seconds=86400))
    
    def _get_default_extractors_for_workflow(self, workflow_type: str) -> List[ResultExtractor]:
        """
        Get default extractors based on workflow type.
        
        Args:
            workflow_type: Type of workflow
            
        Returns:
            List of appropriate extractors
        """
        # Base extractors for all workflows
        base_extractors = [
            self.extractors['portfolio_metrics'],
            self.extractors['signals'],
            self.extractors['fills'],
            self.extractors['orders']
        ]
        
        # Add workflow-specific extractors
        if workflow_type == 'optimization':
            # For optimization, we want all data for analysis
            return base_extractors + [
                self.extractors['risk_decisions'],
                self.extractors['regime_changes'],
                self.extractors['performance_snapshots_1h']
            ]
        
        elif workflow_type == 'backtest':
            # Standard backtest needs core metrics
            return base_extractors + [
                self.extractors['performance_snapshots_1d']
            ]
        
        elif workflow_type == 'regime_analysis':
            # Regime analysis focuses on regime changes
            return base_extractors + [
                self.extractors['regime_changes'],
                self.extractors['performance_snapshots_1h']
            ]
        
        elif workflow_type == 'risk_analysis':
            # Risk analysis needs risk decisions
            return base_extractors + [
                self.extractors['risk_decisions'],
                self.extractors['performance_snapshots_1h']
            ]
        
        elif workflow_type == 'signal_generation':
            # Signal generation only needs signals
            return [
                self.extractors['signals'],
                self.extractors['regime_changes']  # For regime context
            ]
        
        elif workflow_type == 'signal_replay':
            # Signal replay needs execution data
            return [
                self.extractors['portfolio_metrics'],
                self.extractors['fills'],
                self.extractors['orders'],
                self.extractors['performance_snapshots_1h']
            ]
        
        else:
            # Default: return base extractors
            logger.warning(f"Unknown workflow type '{workflow_type}', using base extractors")
            return base_extractors


class ExtractorConfig:
    """Configuration for result extraction in a workflow."""
    
    def __init__(
        self,
        enabled: bool = True,
        buffer_size: int = 1000,
        output_format: str = 'parquet',
        output_directory: str = './results',
        compression: Optional[str] = 'snappy',
        extractors: Optional[List[str]] = None
    ):
        """
        Initialize extractor configuration.
        
        Args:
            enabled: Whether result extraction is enabled
            buffer_size: Number of results to buffer before flushing
            output_format: Output format ('parquet', 'csv', 'jsonl')
            output_directory: Directory for result files
            compression: Compression type for output files
            extractors: List of specific extractors to use (None = use defaults)
        """
        self.enabled = enabled
        self.buffer_size = buffer_size
        self.output_format = output_format
        self.output_directory = output_directory
        self.compression = compression
        self.extractors = extractors or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'enabled': self.enabled,
            'buffer_size': self.buffer_size,
            'output_format': self.output_format,
            'output_directory': self.output_directory,
            'compression': self.compression,
            'extractors': self.extractors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractorConfig':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_workflow_config(cls, workflow_config: Dict[str, Any]) -> 'ExtractorConfig':
        """
        Create from workflow configuration.
        
        Looks for 'result_extraction' section in workflow config.
        """
        extraction_config = workflow_config.get('result_extraction', {})
        
        # Handle simple boolean enable/disable
        if isinstance(extraction_config, bool):
            return cls(enabled=extraction_config)
        
        # Handle full configuration
        return cls.from_dict(extraction_config)
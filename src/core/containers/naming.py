"""
Structured container naming strategy following BACKTEST.MD.

Implements the naming format:
{container_type}_{phase}_{classifier}_{risk_profile}_{timestamp}
"""
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class ContainerType(Enum):
    """Types of containers in the system."""
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"
    SIGNAL_GEN = "signal_gen"
    SIGNAL_REPLAY = "signal_replay"
    LIVE_TRADING = "live"
    CLASSIFIER = "classifier"
    RISK_PORTFOLIO = "risk_portfolio"
    INDICATOR = "indicator"
    DATA = "data"


class Phase(Enum):
    """Workflow phases for container naming."""
    PHASE1_GRID_SEARCH = "phase1_grid"
    PHASE2_ENSEMBLE = "phase2_ensemble"
    PHASE3_VALIDATION = "phase3_validation"
    PHASE4_WALK_FORWARD = "phase4_walkforward"
    INITIALIZATION = "init"
    DATA_PREPARATION = "data_prep"
    COMPUTATION = "compute"
    VALIDATION = "validate"
    AGGREGATION = "aggregate"
    LIVE = "live"
    ANALYSIS = "analysis"


class ClassifierType(Enum):
    """Types of classifiers."""
    HMM = "hmm"
    PATTERN = "pattern"
    TREND_VOL = "trend_vol"
    MULTI_INDICATOR = "multi_ind"
    ENSEMBLE = "ensemble"
    NONE = "none"


class RiskProfile(Enum):
    """Risk profiles for portfolio management."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"
    NONE = "none"


class ContainerNamingStrategy:
    """
    Structured naming strategy for containers.
    
    Creates names that:
    - Enable easy identification of container purpose
    - Support tracking across optimization phases
    - Facilitate debugging and monitoring
    - Allow result aggregation by type
    """
    
    @staticmethod
    def generate_container_id(
        container_type: ContainerType,
        phase: Optional[Phase] = None,
        classifier: Optional[ClassifierType] = None,
        risk_profile: Optional[RiskProfile] = None,
        timestamp: Optional[datetime] = None,
        unique_suffix: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a structured container ID.
        
        Args:
            container_type: Type of container
            phase: Current workflow phase
            classifier: Classifier type (if applicable)
            risk_profile: Risk profile (if applicable)
            timestamp: Timestamp for the container
            unique_suffix: Whether to add UUID suffix
            metadata: Additional metadata to encode in name
            
        Returns:
            Structured container ID
        """
        parts = [container_type.value]
        
        # Add phase if specified
        if phase:
            parts.append(phase.value)
            
        # Add classifier if specified and not none
        if classifier and classifier != ClassifierType.NONE:
            parts.append(classifier.value)
            
        # Add risk profile if specified and not none
        if risk_profile and risk_profile != RiskProfile.NONE:
            parts.append(risk_profile.value)
            
        # Add metadata elements if provided
        if metadata:
            # Add symbol set identifier
            if 'symbols' in metadata and metadata['symbols']:
                if len(metadata['symbols']) == 1:
                    parts.append(metadata['symbols'][0].lower())
                elif len(metadata['symbols']) <= 3:
                    parts.append('_'.join(s.lower() for s in metadata['symbols']))
                else:
                    parts.append(f"{len(metadata['symbols'])}syms")
                    
            # Add strategy identifier
            if 'strategy' in metadata:
                parts.append(metadata['strategy'].lower().replace(' ', '_'))
                
            # Add optimization trial number
            if 'trial' in metadata:
                parts.append(f"t{metadata['trial']}")
                
        # Add timestamp
        if timestamp is None:
            timestamp = datetime.now()
        parts.append(timestamp.strftime('%Y%m%d_%H%M%S'))
        
        # Add unique suffix if requested
        if unique_suffix:
            parts.append(uuid.uuid4().hex[:8])
            
        return '_'.join(parts)
        
    @staticmethod
    def parse_container_id(container_id: str) -> Dict[str, Any]:
        """
        Parse a structured container ID into components.
        
        Args:
            container_id: Container ID to parse
            
        Returns:
            Dictionary with parsed components
        """
        parts = container_id.split('_')
        result = {
            'raw_id': container_id,
            'container_type': None,
            'phase': None,
            'classifier': None,
            'risk_profile': None,
            'timestamp': None,
            'uuid': None,
            'metadata': {}
        }
        
        if not parts:
            return result
            
        # Parse container type (always first)
        try:
            result['container_type'] = ContainerType(parts[0])
            idx = 1
        except ValueError:
            idx = 0
            
        # Try to parse remaining parts
        while idx < len(parts):
            part = parts[idx]
            
            # Check if it's a phase
            try:
                for phase in Phase:
                    if phase.value == part or phase.value.startswith(part):
                        result['phase'] = phase
                        idx += 1
                        continue
            except:
                pass
                
            # Check if it's a classifier
            try:
                for classifier in ClassifierType:
                    if classifier.value == part:
                        result['classifier'] = classifier
                        idx += 1
                        continue
            except:
                pass
                
            # Check if it's a risk profile
            try:
                for profile in RiskProfile:
                    if profile.value == part:
                        result['risk_profile'] = profile
                        idx += 1
                        continue
            except:
                pass
                
            # Check if it's a timestamp (YYYYMMDD format)
            if len(part) == 8 and part.isdigit():
                # Next part should be time (HHMMSS)
                if idx + 1 < len(parts) and len(parts[idx + 1]) == 6 and parts[idx + 1].isdigit():
                    try:
                        timestamp_str = f"{part}_{parts[idx + 1]}"
                        result['timestamp'] = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        idx += 2
                        continue
                    except:
                        pass
                        
            # Check if it's a UUID (8 hex chars at the end)
            if idx == len(parts) - 1 and len(part) == 8:
                try:
                    int(part, 16)
                    result['uuid'] = part
                    idx += 1
                    continue
                except:
                    pass
                    
            # Otherwise, it's metadata
            result['metadata'][f'part_{idx}'] = part
            idx += 1
            
        return result
        
    @staticmethod
    def create_hierarchical_id(
        parent_id: str,
        child_type: str,
        child_descriptor: Optional[str] = None
    ) -> str:
        """
        Create a child container ID based on parent.
        
        Args:
            parent_id: Parent container ID
            child_type: Type of child container
            child_descriptor: Optional descriptor for child
            
        Returns:
            Child container ID
        """
        parts = [parent_id, child_type]
        
        if child_descriptor:
            parts.append(child_descriptor)
            
        # Add short UUID for uniqueness
        parts.append(uuid.uuid4().hex[:6])
        
        return '_'.join(parts)
        
    @staticmethod
    def get_container_family(container_id: str) -> str:
        """
        Get the family/root of a container ID.
        
        Useful for grouping related containers.
        
        Args:
            container_id: Container ID
            
        Returns:
            Container family identifier
        """
        parsed = ContainerNamingStrategy.parse_container_id(container_id)
        
        family_parts = []
        
        if parsed['container_type']:
            family_parts.append(parsed['container_type'].value)
            
        if parsed['phase']:
            family_parts.append(parsed['phase'].value)
            
        if parsed['classifier']:
            family_parts.append(parsed['classifier'].value)
            
        return '_'.join(family_parts) if family_parts else 'unknown'
        
    @staticmethod
    def create_workflow_container_id(
        workflow_id: str,
        container_type: ContainerType,
        phase: Phase,
        iteration: Optional[int] = None
    ) -> str:
        """
        Create container ID for workflow execution.
        
        Args:
            workflow_id: Workflow identifier
            container_type: Type of container
            phase: Current phase
            iteration: Optional iteration number
            
        Returns:
            Workflow container ID
        """
        parts = [
            'wf',
            workflow_id[:8],  # First 8 chars of workflow ID
            container_type.value,
            phase.value
        ]
        
        if iteration is not None:
            parts.append(f'i{iteration}')
            
        parts.append(datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        return '_'.join(parts)


# Convenience functions

def create_backtest_container_id(
    phase: Phase,
    classifier: ClassifierType = ClassifierType.NONE,
    risk_profile: RiskProfile = RiskProfile.BALANCED,
    **kwargs
) -> str:
    """Create a backtest container ID."""
    return ContainerNamingStrategy.generate_container_id(
        ContainerType.BACKTEST,
        phase,
        classifier,
        risk_profile,
        **kwargs
    )


def create_optimization_container_id(
    phase: Phase,
    trial_number: int,
    classifier: ClassifierType = ClassifierType.NONE,
    **kwargs
) -> str:
    """Create an optimization container ID."""
    metadata = kwargs.get('metadata', {})
    metadata['trial'] = trial_number
    kwargs['metadata'] = metadata
    
    return ContainerNamingStrategy.generate_container_id(
        ContainerType.OPTIMIZATION,
        phase,
        classifier,
        metadata=metadata,
        **kwargs
    )


def create_signal_analysis_container_id(
    analysis_type: str = "mae_mfe",
    symbols: Optional[list] = None,
    **kwargs
) -> str:
    """Create a signal analysis container ID."""
    metadata = kwargs.get('metadata', {})
    metadata['analysis'] = analysis_type
    if symbols:
        metadata['symbols'] = symbols
    kwargs['metadata'] = metadata
    
    return ContainerNamingStrategy.generate_container_id(
        ContainerType.SIGNAL_GEN,
        Phase.ANALYSIS,
        metadata=metadata,
        **kwargs
    )
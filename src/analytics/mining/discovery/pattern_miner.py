"""
Pattern discovery and mining for trading strategies.
Implements the scientific method approach to finding successful patterns.
"""
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from collections import defaultdict
import numpy as np
from enum import Enum

from src.core.events.tracing import TracedEvent
from src.analytics.mining.storage.schemas import DiscoveredPattern, PatternType


class PatternDetectionMethod(Enum):
    """Methods for detecting patterns"""
    SEQUENCE_MINING = "sequence_mining"
    REGIME_ANALYSIS = "regime_analysis"
    FAILURE_ANALYSIS = "failure_analysis"
    INTERACTION_ANALYSIS = "interaction_analysis"
    SUCCESS_FACTOR = "success_factor"


@dataclass
class PatternHypothesis:
    """A hypothesis about a potential pattern"""
    hypothesis_id: str
    pattern_type: PatternType
    description: str
    detection_method: PatternDetectionMethod
    confidence: float = 0.0
    evidence_count: int = 0
    
    # Pattern specifics
    event_sequence: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    
    # Validation requirements
    min_occurrences: int = 10
    min_success_rate: float = 0.6
    min_sharpe_improvement: float = 0.1


@dataclass 
class PatternEvidence:
    """Evidence supporting or refuting a pattern hypothesis"""
    correlation_id: str
    supports_hypothesis: bool
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    event_trace: List[TracedEvent]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternMiner:
    """
    Core pattern mining engine.
    Discovers patterns from event traces and optimization results.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.hypotheses: Dict[str, PatternHypothesis] = {}
        self.evidence: Dict[str, List[PatternEvidence]] = defaultdict(list)
        
    def mine_patterns(self, 
                     optimization_runs: List[Dict[str, Any]],
                     event_traces: Dict[str, List[TracedEvent]]) -> List[DiscoveredPattern]:
        """
        Mine patterns from optimization runs and their event traces.
        
        Args:
            optimization_runs: List of optimization run results
            event_traces: Event traces indexed by correlation_id
            
        Returns:
            List of discovered patterns
        """
        self.logger.info(f"Mining patterns from {len(optimization_runs)} optimization runs")
        
        # Step 1: Generate hypotheses
        hypotheses = self._generate_hypotheses(optimization_runs, event_traces)
        
        # Step 2: Gather evidence
        for hypothesis in hypotheses:
            self._gather_evidence(hypothesis, optimization_runs, event_traces)
            
        # Step 3: Validate patterns
        validated_patterns = self._validate_patterns(hypotheses)
        
        # Step 4: Rank patterns by effectiveness
        ranked_patterns = self._rank_patterns(validated_patterns)
        
        self.logger.info(f"Discovered {len(ranked_patterns)} validated patterns")
        return ranked_patterns
        
    def _generate_hypotheses(self,
                           optimization_runs: List[Dict[str, Any]],
                           event_traces: Dict[str, List[TracedEvent]]) -> List[PatternHypothesis]:
        """Generate pattern hypotheses using multiple detection methods"""
        hypotheses = []
        
        # Success factor analysis - what do top performers have in common?
        success_hypotheses = self._analyze_success_factors(optimization_runs, event_traces)
        hypotheses.extend(success_hypotheses)
        
        # Failure analysis - what patterns lead to poor performance?
        failure_hypotheses = self._analyze_failure_patterns(optimization_runs, event_traces)
        hypotheses.extend(failure_hypotheses)
        
        # Regime transition analysis - how do strategies handle market changes?
        regime_hypotheses = self._analyze_regime_transitions(optimization_runs, event_traces)
        hypotheses.extend(regime_hypotheses)
        
        # Event sequence mining - common successful event sequences
        sequence_hypotheses = self._mine_event_sequences(optimization_runs, event_traces)
        hypotheses.extend(sequence_hypotheses)
        
        return hypotheses
        
    def _analyze_success_factors(self,
                                optimization_runs: List[Dict[str, Any]],
                                event_traces: Dict[str, List[TracedEvent]]) -> List[PatternHypothesis]:
        """Analyze what successful strategies have in common"""
        hypotheses = []
        
        # Sort runs by performance
        sorted_runs = sorted(optimization_runs, 
                           key=lambda x: x.get('sharpe_ratio', 0), 
                           reverse=True)
        
        # Take top 20%
        top_percentile = int(len(sorted_runs) * 0.2)
        top_performers = sorted_runs[:top_percentile]
        
        if not top_performers:
            return hypotheses
            
        # Analyze common characteristics
        common_params = self._find_common_parameters(top_performers)
        common_events = self._find_common_event_patterns(top_performers, event_traces)
        
        # Generate hypotheses from findings
        for param_name, param_range in common_params.items():
            hypothesis = PatternHypothesis(
                hypothesis_id=f"success_param_{param_name}",
                pattern_type=PatternType.PARAMETER_REGIME,
                description=f"Parameter {param_name} in range {param_range} correlates with success",
                detection_method=PatternDetectionMethod.SUCCESS_FACTOR,
                conditions={
                    'parameter': param_name,
                    'range': param_range
                },
                expected_outcome="sharpe_ratio > baseline"
            )
            hypotheses.append(hypothesis)
            
        for event_pattern in common_events:
            hypothesis = PatternHypothesis(
                hypothesis_id=f"success_sequence_{len(hypotheses)}",
                pattern_type=PatternType.EVENT_SEQUENCE,
                description=f"Event sequence {event_pattern} correlates with success",
                detection_method=PatternDetectionMethod.SUCCESS_FACTOR,
                event_sequence=event_pattern,
                expected_outcome="positive_returns"
            )
            hypotheses.append(hypothesis)
            
        return hypotheses
        
    def _analyze_failure_patterns(self,
                                optimization_runs: List[Dict[str, Any]],
                                event_traces: Dict[str, List[TracedEvent]]) -> List[PatternHypothesis]:
        """Identify patterns that lead to poor performance"""
        hypotheses = []
        
        # Get bottom performers
        sorted_runs = sorted(optimization_runs, 
                           key=lambda x: x.get('sharpe_ratio', 0))
        
        bottom_percentile = int(len(sorted_runs) * 0.2)
        poor_performers = sorted_runs[:bottom_percentile]
        
        if not poor_performers:
            return hypotheses
            
        # Look for anti-patterns
        for run in poor_performers:
            correlation_id = run.get('correlation_id')
            if correlation_id not in event_traces:
                continue
                
            events = event_traces[correlation_id]
            
            # Check for specific failure modes
            if self._has_excessive_trading(events):
                hypothesis = PatternHypothesis(
                    hypothesis_id=f"failure_excessive_trading_{correlation_id}",
                    pattern_type=PatternType.ANTI_PATTERN,
                    description="Excessive trading (>100 trades/day) leads to poor performance",
                    detection_method=PatternDetectionMethod.FAILURE_ANALYSIS,
                    conditions={'daily_trade_limit': 100},
                    expected_outcome="negative_sharpe"
                )
                hypotheses.append(hypothesis)
                
            if self._has_risk_violations(events):
                hypothesis = PatternHypothesis(
                    hypothesis_id=f"failure_risk_violation_{correlation_id}",
                    pattern_type=PatternType.ANTI_PATTERN,
                    description="Risk limit violations correlate with drawdowns",
                    detection_method=PatternDetectionMethod.FAILURE_ANALYSIS,
                    conditions={'risk_violations': True},
                    expected_outcome="high_drawdown"
                )
                hypotheses.append(hypothesis)
                
        return hypotheses
        
    def _analyze_regime_transitions(self,
                                  optimization_runs: List[Dict[str, Any]],
                                  event_traces: Dict[str, List[TracedEvent]]) -> List[PatternHypothesis]:
        """Analyze how strategies handle market regime changes"""
        hypotheses = []
        
        for run in optimization_runs:
            correlation_id = run.get('correlation_id')
            if correlation_id not in event_traces:
                continue
                
            events = event_traces[correlation_id]
            
            # Detect regime changes in event stream
            regime_changes = self._detect_regime_changes(events)
            
            for regime_change in regime_changes:
                # Analyze strategy behavior around regime change
                pre_regime_perf = self._calculate_performance_window(
                    events, regime_change['timestamp'] - timedelta(days=5), regime_change['timestamp']
                )
                post_regime_perf = self._calculate_performance_window(
                    events, regime_change['timestamp'], regime_change['timestamp'] + timedelta(days=5)
                )
                
                if pre_regime_perf and post_regime_perf:
                    # Did strategy adapt well?
                    if post_regime_perf['sharpe'] > pre_regime_perf['sharpe']:
                        hypothesis = PatternHypothesis(
                            hypothesis_id=f"regime_adapt_{regime_change['from_regime']}_{regime_change['to_regime']}",
                            pattern_type=PatternType.REGIME_TRANSITION,
                            description=f"Successful adaptation from {regime_change['from_regime']} to {regime_change['to_regime']}",
                            detection_method=PatternDetectionMethod.REGIME_ANALYSIS,
                            conditions={
                                'from_regime': regime_change['from_regime'],
                                'to_regime': regime_change['to_regime'],
                                'adaptation_signal': regime_change.get('signal_change')
                            },
                            expected_outcome="maintain_positive_sharpe"
                        )
                        hypotheses.append(hypothesis)
                        
        return hypotheses
        
    def _mine_event_sequences(self,
                            optimization_runs: List[Dict[str, Any]],
                            event_traces: Dict[str, List[TracedEvent]]) -> List[PatternHypothesis]:
        """Mine common event sequences that lead to successful trades"""
        hypotheses = []
        sequence_counts = defaultdict(int)
        sequence_outcomes = defaultdict(list)
        
        for run in optimization_runs:
            correlation_id = run.get('correlation_id')
            if correlation_id not in event_traces:
                continue
                
            events = event_traces[correlation_id]
            
            # Extract event sequences leading to trades
            trade_sequences = self._extract_trade_sequences(events)
            
            for sequence, outcome in trade_sequences:
                sequence_key = "->".join(sequence)
                sequence_counts[sequence_key] += 1
                sequence_outcomes[sequence_key].append(outcome)
                
        # Find sequences with high success rate
        for sequence_key, outcomes in sequence_outcomes.items():
            if len(outcomes) < 10:  # Minimum occurrences
                continue
                
            success_rate = sum(1 for o in outcomes if o > 0) / len(outcomes)
            
            if success_rate > 0.6:  # 60% success rate threshold
                hypothesis = PatternHypothesis(
                    hypothesis_id=f"sequence_{sequence_key[:20]}",
                    pattern_type=PatternType.EVENT_SEQUENCE,
                    description=f"Event sequence with {success_rate:.1%} success rate",
                    detection_method=PatternDetectionMethod.SEQUENCE_MINING,
                    event_sequence=sequence_key.split("->"),
                    expected_outcome="profitable_trade",
                    confidence=success_rate
                )
                hypotheses.append(hypothesis)
                
        return hypotheses
        
    def _gather_evidence(self,
                        hypothesis: PatternHypothesis,
                        optimization_runs: List[Dict[str, Any]],
                        event_traces: Dict[str, List[TracedEvent]]) -> None:
        """Gather evidence supporting or refuting a hypothesis"""
        
        for run in optimization_runs:
            correlation_id = run.get('correlation_id')
            if correlation_id not in event_traces:
                continue
                
            events = event_traces[correlation_id]
            
            # Check if this run matches the hypothesis conditions
            matches_hypothesis = self._check_hypothesis_match(hypothesis, run, events)
            
            if matches_hypothesis is not None:
                evidence = PatternEvidence(
                    correlation_id=correlation_id,
                    supports_hypothesis=matches_hypothesis,
                    sharpe_ratio=run.get('sharpe_ratio', 0),
                    total_return=run.get('total_return', 0),
                    max_drawdown=run.get('max_drawdown', 0),
                    event_trace=events
                )
                self.evidence[hypothesis.hypothesis_id].append(evidence)
                
    def _validate_patterns(self, hypotheses: List[PatternHypothesis]) -> List[DiscoveredPattern]:
        """Validate hypotheses using gathered evidence"""
        validated_patterns = []
        
        for hypothesis in hypotheses:
            evidence_list = self.evidence.get(hypothesis.hypothesis_id, [])
            
            if len(evidence_list) < hypothesis.min_occurrences:
                continue
                
            # Calculate success metrics
            supporting_evidence = [e for e in evidence_list if e.supports_hypothesis]
            success_rate = len(supporting_evidence) / len(evidence_list)
            
            if success_rate < hypothesis.min_success_rate:
                continue
                
            # Calculate performance improvement
            avg_sharpe_with_pattern = np.mean([e.sharpe_ratio for e in supporting_evidence])
            avg_sharpe_without = np.mean([e.sharpe_ratio for e in evidence_list if not e.supports_hypothesis])
            
            sharpe_improvement = avg_sharpe_with_pattern - avg_sharpe_without
            
            if sharpe_improvement < hypothesis.min_sharpe_improvement:
                continue
                
            # Create validated pattern
            pattern = DiscoveredPattern(
                pattern_id=hypothesis.hypothesis_id,
                pattern_type=hypothesis.pattern_type,
                description=hypothesis.description,
                discovery_date=datetime.now(),
                confidence_score=success_rate,
                occurrence_count=len(supporting_evidence),
                avg_sharpe_improvement=sharpe_improvement,
                pattern_data={
                    'conditions': hypothesis.conditions,
                    'event_sequence': hypothesis.event_sequence,
                    'expected_outcome': hypothesis.expected_outcome,
                    'detection_method': hypothesis.detection_method.value
                }
            )
            validated_patterns.append(pattern)
            
        return validated_patterns
        
    def _rank_patterns(self, patterns: List[DiscoveredPattern]) -> List[DiscoveredPattern]:
        """Rank patterns by their effectiveness and reliability"""
        
        # Score patterns based on multiple factors
        scored_patterns = []
        for pattern in patterns:
            score = (
                pattern.confidence_score * 0.3 +  # Reliability
                min(pattern.avg_sharpe_improvement, 1.0) * 0.4 +  # Effectiveness  
                min(pattern.occurrence_count / 100, 1.0) * 0.3  # Robustness
            )
            scored_patterns.append((score, pattern))
            
        # Sort by score descending
        scored_patterns.sort(key=lambda x: x[0], reverse=True)
        
        return [pattern for _, pattern in scored_patterns]
        
    # Helper methods
    def _find_common_parameters(self, runs: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        """Find parameter ranges common to successful runs"""
        param_values = defaultdict(list)
        
        for run in runs:
            params = run.get('parameters', {})
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    param_values[key].append(float(value))
                    
        common_params = {}
        for param, values in param_values.items():
            if len(values) > len(runs) * 0.7:  # Present in 70% of runs
                common_params[param] = (min(values), max(values))
                
        return common_params
        
    def _find_common_event_patterns(self, 
                                  runs: List[Dict[str, Any]], 
                                  event_traces: Dict[str, List[TracedEvent]]) -> List[List[str]]:
        """Find event sequences common to successful runs"""
        all_sequences = []
        
        for run in runs:
            correlation_id = run.get('correlation_id')
            if correlation_id in event_traces:
                events = event_traces[correlation_id]
                sequences = self._extract_event_sequences(events, max_length=5)
                all_sequences.extend(sequences)
                
        # Count sequence occurrences
        sequence_counts = defaultdict(int)
        for seq in all_sequences:
            sequence_counts[tuple(seq)] += 1
            
        # Return sequences that appear in >50% of runs
        threshold = len(runs) * 0.5
        common_sequences = [
            list(seq) for seq, count in sequence_counts.items() 
            if count > threshold
        ]
        
        return common_sequences
        
    def _has_excessive_trading(self, events: List[TracedEvent]) -> bool:
        """Check if event trace shows excessive trading"""
        order_events = [e for e in events if e.event_type == "OrderEvent"]
        
        if not order_events:
            return False
            
        # Group by day and count
        daily_counts = defaultdict(int)
        for event in order_events:
            day = event.timestamp.date()
            daily_counts[day] += 1
            
        # Check if any day exceeds threshold
        return any(count > 100 for count in daily_counts.values())
        
    def _has_risk_violations(self, events: List[TracedEvent]) -> bool:
        """Check if event trace contains risk violations"""
        for event in events:
            if event.event_type == "RiskViolationEvent":
                return True
            if "risk_violation" in str(event.data).lower():
                return True
        return False
        
    def _detect_regime_changes(self, events: List[TracedEvent]) -> List[Dict[str, Any]]:
        """Detect market regime changes from event stream"""
        regime_changes = []
        
        # Look for specific regime change indicators
        for i, event in enumerate(events):
            if event.event_type == "RegimeChangeEvent":
                regime_changes.append({
                    'timestamp': event.timestamp,
                    'from_regime': event.data.get('from_regime'),
                    'to_regime': event.data.get('to_regime'),
                    'signal_change': event.data.get('signal_change')
                })
            # Could also detect from volatility/trend changes in market data events
            
        return regime_changes
        
    def _calculate_performance_window(self, 
                                    events: List[TracedEvent],
                                    start: datetime,
                                    end: datetime) -> Optional[Dict[str, float]]:
        """Calculate performance metrics for a time window"""
        window_events = [e for e in events if start <= e.timestamp <= end]
        
        if not window_events:
            return None
            
        # Extract fills and calculate metrics
        fills = [e for e in window_events if e.event_type == "FillEvent"]
        
        if not fills:
            return None
            
        # Simplified performance calculation
        returns = []
        for fill in fills:
            # Would calculate actual returns from fill prices
            returns.append(fill.data.get('realized_pnl', 0))
            
        if returns:
            return {
                'sharpe': np.mean(returns) / (np.std(returns) + 1e-8),
                'total_return': sum(returns),
                'trade_count': len(fills)
            }
            
        return None
        
    def _extract_trade_sequences(self, events: List[TracedEvent]) -> List[Tuple[List[str], float]]:
        """Extract event sequences that lead to trades with outcomes"""
        sequences = []
        
        # Find all fill events (completed trades)
        fill_indices = [i for i, e in enumerate(events) if e.event_type == "FillEvent"]
        
        for fill_idx in fill_indices:
            # Look back up to 10 events before the fill
            start_idx = max(0, fill_idx - 10)
            sequence_events = events[start_idx:fill_idx]
            
            # Extract event type sequence
            sequence = [e.event_type for e in sequence_events]
            
            # Get trade outcome (simplified - would use actual P&L)
            outcome = events[fill_idx].data.get('realized_pnl', 0)
            
            sequences.append((sequence, outcome))
            
        return sequences
        
    def _extract_event_sequences(self, events: List[TracedEvent], max_length: int = 5) -> List[List[str]]:
        """Extract all event sequences up to max_length"""
        sequences = []
        
        for i in range(len(events) - max_length + 1):
            sequence = [events[j].event_type for j in range(i, i + max_length)]
            sequences.append(sequence)
            
        return sequences
        
    def _check_hypothesis_match(self,
                              hypothesis: PatternHypothesis,
                              run: Dict[str, Any],
                              events: List[TracedEvent]) -> Optional[bool]:
        """Check if a run matches hypothesis conditions"""
        
        # Parameter-based hypotheses
        if hypothesis.pattern_type == PatternType.PARAMETER_REGIME:
            param_name = hypothesis.conditions.get('parameter')
            param_range = hypothesis.conditions.get('range')
            
            if param_name and param_range:
                param_value = run.get('parameters', {}).get(param_name)
                if param_value is not None:
                    in_range = param_range[0] <= param_value <= param_range[1]
                    # Check if outcome matches expectation
                    if hypothesis.expected_outcome == "sharpe_ratio > baseline":
                        return in_range and run.get('sharpe_ratio', 0) > 0
                    return in_range
                    
        # Event sequence hypotheses  
        elif hypothesis.pattern_type == PatternType.EVENT_SEQUENCE:
            # Check if the sequence appears in events
            event_types = [e.event_type for e in events]
            sequence = hypothesis.event_sequence
            
            # Simple substring match (could be more sophisticated)
            for i in range(len(event_types) - len(sequence) + 1):
                if event_types[i:i+len(sequence)] == sequence:
                    # Found sequence, check outcome
                    if hypothesis.expected_outcome == "positive_returns":
                        return run.get('total_return', 0) > 0
                    return True
                    
        # Anti-pattern hypotheses
        elif hypothesis.pattern_type == PatternType.ANTI_PATTERN:
            if "excessive_trading" in hypothesis.hypothesis_id:
                has_excessive = self._has_excessive_trading(events)
                poor_performance = run.get('sharpe_ratio', 0) < 0
                return has_excessive and poor_performance
                
            elif "risk_violation" in hypothesis.hypothesis_id:
                has_violations = self._has_risk_violations(events)
                high_drawdown = run.get('max_drawdown', 0) > 0.2
                return has_violations and high_drawdown
                
        return None
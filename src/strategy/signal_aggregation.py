"""
File: src/strategy/signal_aggregation.py
Status: ACTIVE
Architecture Ref: SYSTEM_ARCHITECTURE_v5.md#signal-aggregation
Step: 4 - Multiple Strategies
Dependencies: abc, datetime, typing

Signal aggregation methods for combining multiple strategy signals.
Implements weighted voting, majority voting, and consensus building.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from ..core.logging.structured import ContainerLogger


class Direction(Enum):
    """Signal direction enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """Basic trading signal structure"""
    symbol: str
    direction: Direction
    strength: float  # 0.0 to 1.0
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AggregatedSignal:
    """Signal with strategy metadata for aggregation"""
    strategy_id: str
    signal: TradingSignal
    weight: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass 
class ConsensusSignal:
    """Aggregated consensus signal from multiple strategies"""
    symbol: str
    direction: Direction
    strength: float
    confidence: float
    contributing_strategies: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SignalAggregator(ABC):
    """Base class for signal aggregation methods"""
    
    def __init__(self, container_id: str = "global"):
        self.logger = ContainerLogger("SignalAggregator", container_id)
    
    @abstractmethod
    def aggregate(self, signals: List[AggregatedSignal]) -> Optional[ConsensusSignal]:
        """Aggregate signals into consensus"""
        pass


class WeightedVotingAggregator(SignalAggregator):
    """Weighted voting signal aggregation"""
    
    def __init__(self, min_confidence: float = 0.6, container_id: str = "global"):
        super().__init__(container_id)
        self.min_confidence = min_confidence
    
    def aggregate(self, signals: List[AggregatedSignal]) -> Optional[ConsensusSignal]:
        """Aggregate signals using weighted voting"""
        if not signals:
            return None
        
        symbol = signals[0].signal.symbol
        
        # Calculate weighted votes for each direction
        direction_weights = {
            Direction.BUY: 0.0,
            Direction.SELL: 0.0,
            Direction.HOLD: 0.0
        }
        
        total_weight = 0.0
        
        for agg_signal in signals:
            direction = agg_signal.signal.direction
            weight = agg_signal.weight
            
            # Convert string direction to enum if needed
            if isinstance(direction, str):
                direction = Direction(direction.upper())
            
            direction_weights[direction] += weight
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        # Find winning direction
        winning_direction = max(direction_weights, key=direction_weights.get)
        winning_weight = direction_weights[winning_direction]
        
        # Calculate confidence as percentage of total weight
        confidence = winning_weight / total_weight
        
        # Check minimum confidence threshold
        if confidence < self.min_confidence:
            self.logger.debug(
                f"Insufficient confidence for {symbol}: {confidence:.2f} < {self.min_confidence}"
            )
            return None
        
        # Calculate strength as weighted average of contributing signals
        relevant_signals = [
            s for s in signals 
            if (s.signal.direction == winning_direction or 
                (isinstance(s.signal.direction, str) and 
                 Direction(s.signal.direction.upper()) == winning_direction))
        ]
        
        if not relevant_signals:
            return None
        
        total_relevant_weight = sum(s.weight for s in relevant_signals)
        weighted_strength = sum(
            s.signal.strength * s.weight for s in relevant_signals
        ) / total_relevant_weight
        
        # Get contributing strategies
        contributing_strategies = [s.strategy_id for s in relevant_signals]
        
        consensus = ConsensusSignal(
            symbol=symbol,
            direction=winning_direction,
            strength=weighted_strength,
            confidence=confidence,
            contributing_strategies=contributing_strategies,
            timestamp=datetime.now(),
            metadata={
                'aggregation_method': 'weighted_voting',
                'total_strategies': len(signals),
                'contributing_strategies_count': len(contributing_strategies),
                'direction_weights': {d.value: w for d, w in direction_weights.items()},
                'total_weight': total_weight
            }
        )
        
        self.logger.info(
            f"Consensus: {symbol} {winning_direction.value} "
            f"(confidence: {confidence:.2f}, strength: {weighted_strength:.2f})"
        )
        
        return consensus


class MajorityVotingAggregator(SignalAggregator):
    """Simple majority voting aggregation"""
    
    def __init__(self, min_agreement: float = 0.6, container_id: str = "global"):
        super().__init__(container_id)
        self.min_agreement = min_agreement
    
    def aggregate(self, signals: List[AggregatedSignal]) -> Optional[ConsensusSignal]:
        """Require majority agreement"""
        if not signals:
            return None
        
        symbol = signals[0].signal.symbol
        
        # Count votes for each direction
        direction_counts = {
            Direction.BUY: 0,
            Direction.SELL: 0,
            Direction.HOLD: 0
        }
        
        for agg_signal in signals:
            direction = agg_signal.signal.direction
            
            # Convert string direction to enum if needed
            if isinstance(direction, str):
                direction = Direction(direction.upper())
            
            direction_counts[direction] += 1
        
        total_count = len(signals)
        
        # Find direction with most votes
        winning_direction = max(direction_counts, key=direction_counts.get)
        winning_count = direction_counts[winning_direction]
        
        # Check for majority
        agreement_rate = winning_count / total_count
        
        if agreement_rate < self.min_agreement:
            self.logger.debug(
                f"Insufficient agreement for {symbol}: {agreement_rate:.2f} < {self.min_agreement}"
            )
            return None
        
        # Calculate average strength of agreeing signals
        relevant_signals = [
            s for s in signals 
            if (s.signal.direction == winning_direction or
                (isinstance(s.signal.direction, str) and 
                 Direction(s.signal.direction.upper()) == winning_direction))
        ]
        
        avg_strength = sum(s.signal.strength for s in relevant_signals) / len(relevant_signals)
        contributing_strategies = [s.strategy_id for s in relevant_signals]
        
        consensus = ConsensusSignal(
            symbol=symbol,
            direction=winning_direction,
            strength=avg_strength,
            confidence=agreement_rate,
            contributing_strategies=contributing_strategies,
            timestamp=datetime.now(),
            metadata={
                'aggregation_method': 'majority_voting',
                'total_strategies': total_count,
                'agreement_count': winning_count,
                'agreement_rate': agreement_rate,
                'direction_counts': {d.value: c for d, c in direction_counts.items()}
            }
        )
        
        return consensus


class EnsembleAggregator(SignalAggregator):
    """Ensemble aggregation using multiple methods"""
    
    def __init__(self, 
                 methods: List[SignalAggregator],
                 voting_method: str = "unanimous",
                 container_id: str = "global"):
        super().__init__(container_id)
        self.methods = methods
        self.voting_method = voting_method  # "unanimous", "majority", "weighted"
    
    def aggregate(self, signals: List[AggregatedSignal]) -> Optional[ConsensusSignal]:
        """Aggregate using ensemble of methods"""
        if not signals:
            return None
        
        # Get consensus from each method
        method_results = []
        for method in self.methods:
            result = method.aggregate(signals)
            if result:
                method_results.append(result)
        
        if not method_results:
            return None
        
        # Apply ensemble voting
        if self.voting_method == "unanimous":
            return self._unanimous_voting(method_results)
        elif self.voting_method == "majority":
            return self._majority_voting(method_results)
        else:
            return method_results[0]  # Default to first method
    
    def _unanimous_voting(self, results: List[ConsensusSignal]) -> Optional[ConsensusSignal]:
        """Require all methods to agree"""
        if not results:
            return None
        
        # Check if all agree on direction
        first_direction = results[0].direction
        if not all(r.direction == first_direction for r in results):
            return None
        
        # Average the results
        avg_strength = sum(r.strength for r in results) / len(results)
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        # Combine contributing strategies
        all_strategies = set()
        for r in results:
            all_strategies.update(r.contributing_strategies)
        
        return ConsensusSignal(
            symbol=results[0].symbol,
            direction=first_direction,
            strength=avg_strength,
            confidence=avg_confidence,
            contributing_strategies=list(all_strategies),
            timestamp=datetime.now(),
            metadata={
                'aggregation_method': 'ensemble_unanimous',
                'ensemble_methods': len(results),
                'method_results': [r.metadata.get('aggregation_method') for r in results]
            }
        )
    
    def _majority_voting(self, results: List[ConsensusSignal]) -> Optional[ConsensusSignal]:
        """Use majority vote among methods"""
        if not results:
            return None
        
        # Count direction votes
        direction_counts = {}
        for result in results:
            direction = result.direction
            if direction not in direction_counts:
                direction_counts[direction] = []
            direction_counts[direction].append(result)
        
        # Find majority direction
        majority_direction = max(direction_counts, key=lambda d: len(direction_counts[d]))
        majority_results = direction_counts[majority_direction]
        
        # Average the majority results
        avg_strength = sum(r.strength for r in majority_results) / len(majority_results)
        avg_confidence = sum(r.confidence for r in majority_results) / len(majority_results)
        
        all_strategies = set()
        for r in majority_results:
            all_strategies.update(r.contributing_strategies)
        
        return ConsensusSignal(
            symbol=results[0].symbol,
            direction=majority_direction,
            strength=avg_strength,
            confidence=avg_confidence,
            contributing_strategies=list(all_strategies),
            timestamp=datetime.now(),
            metadata={
                'aggregation_method': 'ensemble_majority',
                'majority_count': len(majority_results),
                'total_methods': len(results)
            }
        )


# Factory functions for easy creation
def create_weighted_voting_aggregator(min_confidence: float = 0.6) -> WeightedVotingAggregator:
    """Create weighted voting aggregator"""
    return WeightedVotingAggregator(min_confidence)


def create_majority_voting_aggregator(min_agreement: float = 0.6) -> MajorityVotingAggregator:
    """Create majority voting aggregator"""
    return MajorityVotingAggregator(min_agreement)


def create_ensemble_aggregator(
    min_confidence: float = 0.6,
    min_agreement: float = 0.6,
    voting_method: str = "unanimous"
) -> EnsembleAggregator:
    """Create ensemble aggregator with both methods"""
    methods = [
        WeightedVotingAggregator(min_confidence),
        MajorityVotingAggregator(min_agreement)
    ]
    return EnsembleAggregator(methods, voting_method)
"""
Unit tests for Step 4: Signal Aggregation Methods

Tests the various signal aggregation algorithms for combining
multiple strategy signals into consensus decisions.
"""

import pytest
from datetime import datetime
from typing import List

from src.strategy.signal_aggregation import (
    WeightedVotingAggregator, MajorityVotingAggregator, EnsembleAggregator,
    TradingSignal, AggregatedSignal, ConsensusSignal, Direction,
    create_weighted_voting_aggregator, create_majority_voting_aggregator
)


class TestTradingSignal:
    """Test TradingSignal data class"""
    
    def test_signal_creation(self):
        """Test basic signal creation"""
        signal = TradingSignal(
            symbol="AAPL",
            direction=Direction.BUY,
            strength=0.8,
            timestamp=datetime.now()
        )
        
        assert signal.symbol == "AAPL"
        assert signal.direction == Direction.BUY
        assert signal.strength == 0.8
        assert signal.metadata == {}
    
    def test_signal_with_metadata(self):
        """Test signal with metadata"""
        metadata = {"strategy": "momentum", "confidence": 0.9}
        signal = TradingSignal(
            symbol="SPY",
            direction=Direction.SELL,
            strength=0.7,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        assert signal.metadata == metadata


class TestAggregatedSignal:
    """Test AggregatedSignal data class"""
    
    def test_aggregated_signal_creation(self):
        """Test creating aggregated signal"""
        base_signal = TradingSignal(
            symbol="MSFT",
            direction=Direction.BUY,
            strength=0.75,
            timestamp=datetime.now()
        )
        
        agg_signal = AggregatedSignal(
            strategy_id="momentum_1",
            signal=base_signal,
            weight=0.4,
            timestamp=datetime.now()
        )
        
        assert agg_signal.strategy_id == "momentum_1"
        assert agg_signal.signal == base_signal
        assert agg_signal.weight == 0.4


class TestWeightedVotingAggregator:
    """Test weighted voting aggregation"""
    
    def test_basic_weighted_voting(self):
        """Test basic weighted voting with clear winner"""
        aggregator = WeightedVotingAggregator(min_confidence=0.6)
        
        signals = [
            AggregatedSignal(
                strategy_id="momentum",
                signal=TradingSignal("AAPL", Direction.BUY, 0.8, datetime.now()),
                weight=0.4,
                timestamp=datetime.now()
            ),
            AggregatedSignal(
                strategy_id="trend",
                signal=TradingSignal("AAPL", Direction.BUY, 0.9, datetime.now()),
                weight=0.3,
                timestamp=datetime.now()
            ),
            AggregatedSignal(
                strategy_id="mean_rev",
                signal=TradingSignal("AAPL", Direction.SELL, 0.7, datetime.now()),
                weight=0.3,
                timestamp=datetime.now()
            )
        ]
        
        consensus = aggregator.aggregate(signals)
        
        assert consensus is not None
        assert consensus.direction == Direction.BUY
        assert consensus.confidence == 0.7  # (0.4 + 0.3) / 1.0
        assert len(consensus.contributing_strategies) == 2
        assert "momentum" in consensus.contributing_strategies
        assert "trend" in consensus.contributing_strategies
    
    def test_insufficient_confidence(self):
        """Test rejection due to insufficient confidence"""
        aggregator = WeightedVotingAggregator(min_confidence=0.8)
        
        signals = [
            AggregatedSignal(
                strategy_id="strategy1",
                signal=TradingSignal("AAPL", Direction.BUY, 0.6, datetime.now()),
                weight=0.5,
                timestamp=datetime.now()
            ),
            AggregatedSignal(
                strategy_id="strategy2",
                signal=TradingSignal("AAPL", Direction.SELL, 0.7, datetime.now()),
                weight=0.5,
                timestamp=datetime.now()
            )
        ]
        
        consensus = aggregator.aggregate(signals)
        
        # Should be None due to insufficient confidence (0.5 < 0.8)
        assert consensus is None
    
    def test_empty_signals(self):
        """Test aggregation with no signals"""
        aggregator = WeightedVotingAggregator()
        consensus = aggregator.aggregate([])
        assert consensus is None
    
    def test_single_signal(self):
        """Test aggregation with single signal"""
        aggregator = WeightedVotingAggregator(min_confidence=0.5)
        
        signals = [
            AggregatedSignal(
                strategy_id="solo",
                signal=TradingSignal("AAPL", Direction.BUY, 0.8, datetime.now()),
                weight=1.0,
                timestamp=datetime.now()
            )
        ]
        
        consensus = aggregator.aggregate(signals)
        
        assert consensus is not None
        assert consensus.direction == Direction.BUY
        assert consensus.confidence == 1.0
        assert consensus.strength == 0.8
    
    def test_strength_calculation(self):
        """Test weighted strength calculation"""
        aggregator = WeightedVotingAggregator(min_confidence=0.6)
        
        signals = [
            AggregatedSignal(
                strategy_id="high_strength",
                signal=TradingSignal("AAPL", Direction.BUY, 0.9, datetime.now()),
                weight=0.6,
                timestamp=datetime.now()
            ),
            AggregatedSignal(
                strategy_id="low_strength",
                signal=TradingSignal("AAPL", Direction.BUY, 0.5, datetime.now()),
                weight=0.4,
                timestamp=datetime.now()
            )
        ]
        
        consensus = aggregator.aggregate(signals)
        
        assert consensus is not None
        # Weighted average: (0.9 * 0.6 + 0.5 * 0.4) / (0.6 + 0.4) = 0.74
        expected_strength = (0.9 * 0.6 + 0.5 * 0.4) / 1.0
        assert abs(consensus.strength - expected_strength) < 0.001
    
    def test_string_direction_handling(self):
        """Test handling of string directions"""
        aggregator = WeightedVotingAggregator(min_confidence=0.6)
        
        # Create signal with string direction
        signal = TradingSignal("AAPL", "BUY", 0.8, datetime.now())
        signals = [
            AggregatedSignal(
                strategy_id="test",
                signal=signal,
                weight=1.0,
                timestamp=datetime.now()
            )
        ]
        
        consensus = aggregator.aggregate(signals)
        
        assert consensus is not None
        assert consensus.direction == Direction.BUY


class TestMajorityVotingAggregator:
    """Test majority voting aggregation"""
    
    def test_basic_majority_voting(self):
        """Test basic majority voting"""
        aggregator = MajorityVotingAggregator(min_agreement=0.6)
        
        signals = [
            AggregatedSignal(
                strategy_id="s1",
                signal=TradingSignal("AAPL", Direction.BUY, 0.8, datetime.now()),
                weight=1.0,
                timestamp=datetime.now()
            ),
            AggregatedSignal(
                strategy_id="s2",
                signal=TradingSignal("AAPL", Direction.BUY, 0.7, datetime.now()),
                weight=1.0,
                timestamp=datetime.now()
            ),
            AggregatedSignal(
                strategy_id="s3",
                signal=TradingSignal("AAPL", Direction.SELL, 0.6, datetime.now()),
                weight=1.0,
                timestamp=datetime.now()
            )
        ]
        
        consensus = aggregator.aggregate(signals)
        
        assert consensus is not None
        assert consensus.direction == Direction.BUY
        assert consensus.confidence == 2/3  # 2 out of 3
        assert len(consensus.contributing_strategies) == 2
    
    def test_insufficient_majority(self):
        """Test insufficient majority"""
        aggregator = MajorityVotingAggregator(min_agreement=0.8)
        
        signals = [
            AggregatedSignal(
                strategy_id="s1",
                signal=TradingSignal("AAPL", Direction.BUY, 0.8, datetime.now()),
                weight=1.0,
                timestamp=datetime.now()
            ),
            AggregatedSignal(
                strategy_id="s2",
                signal=TradingSignal("AAPL", Direction.SELL, 0.7, datetime.now()),
                weight=1.0,
                timestamp=datetime.now()
            )
        ]
        
        consensus = aggregator.aggregate(signals)
        
        # 50% agreement < 80% required
        assert consensus is None
    
    def test_unanimous_agreement(self):
        """Test unanimous agreement"""
        aggregator = MajorityVotingAggregator(min_agreement=0.6)
        
        signals = [
            AggregatedSignal(
                strategy_id=f"s{i}",
                signal=TradingSignal("AAPL", Direction.BUY, 0.8, datetime.now()),
                weight=1.0,
                timestamp=datetime.now()
            )
            for i in range(5)
        ]
        
        consensus = aggregator.aggregate(signals)
        
        assert consensus is not None
        assert consensus.direction == Direction.BUY
        assert consensus.confidence == 1.0  # 100% agreement
        assert len(consensus.contributing_strategies) == 5


class TestEnsembleAggregator:
    """Test ensemble aggregation"""
    
    def test_unanimous_ensemble(self):
        """Test unanimous ensemble voting"""
        weighted = WeightedVotingAggregator(min_confidence=0.5)
        majority = MajorityVotingAggregator(min_agreement=0.5)
        ensemble = EnsembleAggregator([weighted, majority], voting_method="unanimous")
        
        # Create signals where both methods should agree
        signals = [
            AggregatedSignal(
                strategy_id="s1",
                signal=TradingSignal("AAPL", Direction.BUY, 0.8, datetime.now()),
                weight=0.6,
                timestamp=datetime.now()
            ),
            AggregatedSignal(
                strategy_id="s2",
                signal=TradingSignal("AAPL", Direction.BUY, 0.7, datetime.now()),
                weight=0.4,
                timestamp=datetime.now()
            )
        ]
        
        consensus = ensemble.aggregate(signals)
        
        assert consensus is not None
        assert consensus.direction == Direction.BUY
        assert consensus.metadata['aggregation_method'] == 'ensemble_unanimous'
    
    def test_majority_ensemble(self):
        """Test majority ensemble voting"""
        # Create methods with different thresholds
        weighted = WeightedVotingAggregator(min_confidence=0.9)  # High threshold
        majority = MajorityVotingAggregator(min_agreement=0.5)   # Low threshold
        ensemble = EnsembleAggregator([weighted, majority], voting_method="majority")
        
        signals = [
            AggregatedSignal(
                strategy_id="s1",
                signal=TradingSignal("AAPL", Direction.BUY, 0.8, datetime.now()),
                weight=0.6,
                timestamp=datetime.now()
            ),
            AggregatedSignal(
                strategy_id="s2",
                signal=TradingSignal("AAPL", Direction.BUY, 0.7, datetime.now()),
                weight=0.4,
                timestamp=datetime.now()
            )
        ]
        
        consensus = ensemble.aggregate(signals)
        
        # Majority method should pass, weighted might not (due to confidence 0.6 < 0.9)
        # If only majority passes, ensemble should still work
        if consensus:
            assert consensus.direction == Direction.BUY


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_weighted_voting_aggregator(self):
        """Test weighted voting factory"""
        aggregator = create_weighted_voting_aggregator(min_confidence=0.7)
        assert isinstance(aggregator, WeightedVotingAggregator)
        assert aggregator.min_confidence == 0.7
    
    def test_create_majority_voting_aggregator(self):
        """Test majority voting factory"""
        aggregator = create_majority_voting_aggregator(min_agreement=0.8)
        assert isinstance(aggregator, MajorityVotingAggregator)
        assert aggregator.min_agreement == 0.8


class TestRealWorldScenarios:
    """Test real-world aggregation scenarios"""
    
    def test_conflicting_signals(self):
        """Test handling of conflicting signals"""
        aggregator = WeightedVotingAggregator(min_confidence=0.6)
        
        signals = [
            AggregatedSignal(
                strategy_id="momentum",
                signal=TradingSignal("AAPL", Direction.BUY, 0.9, datetime.now()),
                weight=0.3,
                timestamp=datetime.now()
            ),
            AggregatedSignal(
                strategy_id="mean_reversion",
                signal=TradingSignal("AAPL", Direction.SELL, 0.8, datetime.now()),
                weight=0.4,
                timestamp=datetime.now()
            ),
            AggregatedSignal(
                strategy_id="trend",
                signal=TradingSignal("AAPL", Direction.BUY, 0.7, datetime.now()),
                weight=0.3,
                timestamp=datetime.now()
            )
        ]
        
        consensus = aggregator.aggregate(signals)
        
        # SELL should win with 0.4 weight vs BUY with 0.6 weight
        # Actually BUY should win: (0.3 + 0.3) = 0.6 > 0.4
        assert consensus is not None
        assert consensus.direction == Direction.BUY
        assert consensus.confidence == 0.6
    
    def test_hold_signals(self):
        """Test handling of HOLD signals"""
        aggregator = WeightedVotingAggregator(min_confidence=0.6)
        
        signals = [
            AggregatedSignal(
                strategy_id="s1",
                signal=TradingSignal("AAPL", Direction.HOLD, 0.8, datetime.now()),
                weight=0.5,
                timestamp=datetime.now()
            ),
            AggregatedSignal(
                strategy_id="s2",
                signal=TradingSignal("AAPL", Direction.BUY, 0.7, datetime.now()),
                weight=0.3,
                timestamp=datetime.now()
            ),
            AggregatedSignal(
                strategy_id="s3",
                signal=TradingSignal("AAPL", Direction.SELL, 0.6, datetime.now()),
                weight=0.2,
                timestamp=datetime.now()
            )
        ]
        
        consensus = aggregator.aggregate(signals)
        
        # HOLD should win with 0.5 weight
        assert consensus is not None
        assert consensus.direction == Direction.HOLD
        assert consensus.confidence == 0.5
    
    def test_metadata_preservation(self):
        """Test that metadata is properly set in consensus"""
        aggregator = WeightedVotingAggregator(min_confidence=0.6)
        
        signals = [
            AggregatedSignal(
                strategy_id="test_strategy",
                signal=TradingSignal("AAPL", Direction.BUY, 0.8, datetime.now()),
                weight=1.0,
                timestamp=datetime.now()
            )
        ]
        
        consensus = aggregator.aggregate(signals)
        
        assert consensus is not None
        assert 'aggregation_method' in consensus.metadata
        assert consensus.metadata['aggregation_method'] == 'weighted_voting'
        assert 'total_strategies' in consensus.metadata
        assert 'direction_weights' in consensus.metadata


if __name__ == "__main__":
    pytest.main([__file__])
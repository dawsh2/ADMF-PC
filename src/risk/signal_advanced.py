"""Advanced signal processing components.

This module provides advanced signal processing capabilities:
- SignalRouter: Routes signals to appropriate processors
- SignalValidator: Validates signal integrity and feasibility
- RiskAdjustedSignalProcessor: Enhanced processor with risk adjustments
- SignalCache: Caches and deduplicates signals
- SignalPrioritizer: Prioritizes signals for execution
"""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import defaultdict, deque
from enum import IntEnum

import logging

from .protocols import (
    SignalProcessorProtocol,
    Signal,
    Order,
    OrderType,
    OrderSide,
    SignalType,
    PortfolioStateProtocol,
    PositionSizerProtocol,
    RiskLimitProtocol,
)
from .signal_processing import SignalProcessor


class SignalPriority(IntEnum):
    """Signal priority levels."""
    CRITICAL = 1  # Risk exits, stop losses
    HIGH = 2      # Strong signals, exits
    NORMAL = 3    # Regular entry signals
    LOW = 4       # Weak signals, rebalances


class SignalRouter:
    """Route signals to appropriate processors based on type and strategy."""
    
    def __init__(self):
        """Initialize signal router."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._processors: Dict[str, SignalProcessorProtocol] = {}
        self._strategy_processors: Dict[str, SignalProcessorProtocol] = {}
        self._signal_type_processors: Dict[SignalType, SignalProcessorProtocol] = {}
        self._default_processor: Optional[SignalProcessorProtocol] = None
    
    def set_default_processor(self, processor: SignalProcessorProtocol) -> None:
        """Set default processor for unmatched signals."""
        self._default_processor = processor
    
    def add_strategy_processor(
        self,
        strategy_id: str,
        processor: SignalProcessorProtocol
    ) -> None:
        """Add processor for specific strategy."""
        self._strategy_processors[strategy_id] = processor
        self.logger.info(
            f"Strategy processor added - Strategy ID: {strategy_id}, Processor: {type(processor).__name__}"
        )
    
    def add_signal_type_processor(
        self,
        signal_type: SignalType,
        processor: SignalProcessorProtocol
    ) -> None:
        """Add processor for specific signal type."""
        self._signal_type_processors[signal_type] = processor
        self.logger.info(
            f"Signal type processor added - Signal type: {signal_type.value}, Processor: {type(processor).__name__}"
        )
    
    def route_signal(
        self,
        signal: Signal,
        portfolio_state: PortfolioStateProtocol,
        position_sizer: PositionSizerProtocol,
        risk_limits: List[RiskLimitProtocol],
        market_data: Dict[str, Any]
    ) -> Optional[Order]:
        """Route signal to appropriate processor.
        
        Priority:
        1. Strategy-specific processor
        2. Signal type processor
        3. Default processor
        """
        # Try strategy-specific processor first
        processor = self._strategy_processors.get(signal.strategy_id)
        
        # Try signal type processor
        if not processor:
            processor = self._signal_type_processors.get(signal.signal_type)
        
        # Use default processor
        if not processor:
            processor = self._default_processor
        
        if not processor:
            self.logger.warning(
                f"No processor found - Signal: {signal}, Strategy ID: {signal.strategy_id}, Signal type: {signal.signal_type.value}"
            )
            return None
        
        return processor.process_signal(
            signal, portfolio_state, position_sizer, risk_limits, market_data
        )


class SignalValidator:
    """Validate signals before processing."""
    
    def __init__(self, max_signal_age: int = 300):
        """Initialize signal validator.
        
        Args:
            max_signal_age: Maximum age of signal in seconds
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_signal_age = max_signal_age
        self._validation_rules: List[Tuple[str, callable]] = []
        self._add_default_rules()
    
    def _add_default_rules(self) -> None:
        """Add default validation rules."""
        self.add_rule("strength_range", self._validate_strength_range)
        self.add_rule("required_fields", self._validate_required_fields)
        self.add_rule("timestamp_valid", self._validate_timestamp)
        self.add_rule("metadata_valid", self._validate_metadata)
    
    def add_rule(self, name: str, validator: callable) -> None:
        """Add validation rule.
        
        Args:
            name: Rule name
            validator: Function that takes signal and returns (bool, Optional[str])
        """
        self._validation_rules.append((name, validator))
    
    def validate(self, signal: Signal) -> Tuple[bool, List[str]]:
        """Validate signal against all rules.
        
        Returns:
            Tuple of (is_valid, list_of_failures)
        """
        failures = []
        
        for rule_name, validator in self._validation_rules:
            try:
                is_valid, reason = validator(signal)
                if not is_valid:
                    failures.append(f"{rule_name}: {reason}")
            except Exception as e:
                failures.append(f"{rule_name}: {str(e)}")
        
        is_valid = len(failures) == 0
        
        if not is_valid:
            self.logger.warning(
                f"Signal validation failed - Signal ID: {signal.signal_id}, Failures: {failures}"
            )
        
        return is_valid, failures
    
    def _validate_strength_range(self, signal: Signal) -> Tuple[bool, Optional[str]]:
        """Validate signal strength is in valid range."""
        if not -1 <= signal.strength <= 1:
            return False, f"Strength {signal.strength} not in range [-1, 1]"
        return True, None
    
    def _validate_required_fields(self, signal: Signal) -> Tuple[bool, Optional[str]]:
        """Validate required fields are present."""
        if not signal.signal_id:
            return False, "Missing signal_id"
        if not signal.strategy_id:
            return False, "Missing strategy_id"
        if not signal.symbol:
            return False, "Missing symbol"
        return True, None
    
    def _validate_timestamp(self, signal: Signal) -> Tuple[bool, Optional[str]]:
        """Validate timestamp is reasonable."""
        now = datetime.now()
        age = now - signal.timestamp
        
        # Signal shouldn't be from the future
        if age.total_seconds() < -60:  # Allow 1 minute clock skew
            return False, "Signal timestamp is in the future"
        
        # Signal shouldn't be too old
        if age.total_seconds() > self.max_signal_age:
            return False, f"Signal is too old ({age.total_seconds():.1f}s)"
        
        return True, None
    
    def _validate_metadata(self, signal: Signal) -> Tuple[bool, Optional[str]]:
        """Validate signal metadata."""
        if not isinstance(signal.metadata, dict):
            return False, "Metadata must be a dictionary"
        return True, None


class RiskAdjustedSignalProcessor(SignalProcessor):
    """Enhanced signal processor with risk-based adjustments."""
    
    def __init__(self, risk_multiplier: Decimal = Decimal("1.0")):
        """Initialize risk-adjusted processor.
        
        Args:
            risk_multiplier: Multiplier for risk-based size adjustments
        """
        super().__init__()
        self.risk_multiplier = risk_multiplier
    
    def _create_order(
        self,
        signal: Signal,
        size: Decimal,
        market_data: Dict[str, Any]
    ) -> Order:
        """Create order with risk adjustments."""
        # Get base order
        order = super()._create_order(signal, size, market_data)
        
        # Apply risk adjustments based on signal type
        if signal.signal_type == SignalType.RISK_EXIT:
            # Risk exits should be prioritized
            order = Order(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.MARKET,  # Always use market for risk exits
                quantity=order.quantity,
                price=None,  # Market order
                stop_price=order.stop_price,
                time_in_force="IOC",  # Immediate or cancel
                source_signal=order.source_signal,
                risk_checks_passed=order.risk_checks_passed,
                timestamp=order.timestamp,
                metadata={
                    **order.metadata,
                    "priority": "high",
                    "risk_exit": True
                }
            )
        
        return order


class SignalCache:
    """Cache and deduplicate signals."""
    
    def __init__(
        self,
        cache_duration: int = 60,
        max_cache_size: int = 1000
    ):
        """Initialize signal cache.
        
        Args:
            cache_duration: How long to cache signals (seconds)
            max_cache_size: Maximum number of signals to cache
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cache_duration = cache_duration
        self.max_cache_size = max_cache_size
        
        # Cache structure: signal_hash -> (signal, timestamp)
        self._cache: Dict[str, Tuple[Signal, datetime]] = {}
        self._cache_order: deque = deque()
        
        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._duplicates_rejected = 0
    
    def _compute_signal_hash(self, signal: Signal) -> str:
        """Compute hash for signal deduplication."""
        # Include key fields in hash
        key_parts = [
            signal.strategy_id,
            signal.symbol,
            signal.signal_type.value,
            signal.side.value,
            str(signal.strength),
        ]
        return "-".join(key_parts)
    
    def is_duplicate(self, signal: Signal) -> bool:
        """Check if signal is a duplicate.
        
        Returns:
            True if signal is a duplicate within cache duration
        """
        signal_hash = self._compute_signal_hash(signal)
        
        # Check cache
        if signal_hash in self._cache:
            cached_signal, cached_time = self._cache[signal_hash]
            age = datetime.now() - cached_time
            
            if age.total_seconds() < self.cache_duration:
                self._cache_hits += 1
                self._duplicates_rejected += 1
                self.logger.debug(
                    f"Duplicate signal detected - Signal ID: {signal.signal_id}, Cached signal ID: {cached_signal.signal_id}, Age: {age.total_seconds()}s"
                )
                return True
            else:
                # Expired, remove from cache
                del self._cache[signal_hash]
        
        self._cache_misses += 1
        return False
    
    def add_signal(self, signal: Signal) -> None:
        """Add signal to cache."""
        signal_hash = self._compute_signal_hash(signal)
        
        # Enforce cache size limit
        if len(self._cache) >= self.max_cache_size:
            # Remove oldest
            if self._cache_order:
                oldest_hash = self._cache_order.popleft()
                if oldest_hash in self._cache:
                    del self._cache[oldest_hash]
        
        # Add to cache
        self._cache[signal_hash] = (signal, datetime.now())
        self._cache_order.append(signal_hash)
    
    def clean_expired(self) -> int:
        """Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired = []
        
        for signal_hash, (signal, cached_time) in self._cache.items():
            age = now - cached_time
            if age.total_seconds() > self.cache_duration:
                expired.append(signal_hash)
        
        for signal_hash in expired:
            del self._cache[signal_hash]
            # Remove from order tracking
            try:
                self._cache_order.remove(signal_hash)
            except ValueError:
                pass
        
        return len(expired)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0
        if self._cache_hits + self._cache_misses > 0:
            hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses)
        
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": f"{hit_rate:.1%}",
            "duplicates_rejected": self._duplicates_rejected
        }


class SignalPrioritizer:
    """Prioritize signals for execution."""
    
    def __init__(self):
        """Initialize signal prioritizer."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._priority_rules: List[Tuple[str, callable]] = []
        self._add_default_rules()
    
    def _add_default_rules(self) -> None:
        """Add default prioritization rules."""
        self.add_rule("signal_type", self._prioritize_by_type)
        self.add_rule("signal_strength", self._prioritize_by_strength)
        self.add_rule("position_exit", self._prioritize_exits)
    
    def add_rule(self, name: str, scorer: callable) -> None:
        """Add prioritization rule.
        
        Args:
            name: Rule name
            scorer: Function that takes signal and returns priority score (lower is higher priority)
        """
        self._priority_rules.append((name, scorer))
    
    def prioritize(self, signals: List[Signal]) -> List[Signal]:
        """Sort signals by priority.
        
        Returns:
            Signals sorted by priority (highest priority first)
        """
        if not signals:
            return []
        
        # Calculate composite priority score for each signal
        signal_scores: List[Tuple[Signal, float]] = []
        
        for signal in signals:
            total_score = 0.0
            
            for rule_name, scorer in self._priority_rules:
                try:
                    score = scorer(signal)
                    total_score += score
                except Exception as e:
                    self.logger.error(
                        f"Priority rule error - Rule: {rule_name}, Signal ID: {signal.signal_id}, Error: {str(e)}"
                    )
            
            signal_scores.append((signal, total_score))
        
        # Sort by score (lower is higher priority)
        signal_scores.sort(key=lambda x: x[1])
        
        return [signal for signal, _ in signal_scores]
    
    def _prioritize_by_type(self, signal: Signal) -> float:
        """Priority based on signal type."""
        if signal.signal_type == SignalType.RISK_EXIT:
            return SignalPriority.CRITICAL
        elif signal.signal_type == SignalType.EXIT:
            return SignalPriority.HIGH
        elif signal.signal_type == SignalType.ENTRY:
            return SignalPriority.NORMAL
        else:  # REBALANCE
            return SignalPriority.LOW
    
    def _prioritize_by_strength(self, signal: Signal) -> float:
        """Priority based on signal strength."""
        # Stronger signals get higher priority
        # Invert strength so higher strength = lower score
        # Convert Decimal to float for priority calculation
        signal_strength = float(signal.strength) if isinstance(signal.strength, Decimal) else signal.strength
        return 1.0 - abs(signal_strength)
    
    def _prioritize_exits(self, signal: Signal) -> float:
        """Prioritize exit signals."""
        # Exit signals should generally come before entries
        if signal.signal_type in [SignalType.EXIT, SignalType.RISK_EXIT]:
            return -1.0  # Negative score for higher priority
        return 0.0
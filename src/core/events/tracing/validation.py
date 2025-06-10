"""Type flow validation for compile-time safety."""

from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
import logging

from ..types import Event

logger = logging.getLogger(__name__)

class EventFlowValidator:
    """
    Validates event flows match expected patterns.
    
    Provides compile-time validation and compliance checking.
    """
    
    def __init__(self, expected_flows: Dict[str, List[str]]):
        """
        Initialize with expected event flows.
        
        Args:
            expected_flows: Maps initial event to expected sequence
                           e.g., {'SIGNAL': ['ORDER_REQUEST', 'ORDER', 'FILL']}
        """
        self.expected_flows = expected_flows
        self.flow_violations: List[Dict[str, Any]] = []
        
    def validate_sequence(self, events: List[Event]) -> bool:
        """Validate a sequence of events follows expected patterns."""
        # Group by correlation ID
        correlation_groups = self._group_by_correlation(events)
        
        for correlation_id, event_group in correlation_groups.items():
            if not self._validate_group(event_group):
                self.flow_violations.append({
                    'correlation_id': correlation_id,
                    'events': event_group,
                    'reason': 'Invalid event sequence'
                })
                return False
        return True
        
    def _validate_group(self, events: List[Event]) -> bool:
        """Validate a correlated group of events."""
        if not events:
            return True
            
        initial_type = events[0].event_type
        if initial_type not in self.expected_flows:
            return True  # No validation rules for this type
            
        expected = self.expected_flows[initial_type]
        actual = [e.event_type for e in events[1:]]
        
        # Check if actual matches expected (allowing skips)
        expected_idx = 0
        for event_type in actual:
            if expected_idx < len(expected) and event_type == expected[expected_idx]:
                expected_idx += 1
                
        return expected_idx == len(expected)
    
    def _group_by_correlation(self, events: List[Event]) -> Dict[str, List[Event]]:
        """Group events by correlation ID."""
        groups = defaultdict(list)
        
        for event in events:
            if event.correlation_id:
                groups[event.correlation_id].append(event)
                
        return dict(groups)
    
    def get_violations(self) -> List[Dict[str, Any]]:
        """Get list of flow violations."""
        return self.flow_violations
    
    def clear_violations(self) -> None:
        """Clear recorded violations."""
        self.flow_violations.clear()
        
    def validate_compliance(self, events: List[Event], rules: Dict[str, Any]) -> bool:
        """Validate events comply with business rules."""
        # Example rules:
        # - No duplicate orders
        # - Risk limits respected
        # - Proper position lifecycle
        
        violations = []
        
        # Check for duplicate orders
        if 'no_duplicate_orders' in rules and rules['no_duplicate_orders']:
            order_ids = set()
            for event in events:
                if event.event_type == 'ORDER':
                    order_id = event.payload.get('order_id')
                    if order_id in order_ids:
                        violations.append({
                            'rule': 'no_duplicate_orders',
                            'event': event,
                            'reason': f'Duplicate order ID: {order_id}'
                        })
                    order_ids.add(order_id)
                    
        # Check position lifecycle
        if 'position_lifecycle' in rules and rules['position_lifecycle']:
            position_states = {}
            for event in events:
                if event.event_type == 'POSITION_OPEN':
                    pos_id = event.correlation_id
                    if pos_id in position_states:
                        violations.append({
                            'rule': 'position_lifecycle',
                            'event': event,
                            'reason': f'Position {pos_id} already open'
                        })
                    position_states[pos_id] = 'open'
                    
                elif event.event_type == 'POSITION_CLOSE':
                    pos_id = event.correlation_id
                    if pos_id not in position_states or position_states[pos_id] != 'open':
                        violations.append({
                            'rule': 'position_lifecycle',
                            'event': event,
                            'reason': f'Closing unopened position {pos_id}'
                        })
                    else:
                        position_states[pos_id] = 'closed'
                        
        if violations:
            self.flow_violations.extend(violations)
            return False
            
        return True
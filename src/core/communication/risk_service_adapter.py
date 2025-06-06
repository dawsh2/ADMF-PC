"""
Risk Service Adapter for EVENT_FLOW_ARCHITECTURE

This adapter bridges isolated portfolio containers to the root event bus,
implementing the linear flow: Portfolio → ORDER_REQUEST → Risk → ORDER → Execution

Key responsibilities:
1. Subscribe to ORDER_REQUEST events from portfolio containers
2. Validate orders using stateless risk validators
3. Emit ORDER events to root bus after approval
4. Maintain portfolio isolation
"""

from typing import Dict, Any, List, Optional
import logging

from ..events import Event, EventType
from ..components.protocols import StatelessRiskValidator
from .protocols import CommunicationAdapter, Container

logger = logging.getLogger(__name__)


class RiskServiceAdapter:
    """
    Adapter that bridges portfolio ORDER_REQUEST events to validated ORDER events.
    
    This ensures portfolios remain isolated while still participating in the
    system-wide order flow through the root event bus.
    """
    
    def __init__(self, 
                 name: str,
                 config: Dict[str, Any]):
        """
        Initialize risk service adapter.
        
        Args:
            name: Adapter name for logging
            config: Adapter configuration containing:
                - risk_validators: Map of risk profile names to validators
                - root_event_bus: Root event bus for validated orders
        """
        self.name = name
        self.config = config
        self.risk_validators = config.get('risk_validators', {})
        self.root_event_bus = config.get('root_event_bus')
        self.portfolio_containers: Dict[str, Container] = {}  # Will be set in setup()
        self.active = False
        
        # Metrics
        self.orders_processed = 0
        self.orders_approved = 0
        self.orders_rejected = 0
        
        logger.info(f"Created RiskServiceAdapter '{name}'")
    
    def setup(self, containers: Dict[str, Container]) -> None:
        """Configure adapter with container references.
        
        Called once during initialization to establish connections.
        
        Args:
            containers: Map of container names to container instances
        """
        # Find all portfolio containers
        self.portfolio_containers = {
            name: container for name, container in containers.items()
            if name.startswith('portfolio_')
        }
        logger.info(f"RiskServiceAdapter setup with {len(self.portfolio_containers)} portfolio containers")
    
    def start(self) -> None:
        """Start the adapter by subscribing to portfolio ORDER_REQUEST events."""
        if self.active:
            logger.warning(f"RiskServiceAdapter {self.name} already active")
            return
            
        # Subscribe to ORDER_REQUEST events from each portfolio
        for combo_id, portfolio_container in self.portfolio_containers.items():
            portfolio_container.event_bus.subscribe(EventType.ORDER_REQUEST, self._handle_order_request)
            logger.debug(f"Subscribed to ORDER_REQUEST events from portfolio {combo_id}")
        
        self.active = True
        logger.info(f"RiskServiceAdapter {self.name} started")
    
    def stop(self) -> None:
        """Stop the adapter by unsubscribing from events."""
        if not self.active:
            return
            
        # Unsubscribe from all portfolio buses
        for combo_id, portfolio_container in self.portfolio_containers.items():
            portfolio_container.event_bus.unsubscribe(EventType.ORDER_REQUEST, self._handle_order_request)
            
        self.active = False
        logger.info(f"RiskServiceAdapter {self.name} stopped")
    
    def _handle_order_request(self, event: Event) -> None:
        """
        Handle ORDER_REQUEST event from portfolio.
        
        Flow:
        1. Extract order and portfolio state from event
        2. Determine appropriate risk validator
        3. Validate order
        4. If approved, emit ORDER to root bus
        5. If rejected, log rejection (portfolio already knows)
        """
        try:
            self.orders_processed += 1
            
            # Extract payload
            order = event.payload.get('order')
            portfolio_state = event.payload.get('portfolio_state')
            risk_params = event.payload.get('risk_params', {})
            market_data = event.payload.get('market_data', {})
            
            if not all([order, portfolio_state]):
                logger.error(f"Incomplete ORDER_REQUEST payload from {event.source_id}")
                return
            
            # Get portfolio ID from order
            portfolio_id = order.get('portfolio_id')
            if not portfolio_id:
                logger.error(f"ORDER_REQUEST missing portfolio_id")
                return
                
            logger.info(f"RiskServiceAdapter processing ORDER_REQUEST from portfolio {portfolio_id}")
            
            # Determine risk validator based on risk profile
            risk_type = risk_params.get('type', 'default')
            validator = self.risk_validators.get(risk_type)
            
            if not validator:
                # If no specific validator, use default/composite
                validator = self.risk_validators.get('default')
                if not validator:
                    logger.warning(f"No risk validator found for type '{risk_type}', approving by default")
                    validation_result = {'approved': True, 'reason': 'No validator configured'}
                else:
                    validation_result = validator.validate_order(order, portfolio_state, risk_params, market_data)
            else:
                # Validate order
                validation_result = validator.validate_order(order, portfolio_state, risk_params, market_data)
            
            if validation_result.get('approved', False):
                self.orders_approved += 1
                
                # Adjust quantity if validator suggests it
                if validation_result.get('adjusted_quantity'):
                    order['quantity'] = validation_result['adjusted_quantity']
                    logger.info(f"Order quantity adjusted to {order['quantity']} by risk validator")
                
                # Add risk metrics to order metadata
                if 'metadata' not in order:
                    order['metadata'] = {}
                order['metadata']['risk_metrics'] = validation_result.get('risk_metrics', {})
                
                # Emit ORDER event to root bus
                order_event = Event(
                    event_type=EventType.ORDER,
                    payload={'order': order},
                    source_id=f'risk_service_{portfolio_id}',
                    metadata={
                        'risk_validated': True,
                        'risk_type': risk_type,
                        'original_source': event.source_id
                    }
                )
                
                self.root_event_bus.publish(order_event)
                logger.info(f"Risk service approved and published ORDER for portfolio {portfolio_id}")
                
            else:
                self.orders_rejected += 1
                reason = validation_result.get('reason', 'Unknown')
                logger.info(f"Risk service rejected ORDER from portfolio {portfolio_id}: {reason}")
                
                # Note: We don't need to notify the portfolio of rejection
                # The portfolio is already aware it sent an ORDER_REQUEST
                # and can implement timeout logic if needed
                
        except Exception as e:
            logger.error(f"Error processing ORDER_REQUEST: {e}", exc_info=True)
    
    def handle_event(self, event: Event, source: Container) -> None:
        """Process event with error handling and metrics.
        
        This is the main entry point for events flowing through the adapter.
        
        Args:
            event: Event to process
            source: Container that published the event
        """
        # For risk adapter, we only care about ORDER_REQUEST events
        # which are already handled via subscriptions
        if event.event_type == EventType.ORDER_REQUEST:
            self._handle_order_request(event)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics."""
        return {
            'name': self.name,
            'active': self.active,
            'portfolios_connected': len(self.portfolio_containers),
            'orders_processed': self.orders_processed,
            'orders_approved': self.orders_approved,
            'orders_rejected': self.orders_rejected,
            'approval_rate': self.orders_approved / self.orders_processed if self.orders_processed > 0 else 0
        }
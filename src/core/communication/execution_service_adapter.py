"""
Execution Service Adapter for dynamic execution model routing.

This adapter enables different parameter combinations to use different
execution models (slippage, commission, liquidity) in the execution pipeline.
"""

import logging
from typing import Dict, Any, Optional
from decimal import Decimal

from ..events import Event, EventType
from ..types.trading import Order
from .protocols import CommunicationAdapter

logger = logging.getLogger(__name__)


class ExecutionServiceAdapter(CommunicationAdapter):
    """
    Routes orders through appropriate execution models based on combo_id.
    
    This adapter:
    1. Intercepts ORDER events from portfolios
    2. Applies combo-specific execution models (slippage, commission)
    3. Forwards modified orders to execution container
    4. Ensures fills include proper execution costs
    """
    
    def __init__(self, 
                 name: str,
                 execution_models: Dict[str, Dict[str, Any]],
                 combo_execution_mapping: Dict[str, str],
                 root_event_bus: Any):
        """
        Initialize execution service adapter.
        
        Args:
            name: Adapter name
            execution_models: Dict of execution model configs by type
            combo_execution_mapping: Maps combo_id to execution model type
            root_event_bus: Root event bus for publishing
        """
        self.name = name
        self.execution_models = execution_models
        self.combo_execution_mapping = combo_execution_mapping
        self.root_event_bus = root_event_bus
        self._started = False
        
        logger.info(f"Created ExecutionServiceAdapter with {len(execution_models)} execution model configs")
    
    def start(self) -> None:
        """Start the adapter."""
        if not self._started:
            # Subscribe to ORDER events
            self.root_event_bus.subscribe(EventType.ORDER, self._handle_order)
            self._started = True
            logger.info(f"ExecutionServiceAdapter started")
    
    def stop(self) -> None:
        """Stop the adapter."""
        if self._started:
            # Unsubscribe from events
            try:
                self.root_event_bus.unsubscribe(EventType.ORDER, self._handle_order)
            except Exception as e:
                logger.warning(f"Error unsubscribing: {e}")
            self._started = False
    
    def _handle_order(self, event: Event) -> None:
        """Process ORDER event with appropriate execution models."""
        try:
            order = event.payload.get('order', {})
            portfolio_id = order.get('portfolio_id')
            
            if not portfolio_id:
                logger.warning("Order missing portfolio_id, using default execution")
                return
            
            # Extract combo_id from portfolio_id (e.g., 'portfolio_c0001' -> 'c0001')
            combo_id = portfolio_id.replace('portfolio_', '') if portfolio_id.startswith('portfolio_') else portfolio_id
            
            # Get execution model type for this combo
            exec_type = self.combo_execution_mapping.get(combo_id, 'standard')
            exec_models = self.execution_models.get(exec_type, {})
            
            if not exec_models:
                logger.debug(f"No execution models for combo {combo_id}, using defaults")
                return
            
            # Apply slippage model
            if 'slippage' in exec_models and hasattr(exec_models['slippage'], 'calculate_slippage'):
                slippage_model = exec_models['slippage']
                
                # Get market data from event
                market_data = event.payload.get('market_data', {})
                market_price = Decimal(str(order.get('price', 100)))
                
                # Calculate slippage
                slippage = slippage_model.calculate_slippage(
                    order=self._order_dict_to_object(order),
                    market_price=market_price,
                    market_data=market_data
                )
                
                # Add slippage info to order metadata
                if 'metadata' not in order:
                    order['metadata'] = {}
                order['metadata']['expected_slippage'] = float(slippage)
                order['metadata']['execution_model'] = exec_type
                
                logger.debug(f"Applied {exec_type} slippage model to order from {combo_id}: {slippage}")
            
            # Note: Commission is typically calculated post-fill, not pre-order
            # But we can add expected commission to metadata
            if 'commission' in exec_models and hasattr(exec_models['commission'], 'calculate_commission'):
                commission_model = exec_models['commission']
                
                # Estimate commission for metadata
                estimated_commission = self._estimate_commission(
                    commission_model, order
                )
                
                order['metadata']['expected_commission'] = float(estimated_commission)
                
            # Re-publish the enriched order event
            enriched_event = Event(
                event_type=EventType.ORDER,
                payload={
                    'order': order,
                    'portfolio_state': event.payload.get('portfolio_state'),
                    'risk_params': event.payload.get('risk_params'),
                    'market_data': event.payload.get('market_data'),
                    'execution_models': {
                        'slippage': exec_type,
                        'commission': exec_type
                    }
                },
                source_id=self.name,
                metadata={
                    'original_source': event.source_id,
                    'execution_model_applied': exec_type
                }
            )
            
            # Note: We're modifying the event in-place, so execution container
            # will receive the enriched version automatically
            
        except Exception as e:
            logger.error(f"Error in execution service adapter: {e}")
    
    def _order_dict_to_object(self, order_dict: Dict[str, Any]) -> Any:
        """Convert order dict to object-like structure for models."""
        class OrderProxy:
            def __init__(self, data):
                self._data = data
            
            @property
            def quantity(self):
                return self._data.get('quantity', 0)
            
            @property
            def side(self):
                # Convert to OrderSide enum if needed
                side = self._data.get('side', 'buy')
                if side == 'buy':
                    return 1  # OrderSide.BUY
                else:
                    return 2  # OrderSide.SELL
            
            @property
            def symbol(self):
                return self._data.get('symbol', '')
        
        return OrderProxy(order_dict)
    
    def _estimate_commission(self, commission_model: Any, order: Dict[str, Any]) -> Decimal:
        """Estimate commission for an order."""
        try:
            # Use order price as estimate
            price = Decimal(str(order.get('price', 100)))
            quantity = Decimal(str(order.get('quantity', 0)))
            
            return commission_model.calculate_commission(
                order=self._order_dict_to_object(order),
                fill_price=price,
                fill_quantity=quantity
            )
        except Exception as e:
            logger.warning(f"Error estimating commission: {e}")
            return Decimal("0")
    
    def setup(self, containers: Dict[str, Any]) -> None:
        """Setup is handled in __init__."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            'name': self.name,
            'started': self._started,
            'execution_models': len(self.execution_models),
            'combo_mappings': len(self.combo_execution_mapping)
        }
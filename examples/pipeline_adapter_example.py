"""Example usage of the Pipeline Communication Adapter in ADMF-PC.

This example demonstrates how to set up a pipeline adapter for a typical
trading workflow with data → indicators → strategy → risk → execution.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any

from src.core.communication import AdapterConfig, PipelineCommunicationAdapter
from src.core.events.types import Event, EventType
from src.core.logging.container_logger import ContainerLogger
from src.core.containers.protocols import Container


class ExampleContainer(Container):
    """Example container that demonstrates pipeline integration."""
    
    def __init__(self, container_id: str, expected_input_type: EventType = None):
        self.container_id = container_id
        self.expected_input_type = expected_input_type
        self._output_handlers = []
        self.logger = ContainerLogger(
            container_id=container_id,
            component_name=f"{container_id}_component",
            log_level="INFO"
        )
        
    def on_output_event(self, handler):
        """Register handler for output events."""
        self._output_handlers.append(handler)
        
    def emit_output_event(self, event: Event):
        """Emit event to all registered handlers."""
        for handler in self._output_handlers:
            handler(event)
    
    async def receive_event(self, event: Event):
        """Process received event and potentially emit a new one."""
        self.logger.info(
            f"Received {event.event_type.name if isinstance(event.event_type, EventType) else event.event_type} event",
            event_id=event.metadata.get('event_id'),
            correlation_id=event.metadata.get('correlation_id')
        )
        
        # Process based on container type
        if self.container_id == "data_container":
            # Data container doesn't receive events in this example
            pass
            
        elif self.container_id == "indicator_container":
            # Calculate indicators and emit
            indicators = self._calculate_indicators(event.payload)
            new_event = Event(
                event_type=EventType.INDICATOR,
                payload={
                    **event.payload,
                    'indicators': indicators
                },
                timestamp=datetime.now(),
                source_id=self.container_id,
                metadata=event.metadata
            )
            self.emit_output_event(new_event)
            
        elif self.container_id == "strategy_container":
            # Generate signal based on indicators
            signal = self._generate_signal(event.payload)
            if signal:
                signal_event = Event(
                    event_type=EventType.SIGNAL,
                    payload=signal,
                    timestamp=datetime.now(),
                    source_id=self.container_id,
                    metadata=event.metadata
                )
                self.emit_output_event(signal_event)
                
        elif self.container_id == "risk_container":
            # Apply risk checks and create order
            order = self._apply_risk_checks(event.payload)
            if order:
                order_event = Event(
                    event_type=EventType.ORDER,
                    payload=order,
                    timestamp=datetime.now(),
                    source_id=self.container_id,
                    metadata=event.metadata
                )
                self.emit_output_event(order_event)
                
        elif self.container_id == "execution_container":
            # Execute order
            fill = self._execute_order(event.payload)
            self.logger.info(
                "Order executed",
                order_id=event.metadata.get('event_id'),
                fill=fill
            )
    
    def _calculate_indicators(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate technical indicators."""
        price = data.get('price', 0)
        return {
            'sma_20': price * 0.98,  # Simplified
            'rsi': 55.0,
            'macd': 0.5
        }
    
    def _generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal."""
        indicators = data.get('indicators', {})
        if indicators.get('rsi', 0) > 50:
            return {
                'symbol': data.get('symbol'),
                'side': 'BUY',
                'confidence': 0.75,
                'quantity': 100
            }
        return None
    
    def _apply_risk_checks(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Apply risk management rules."""
        # Simplified risk check
        if signal.get('confidence', 0) > 0.6:
            return {
                'symbol': signal.get('symbol'),
                'side': signal.get('side'),
                'quantity': min(signal.get('quantity', 0), 50),  # Risk limit
                'order_type': 'LIMIT',
                'price': 150.5  # Simplified
            }
        return None
    
    def _execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate order execution."""
        return {
            'symbol': order.get('symbol'),
            'side': order.get('side'),
            'quantity': order.get('quantity'),
            'fill_price': order.get('price', 0),
            'status': 'FILLED'
        }


async def run_pipeline_example():
    """Run the pipeline adapter example."""
    
    print("=== Pipeline Communication Adapter Example ===\n")
    
    # Create adapter configuration
    adapter_config = AdapterConfig(
        name="trading_pipeline",
        adapter_type="pipeline",
        retry_attempts=3,
        timeout_ms=5000,
        custom_settings={
            'log_level': 'DEBUG',
            'enable_transformation': True
        }
    )
    
    # Create containers
    containers = [
        ExampleContainer("data_container"),
        ExampleContainer("indicator_container", EventType.BAR),
        ExampleContainer("strategy_container", EventType.INDICATOR),
        ExampleContainer("risk_container", EventType.SIGNAL),
        ExampleContainer("execution_container", EventType.ORDER)
    ]
    
    # Create pipeline adapter
    logger = ContainerLogger(
        container_id="coordinator",
        component_name="pipeline_example",
        log_level="INFO"
    )
    
    adapter = PipelineCommunicationAdapter(adapter_config, logger)
    
    # Setup pipeline
    print("Setting up pipeline...")
    adapter.setup_pipeline(containers)
    
    # Connect adapter
    connected = await adapter.connect()
    if connected:
        print("Pipeline connected successfully\n")
    else:
        print("Failed to connect pipeline")
        return
    
    # Simulate market data events
    data_container = containers[0]
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    for i, symbol in enumerate(symbols):
        print(f"\n--- Processing {symbol} ---")
        
        # Create market data event
        market_event = Event(
            event_type=EventType.BAR,
            payload={
                'symbol': symbol,
                'price': 150.0 + i * 10,
                'volume': 1000000 + i * 100000,
                'timestamp': datetime.now()
            },
            timestamp=datetime.now(),
            source_id='market_data_feed',
            metadata={
                'event_id': f'mkt_{symbol}_{i:03d}',
                'sequence': i
            }
        )
        
        # Inject into pipeline
        data_container.emit_output_event(market_event)
        
        # Wait for processing
        await asyncio.sleep(0.1)
    
    # Get pipeline metrics
    print("\n\n=== Pipeline Performance Metrics ===")
    metrics = adapter.get_pipeline_metrics()
    
    print(f"\nEnd-to-end metrics:")
    print(f"  Total events: {metrics['end_to_end_metrics']['total_events']}")
    print(f"  Average latency: {metrics['end_to_end_metrics']['average_latency_ms']:.2f}ms")
    print(f"  Error rate: {metrics['end_to_end_metrics']['error_rate']:.2%}")
    
    print(f"\nPer-stage breakdown:")
    for stage in metrics['pipeline_flow']:
        if stage['events_processed'] > 0:
            print(f"  {stage['container']}:")
            print(f"    Events processed: {stage['events_processed']}")
            print(f"    Average latency: {stage['average_latency_ms']:.2f}ms")
    
    # Demonstrate custom transformation
    print("\n\n=== Custom Transformation Example ===")
    
    def enrich_signal(event: Event) -> Event:
        """Custom transformation to enrich signals with additional data."""
        enriched_payload = {
            **event.payload,
            'enrichment_timestamp': datetime.now().isoformat(),
            'risk_score': 0.3,  # Add risk scoring
            'market_conditions': 'NORMAL'
        }
        
        return Event(
            event_type=EventType.ORDER,
            payload=enriched_payload,
            timestamp=event.timestamp,
            source_id=event.source_id,
            container_id=event.container_id,
            metadata={
                **event.metadata,
                'enriched': True,
                'transformation': 'signal_enrichment'
            }
        )
    
    # Add the custom transformation
    adapter.add_transformation_rule(
        EventType.SIGNAL,
        EventType.ORDER,
        enrich_signal
    )
    print("Added custom signal enrichment transformation")
    
    # Cleanup
    await adapter.disconnect()
    await adapter.cleanup()
    
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(run_pipeline_example())
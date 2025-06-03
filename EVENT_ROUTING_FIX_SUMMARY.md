# END_OF_DATA Event Routing Fix Summary

## Problem
The END_OF_DATA SYSTEM event published by DataContainer was not reaching RiskContainer. This prevented RiskContainer from closing open positions at the end of the backtest.

## Root Cause
The pipeline adapter was treating all events equally and only forwarding them to the next stage in the linear pipeline:
- DataContainer -> IndicatorContainer -> StrategyContainer -> RiskContainer -> ExecutionContainer

Since DataContainer publishes the END_OF_DATA event, it was only being sent to IndicatorContainer (the next stage), not to RiskContainer which needs it.

## Solution
Modified the pipeline adapter (`src/core/communication/pipeline_adapter.py`) to:

1. **Added special handling for SYSTEM events** in the `_wire_pipeline_stage` method:
   ```python
   # Special handling for SYSTEM events - broadcast to all containers
   if event.event_type == EventType.SYSTEM:
       self._broadcast_system_event(event)
       return
   ```

2. **Added `_broadcast_system_event` method** to send SYSTEM events to all containers:
   ```python
   def _broadcast_system_event(self, event: Event) -> None:
       """Broadcast SYSTEM events to all containers in the pipeline."""
       message = event.payload.get('message', '')
       self.logger.info(
           f"📢 Broadcasting SYSTEM event '{message}' to all {len(self.pipeline_stages)} containers"
       )
       
       # Send to all containers
       for stage in self.pipeline_stages:
           container = stage.container
           if hasattr(container, 'receive_event'):
               try:
                   container.receive_event(event)
                   self.logger.debug(
                       f"SYSTEM event '{message}' sent to {container.metadata.name}"
                   )
               except Exception as e:
                   self.logger.error(
                       f"Failed to send SYSTEM event to {container.metadata.name}",
                       error=str(e),
                       error_type=type(e).__name__
                   )
       
       # Update metrics
       self.metrics.events_sent += len(self.pipeline_stages)
   ```

3. **Added `_setup_system_event_broadcasting` method** for setup logging:
   ```python
   def _setup_system_event_broadcasting(self) -> None:
       """Setup broadcasting for SYSTEM events from DataContainer to all containers."""
       # Find DataContainer in pipeline
       data_container = None
       
       for stage in self.pipeline_stages:
           container = stage.container
           if hasattr(container, 'metadata'):
               role = getattr(container.metadata, 'role', None)
               if role and hasattr(role, 'value'):
                   role_value = role.value
               elif role:
                   role_value = str(role)
               else:
                   role_value = 'unknown'
               
               if role_value == 'data':
                   data_container = container
                   self.logger.info(f"Found DataContainer: {container.metadata.name}")
                   break
       
       if data_container:
           self.logger.info("✅ SYSTEM event broadcasting configured (handled in stage handler)")
       else:
           self.logger.warning("Cannot setup SYSTEM event broadcasting - DataContainer not found")
   ```

4. **Updated setup_pipeline method** to include system event broadcasting setup:
   ```python
   # Wire up SYSTEM event broadcasting from DataContainer
   self.logger.info("🔄 Setting up SYSTEM event broadcasting...")
   self._setup_system_event_broadcasting()
   ```

## Result
- SYSTEM events (including END_OF_DATA) are now broadcast to all containers in the pipeline
- RiskContainer successfully receives the END_OF_DATA event and can close positions
- The workflow completes normally as expected

## Test Results
The logs confirm the fix is working:
```
2025-06-02 16:45:32,150 - src.execution.containers_pipeline - INFO - 📢 Published END_OF_DATA event
2025-06-02 16:45:32,150 - src.execution.containers_pipeline - INFO - 🔍 IndicatorContainer received event: EventType.SYSTEM
2025-06-02 16:45:32,150 - src.execution.containers_pipeline - INFO - 🎯 StrategyContainer received event: EventType.SYSTEM
2025-06-02 16:45:32,151 - src.execution.containers_pipeline - INFO - 🏁 RiskContainer received END_OF_DATA event
2025-06-02 16:45:32,151 - src.core.coordinator.composable_workflow_manager_pipeline - INFO - Workflow completed normally
```
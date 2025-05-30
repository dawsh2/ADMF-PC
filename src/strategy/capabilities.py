"""
Strategy-specific capabilities for ADMF-PC.

These capabilities add strategy-related functionality to components
through composition rather than inheritance.
"""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import logging

from ..core.components.protocols import Capability
from ..core.events import Event, EventType
from .protocols import SignalDirection


logger = logging.getLogger(__name__)


class StrategyCapability(Capability):
    """
    Adds core strategy functionality to any component.
    
    This capability provides:
    - Signal generation tracking
    - Event-based market data handling
    - Signal emission through event bus
    - Strategy metadata management
    """
    
    def get_name(self) -> str:
        return "strategy"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        """Apply strategy capability to component."""
        
        # Ensure component has generate_signal method
        if not hasattr(component, 'generate_signal'):
            raise ValueError(f"Component {component.__class__.__name__} must implement generate_signal method")
        
        # Add strategy metadata
        component._strategy_metadata = {
            'name': spec.get('name', component.__class__.__name__),
            'version': spec.get('version', '1.0'),
            'created': datetime.now(),
            'signal_count': 0,
            'last_signal': None,
            'signal_history': []
        }
        
        # Wrap generate_signal to track metrics
        original_generate = component.generate_signal
        
        def tracked_generate_signal(market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Generate signal with tracking."""
            signal = original_generate(market_data)
            
            if signal:
                # Ensure required fields
                signal['strategy'] = component._strategy_metadata['name']
                signal['timestamp'] = signal.get('timestamp', datetime.now())
                signal['direction'] = signal.get('direction', SignalDirection.HOLD)
                
                # Track signal
                component._strategy_metadata['signal_count'] += 1
                component._strategy_metadata['last_signal'] = signal
                component._strategy_metadata['signal_history'].append({
                    'signal': signal.copy(),
                    'timestamp': datetime.now()
                })
                
                # Limit history size
                if len(component._strategy_metadata['signal_history']) > 1000:
                    component._strategy_metadata['signal_history'].pop(0)
            
            return signal
        
        component.generate_signal = tracked_generate_signal
        
        # Add helper methods
        component.get_signal_count = lambda: component._strategy_metadata['signal_count']
        component.get_last_signal = lambda: component._strategy_metadata['last_signal']
        component.get_signal_history = lambda: component._strategy_metadata['signal_history'].copy()
        
        # Add name property if not present
        if not hasattr(component, 'name'):
            component.name = component._strategy_metadata['name']
        
        # Add event handling support
        if hasattr(component, '_event_bus'):
            self._add_event_handling(component)
        
        return component
    
    def _add_event_handling(self, component: Any) -> None:
        """Add event-based market data handling."""
        
        def on_bar(event: Event) -> None:
            """Handle market bar events."""
            market_data = event.payload
            signal = component.generate_signal(market_data)
            
            if signal and hasattr(component, '_event_bus'):
                # Emit signal event
                signal_event = Event(
                    event_type=EventType.SIGNAL,
                    payload=signal,
                    source_id=component._strategy_metadata['name']
                )
                component._event_bus.publish(signal_event)
        
        # Store handler for cleanup
        component._strategy_bar_handler = on_bar
        
        # Subscribe to bar events
        if hasattr(component, '_event_bus'):
            component._event_bus.subscribe(EventType.BAR, on_bar)


class IndicatorCapability(Capability):
    """
    Adds indicator management to strategies.
    
    This capability provides:
    - Indicator registration and tracking
    - Bulk indicator updates
    - Indicator value access
    - Automatic indicator creation from spec
    """
    
    def get_name(self) -> str:
        return "indicators"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        """Apply indicator capability to component."""
        
        # Initialize indicator storage
        component._indicators = {}
        component._indicator_history = {}
        
        # Add indicator registration
        def register_indicator(name: str, indicator: Any) -> None:
            """Register an indicator with the strategy."""
            component._indicators[name] = indicator
            component._indicator_history[name] = []
            logger.debug(f"Registered indicator '{name}' with strategy")
        
        component.register_indicator = register_indicator
        
        # Add indicator update
        def update_indicators(price_data: Dict[str, float]) -> Dict[str, Optional[float]]:
            """Update all indicators and return current values."""
            results = {}
            timestamp = price_data.get('timestamp', datetime.now())
            close_price = price_data.get('close', price_data.get('price'))
            
            for name, indicator in component._indicators.items():
                if hasattr(indicator, 'calculate'):
                    value = indicator.calculate(close_price, timestamp)
                    results[name] = value
                    
                    # Track history
                    if value is not None:
                        component._indicator_history[name].append({
                            'timestamp': timestamp,
                            'value': value
                        })
                        
                        # Limit history size
                        if len(component._indicator_history[name]) > 1000:
                            component._indicator_history[name].pop(0)
            
            return results
        
        component.update_indicators = update_indicators
        
        # Add indicator access
        def get_indicator_value(name: str) -> Optional[float]:
            """Get current value of named indicator."""
            if name in component._indicators:
                indicator = component._indicators[name]
                return indicator.value if hasattr(indicator, 'value') else None
            return None
        
        def get_all_indicator_values() -> Dict[str, Optional[float]]:
            """Get all current indicator values."""
            return {
                name: get_indicator_value(name)
                for name in component._indicators
            }
        
        component.get_indicator_value = get_indicator_value
        component.get_all_indicator_values = get_all_indicator_values
        
        # Add indicator reset
        def reset_indicators() -> None:
            """Reset all indicators."""
            for indicator in component._indicators.values():
                if hasattr(indicator, 'reset'):
                    indicator.reset()
            
            # Clear history
            for name in component._indicator_history:
                component._indicator_history[name].clear()
        
        component.reset_indicators = reset_indicators
        
        # Create indicators from spec if provided
        if 'indicators' in spec:
            for ind_spec in spec['indicators']:
                # This would use a factory to create indicators
                # For now, we'll skip auto-creation
                logger.info(f"Auto-creation of indicator '{ind_spec}' not yet implemented")
        
        return component


class RuleManagementCapability(Capability):
    """
    Adds rule-based signal generation to strategies.
    
    This capability provides:
    - Rule registration and management
    - Rule evaluation with aggregation
    - Weighted voting mechanisms
    - Rule state tracking
    """
    
    def get_name(self) -> str:
        return "rules"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        """Apply rule management capability to component."""
        
        # Initialize rule storage
        component._rules = []
        component._rule_weights = {}
        component._rule_history = []
        
        # Add rule registration
        def add_rule(rule: Any, weight: float = 1.0) -> None:
            """Add a trading rule with optional weight."""
            component._rules.append(rule)
            rule_name = rule.name if hasattr(rule, 'name') else str(rule)
            component._rule_weights[rule_name] = weight
            logger.debug(f"Added rule '{rule_name}' with weight {weight}")
        
        component.add_rule = add_rule
        
        # Add rule evaluation
        def evaluate_rules(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Evaluate all rules and generate aggregated signal."""
            if not component._rules:
                return None
            
            signals = []
            total_weight = 0
            
            for rule in component._rules:
                if hasattr(rule, 'evaluate'):
                    triggered, strength = rule.evaluate(data)
                    
                    if triggered:
                        rule_name = rule.name if hasattr(rule, 'name') else str(rule)
                        weight = component._rule_weights.get(rule_name, 1.0)
                        
                        signals.append({
                            'rule': rule_name,
                            'strength': strength,
                            'weight': weight,
                            'direction': SignalDirection.BUY if strength > 0 else SignalDirection.SELL
                        })
                        total_weight += weight
            
            if not signals:
                return None
            
            # Aggregate signals
            aggregation_method = spec.get('aggregation_method', 'weighted_average')
            signal = self._aggregate_signals(signals, total_weight, aggregation_method, data)
            
            if signal:
                # Track rule evaluation
                component._rule_history.append({
                    'timestamp': datetime.now(),
                    'signals': signals,
                    'result': signal
                })
                
                # Limit history
                if len(component._rule_history) > 1000:
                    component._rule_history.pop(0)
            
            return signal
        
        component.evaluate_rules = evaluate_rules
        
        # Add rule weight adjustment
        def set_rule_weight(rule_name: str, weight: float) -> None:
            """Update weight for a specific rule."""
            component._rule_weights[rule_name] = weight
        
        def get_rule_weights() -> Dict[str, float]:
            """Get current rule weights."""
            return component._rule_weights.copy()
        
        component.set_rule_weight = set_rule_weight
        component.get_rule_weights = get_rule_weights
        
        # If component has generate_signal, wrap it to use rules
        if hasattr(component, 'generate_signal') and spec.get('use_rules_for_signals', False):
            original_generate = component.generate_signal
            
            def rule_based_generate(market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                """Generate signal using rules."""
                # First check rules
                rule_signal = evaluate_rules(market_data)
                if rule_signal:
                    return rule_signal
                
                # Fall back to original if no rule signal
                return original_generate(market_data)
            
            component.generate_signal = rule_based_generate
        
        return component
    
    def _aggregate_signals(self, signals: List[Dict], total_weight: float, 
                          method: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Aggregate multiple rule signals into one."""
        
        if method == 'weighted_average':
            # Calculate weighted average strength
            buy_strength = sum(
                s['strength'] * s['weight'] 
                for s in signals 
                if s['strength'] > 0
            )
            sell_strength = sum(
                abs(s['strength']) * s['weight'] 
                for s in signals 
                if s['strength'] < 0
            )
            
            if buy_strength > sell_strength:
                direction = SignalDirection.BUY
                strength = buy_strength / total_weight
            else:
                direction = SignalDirection.SELL
                strength = sell_strength / total_weight
            
        elif method == 'majority_vote':
            # Count votes
            buy_votes = sum(1 for s in signals if s['strength'] > 0)
            sell_votes = sum(1 for s in signals if s['strength'] < 0)
            
            if buy_votes > sell_votes:
                direction = SignalDirection.BUY
                strength = buy_votes / len(signals)
            else:
                direction = SignalDirection.SELL
                strength = sell_votes / len(signals)
            
        else:
            # Default to first signal
            direction = signals[0]['direction']
            strength = abs(signals[0]['strength'])
        
        # Create aggregated signal
        return {
            'symbol': data.get('symbol', 'UNKNOWN'),
            'direction': direction,
            'strength': strength,
            'timestamp': data.get('timestamp', datetime.now()),
            'metadata': {
                'rule_signals': signals,
                'aggregation_method': method
            }
        }


class RegimeAdaptiveCapability(Capability):
    """
    Adds regime-adaptive behavior to strategies.
    
    This capability provides:
    - Regime change handling
    - Parameter switching based on regime
    - Regime-specific performance tracking
    - Smooth parameter transitions
    """
    
    def get_name(self) -> str:
        return "regime_adaptive"
    
    def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
        """Apply regime-adaptive capability to component."""
        
        # Initialize regime state
        component._regime_state = {
            'current_regime': None,
            'regime_parameters': spec.get('regime_parameters', {}),
            'regime_history': [],
            'transition_mode': spec.get('transition_mode', 'immediate'),
            'active_parameters': {}
        }
        
        # Add regime change handler
        def on_regime_change(new_regime: str, metadata: Dict[str, Any] = None) -> None:
            """Handle regime change notification."""
            old_regime = component._regime_state['current_regime']
            
            if old_regime != new_regime:
                logger.info(f"Regime change: {old_regime} -> {new_regime}")
                
                # Record regime change
                component._regime_state['regime_history'].append({
                    'timestamp': datetime.now(),
                    'old_regime': old_regime,
                    'new_regime': new_regime,
                    'metadata': metadata or {}
                })
                
                # Update current regime
                component._regime_state['current_regime'] = new_regime
                
                # Apply new parameters
                if new_regime in component._regime_state['regime_parameters']:
                    new_params = component._regime_state['regime_parameters'][new_regime]
                    
                    if component._regime_state['transition_mode'] == 'immediate':
                        # Immediate switch
                        if hasattr(component, 'set_parameters'):
                            component.set_parameters(new_params)
                        component._regime_state['active_parameters'] = new_params.copy()
                        
                    elif component._regime_state['transition_mode'] == 'gradual':
                        # Gradual transition (would implement blending)
                        logger.info("Gradual parameter transition not yet implemented")
                        component._regime_state['active_parameters'] = new_params.copy()
                        
                    elif component._regime_state['transition_mode'] == 'on_flat':
                        # Wait until flat to switch
                        component._regime_state['pending_parameters'] = new_params
                        logger.info("Parameter switch pending until position is flat")
        
        component.on_regime_change = on_regime_change
        
        # Add parameter management
        def set_regime_parameters(regime: str, parameters: Dict[str, Any]) -> None:
            """Set parameters for a specific regime."""
            component._regime_state['regime_parameters'][regime] = parameters.copy()
        
        def get_regime_parameters(regime: Optional[str] = None) -> Dict[str, Any]:
            """Get parameters for a regime (current if not specified)."""
            if regime is None:
                regime = component._regime_state['current_regime']
            
            return component._regime_state['regime_parameters'].get(regime, {})
        
        def get_active_parameters() -> Dict[str, Any]:
            """Get currently active parameters."""
            return component._regime_state['active_parameters'].copy()
        
        component.set_regime_parameters = set_regime_parameters
        component.get_regime_parameters = get_regime_parameters
        component.get_active_parameters = get_active_parameters
        
        # Add regime history access
        def get_regime_history() -> List[Dict[str, Any]]:
            """Get regime change history."""
            return component._regime_state['regime_history'].copy()
        
        component.get_regime_history = get_regime_history
        
        # Subscribe to regime events if event bus available
        if hasattr(component, '_event_bus'):
            def handle_classification_event(event: Event) -> None:
                """Handle classification events from regime detector."""
                if event.event_type == EventType.CLASSIFICATION:
                    new_regime = event.payload.get('classification')
                    metadata = event.payload.get('metadata', {})
                    on_regime_change(new_regime, metadata)
            
            component._event_bus.subscribe(EventType.CLASSIFICATION, handle_classification_event)
            component._regime_classification_handler = handle_classification_event
        
        return component
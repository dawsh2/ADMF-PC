  # Strategy Module with Protocol + Composition Architecture

  ## Overview

  The Strategy module implements trading strategies using protocols and composition, eliminating inheritance 
  hierarchies while providing maximum flexibility. Any function, class, or complex system can become a 
  strategy by implementing the appropriate protocols.

  ## 1. Core Strategy Protocols

  ```python
  from typing import Protocol, runtime_checkable, Dict, Any, Optional, List, Tuple
  from abc import abstractmethod
  from datetime import datetime
  from enum import Enum

  # === Core Strategy Protocol ===
  @runtime_checkable
  class Strategy(Protocol):
      """Base protocol for all trading strategies"""
      
      @abstractmethod
      def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
          """Generate trading signal from market data"""
          ...
      
      @abstractmethod
      def get_name(self) -> str:
          """Get strategy name"""
          ...

  # === Signal Generation Protocols ===
  @runtime_checkable
  class SignalGenerator(Protocol):
      """Protocol for components that generate trading signals"""
      
      @abstractmethod
      def evaluate(self, data: Any) -> Optional[Dict[str, Any]]:
          """Evaluate data and potentially generate signal"""
          ...
      
      @abstractmethod
      def get_signal_strength(self) -> float:
          """Get current signal strength"""
          ...

  @runtime_checkable
  class MultiTimeframeStrategy(Protocol):
      """Protocol for strategies using multiple timeframes"""
      
      @abstractmethod
      def set_timeframes(self, timeframes: List[str]) -> None:
          """Set timeframes for analysis"""
          ...
      
      @abstractmethod
      def evaluate_timeframe(self, timeframe: str, data: Any) -> Dict[str, Any]:
          """Evaluate specific timeframe"""
          ...

  # === Indicator Protocols ===
  @runtime_checkable
  class Indicator(Protocol):
      """Protocol for technical indicators"""
      
      @abstractmethod
      def calculate(self, value: float, timestamp: datetime) -> Optional[float]:
          """Calculate indicator value"""
          ...
      
      @property
      @abstractmethod
      def value(self) -> Optional[float]:
          """Current indicator value"""
          ...
      
      @property
      @abstractmethod
      def ready(self) -> bool:
          """Whether indicator has enough data"""
          ...
      
      @abstractmethod
      def reset(self) -> None:
          """Reset indicator state"""
          ...

  # === Strategy State Protocols ===
  @runtime_checkable
  class StatefulStrategy(Protocol):
      """Protocol for strategies with internal state"""
      
      @abstractmethod
      def get_state(self) -> Dict[str, Any]:
          """Get current strategy state"""
          ...
      
      @abstractmethod
      def set_state(self, state: Dict[str, Any]) -> None:
          """Set strategy state"""
          ...
      
      @abstractmethod
      def reset_state(self) -> None:
          """Reset to initial state"""
          ...

  # === Ensemble Protocols ===
  @runtime_checkable
  class EnsembleMember(Protocol):
      """Protocol for strategies that can participate in ensembles"""
      
      @abstractmethod
      def get_vote(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
          """Get strategy vote: (direction, confidence)"""
          ...
      
      @abstractmethod
      def get_weight(self) -> float:
          """Get strategy weight in ensemble"""
          ...
  ```

  ## 2. Strategy Capabilities

  ### 2.1 Core Strategy Capability

  ```python
  class StrategyCapability(Capability):
      """Adds basic strategy functionality to any component"""
      
      def get_name(self) -> str:
          return "strategy"
      
      def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
          # Ensure component has generate_signal method
          if not hasattr(component, 'generate_signal'):
              # Add default implementation
              component.generate_signal = lambda data: None
          
          # Add strategy metadata
          component._strategy_metadata = {
              'name': spec.get('name', component.__class__.__name__),
              'version': spec.get('version', '1.0'),
              'created': datetime.now(),
              'signal_count': 0,
              'last_signal': None
          }
          
          # Wrap generate_signal to track metrics
          original_generate = component.generate_signal
          
          def tracked_generate_signal(market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
              signal = original_generate(market_data)
              if signal:
                  component._strategy_metadata['signal_count'] += 1
                  component._strategy_metadata['last_signal'] = signal
                  
                  # Add strategy name to signal
                  signal['strategy'] = component._strategy_metadata['name']
                  signal['timestamp'] = signal.get('timestamp', datetime.now())
              
              return signal
          
          component.generate_signal = tracked_generate_signal
          
          # Add helper methods
          component.get_name = lambda: component._strategy_metadata['name']
          component.get_signal_count = lambda: component._strategy_metadata['signal_count']
          component.get_last_signal = lambda: component._strategy_metadata['last_signal']
          
          return component
  ```

  ### 2.2 Indicator Management Capability

  ```python
  class IndicatorManagementCapability(Capability):
      """Adds indicator management to strategies"""
      
      def get_name(self) -> str:
          return "indicator_management"
      
      def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
          # Initialize indicator storage
          component._indicators = {}
          component._indicator_history = {}
          
          # Add indicator registration
          def register_indicator(name: str, indicator: Indicator) -> None:
              """Register an indicator with the strategy"""
              component._indicators[name] = indicator
              component._indicator_history[name] = []
          
          component.register_indicator = register_indicator
          
          # Add indicator update
          def update_indicators(price_data: Dict[str, float]) -> Dict[str, Optional[float]]:
              """Update all indicators and return current values"""
              results = {}
              timestamp = price_data.get('timestamp', datetime.now())
              
              for name, indicator in component._indicators.items():
                  value = indicator.calculate(price_data['close'], timestamp)
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
              """Get current value of named indicator"""
              if name in component._indicators:
                  return component._indicators[name].value
              return None
          
          component.get_indicator_value = get_indicator_value
          
          # Add bulk indicator creation from spec
          if 'indicators' in spec:
              for ind_spec in spec['indicators']:
                  indicator = create_indicator(ind_spec)
                  component.register_indicator(ind_spec['name'], indicator)
          
          return component
  ```

  ### 2.3 Signal Filtering Capability

  ```python
  class SignalFilteringCapability(Capability):
      """Adds signal filtering and validation"""
      
      def get_name(self) -> str:
          return "signal_filtering"
      
      def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
          # Initialize filters
          component._signal_filters = []
          
          # Add filter registration
          def add_signal_filter(filter_func: Callable[[Dict[str, Any]], bool]) -> None:
              """Add a signal filter function"""
              component._signal_filters.append(filter_func)
          
          component.add_signal_filter = add_signal_filter
          
          # Add default filters from spec
          if spec.get('min_signal_strength'):
              min_strength = spec['min_signal_strength']
              component.add_signal_filter(
                  lambda signal: signal.get('strength', 0) >= min_strength
              )
          
          if spec.get('allowed_hours'):
              allowed_hours = spec['allowed_hours']
              component.add_signal_filter(
                  lambda signal: signal['timestamp'].hour in allowed_hours
              )
          
          if spec.get('require_volume'):
              min_volume = spec['require_volume']
              component.add_signal_filter(
                  lambda signal: signal.get('volume', 0) >= min_volume
              )
          
          # Wrap generate_signal to apply filters
          if hasattr(component, 'generate_signal'):
              original_generate = component.generate_signal
              
              def filtered_generate_signal(market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                  signal = original_generate(market_data)
                  
                  if signal:
                      # Apply all filters
                      for filter_func in component._signal_filters:
                          if not filter_func(signal):
                              # Log filtered signal if logging capability exists
                              if hasattr(component, 'logger'):
                                  component.logger.debug(f"Signal filtered: {signal}")
                              return None
                  
                  return signal
              
              component.generate_signal = filtered_generate_signal
          
          return component
  ```

  ### 2.4 State Management Capability

  ```python
  class StateManagementCapability(Capability):
      """Adds state management for stateful strategies"""
      
      def get_name(self) -> str:
          return "state_management"
      
      def apply(self, component: Any, spec: Dict[str, Any]) -> Any:
          # Initialize state storage
          initial_state = spec.get('initial_state', {})
          component._state = initial_state.copy()
          component._state_history = []
          component._state_snapshots = {}
          
          # Add state access methods
          def get_state() -> Dict[str, Any]:
              """Get current strategy state"""
              return component._state.copy()
          
          def set_state(state: Dict[str, Any]) -> None:
              """Set strategy state"""
              component._state = state.copy()
              component._state_history.append({
                  'timestamp': datetime.now(),
                  'state': state.copy()
              })
          
          def update_state(updates: Dict[str, Any]) -> None:
              """Update specific state values"""
              component._state.update(updates)
              component._state_history.append({
                  'timestamp': datetime.now(),
                  'updates': updates.copy()
              })
          
          def reset_state() -> None:
              """Reset to initial state"""
              component._state = initial_state.copy()
              component._state_history.clear()
          
          def save_state_snapshot(name: str) -> None:
              """Save named state snapshot"""
              component._state_snapshots[name] = {
                  'state': component._state.copy(),
                  'timestamp': datetime.now()
              }
          
          def restore_state_snapshot(name: str) -> bool:
              """Restore named state snapshot"""
              if name in component._state_snapshots:
                  component._state = component._state_snapshots[name]['state'].copy()
                  return True
              return False
          
          component.get_state = get_state
          component.set_state = set_state
          component.update_state = update_state
          component.reset_state = reset_state
          component.save_state_snapshot = save_state_snapshot
          component.restore_state_snapshot = restore_state_snapshot
          
          return component
  ```

  ## 3. Strategy Implementations

  ### 3.1 Simple Moving Average Crossover

  ```python
  class MACrossoverStrategy:
      """Simple MA crossover strategy - no inheritance needed"""
      
      def __init__(self, fast_period: int = 10, slow_period: int = 30,
                   signal_threshold: float = 0.0):
          self.fast_period = fast_period
          self.slow_period = slow_period
          self.signal_threshold = signal_threshold
          
          # State
          self.last_signal = None
          self.position_side = None
          
      def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
          """Generate signal from MA crossover"""
          # This would typically use indicators from shared indicator hub
          # For demonstration, assuming indicators are provided in market_data
          fast_ma = market_data.get(f'ma_{self.fast_period}')
          slow_ma = market_data.get(f'ma_{self.slow_period}')
          
          if fast_ma is None or slow_ma is None:
              return None
          
          # Calculate signal
          ma_diff = (fast_ma - slow_ma) / slow_ma
          
          # Check for crossover
          if ma_diff > self.signal_threshold and self.position_side != 'LONG':
              self.position_side = 'LONG'
              return {
                  'symbol': market_data['symbol'],
                  'direction': 'BUY',
                  'strength': abs(ma_diff),
                  'price': market_data['close'],
                  'reason': f'MA({self.fast_period}) crossed above MA({self.slow_period})',
                  'indicators': {
                      'fast_ma': fast_ma,
                      'slow_ma': slow_ma,
                      'ma_diff': ma_diff
                  }
              }
          
          elif ma_diff < -self.signal_threshold and self.position_side != 'SHORT':
              self.position_side = 'SHORT'
              return {
                  'symbol': market_data['symbol'],
                  'direction': 'SELL',
                  'strength': abs(ma_diff),
                  'price': market_data['close'],
                  'reason': f'MA({self.fast_period}) crossed below MA({self.slow_period})',
                  'indicators': {
                      'fast_ma': fast_ma,
                      'slow_ma': slow_ma,
                      'ma_diff': ma_diff
                  }
              }
          
          return None

  # Create with capabilities
  ma_strategy = ComponentFactory().create_component({
      'class': 'MACrossoverStrategy',
      'params': {'fast_period': 10, 'slow_period': 30},
      'capabilities': ['strategy', 'lifecycle', 'events', 'optimization'],
      
      # Strategy-specific configuration
      'min_signal_strength': 0.001,
      'allowed_hours': list(range(8, 17)),  # Trading hours only
      
      # Optimization parameters
      'parameter_space': {
          'fast_period': [5, 10, 15, 20],
          'slow_period': [20, 30, 40, 50],
          'signal_threshold': [0.0, 0.001, 0.002]
      }
  })
  ```

  ### 3.2 RSI Mean Reversion Strategy

  ```python
  class RSIMeanReversionStrategy:
      """RSI-based mean reversion strategy"""
      
      def __init__(self, rsi_period: int = 14, 
                   oversold_threshold: float = 30,
                   overbought_threshold: float = 70,
                   exit_threshold: float = 50):
          self.rsi_period = rsi_period
          self.oversold_threshold = oversold_threshold
          self.overbought_threshold = overbought_threshold
          self.exit_threshold = exit_threshold
          
          # State tracking
          self.in_position = False
          self.position_direction = None
          self.entry_reason = None
          
      def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
          """Generate mean reversion signals based on RSI"""
          rsi = market_data.get(f'rsi_{self.rsi_period}')
          
          if rsi is None:
              return None
          
          # Entry signals
          if not self.in_position:
              if rsi < self.oversold_threshold:
                  self.in_position = True
                  self.position_direction = 'LONG'
                  self.entry_reason = f'RSI oversold ({rsi:.1f})'
                  
                  return {
                      'symbol': market_data['symbol'],
                      'direction': 'BUY',
                      'strength': (self.oversold_threshold - rsi) / self.oversold_threshold,
                      'price': market_data['close'],
                      'reason': self.entry_reason,
                      'indicators': {'rsi': rsi},
                      'signal_type': 'entry'
                  }
              
              elif rsi > self.overbought_threshold:
                  self.in_position = True
                  self.position_direction = 'SHORT'
                  self.entry_reason = f'RSI overbought ({rsi:.1f})'
                  
                  return {
                      'symbol': market_data['symbol'],
                      'direction': 'SELL',
                      'strength': (rsi - self.overbought_threshold) / (100 - self.overbought_threshold),
                      'price': market_data['close'],
                      'reason': self.entry_reason,
                      'indicators': {'rsi': rsi},
                      'signal_type': 'entry'
                  }
          
          # Exit signals
          else:
              should_exit = False
              
              if self.position_direction == 'LONG' and rsi > self.exit_threshold:
                  should_exit = True
                  exit_direction = 'SELL'
              elif self.position_direction == 'SHORT' and rsi < self.exit_threshold:
                  should_exit = True
                  exit_direction = 'BUY'
              
              if should_exit:
                  self.in_position = False
                  position_dir = self.position_direction
                  self.position_direction = None
                  
                  return {
                      'symbol': market_data['symbol'],
                      'direction': exit_direction,
                      'strength': 1.0,  # Full exit
                      'price': market_data['close'],
                      'reason': f'RSI mean reversion exit ({rsi:.1f})',
                      'indicators': {'rsi': rsi},
                      'signal_type': 'exit',
                      'closing_position': position_dir
                  }
          
          return None
  ```

  ### 3.3 Ensemble Strategy

  ```python
  class VotingEnsembleStrategy:
      """Ensemble strategy that combines multiple strategies"""
      
      def __init__(self, voting_method: str = 'weighted',
                   min_agreement: float = 0.6):
          self.voting_method = voting_method
          self.min_agreement = min_agreement
          self.strategies: List[Tuple[Any, float]] = []  # (strategy, weight)
          
      def add_strategy(self, strategy: Any, weight: float = 1.0) -> None:
          """Add a strategy to the ensemble"""
          self.strategies.append((strategy, weight))
          
      def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
          """Generate ensemble signal from member votes"""
          if not self.strategies:
              return None
          
          # Collect votes from all strategies
          votes = []
          total_weight = 0
          
          for strategy, weight in self.strategies:
              if hasattr(strategy, 'generate_signal'):
                  signal = strategy.generate_signal(market_data)
                  if signal:
                      votes.append({
                          'direction': signal['direction'],
                          'strength': signal.get('strength', 1.0),
                          'weight': weight,
                          'strategy': signal.get('strategy', strategy.__class__.__name__),
                          'reason': signal.get('reason', '')
                      })
                      total_weight += weight
          
          if not votes:
              return None
          
          # Aggregate votes
          if self.voting_method == 'weighted':
              result = self._weighted_vote(votes, total_weight)
          elif self.voting_method == 'majority':
              result = self._majority_vote(votes)
          else:
              raise ValueError(f"Unknown voting method: {self.voting_method}")
          
          if result and result['agreement'] >= self.min_agreement:
              return {
                  'symbol': market_data['symbol'],
                  'direction': result['direction'],
                  'strength': result['strength'],
                  'price': market_data['close'],
                  'reason': f"Ensemble vote: {result['agreement']:.1%} agreement",
                  'ensemble_details': {
                      'voting_method': self.voting_method,
                      'agreement': result['agreement'],
                      'votes': votes,
                      'contributing_strategies': result['contributors']
                  }
              }
          
          return None
      
      def _weighted_vote(self, votes: List[Dict], total_weight: float) -> Dict[str, Any]:
          """Perform weighted voting"""
          buy_weight = sum(v['weight'] * v['strength'] 
                          for v in votes if v['direction'] == 'BUY')
          sell_weight = sum(v['weight'] * v['strength'] 
                           for v in votes if v['direction'] == 'SELL')
          
          if buy_weight > sell_weight:
              direction = 'BUY'
              agreement = buy_weight / total_weight
              strength = buy_weight / sum(v['weight'] for v in votes if v['direction'] == 'BUY')
              contributors = [v['strategy'] for v in votes if v['direction'] == 'BUY']
          else:
              direction = 'SELL'
              agreement = sell_weight / total_weight
              strength = sell_weight / sum(v['weight'] for v in votes if v['direction'] == 'SELL')
              contributors = [v['strategy'] for v in votes if v['direction'] == 'SELL']
          
          return {
              'direction': direction,
              'strength': strength,
              'agreement': agreement,
              'contributors': contributors
          }
  ```

  ### 3.4 Regime-Adaptive Strategy

  ```python
  class RegimeAdaptiveStrategy:
      """Strategy that adapts behavior based on market regime"""
      
      def __init__(self, regime_strategies: Dict[str, Any]):
          self.regime_strategies = regime_strategies
          self.current_regime = None
          self.active_strategy = None
          
      def on_regime_change(self, new_regime: str) -> None:
          """Handle regime change notification"""
          if new_regime != self.current_regime:
              self.current_regime = new_regime
              self.active_strategy = self.regime_strategies.get(new_regime)
              
              # Reset new strategy if it has reset method
              if self.active_strategy and hasattr(self.active_strategy, 'reset_state'):
                  self.active_strategy.reset_state()
      
      def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
          """Generate signal using regime-appropriate strategy"""
          if not self.active_strategy:
              return None
          
          # Generate signal from active strategy
          signal = self.active_strategy.generate_signal(market_data)
          
          if signal:
              # Add regime information
              signal['regime'] = self.current_regime
              signal['regime_strategy'] = self.active_strategy.get_name() if hasattr(
                  self.active_strategy, 'get_name'
              ) else self.active_strategy.__class__.__name__
          
          return signal

  # Create with regime-specific strategies
  regime_adaptive = ComponentFactory().create_component({
      'class': 'RegimeAdaptiveStrategy',
      'params': {
          'regime_strategies': {
              'trending_up': ma_strategy,      # Use trend following
              'trending_down': ma_strategy,     # Use trend following
              'ranging': rsi_strategy,          # Use mean reversion
              'high_volatility': None           # No trading
          }
      },
      'capabilities': ['strategy', 'lifecycle', 'events', 'state_management']
  })
  ```

  ## 4. Strategy Factory Functions

  ```python
  def create_simple_strategy(signal_function: Callable) -> Any:
      """Create a strategy from a simple signal function"""
      
      class SimpleStrategyWrapper:
          def __init__(self, func):
              self.func = func
              
          def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
              return self.func(market_data)
      
      # Create instance with capabilities
      return ComponentFactory().create_component({
          'class': SimpleStrategyWrapper,
          'params': {'func': signal_function},
          'capabilities': ['strategy', 'lifecycle']
      })

  # Example: Create strategy from lambda
  momentum_strategy = create_simple_strategy(
      lambda data: {
          'symbol': data['symbol'],
          'direction': 'BUY' if data['close'] > data['open'] else 'SELL',
          'strength': abs(data['close'] - data['open']) / data['open'],
          'price': data['close']
      } if abs(data['close'] - data['open']) / data['open'] > 0.01 else None
  )
  ```

  ## 5. Integration with Container Architecture

  ```python
  class StrategyContainer(UniversalScopedContainer):
      """Container specialized for strategy execution"""
      
      def create_strategy_stack(self, strategy_spec: Dict[str, Any]) -> None:
          """Create complete strategy execution stack"""
          
          # Create strategy with all needed capabilities
          strategy_config = {
              'name': 'strategy',
              'class': strategy_spec['class'],
              'params': strategy_spec.get('params', {}),
              'capabilities': [
                  'strategy',
                  'lifecycle',
                  'events',
                  'indicator_management',
                  'state_management'
              ]
          }
          
          # Add optional capabilities
          if strategy_spec.get('use_filtering', True):
              strategy_config['capabilities'].append('signal_filtering')
              
          if strategy_spec.get('optimizable', False):
              strategy_config['capabilities'].append('optimization')
              strategy_config['parameter_space'] = strategy_spec.get('parameter_space', {})
          
          self.create_component(strategy_config)
          
          # Set up event subscriptions if using shared indicators
          strategy = self.resolve('strategy')
          if hasattr(strategy, 'setup_subscriptions'):
              strategy.setup_subscriptions()
  ```

  ## 6. Configuration Examples

  ```yaml
  # Strategy configuration examples
  strategies:
    ma_crossover:
      class: "MACrossoverStrategy"
      capabilities: 
        - "strategy"
        - "lifecycle"
        - "events"
        - "optimization"
        - "signal_filtering"
      
      params:
        fast_period: 10
        slow_period: 30
        signal_threshold: 0.001
      
      # Signal filtering
      min_signal_strength: 0.002
      allowed_hours: [9, 10, 11, 12, 13, 14, 15]
      require_volume: 1000000
      
      # Optimization
      parameter_space:
        fast_period: [5, 10, 15, 20]
        slow_period: [20, 30, 40, 50]
        signal_threshold: [0.0, 0.001, 0.002, 0.005]
    
    ensemble:
      class: "VotingEnsembleStrategy"
      capabilities: ["strategy", "lifecycle", "events"]
      
      params:
        voting_method: "weighted"
        min_agreement: 0.6
      
      members:
        - strategy: "ma_crossover"
          weight: 0.4
        - strategy: "rsi_mean_reversion"
          weight: 0.3
        - strategy: "momentum"
          weight: 0.3
  ```

  ## 7. Benefits of Protocol-Based Strategies

  1. **Zero Inheritance**: Strategies don't need to inherit from base classes
  2. **Flexible Composition**: Add only needed capabilities
  3. **Easy Testing**: Test strategies with minimal setup
  4. **Container Compatible**: Natural isolation for parallel execution
  5. **Type Safe**: Protocols provide compile-time checks
  6. **Reusable Components**: Indicators and filters can be shared

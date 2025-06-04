"""
Transform traced events into structured SQL records.
Core ETL pipeline for the data mining architecture.
"""
import logging
from typing import List, Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from collections import defaultdict
import json
import uuid

from src.core.events.tracing.traced_event import TracedEvent
from src.core.events.tracing.event_store import EventStore
from src.core.data_mining.storage.schemas import (
    OptimizationRun, Trade, MarketRegime, VolatilityRegime
)
from src.core.data_mining.storage.connections import DatabaseConnection


@dataclass
class EventChain:
    """Represents a chain of related events."""
    correlation_id: str
    events: List[TracedEvent]
    start_time: datetime
    end_time: datetime
    
    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time
        
    def get_events_by_type(self, event_type: str) -> List[TracedEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]
        
    def get_event_sequence(self) -> List[str]:
        """Get the sequence of event types."""
        return [e.event_type for e in self.events]


class EventToSQLTransformer:
    """
    Transforms event streams into structured SQL records.
    This is the bridge between the event layer and analytics layer.
    """
    
    def __init__(self, event_store: EventStore, db_connection: DatabaseConnection):
        self.event_store = event_store
        self.db = db_connection
        self.logger = logging.getLogger(f"{__name__}.EventToSQLTransformer")
        
    def transform_optimization_run(self, correlation_id: str) -> OptimizationRun:
        """Transform events from an optimization run into SQL record."""
        # Load all events for this correlation ID
        events = list(self.event_store.query_events(correlation_id=correlation_id))
        
        if not events:
            raise ValueError(f"No events found for correlation_id: {correlation_id}")
            
        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)
        
        # Create event chain
        chain = EventChain(
            correlation_id=correlation_id,
            events=events,
            start_time=events[0].timestamp,
            end_time=events[-1].timestamp
        )
        
        # Extract optimization run data
        run = self._extract_optimization_metrics(chain)
        
        # Extract market conditions
        self._extract_market_conditions(chain, run)
        
        # Extract execution statistics
        self._extract_execution_stats(chain, run)
        
        # Set event metadata
        run.event_count = len(events)
        run.first_event_id = events[0].event_id
        run.last_event_id = events[-1].event_id
        run.event_storage_path = f"events/{correlation_id}"
        
        return run
        
    def _extract_optimization_metrics(self, chain: EventChain) -> OptimizationRun:
        """Extract performance metrics from event chain."""
        run = OptimizationRun(
            run_id=str(uuid.uuid4()),
            correlation_id=chain.correlation_id,
            strategy_type="unknown",
            parameters={}
        )
        
        # Look for strategy configuration events
        config_events = chain.get_events_by_type("StrategyConfig")
        if config_events:
            config = config_events[0].data
            run.strategy_type = config.get('strategy_type', 'unknown')
            run.parameters = config.get('parameters', {})
            run.strategy_version = config.get('version')
            
        # Look for performance summary events
        summary_events = chain.get_events_by_type("PerformanceSummary")
        if summary_events:
            summary = summary_events[-1].data  # Use last summary
            run.total_return = Decimal(str(summary.get('total_return', 0)))
            run.sharpe_ratio = Decimal(str(summary.get('sharpe_ratio', 0)))
            run.sortino_ratio = Decimal(str(summary.get('sortino_ratio', 0)))
            run.max_drawdown = Decimal(str(summary.get('max_drawdown', 0)))
            run.win_rate = Decimal(str(summary.get('win_rate', 0)))
            run.profit_factor = Decimal(str(summary.get('profit_factor', 0)))
            
        # Extract from portfolio updates
        portfolio_events = chain.get_events_by_type("PortfolioUpdateEvent")
        if portfolio_events:
            # Calculate metrics from portfolio history
            returns = self._calculate_returns_from_portfolio(portfolio_events)
            if not run.total_return and returns:
                run.total_return = Decimal(str(returns[-1]))
                
        return run
        
    def _extract_market_conditions(self, chain: EventChain, run: OptimizationRun) -> None:
        """Extract market regime information."""
        # Look for regime classification events
        regime_events = chain.get_events_by_type("RegimeClassification")
        if regime_events:
            # Count regime occurrences
            regime_counts = defaultdict(int)
            vol_regime_counts = defaultdict(int)
            
            for event in regime_events:
                regime_counts[event.data.get('market_regime')] += 1
                vol_regime_counts[event.data.get('volatility_regime')] += 1
                
            # Set dominant regimes
            if regime_counts:
                run.market_regime = max(regime_counts, key=regime_counts.get)
            if vol_regime_counts:
                run.volatility_regime = max(vol_regime_counts, key=vol_regime_counts.get)
                
        # Calculate average volatility
        volatility_values = []
        for event in chain.events:
            if 'volatility' in event.data:
                volatility_values.append(event.data['volatility'])
                
        if volatility_values:
            run.avg_market_volatility = Decimal(str(sum(volatility_values) / len(volatility_values)))
            
    def _extract_execution_stats(self, chain: EventChain, run: OptimizationRun) -> None:
        """Extract execution statistics."""
        # Count trades
        fill_events = chain.get_events_by_type("FillEvent")
        run.total_trades = len(fill_events)
        
        # Calculate average trade duration
        trade_durations = []
        order_events = chain.get_events_by_type("OrderEvent")
        
        for fill in fill_events:
            # Find corresponding order
            order = self._find_causation_event(fill, order_events)
            if order:
                duration = fill.timestamp - order.timestamp
                trade_durations.append(duration)
                
        if trade_durations:
            avg_duration = sum(trade_durations, timedelta()) / len(trade_durations)
            run.avg_trade_duration = avg_duration
            
        # Calculate costs
        total_slippage = Decimal('0')
        total_commission = Decimal('0')
        
        for fill in fill_events:
            if 'slippage' in fill.data:
                total_slippage += Decimal(str(fill.data['slippage']))
            if 'commission' in fill.data:
                total_commission += Decimal(str(fill.data['commission']))
                
        if fill_events:
            run.avg_slippage = total_slippage / len(fill_events)
        run.total_commission = total_commission
        
    def _find_causation_event(self, event: TracedEvent, candidates: List[TracedEvent]) -> Optional[TracedEvent]:
        """Find the event that caused this event."""
        if not event.causation_id:
            return None
            
        for candidate in candidates:
            if candidate.event_id == event.causation_id:
                return candidate
        return None
        
    def _calculate_returns_from_portfolio(self, portfolio_events: List[TracedEvent]) -> List[float]:
        """Calculate returns series from portfolio events."""
        returns = []
        
        for i in range(1, len(portfolio_events)):
            prev_value = portfolio_events[i-1].data.get('total_value', 0)
            curr_value = portfolio_events[i].data.get('total_value', 0)
            
            if prev_value > 0:
                ret = (curr_value - prev_value) / prev_value
                returns.append(ret)
                
        return returns


class TradeExtractor:
    """Extract individual trades from event streams."""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.logger = logging.getLogger(f"{__name__}.TradeExtractor")
        
    def extract_trades(self, correlation_id: str, run_id: str) -> List[Trade]:
        """Extract all trades from an optimization run."""
        events = list(self.event_store.query_events(correlation_id=correlation_id))
        events.sort(key=lambda e: e.timestamp)
        
        # Group events by trade
        trades = []
        open_positions = {}  # symbol -> trade info
        
        for event in events:
            if event.event_type == "OrderEvent" and event.data.get('order_type') == 'MARKET':
                # New position or add to position
                symbol = event.data['symbol']
                
                if symbol not in open_positions:
                    # New trade
                    trade = Trade(
                        trade_id=str(uuid.uuid4()),
                        run_id=run_id,
                        correlation_id=correlation_id,
                        entry_time=event.timestamp,
                        symbol=symbol,
                        direction='LONG' if event.data.get('side') == 'BUY' else 'SHORT',
                        entry_price=Decimal(str(event.data.get('price', 0))),
                        position_size=int(event.data.get('quantity', 0)),
                        entry_signal_event_id=event.causation_id
                    )
                    open_positions[symbol] = trade
                    
            elif event.event_type == "FillEvent":
                # Position filled
                symbol = event.data['symbol']
                if symbol in open_positions:
                    trade = open_positions[symbol]
                    
                    # Update with fill info
                    if not trade.entry_price or trade.entry_price == 0:
                        trade.entry_price = Decimal(str(event.data.get('fill_price', 0)))
                        
                    # Calculate slippage
                    if 'expected_price' in event.data and 'fill_price' in event.data:
                        trade.slippage = abs(
                            Decimal(str(event.data['fill_price'])) - 
                            Decimal(str(event.data['expected_price']))
                        )
                        
                    # Commission
                    if 'commission' in event.data:
                        trade.commission = Decimal(str(event.data['commission']))
                        
            elif event.event_type in ["CloseSignal", "StopLoss", "TakeProfit"]:
                # Position close signal
                symbol = event.data.get('symbol')
                if symbol in open_positions:
                    trade = open_positions[symbol]
                    trade.exit_time = event.timestamp
                    trade.exit_signal_event_id = event.event_id
                    
                    # Look for fill event after this
                    exit_fill = self._find_exit_fill(events, event)
                    if exit_fill:
                        trade.exit_price = Decimal(str(exit_fill.data.get('fill_price', 0)))
                        
                        # Calculate P&L
                        if trade.direction == 'LONG':
                            trade.pnl = (trade.exit_price - trade.entry_price) * trade.position_size
                        else:
                            trade.pnl = (trade.entry_price - trade.exit_price) * trade.position_size
                            
                        trade.pnl_percent = (trade.pnl / (trade.entry_price * trade.position_size)) * 100
                        
                    # Add to trades list
                    trades.append(trade)
                    del open_positions[symbol]
                    
        # Add any remaining open positions
        for trade in open_positions.values():
            trades.append(trade)
            
        # Extract market context for each trade
        self._extract_trade_context(trades, events)
        
        return trades
        
    def _find_exit_fill(self, events: List[TracedEvent], exit_signal: TracedEvent) -> Optional[TracedEvent]:
        """Find the fill event for an exit signal."""
        for event in events:
            if (event.event_type == "FillEvent" and 
                event.causation_id == exit_signal.event_id and
                event.timestamp > exit_signal.timestamp):
                return event
        return None
        
    def _extract_trade_context(self, trades: List[Trade], events: List[TracedEvent]) -> None:
        """Extract market context for each trade."""
        # Create time-indexed regime map
        regime_map = {}
        volatility_map = {}
        
        for event in events:
            if event.event_type == "RegimeClassification":
                regime_map[event.timestamp] = event.data.get('market_regime')
                volatility_map[event.timestamp] = event.data.get('volatility_level')
                
        # Assign regimes to trades
        for trade in trades:
            # Find regime at entry
            trade.entry_regime = self._find_regime_at_time(trade.entry_time, regime_map)
            trade.entry_volatility = self._find_value_at_time(trade.entry_time, volatility_map)
            
            # Find regime at exit
            if trade.exit_time:
                trade.exit_regime = self._find_regime_at_time(trade.exit_time, regime_map)
                trade.exit_volatility = self._find_value_at_time(trade.exit_time, volatility_map)
                
    def _find_regime_at_time(self, timestamp: datetime, regime_map: Dict[datetime, str]) -> Optional[str]:
        """Find the regime at a specific time."""
        # Find the most recent regime before this timestamp
        valid_times = [t for t in regime_map if t <= timestamp]
        if valid_times:
            latest_time = max(valid_times)
            return regime_map[latest_time]
        return None
        
    def _find_value_at_time(self, timestamp: datetime, value_map: Dict[datetime, Any]) -> Optional[Any]:
        """Find a value at a specific time."""
        valid_times = [t for t in value_map if t <= timestamp]
        if valid_times:
            latest_time = max(valid_times)
            return value_map[latest_time]
        return None


class OptimizationRunExtractor:
    """
    High-level extractor that coordinates the ETL process.
    """
    
    def __init__(self, 
                 event_store: EventStore,
                 db_connection: DatabaseConnection):
        self.event_store = event_store
        self.db = db_connection
        self.event_transformer = EventToSQLTransformer(event_store, db_connection)
        self.trade_extractor = TradeExtractor(event_store)
        self.logger = logging.getLogger(f"{__name__}.OptimizationRunExtractor")
        
    def process_optimization_run(self, correlation_id: str) -> Tuple[str, int]:
        """
        Process a complete optimization run.
        Returns (run_id, trade_count).
        """
        try:
            # Extract optimization run data
            run = self.event_transformer.transform_optimization_run(correlation_id)
            
            # Insert optimization run
            self._insert_optimization_run(run)
            
            # Extract and insert trades
            trades = self.trade_extractor.extract_trades(correlation_id, run.run_id)
            self._insert_trades(trades)
            
            # Extract and insert market conditions
            self._extract_market_conditions(correlation_id, run.run_id)
            
            self.logger.info(
                f"Processed optimization run {run.run_id} with {len(trades)} trades"
            )
            
            return run.run_id, len(trades)
            
        except Exception as e:
            self.logger.error(f"Failed to process optimization run {correlation_id}: {e}")
            raise
            
    def _insert_optimization_run(self, run: OptimizationRun) -> None:
        """Insert optimization run into database."""
        # Convert to dict and handle special types
        run_dict = asdict(run)
        
        # Convert parameters to JSON string
        run_dict['parameters'] = json.dumps(run_dict['parameters'])
        
        # Handle None values
        run_dict = {k: v for k, v in run_dict.items() if v is not None}
        
        # Insert
        self.db.insert_many('optimization_runs', [run_dict])
        
    def _insert_trades(self, trades: List[Trade]) -> None:
        """Insert trades into database."""
        if not trades:
            return
            
        trade_dicts = []
        for trade in trades:
            trade_dict = asdict(trade)
            # Handle None values
            trade_dict = {k: v for k, v in trade_dict.items() if v is not None}
            trade_dicts.append(trade_dict)
            
        self.db.insert_many('trades', trade_dicts)
        
    def _extract_market_conditions(self, correlation_id: str, run_id: str) -> None:
        """Extract and store market condition snapshots."""
        events = list(self.event_store.query_events(
            correlation_id=correlation_id,
            event_type="MarketSnapshot"
        ))
        
        if not events:
            return
            
        conditions = []
        for event in events:
            condition = {
                'condition_id': str(uuid.uuid4()),
                'run_id': run_id,
                'timestamp': event.timestamp,
                'vix_level': event.data.get('vix'),
                'market_regime': event.data.get('regime'),
                'sector_rotation_score': event.data.get('sector_rotation'),
                'equity_bond_correlation': event.data.get('equity_bond_corr'),
                'avg_volume': event.data.get('volume'),
                'liquidity_score': event.data.get('liquidity_score')
            }
            
            # Remove None values
            condition = {k: v for k, v in condition.items() if v is not None}
            conditions.append(condition)
            
        if conditions:
            self.db.insert_many('market_conditions', conditions)
            
            
class BatchETLProcessor:
    """Process multiple optimization runs in batch."""
    
    def __init__(self,
                 event_store: EventStore,
                 db_connection: DatabaseConnection,
                 batch_size: int = 100):
        self.extractor = OptimizationRunExtractor(event_store, db_connection)
        self.event_store = event_store
        self.batch_size = batch_size
        self.logger = logging.getLogger(f"{__name__}.BatchETLProcessor")
        
    def process_date_range(self, 
                          start_date: datetime,
                          end_date: datetime) -> Dict[str, Any]:
        """Process all optimization runs in a date range."""
        # Get unique correlation IDs in date range
        correlation_ids = self.event_store.get_correlation_ids(
            start_time=start_date,
            end_time=end_date
        )
        
        self.logger.info(
            f"Found {len(correlation_ids)} optimization runs to process"
        )
        
        # Process in batches
        results = {
            'processed': 0,
            'failed': 0,
            'total_trades': 0,
            'errors': []
        }
        
        for i in range(0, len(correlation_ids), self.batch_size):
            batch = correlation_ids[i:i + self.batch_size]
            
            for correlation_id in batch:
                try:
                    run_id, trade_count = self.extractor.process_optimization_run(
                        correlation_id
                    )
                    results['processed'] += 1
                    results['total_trades'] += trade_count
                    
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append({
                        'correlation_id': correlation_id,
                        'error': str(e)
                    })
                    
            self.logger.info(
                f"Processed batch {i//self.batch_size + 1}: "
                f"{results['processed']} successful, {results['failed']} failed"
            )
            
        return results
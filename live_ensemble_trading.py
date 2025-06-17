#!/usr/bin/env python3
"""
Live Ensemble Trading System
Connects Alpaca WebSocket data to ADMF-PC ensemble strategy with comprehensive logging.
"""

import asyncio
import websockets
import json
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print(f"🔍 Added to Python path: {src_path}")
print(f"🔍 Current working directory: {os.getcwd()}")

# Simple feature calculation functions (avoiding complex imports for now)
def calculate_simple_features(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate basic technical features without complex imports."""
    if len(df) < 20:
        return {}
    
    features = {}
    
    # Basic price features
    features['close'] = df['close'].iloc[-1] if not df.empty else 0
    features['high'] = df['high'].iloc[-1] if not df.empty else 0
    features['low'] = df['low'].iloc[-1] if not df.empty else 0
    features['volume'] = df['volume'].iloc[-1] if not df.empty else 0
    
    # Simple moving averages
    if len(df) >= 20:
        features['sma_20'] = df['close'].rolling(20).mean().iloc[-1]
    if len(df) >= 14:
        features['sma_14'] = df['close'].rolling(14).mean().iloc[-1]
        
        # Simple RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs)).iloc[-1]
    
    # Volatility (simple ATR approximation)
    if len(df) >= 14:
        high_low = df['high'] - df['low']
        features['volatility'] = high_low.rolling(14).mean().iloc[-1]
    
    return features

def simple_regime_classifier(features: Dict[str, Any]) -> str:
    """Simple regime classification without complex imports."""
    try:
        rsi = features.get('rsi_14', 50)
        volatility = features.get('volatility', 0)
        price = features.get('close', 0)
        sma_20 = features.get('sma_20', price)
        
        # Simple regime detection logic
        is_trending_up = price > sma_20 * 1.001  # Price above SMA
        is_trending_down = price < sma_20 * 0.999  # Price below SMA
        is_high_vol = volatility > (price * 0.01)  # High volatility
        is_oversold = rsi < 30
        is_overbought = rsi > 70
        
        if is_trending_up and not is_high_vol:
            return 'low_vol_bullish'
        elif is_trending_down and not is_high_vol:
            return 'low_vol_bearish'
        elif is_high_vol and is_trending_up:
            return 'high_vol_bullish'
        elif is_high_vol and is_trending_down:
            return 'high_vol_bearish'
        else:
            return 'neutral'
            
    except Exception as e:
        logger.error(f"Regime classification error: {e}")
        return 'neutral'

def simple_ensemble_signal(features: Dict[str, Any], regime: str) -> Dict[str, Any]:
    """Simple ensemble signal generation without complex imports."""
    try:
        rsi = features.get('rsi_14', 50)
        price = features.get('close', 0)
        sma_20 = features.get('sma_20', price)
        
        signals = []
        strategy_details = []
        
        # Simple RSI strategy
        if rsi < 30:  # Oversold
            signals.append(1)
            strategy_details.append({'strategy': 'rsi_oversold', 'signal': 1})
        elif rsi > 70:  # Overbought
            signals.append(-1)
            strategy_details.append({'strategy': 'rsi_overbought', 'signal': -1})
        
        # Simple moving average strategy
        if price > sma_20 * 1.002:  # Price significantly above SMA
            signals.append(1)
            strategy_details.append({'strategy': 'sma_breakout', 'signal': 1})
        elif price < sma_20 * 0.998:  # Price significantly below SMA
            signals.append(-1)
            strategy_details.append({'strategy': 'sma_breakdown', 'signal': -1})
        
        # Aggregate signals
        if not signals:
            return None
            
        bullish_count = sum(1 for s in signals if s > 0)
        bearish_count = sum(1 for s in signals if s < 0)
        total_count = len(signals)
        
        # Determine ensemble signal
        if bullish_count > bearish_count:
            agreement_ratio = bullish_count / total_count
            if agreement_ratio >= 0.5:  # 50% agreement threshold
                ensemble_signal = 1
            else:
                ensemble_signal = 0
        elif bearish_count > bullish_count:
            agreement_ratio = bearish_count / total_count
            if agreement_ratio >= 0.5:
                ensemble_signal = -1
            else:
                ensemble_signal = 0
        else:
            ensemble_signal = 0
            agreement_ratio = 0.5
        
        return {
            'signal_value': ensemble_signal,
            'timestamp': datetime.now(),
            'strategy_id': 'simple_ensemble',
            'symbol_timeframe': 'SPY_1m',
            'metadata': {
                'regime': regime,
                'active_strategies': 2,  # RSI + SMA
                'signals_generated': len(signals),
                'bullish_signals': bullish_count,
                'bearish_signals': bearish_count,
                'agreement_ratio': agreement_ratio,
                'strategy_details': strategy_details,
                'price': price,
                'rsi': rsi,
                'sma_20': sma_20
            }
        }
        
    except Exception as e:
        logger.error(f"Ensemble signal error: {e}")
        return None

print("✅ Using simplified ensemble implementation (no complex imports)")


# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(name)20s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'live_ensemble_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

# Set strategy logging to DEBUG for detailed signal info
logging.getLogger('strategy.strategies.ensemble.duckdb_ensemble').setLevel(logging.DEBUG)
logging.getLogger('strategy.strategies.classifiers').setLevel(logging.DEBUG)


class LiveEnsembleTrader:
    """Live trading system connecting Alpaca data to ADMF-PC ensemble."""
    
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_API_SECRET")
        self.websocket = None
        self.is_running = False
        
        # Market data storage (rolling window)
        self.market_data = []
        self.max_history = 200  # Keep 200 bars for indicators
        
        # Signal tracking
        self.last_signal = None
        self.signal_count = 0
        self.regime_changes = 0
        self.last_regime = None
        
        # Ensemble configuration
        self.ensemble_config = {
            'classifier_name': 'volatility_momentum_classifier',
            'min_agreement': 0.3,
            'regime_strategies': {
                'neutral': [
                    {'name': 'rsi_strategy', 'params': {'rsi_period': 14, 'entry_rsi_oversold': 30, 'entry_rsi_overbought': 70}}
                ]
            }
        }
        
        # Classifier configuration  
        self.classifier_config = {
            'vol_threshold': 0.8,
            'rsi_overbought': 60,
            'rsi_oversold': 40,
            'atr_period': 14,
            'rsi_period': 14,
            'sma_period': 20
        }
        
        logger.info("🚀 LiveEnsembleTrader initialized")
        logger.info(f"📊 Ensemble config: {self.ensemble_config}")
        logger.info(f"🧠 Classifier config: {self.classifier_config}")
    
    async def connect_alpaca(self) -> bool:
        """Connect to Alpaca WebSocket and authenticate."""
        try:
            if not self.api_key or not self.secret_key:
                logger.error("❌ Missing Alpaca credentials")
                return False
            
            logger.info("🔌 Connecting to Alpaca WebSocket...")
            self.websocket = await websockets.connect("wss://stream.data.alpaca.markets/v2/iex")
            logger.info("✅ WebSocket connected")
            
            # Authenticate
            await self.websocket.send(json.dumps({
                "action": "auth",
                "key": self.api_key,
                "secret": self.secret_key
            }))
            
            # Subscribe to SPY bars
            await self.websocket.send(json.dumps({
                "action": "subscribe",
                "bars": ["SPY"]
            }))
            
            logger.info("📡 Sent auth + subscription to SPY bars")
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    def calculate_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical features for the ensemble strategy."""
        return calculate_simple_features(df)
    
    def create_bar_dict(self, bar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Alpaca bar data to ADMF-PC format."""
        return {
            'timestamp': pd.to_datetime(bar_data.get('t')),
            'symbol': bar_data.get('S', 'SPY'),
            'timeframe': '1m',
            'open': bar_data.get('o', 0),
            'high': bar_data.get('h', 0),
            'low': bar_data.get('l', 0),
            'close': bar_data.get('c', 0),
            'volume': bar_data.get('v', 0)
        }
    
    async def process_bar(self, bar_data: Dict[str, Any]):
        """Process incoming bar and generate ensemble signal."""
        try:
            # Convert to ADMF-PC format
            bar_dict = self.create_bar_dict(bar_data)
            
            # Add to rolling history
            self.market_data.append(bar_dict)
            if len(self.market_data) > self.max_history:
                self.market_data.pop(0)
            
            # Create DataFrame for feature calculation
            df = pd.DataFrame(self.market_data)
            if len(df) < 2:
                logger.debug("⏰ Need more data for processing")
                return
            
            # Calculate features
            features = self.calculate_features(df)
            if not features:
                logger.debug("⏰ Features not ready yet")
                return
            
            # Detect regime using simplified classifier
            current_regime = simple_regime_classifier(features)
            
            # Track regime changes
            if current_regime != self.last_regime:
                self.regime_changes += 1
                logger.info(f"🔄 REGIME CHANGE: {self.last_regime} → {current_regime} (#{self.regime_changes})")
                self.last_regime = current_regime
            
            # Generate ensemble signal using simplified ensemble
            signal_result = simple_ensemble_signal(features, current_regime)
            
            if signal_result:
                self.signal_count += 1
                signal_value = signal_result.get('signal_value', 0)
                metadata = signal_result.get('metadata', {})
                
                # Log detailed signal information
                timestamp = bar_dict['timestamp'].strftime('%H:%M:%S')
                price = bar_dict['close']
                
                if signal_value != 0:
                    signal_type = "🟢 BUY" if signal_value > 0 else "🔴 SELL"
                    logger.info(f"🎯 SIGNAL #{self.signal_count}: {signal_type}")
                    logger.info(f"   ⏰ Time: {timestamp}")
                    logger.info(f"   💰 Price: ${price:.2f}")
                    logger.info(f"   🧠 Regime: {metadata.get('regime', 'unknown')}")
                    logger.info(f"   📊 Active Strategies: {metadata.get('active_strategies', 0)}")
                    logger.info(f"   ✅ Signals Generated: {metadata.get('signals_generated', 0)}")
                    logger.info(f"   📈 Bullish: {metadata.get('bullish_signals', 0)}")
                    logger.info(f"   📉 Bearish: {metadata.get('bearish_signals', 0)}")
                    logger.info(f"   🤝 Agreement: {metadata.get('agreement_ratio', 0):.2%}")
                    
                    # Log strategy details
                    strategy_details = metadata.get('strategy_details', [])
                    if strategy_details:
                        logger.info(f"   🔧 Strategy Details:")
                        for detail in strategy_details:
                            strategy_name = detail.get('strategy', 'unknown')
                            strategy_signal = detail.get('signal', 0)
                            signal_emoji = "📈" if strategy_signal > 0 else "📉"
                            logger.info(f"      {signal_emoji} {strategy_name}: {strategy_signal}")
                    
                    self.last_signal = signal_result
                else:
                    logger.debug(f"📊 No signal | {timestamp} | ${price:.2f} | {current_regime}")
            
            else:
                logger.debug(f"❌ No ensemble result for bar at {bar_dict['timestamp']}")
                
        except Exception as e:
            logger.error(f"❌ Error processing bar: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    async def listen_for_bars(self):
        """Listen for incoming market data and process bars."""
        logger.info("📊 Starting market data listener...")
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    messages = data if isinstance(data, list) else [data]
                    
                    for msg in messages:
                        msg_type = msg.get("T", "?")
                        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        
                        if msg_type == "success":
                            logger.info(f"[{timestamp}] ✅ {msg.get('msg', '')}")
                        elif msg_type == "subscription":
                            logger.info(f"[{timestamp}] 📡 Subscribed: {msg.get('bars', [])}")
                            logger.info("🔴 LIVE ENSEMBLE TRADING ACTIVE")
                            logger.info("=" * 60)
                        elif msg_type == "b":  # Bar data - process for ensemble
                            await self.process_bar(msg)
                        else:
                            logger.debug(f"[{timestamp}] 📩 {msg}")
                            
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"Message processing error: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 WebSocket connection closed")
        except Exception as e:
            logger.error(f"❌ Listener error: {e}")
    
    async def run(self):
        """Main trading loop."""
        logger.info("🚀 Starting Live Ensemble Trading System")
        logger.info("=" * 60)
        
        try:
            # Connect to Alpaca
            if not await self.connect_alpaca():
                return
            
            self.is_running = True
            
            # Start listening for market data
            await self.listen_for_bars()
            
        except KeyboardInterrupt:
            logger.info("🛑 Stopped by user")
        except Exception as e:
            logger.error(f"❌ Trading system error: {e}")
        finally:
            self.is_running = False
            if self.websocket:
                await self.websocket.close()
            
            # Print summary
            logger.info("📊 TRADING SESSION SUMMARY")
            logger.info("=" * 40)
            logger.info(f"📈 Total Signals Generated: {self.signal_count}")
            logger.info(f"🔄 Regime Changes: {self.regime_changes}")
            logger.info(f"📊 Market Data Points: {len(self.market_data)}")
            if self.last_signal:
                logger.info(f"🎯 Last Signal: {self.last_signal.get('signal_value', 0)}")
            logger.info("✅ Session complete")


async def main():
    """Main entry point."""
    print("🎯 ADMF-PC Live Ensemble Trading System")
    print("=" * 50)
    print("💡 This will:")
    print("   • Connect to Alpaca WebSocket for live SPY bars")
    print("   • Calculate technical features in real-time")
    print("   • Detect market regime using classifier")
    print("   • Generate ensemble trading signals")
    print("   • Log all activity with detailed signal info")
    print("")
    print("💡 Press Ctrl+C to stop")
    print("=" * 50)
    print("")
    
    trader = LiveEnsembleTrader()
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
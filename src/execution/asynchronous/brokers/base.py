"""
Common async broker functionality.

Provides composable components for async broker implementations.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ...types import Order, Fill, Position
from ...async_protocols import AsyncBroker

logger = logging.getLogger(__name__)


@dataclass
class BrokerConfig:
    """Configuration for async broker."""
    broker_name: str
    api_key: str
    secret_key: str
    base_url: str = ""
    paper_trading: bool = True
    rate_limit: int = 10  # concurrent requests
    min_request_interval: float = 0.1  # seconds
    cache_ttl: float = 30.0  # seconds


class RateLimiter:
    """Rate limiting functionality for API requests."""
    
    def __init__(self, max_concurrent: int = 10, min_interval: float = 0.1):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.min_interval = min_interval
        self.last_request_time = 0
    
    async def execute(self, request_func, *args, **kwargs):
        """Execute request with rate limiting."""
        async with self.semaphore:
            # Ensure minimum interval between requests
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            
            try:
                result = await request_func(*args, **kwargs)
                self.last_request_time = asyncio.get_event_loop().time()
                return result
                
            except Exception as e:
                logger.error(f"Rate limited request failed: {e}")
                raise


class CacheManager:
    """Cache management for broker data."""
    
    def __init__(self, ttl: float = 30.0):
        self.ttl = ttl
        self._account_info_cache = {}
        self._positions_cache = {}
        self._last_update = 0
    
    def is_valid(self) -> bool:
        """Check if cache is still valid."""
        current_time = asyncio.get_event_loop().time()
        return (current_time - self._last_update) < self.ttl
    
    def update_timestamp(self) -> None:
        """Update cache timestamp."""
        self._last_update = asyncio.get_event_loop().time()
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._account_info_cache.clear()
        self._positions_cache.clear()
        self._last_update = 0
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get cached account info if valid."""
        return self._account_info_cache if self.is_valid() else None
    
    def set_account_info(self, data: Dict[str, Any]) -> None:
        """Set account info cache."""
        self._account_info_cache = data
        self.update_timestamp()
    
    def get_positions(self) -> Optional[Dict[str, Position]]:
        """Get cached positions if valid."""
        return self._positions_cache if self.is_valid() else None
    
    def set_positions(self, data: Dict[str, Position]) -> None:
        """Set positions cache."""
        self._positions_cache = data
        self.update_timestamp()


class ConnectionManager:
    """Connection state management."""
    
    def __init__(self, broker_name: str):
        self.broker_name = broker_name
        self.connected = False
        self.session = None
        self.logger = logger.getChild(broker_name)
    
    async def connect(self, connect_func, auth_func) -> None:
        """Connect to broker using provided functions."""
        if self.connected:
            return
        
        self.logger.info(f"Connecting to {self.broker_name}")
        
        try:
            await connect_func()
            await auth_func()
            
            self.connected = True
            self.logger.info("Successfully connected to broker")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to broker: {e}")
            raise
    
    async def disconnect(self, disconnect_func) -> None:
        """Disconnect from broker using provided function."""
        if not self.connected:
            return
        
        self.logger.info("Disconnecting from broker")
        
        try:
            await disconnect_func()
            self.connected = False
            self.logger.info("Successfully disconnected from broker")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")


class OrderValidator:
    """Order validation functionality."""
    
    def __init__(self, supported_types: List[str], min_size: float = 1.0):
        self.supported_types = [t.upper() for t in supported_types]
        self.min_size = min_size
    
    async def validate(self, order: Order, is_connected: bool, 
                      broker_validator = None) -> tuple[bool, Optional[str]]:
        """Validate order with basic and broker-specific checks."""
        # Basic validation
        if float(order.quantity) < self.min_size:
            return False, f"Order quantity {order.quantity} below minimum {self.min_size}"
        
        if order.order_type.value.upper() not in self.supported_types:
            return False, f"Order type {order.order_type} not supported"
        
        # Check connection
        if not is_connected:
            return False, "Not connected to broker"
        
        # Broker-specific validation if provided
        if broker_validator:
            return await broker_validator(order)
        
        return True, None


# Example of how to compose these components into a broker implementation:
"""
class AlpacaBroker:
    # This is an implementation that satisfies AsyncBroker protocol
    # using composition of the above components
    
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit, config.min_request_interval)
        self.cache = CacheManager(config.cache_ttl)
        self.connection = ConnectionManager(config.broker_name)
        self.validator = OrderValidator(['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT'])
    
    async def submit_order(self, order: Order) -> str:
        # Implementation using rate_limiter
        return await self.rate_limiter.execute(self._submit_order_impl, order)
    
    async def get_positions(self) -> Dict[str, Position]:
        # Implementation using cache
        cached = self.cache.get_positions()
        if cached is not None:
            return cached
        
        positions = await self._fetch_positions()
        self.cache.set_positions(positions)
        return positions
"""
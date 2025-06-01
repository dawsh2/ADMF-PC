# Step 12: Crypto & DeFi Integration

**Status**: Advanced Step  
**Complexity**: Very High  
**Prerequisites**: [Step 11: Alternative Data Integration](step-11-alternative-data.md) completed  
**Architecture Ref**: [Crypto/DeFi Architecture](../architecture/crypto-defi-architecture.md)

## 🎯 Objective

Integrate cryptocurrency and decentralized finance protocols:
- Connect to major cryptocurrency exchanges (CEX)
- Integrate with decentralized exchanges (DEX)
- Implement DeFi protocol interactions
- Handle blockchain data and on-chain analytics
- Multi-chain wallet management
- Smart contract interaction and monitoring
- Cross-chain bridge integration

## 📋 Required Reading

Before starting:
1. [Cryptocurrency Exchange Integration](../references/crypto-exchange-integration.md)
2. [DeFi Protocol Fundamentals](../references/defi-fundamentals.md)
3. [Web3 Development Guide](../references/web3-development.md)
4. [Smart Contract Security](../references/smart-contract-security.md)

## 🏗️ Implementation Tasks

### 1. Crypto Exchange Framework

```python
# src/crypto/exchange_base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import asyncio
import aiohttp
import websockets
import hmac
import hashlib
import json
import ccxt.async_support as ccxt
from web3 import Web3
from eth_account import Account

class ExchangeType(Enum):
    """Types of cryptocurrency exchanges"""
    CENTRALIZED = "centralized"
    DECENTRALIZED = "decentralized"
    HYBRID = "hybrid"

class OrderType(Enum):
    """Cryptocurrency order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class NetworkType(Enum):
    """Blockchain networks"""
    ETHEREUM = "ethereum"
    BSC = "bsc"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    AVALANCHE = "avalanche"
    SOLANA = "solana"

@dataclass
class CryptoAsset:
    """Cryptocurrency asset information"""
    symbol: str
    name: str
    network: NetworkType
    contract_address: Optional[str] = None
    decimals: int = 18
    
    # Market data
    price_usd: Optional[Decimal] = None
    market_cap: Optional[Decimal] = None
    volume_24h: Optional[Decimal] = None
    
    # DeFi specific
    is_wrapped: bool = False
    underlying_asset: Optional[str] = None
    
    # Risk metrics
    volatility: Optional[float] = None
    liquidity_score: Optional[float] = None

@dataclass
class CryptoOrder:
    """Cryptocurrency order"""
    order_id: str
    exchange: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    
    # Amounts
    amount: Decimal
    price: Optional[Decimal] = None
    
    # Status
    status: str = "pending"
    filled_amount: Decimal = Decimal("0")
    average_price: Optional[Decimal] = None
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    # Fees
    fee_amount: Decimal = Decimal("0")
    fee_currency: Optional[str] = None
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeFiPosition:
    """DeFi protocol position"""
    protocol: str
    position_type: str  # 'lending', 'borrowing', 'liquidity', 'staking'
    asset: CryptoAsset
    
    # Position details
    amount: Decimal
    value_usd: Decimal
    
    # Protocol specific
    pool_address: Optional[str] = None
    apy: Optional[float] = None
    rewards_pending: Optional[Decimal] = None
    
    # Risk metrics
    health_factor: Optional[float] = None
    liquidation_price: Optional[Decimal] = None

class BaseCryptoExchange(ABC):
    """Base class for cryptocurrency exchanges"""
    
    def __init__(self, exchange_id: str, api_key: str = None, 
                 api_secret: str = None, testnet: bool = False):
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Exchange client
        self.client = None
        
        # WebSocket connections
        self.ws_connections = {}
        
        # Order tracking
        self.active_orders = {}
        
        # Rate limiting
        self.rate_limiter = RateLimiter()
        
        # Logger
        self.logger = ComponentLogger(f"CryptoExchange_{exchange_id}", "crypto")
        
        # Initialize exchange
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize exchange connection"""
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, Decimal]:
        """Get account balances"""
        pass
    
    @abstractmethod
    async def place_order(self, symbol: str, side: str, order_type: OrderType,
                         amount: Decimal, price: Optional[Decimal] = None) -> CryptoOrder:
        """Place an order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> CryptoOrder:
        """Get order status"""
        pass
    
    async def get_ticker(self, symbol: str) -> Dict[str, Decimal]:
        """Get current ticker data"""
        await self.rate_limiter.acquire()
        
        try:
            ticker = await self.client.fetch_ticker(symbol)
            
            return {
                'bid': Decimal(str(ticker['bid'])),
                'ask': Decimal(str(ticker['ask'])),
                'last': Decimal(str(ticker['last'])),
                'volume': Decimal(str(ticker['quoteVolume']))
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            raise
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, List]:
        """Get order book"""
        await self.rate_limiter.acquire()
        
        try:
            order_book = await self.client.fetch_order_book(symbol, limit)
            
            return {
                'bids': [(Decimal(str(price)), Decimal(str(amount))) 
                        for price, amount in order_book['bids']],
                'asks': [(Decimal(str(price)), Decimal(str(amount))) 
                        for price, amount in order_book['asks']],
                'timestamp': order_book['timestamp']
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch order book for {symbol}: {e}")
            raise
    
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades"""
        await self.rate_limiter.acquire()
        
        try:
            trades = await self.client.fetch_trades(symbol, limit=limit)
            
            return [{
                'id': trade['id'],
                'timestamp': trade['timestamp'],
                'price': Decimal(str(trade['price'])),
                'amount': Decimal(str(trade['amount'])),
                'side': trade['side']
            } for trade in trades]
        except Exception as e:
            self.logger.error(f"Failed to fetch trades for {symbol}: {e}")
            raise
    
    async def subscribe_ticker(self, symbol: str, callback: Callable):
        """Subscribe to ticker updates via WebSocket"""
        # Implementation depends on exchange
        pass
    
    async def subscribe_order_book(self, symbol: str, callback: Callable):
        """Subscribe to order book updates"""
        # Implementation depends on exchange
        pass

class BinanceExchange(BaseCryptoExchange):
    """Binance exchange implementation"""
    
    def _initialize(self):
        """Initialize Binance connection"""
        self.client = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future' if not self.testnet else 'testnet'
            }
        })
        
        if self.testnet:
            self.client.set_sandbox_mode(True)
    
    async def get_balance(self) -> Dict[str, Decimal]:
        """Get Binance account balances"""
        try:
            balance = await self.client.fetch_balance()
            
            return {
                asset: Decimal(str(amount['free'] + amount['used']))
                for asset, amount in balance['total'].items()
                if amount > 0
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch balance: {e}")
            raise
    
    async def place_order(self, symbol: str, side: str, order_type: OrderType,
                         amount: Decimal, price: Optional[Decimal] = None) -> CryptoOrder:
        """Place order on Binance"""
        try:
            # Convert order type
            exchange_order_type = self._convert_order_type(order_type)
            
            # Place order
            params = {}
            if price is not None:
                params['price'] = float(price)
            
            result = await self.client.create_order(
                symbol=symbol,
                type=exchange_order_type,
                side=side,
                amount=float(amount),
                **params
            )
            
            # Create order object
            order = CryptoOrder(
                order_id=result['id'],
                exchange=self.exchange_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                amount=amount,
                price=price,
                status=result['status'],
                filled_amount=Decimal(str(result.get('filled', 0))),
                average_price=Decimal(str(result.get('average', 0))) if result.get('average') else None
            )
            
            # Track order
            self.active_orders[order.order_id] = order
            
            return order
            
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            raise
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert internal order type to exchange format"""
        mapping = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP_LOSS: 'stop_loss',
            OrderType.STOP_LIMIT: 'stop_loss_limit'
        }
        return mapping.get(order_type, 'market')
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on Binance"""
        try:
            await self.client.cancel_order(order_id, symbol)
            
            if order_id in self.active_orders:
                self.active_orders[order_id].status = 'cancelled'
                self.active_orders[order_id].updated_at = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> CryptoOrder:
        """Get order status from Binance"""
        try:
            order_info = await self.client.fetch_order(order_id, symbol)
            
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                order.status = order_info['status']
                order.filled_amount = Decimal(str(order_info.get('filled', 0)))
                order.average_price = Decimal(str(order_info.get('average', 0))) if order_info.get('average') else None
                order.updated_at = datetime.now()
                
                # Calculate fees
                if 'fee' in order_info:
                    order.fee_amount = Decimal(str(order_info['fee']['cost']))
                    order.fee_currency = order_info['fee']['currency']
                
                return order
            else:
                # Create new order object
                return CryptoOrder(
                    order_id=order_id,
                    exchange=self.exchange_id,
                    symbol=symbol,
                    side=order_info['side'],
                    order_type=OrderType.LIMIT if order_info['type'] == 'limit' else OrderType.MARKET,
                    amount=Decimal(str(order_info['amount'])),
                    price=Decimal(str(order_info['price'])) if order_info['price'] else None,
                    status=order_info['status'],
                    filled_amount=Decimal(str(order_info.get('filled', 0))),
                    average_price=Decimal(str(order_info.get('average', 0))) if order_info.get('average') else None
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {e}")
            raise

class CoinbaseExchange(BaseCryptoExchange):
    """Coinbase exchange implementation"""
    
    def _initialize(self):
        """Initialize Coinbase connection"""
        self.client = ccxt.coinbasepro({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True
        })
        
        if self.testnet:
            self.client.urls['api'] = 'https://api-public.sandbox.pro.coinbase.com'
    
    async def get_balance(self) -> Dict[str, Decimal]:
        """Get Coinbase account balances"""
        try:
            balance = await self.client.fetch_balance()
            
            return {
                asset: Decimal(str(amount['free'] + amount['used']))
                for asset, amount in balance['total'].items()
                if amount > 0
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch balance: {e}")
            raise
```

### 2. DeFi Protocol Integration

```python
# src/crypto/defi_protocols.py
class DeFiProtocol(ABC):
    """Base class for DeFi protocol integration"""
    
    def __init__(self, network: NetworkType, provider_url: str, 
                 private_key: Optional[str] = None):
        self.network = network
        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        
        # Account setup
        if private_key:
            self.account = Account.from_key(private_key)
            self.address = self.account.address
        else:
            self.account = None
            self.address = None
        
        # Contract instances
        self.contracts = {}
        
        # Gas price oracle
        self.gas_oracle = GasPriceOracle(self.w3)
        
        self.logger = ComponentLogger(f"DeFi_{self.__class__.__name__}", "defi")
    
    @abstractmethod
    async def get_pool_info(self, pool_address: str) -> Dict[str, Any]:
        """Get pool information"""
        pass
    
    @abstractmethod
    async def get_position(self, user_address: str) -> List[DeFiPosition]:
        """Get user positions"""
        pass
    
    @abstractmethod
    async def swap(self, token_in: str, token_out: str, 
                   amount_in: Decimal, min_amount_out: Decimal) -> Dict[str, Any]:
        """Execute token swap"""
        pass
    
    def load_contract(self, address: str, abi: List[Dict]) -> Any:
        """Load smart contract"""
        return self.w3.eth.contract(
            address=Web3.toChecksumAddress(address),
            abi=abi
        )
    
    async def estimate_gas(self, transaction: Dict) -> int:
        """Estimate gas for transaction"""
        try:
            return self.w3.eth.estimate_gas(transaction)
        except Exception as e:
            self.logger.error(f"Gas estimation failed: {e}")
            # Return default high gas limit
            return 500000
    
    async def send_transaction(self, transaction: Dict) -> str:
        """Send transaction to blockchain"""
        if not self.account:
            raise ValueError("No account configured for transactions")
        
        # Add gas price
        gas_price = await self.gas_oracle.get_gas_price()
        transaction['gasPrice'] = gas_price
        
        # Add nonce
        transaction['nonce'] = self.w3.eth.get_transaction_count(self.address)
        
        # Estimate gas if not provided
        if 'gas' not in transaction:
            transaction['gas'] = await self.estimate_gas(transaction)
        
        # Sign transaction
        signed_tx = self.account.sign_transaction(transaction)
        
        # Send transaction
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        self.logger.info(f"Transaction sent: {tx_hash.hex()}")
        
        return tx_hash.hex()
    
    async def wait_for_confirmation(self, tx_hash: str, timeout: int = 120) -> Dict:
        """Wait for transaction confirmation"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                if receipt:
                    if receipt['status'] == 1:
                        self.logger.info(f"Transaction confirmed: {tx_hash}")
                        return receipt
                    else:
                        raise Exception(f"Transaction failed: {tx_hash}")
            except Exception as e:
                if "not found" not in str(e).lower():
                    raise
            
            await asyncio.sleep(2)
        
        raise TimeoutError(f"Transaction not confirmed within {timeout} seconds")

class UniswapV3Protocol(DeFiProtocol):
    """Uniswap V3 protocol integration"""
    
    def __init__(self, network: NetworkType, provider_url: str, 
                 private_key: Optional[str] = None):
        super().__init__(network, provider_url, private_key)
        
        # Load contracts
        self._load_contracts()
    
    def _load_contracts(self):
        """Load Uniswap V3 contracts"""
        # Contract addresses (mainnet)
        addresses = {
            'factory': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
            'router': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            'quoter': '0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6',
            'position_manager': '0xC36442b4a4522E871399CD717aBDD847Ab11FE88'
        }
        
        # Load ABIs
        with open('abi/uniswap_v3_factory.json', 'r') as f:
            factory_abi = json.load(f)
        with open('abi/uniswap_v3_router.json', 'r') as f:
            router_abi = json.load(f)
        
        # Create contract instances
        self.contracts['factory'] = self.load_contract(
            addresses['factory'], factory_abi
        )
        self.contracts['router'] = self.load_contract(
            addresses['router'], router_abi
        )
    
    async def get_pool_info(self, pool_address: str) -> Dict[str, Any]:
        """Get Uniswap V3 pool information"""
        pool_abi = [
            # Minimal pool ABI
            {"inputs":[],"name":"token0","outputs":[{"type":"address"}],"type":"function"},
            {"inputs":[],"name":"token1","outputs":[{"type":"address"}],"type":"function"},
            {"inputs":[],"name":"fee","outputs":[{"type":"uint24"}],"type":"function"},
            {"inputs":[],"name":"liquidity","outputs":[{"type":"uint128"}],"type":"function"},
            {"inputs":[],"name":"slot0","outputs":[
                {"type":"uint160","name":"sqrtPriceX96"},
                {"type":"int24","name":"tick"},
                {"type":"uint16","name":"observationIndex"},
                {"type":"uint16","name":"observationCardinality"},
                {"type":"uint16","name":"observationCardinalityNext"},
                {"type":"uint8","name":"feeProtocol"},
                {"type":"bool","name":"unlocked"}
            ],"type":"function"}
        ]
        
        pool = self.load_contract(pool_address, pool_abi)
        
        # Get pool data
        token0 = pool.functions.token0().call()
        token1 = pool.functions.token1().call()
        fee = pool.functions.fee().call()
        liquidity = pool.functions.liquidity().call()
        slot0 = pool.functions.slot0().call()
        
        # Calculate price from sqrtPriceX96
        sqrt_price_x96 = slot0[0]
        price = (sqrt_price_x96 / (2**96)) ** 2
        
        return {
            'address': pool_address,
            'token0': token0,
            'token1': token1,
            'fee': fee / 10000,  # Convert to percentage
            'liquidity': liquidity,
            'price': price,
            'tick': slot0[1]
        }
    
    async def swap(self, token_in: str, token_out: str,
                   amount_in: Decimal, min_amount_out: Decimal) -> Dict[str, Any]:
        """Execute swap on Uniswap V3"""
        
        # Build swap parameters
        params = {
            'tokenIn': Web3.toChecksumAddress(token_in),
            'tokenOut': Web3.toChecksumAddress(token_out),
            'fee': 3000,  # 0.3% fee tier
            'recipient': self.address,
            'deadline': int(time.time()) + 300,  # 5 minutes
            'amountIn': int(amount_in * 10**18),  # Assuming 18 decimals
            'amountOutMinimum': int(min_amount_out * 10**18),
            'sqrtPriceLimitX96': 0
        }
        
        # Build transaction
        router = self.contracts['router']
        tx = router.functions.exactInputSingle(params).build_transaction({
            'from': self.address,
            'value': 0
        })
        
        # Send transaction
        tx_hash = await self.send_transaction(tx)
        
        # Wait for confirmation
        receipt = await self.wait_for_confirmation(tx_hash)
        
        return {
            'tx_hash': tx_hash,
            'gas_used': receipt['gasUsed'],
            'status': 'success' if receipt['status'] == 1 else 'failed'
        }
    
    async def add_liquidity(self, token0: str, token1: str,
                           amount0: Decimal, amount1: Decimal,
                           tick_lower: int, tick_upper: int) -> Dict[str, Any]:
        """Add liquidity to Uniswap V3 pool"""
        # Implementation for adding liquidity
        pass

class AaveProtocol(DeFiProtocol):
    """Aave lending protocol integration"""
    
    def __init__(self, network: NetworkType, provider_url: str,
                 private_key: Optional[str] = None):
        super().__init__(network, provider_url, private_key)
        self._load_contracts()
    
    def _load_contracts(self):
        """Load Aave contracts"""
        # Aave V3 addresses
        addresses = {
            'pool': '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2',
            'pool_data_provider': '0x7B4EB56E7CD4b454BA8ff71E4518426369a138a3',
            'price_oracle': '0x54586bE62E3c3580375aE3723C145253060Ca0C2'
        }
        
        # Load pool contract
        with open('abi/aave_v3_pool.json', 'r') as f:
            pool_abi = json.load(f)
        
        self.contracts['pool'] = self.load_contract(
            addresses['pool'], pool_abi
        )
    
    async def get_position(self, user_address: str) -> List[DeFiPosition]:
        """Get user's Aave positions"""
        positions = []
        
        # Get user account data
        user_data = self.contracts['pool'].functions.getUserAccountData(
            Web3.toChecksumAddress(user_address)
        ).call()
        
        total_collateral_eth = user_data[0] / 10**18
        total_debt_eth = user_data[1] / 10**18
        available_borrow_eth = user_data[2] / 10**18
        ltv = user_data[3] / 10000
        health_factor = user_data[5] / 10**18
        
        # Create position summary
        if total_collateral_eth > 0:
            positions.append(DeFiPosition(
                protocol="Aave",
                position_type="lending",
                asset=CryptoAsset(symbol="ETH", name="Ethereum", network=self.network),
                amount=Decimal(str(total_collateral_eth)),
                value_usd=Decimal(str(total_collateral_eth * 2000)),  # Placeholder price
                health_factor=health_factor
            ))
        
        if total_debt_eth > 0:
            positions.append(DeFiPosition(
                protocol="Aave",
                position_type="borrowing",
                asset=CryptoAsset(symbol="ETH", name="Ethereum", network=self.network),
                amount=Decimal(str(total_debt_eth)),
                value_usd=Decimal(str(total_debt_eth * 2000)),  # Placeholder price
                health_factor=health_factor
            ))
        
        return positions
    
    async def supply(self, asset: str, amount: Decimal) -> Dict[str, Any]:
        """Supply asset to Aave"""
        # Build supply transaction
        tx = self.contracts['pool'].functions.supply(
            Web3.toChecksumAddress(asset),
            int(amount * 10**18),  # Assuming 18 decimals
            self.address,
            0  # No referral
        ).build_transaction({
            'from': self.address,
            'value': 0
        })
        
        # Send transaction
        tx_hash = await self.send_transaction(tx)
        receipt = await self.wait_for_confirmation(tx_hash)
        
        return {
            'tx_hash': tx_hash,
            'gas_used': receipt['gasUsed'],
            'status': 'success' if receipt['status'] == 1 else 'failed'
        }
```

### 3. Multi-Chain Wallet Management

```python
# src/crypto/wallet_manager.py
class MultiChainWallet:
    """Multi-chain wallet management"""
    
    def __init__(self):
        self.wallets = {}
        self.networks = {}
        self.balances = {}
        self.logger = ComponentLogger("MultiChainWallet", "crypto")
    
    def add_network(self, network: NetworkType, provider_url: str):
        """Add network configuration"""
        self.networks[network] = {
            'provider_url': provider_url,
            'w3': Web3(Web3.HTTPProvider(provider_url))
        }
    
    def import_wallet(self, private_key: str, networks: List[NetworkType]):
        """Import wallet for multiple networks"""
        account = Account.from_key(private_key)
        
        for network in networks:
            if network not in self.networks:
                raise ValueError(f"Network {network} not configured")
            
            self.wallets[network] = {
                'account': account,
                'address': account.address
            }
        
        self.logger.info(f"Wallet imported for networks: {networks}")
    
    async def get_balance(self, network: NetworkType, token_address: Optional[str] = None) -> Decimal:
        """Get balance for specific network and token"""
        if network not in self.wallets:
            raise ValueError(f"No wallet for network {network}")
        
        w3 = self.networks[network]['w3']
        address = self.wallets[network]['address']
        
        if token_address is None:
            # Get native token balance
            balance = w3.eth.get_balance(address)
            return Decimal(str(balance)) / Decimal(10**18)
        else:
            # Get ERC20 token balance
            token_abi = [
                {"inputs":[{"type":"address"}],"name":"balanceOf",
                 "outputs":[{"type":"uint256"}],"type":"function"},
                {"inputs":[],"name":"decimals",
                 "outputs":[{"type":"uint8"}],"type":"function"}
            ]
            
            token = w3.eth.contract(
                address=Web3.toChecksumAddress(token_address),
                abi=token_abi
            )
            
            balance = token.functions.balanceOf(address).call()
            decimals = token.functions.decimals().call()
            
            return Decimal(str(balance)) / Decimal(10**decimals)
    
    async def get_all_balances(self) -> Dict[NetworkType, Dict[str, Decimal]]:
        """Get all balances across networks"""
        all_balances = {}
        
        for network in self.wallets:
            balances = {}
            
            # Get native token balance
            native_balance = await self.get_balance(network)
            native_symbol = self._get_native_symbol(network)
            balances[native_symbol] = native_balance
            
            # Get common token balances
            common_tokens = self._get_common_tokens(network)
            for symbol, address in common_tokens.items():
                try:
                    balance = await self.get_balance(network, address)
                    if balance > 0:
                        balances[symbol] = balance
                except Exception as e:
                    self.logger.error(f"Failed to get {symbol} balance on {network}: {e}")
            
            all_balances[network] = balances
        
        return all_balances
    
    def _get_native_symbol(self, network: NetworkType) -> str:
        """Get native token symbol for network"""
        mapping = {
            NetworkType.ETHEREUM: "ETH",
            NetworkType.BSC: "BNB",
            NetworkType.POLYGON: "MATIC",
            NetworkType.ARBITRUM: "ETH",
            NetworkType.OPTIMISM: "ETH",
            NetworkType.AVALANCHE: "AVAX"
        }
        return mapping.get(network, "UNKNOWN")
    
    def _get_common_tokens(self, network: NetworkType) -> Dict[str, str]:
        """Get common token addresses for network"""
        # This would be loaded from configuration
        if network == NetworkType.ETHEREUM:
            return {
                "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
                "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
            }
        elif network == NetworkType.BSC:
            return {
                "USDC": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d",
                "USDT": "0x55d398326f99059fF775485246999027B3197955",
                "BUSD": "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56"
            }
        return {}

class GasPriceOracle:
    """Gas price oracle for multiple networks"""
    
    def __init__(self, w3: Web3):
        self.w3 = w3
        self.cache = {}
        self.cache_duration = 30  # seconds
    
    async def get_gas_price(self, priority: str = "standard") -> int:
        """Get current gas price"""
        
        # Check cache
        cache_key = f"gas_price_{priority}"
        if cache_key in self.cache:
            cached_time, cached_price = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return cached_price
        
        # Get fresh gas price
        if priority == "fast":
            multiplier = 1.2
        elif priority == "slow":
            multiplier = 0.8
        else:
            multiplier = 1.0
        
        base_price = self.w3.eth.gas_price
        gas_price = int(base_price * multiplier)
        
        # Cache result
        self.cache[cache_key] = (time.time(), gas_price)
        
        return gas_price
```

### 4. Cross-Chain Bridge Integration

```python
# src/crypto/bridges.py
class CrossChainBridge(ABC):
    """Base class for cross-chain bridges"""
    
    def __init__(self, name: str):
        self.name = name
        self.supported_chains = {}
        self.logger = ComponentLogger(f"Bridge_{name}", "crypto")
    
    @abstractmethod
    async def get_supported_pairs(self) -> List[Tuple[NetworkType, NetworkType]]:
        """Get supported chain pairs"""
        pass
    
    @abstractmethod
    async def estimate_bridge_fee(self, from_chain: NetworkType, to_chain: NetworkType,
                                 token: str, amount: Decimal) -> Dict[str, Decimal]:
        """Estimate bridge fees"""
        pass
    
    @abstractmethod
    async def bridge_tokens(self, from_chain: NetworkType, to_chain: NetworkType,
                           token: str, amount: Decimal, recipient: str) -> str:
        """Bridge tokens between chains"""
        pass

class LayerZeroBridge(CrossChainBridge):
    """LayerZero cross-chain bridge"""
    
    def __init__(self, endpoint_addresses: Dict[NetworkType, str]):
        super().__init__("LayerZero")
        self.endpoints = endpoint_addresses
        self._load_contracts()
    
    def _load_contracts(self):
        """Load LayerZero contracts"""
        # Load endpoint contracts for each chain
        pass
    
    async def get_supported_pairs(self) -> List[Tuple[NetworkType, NetworkType]]:
        """Get LayerZero supported chain pairs"""
        # LayerZero supports most EVM chains
        chains = [
            NetworkType.ETHEREUM,
            NetworkType.BSC,
            NetworkType.POLYGON,
            NetworkType.ARBITRUM,
            NetworkType.OPTIMISM,
            NetworkType.AVALANCHE
        ]
        
        pairs = []
        for i, chain1 in enumerate(chains):
            for chain2 in chains[i+1:]:
                pairs.append((chain1, chain2))
                pairs.append((chain2, chain1))
        
        return pairs
    
    async def estimate_bridge_fee(self, from_chain: NetworkType, to_chain: NetworkType,
                                 token: str, amount: Decimal) -> Dict[str, Decimal]:
        """Estimate LayerZero bridge fee"""
        # Get chain IDs
        chain_id_map = {
            NetworkType.ETHEREUM: 101,
            NetworkType.BSC: 102,
            NetworkType.POLYGON: 109,
            NetworkType.ARBITRUM: 110,
            NetworkType.OPTIMISM: 111,
            NetworkType.AVALANCHE: 106
        }
        
        dst_chain_id = chain_id_map[to_chain]
        
        # Estimate fee (simplified)
        base_fee = Decimal("0.001")  # 0.1% base fee
        gas_fee = Decimal("10")  # $10 gas estimate
        
        return {
            'protocol_fee': amount * base_fee,
            'gas_fee': gas_fee,
            'total_fee': amount * base_fee + gas_fee
        }
```

### 5. Crypto Trading Strategy

```python
# src/crypto/strategies.py
class CryptoTradingStrategy:
    """Cryptocurrency trading strategy"""
    
    def __init__(self, exchanges: Dict[str, BaseCryptoExchange],
                 defi_protocols: Dict[str, DeFiProtocol]):
        self.exchanges = exchanges
        self.defi_protocols = defi_protocols
        self.positions = {}
        self.logger = ComponentLogger("CryptoStrategy", "crypto")
    
    async def execute_cex_dex_arbitrage(self, token: str, amount: Decimal):
        """Execute CEX-DEX arbitrage"""
        
        # Get prices from CEX
        cex_ticker = await self.exchanges['binance'].get_ticker(f"{token}/USDT")
        cex_price = cex_ticker['last']
        
        # Get price from DEX (Uniswap)
        pool_info = await self.defi_protocols['uniswap'].get_pool_info(
            self._get_pool_address(token, 'USDT')
        )
        dex_price = Decimal(str(pool_info['price']))
        
        # Calculate price difference
        price_diff = abs(cex_price - dex_price) / cex_price
        
        if price_diff > Decimal("0.01"):  # 1% threshold
            self.logger.info(f"Arbitrage opportunity: {token} CEX: {cex_price} DEX: {dex_price}")
            
            if cex_price < dex_price:
                # Buy on CEX, sell on DEX
                await self._execute_cex_to_dex_arb(token, amount, cex_price, dex_price)
            else:
                # Buy on DEX, sell on CEX
                await self._execute_dex_to_cex_arb(token, amount, cex_price, dex_price)
    
    async def execute_cross_chain_arbitrage(self, token: str, amount: Decimal):
        """Execute cross-chain arbitrage"""
        
        # Get prices on different chains
        prices = {}
        
        for chain, protocol in self.defi_protocols.items():
            try:
                pool_info = await protocol.get_pool_info(
                    self._get_pool_address(token, 'USDT', chain)
                )
                prices[chain] = Decimal(str(pool_info['price']))
            except Exception as e:
                self.logger.error(f"Failed to get price on {chain}: {e}")
        
        # Find arbitrage opportunities
        if len(prices) < 2:
            return
        
        chains = list(prices.keys())
        for i, chain1 in enumerate(chains):
            for chain2 in chains[i+1:]:
                price_diff = abs(prices[chain1] - prices[chain2]) / prices[chain1]
                
                if price_diff > Decimal("0.015"):  # 1.5% threshold
                    self.logger.info(
                        f"Cross-chain arbitrage: {token} "
                        f"{chain1}: {prices[chain1]} {chain2}: {prices[chain2]}"
                    )
                    
                    # Execute arbitrage
                    if prices[chain1] < prices[chain2]:
                        await self._execute_cross_chain_arb(
                            token, amount, chain1, chain2, 
                            prices[chain1], prices[chain2]
                        )
    
    async def yield_farming_optimization(self):
        """Optimize yield farming positions"""
        
        # Get current positions
        positions = []
        for protocol_name, protocol in self.defi_protocols.items():
            user_positions = await protocol.get_position(protocol.address)
            positions.extend(user_positions)
        
        # Calculate APYs
        position_apys = []
        for position in positions:
            if position.apy:
                position_apys.append({
                    'position': position,
                    'apy': position.apy,
                    'risk_adjusted_apy': position.apy / self._calculate_risk_score(position)
                })
        
        # Sort by risk-adjusted APY
        position_apys.sort(key=lambda x: x['risk_adjusted_apy'], reverse=True)
        
        # Rebalance if needed
        if len(position_apys) > 1:
            best_position = position_apys[0]
            worst_position = position_apys[-1]
            
            if best_position['risk_adjusted_apy'] > worst_position['risk_adjusted_apy'] * 1.2:
                # Move funds from worst to best
                await self._rebalance_positions(worst_position['position'], 
                                              best_position['position'])
    
    def _calculate_risk_score(self, position: DeFiPosition) -> float:
        """Calculate risk score for position"""
        risk_score = 1.0
        
        # Protocol risk
        protocol_risk = {
            'Aave': 0.1,
            'Compound': 0.1,
            'Uniswap': 0.2,
            'SushiSwap': 0.3,
            'Unknown': 0.5
        }
        risk_score *= (1 + protocol_risk.get(position.protocol, 0.5))
        
        # Asset risk
        if position.asset.volatility:
            risk_score *= (1 + position.asset.volatility)
        
        # Liquidity risk
        if position.asset.liquidity_score:
            risk_score *= (2 - position.asset.liquidity_score)
        
        return risk_score
```

### 6. Testing Framework

```python
# tests/unit/test_step12_crypto_defi.py
class TestCryptoExchangeIntegration:
    """Test cryptocurrency exchange integration"""
    
    async def test_exchange_connection(self):
        """Test exchange connection"""
        # Create test exchange
        exchange = BinanceExchange(
            exchange_id="binance_test",
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        # Test balance fetch
        try:
            balance = await exchange.get_balance()
            assert isinstance(balance, dict)
        except Exception as e:
            # Expected in test environment
            assert "Invalid API key" in str(e)
    
    async def test_order_placement(self):
        """Test order placement logic"""
        exchange = MockCryptoExchange("test_exchange")
        
        # Place market order
        order = await exchange.place_order(
            symbol="BTC/USDT",
            side="buy",
            order_type=OrderType.MARKET,
            amount=Decimal("0.01")
        )
        
        assert order.order_id is not None
        assert order.symbol == "BTC/USDT"
        assert order.side == "buy"
        assert order.amount == Decimal("0.01")
    
    def test_order_type_conversion(self):
        """Test order type conversion"""
        exchange = BinanceExchange("test", testnet=True)
        
        assert exchange._convert_order_type(OrderType.MARKET) == "market"
        assert exchange._convert_order_type(OrderType.LIMIT) == "limit"
        assert exchange._convert_order_type(OrderType.STOP_LOSS) == "stop_loss"

class TestDeFiProtocols:
    """Test DeFi protocol integration"""
    
    def test_contract_loading(self):
        """Test smart contract loading"""
        protocol = MockDeFiProtocol(
            NetworkType.ETHEREUM,
            "https://eth-mainnet.example.com"
        )
        
        # Test contract loading
        contract_address = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
        abi = [{"inputs":[],"name":"test","outputs":[],"type":"function"}]
        
        contract = protocol.load_contract(contract_address, abi)
        assert contract is not None
    
    async def test_gas_estimation(self):
        """Test gas estimation"""
        protocol = MockDeFiProtocol(
            NetworkType.ETHEREUM,
            "https://eth-mainnet.example.com"
        )
        
        # Mock transaction
        tx = {
            'to': '0x0000000000000000000000000000000000000000',
            'value': 0,
            'data': '0x'
        }
        
        gas_estimate = await protocol.estimate_gas(tx)
        assert gas_estimate > 21000  # Minimum gas
    
    async def test_uniswap_pool_info(self):
        """Test Uniswap pool information retrieval"""
        uniswap = MockUniswapProtocol(
            NetworkType.ETHEREUM,
            "https://eth-mainnet.example.com"
        )
        
        pool_address = "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8"  # USDC/ETH pool
        
        pool_info = await uniswap.get_pool_info(pool_address)
        
        assert 'token0' in pool_info
        assert 'token1' in pool_info
        assert 'fee' in pool_info
        assert 'liquidity' in pool_info
        assert 'price' in pool_info

class TestMultiChainWallet:
    """Test multi-chain wallet management"""
    
    def test_wallet_import(self):
        """Test wallet import"""
        wallet = MultiChainWallet()
        
        # Add networks
        wallet.add_network(NetworkType.ETHEREUM, "https://eth.example.com")
        wallet.add_network(NetworkType.BSC, "https://bsc.example.com")
        
        # Import wallet
        private_key = "0x" + "1" * 64  # Test private key
        wallet.import_wallet(private_key, [NetworkType.ETHEREUM, NetworkType.BSC])
        
        assert NetworkType.ETHEREUM in wallet.wallets
        assert NetworkType.BSC in wallet.wallets
    
    async def test_balance_retrieval(self):
        """Test balance retrieval"""
        wallet = MockMultiChainWallet()
        
        # Get ETH balance
        balance = await wallet.get_balance(NetworkType.ETHEREUM)
        assert isinstance(balance, Decimal)
        assert balance >= 0
        
        # Get token balance
        usdc_address = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
        token_balance = await wallet.get_balance(NetworkType.ETHEREUM, usdc_address)
        assert isinstance(token_balance, Decimal)

class TestCryptoArbitrage:
    """Test crypto arbitrage strategies"""
    
    async def test_cex_dex_arbitrage(self):
        """Test CEX-DEX arbitrage detection"""
        exchanges = {'binance': MockCryptoExchange('binance')}
        defi = {'uniswap': MockUniswapProtocol(NetworkType.ETHEREUM, "")}
        
        strategy = CryptoTradingStrategy(exchanges, defi)
        
        # Set different prices
        exchanges['binance'].set_price("ETH/USDT", Decimal("2000"))
        defi['uniswap'].set_price("ETH", "USDT", Decimal("2050"))
        
        # Should detect arbitrage opportunity
        opportunities = await strategy.detect_arbitrage_opportunities()
        assert len(opportunities) > 0
        assert opportunities[0]['profit_percentage'] > 1
```

### 7. Integration Tests

```python
# tests/integration/test_step12_crypto_integration.py
async def test_complete_crypto_workflow():
    """Test complete crypto trading workflow"""
    
    # Initialize components
    exchange = create_test_exchange()
    defi_protocol = create_test_defi_protocol()
    wallet = create_test_wallet()
    
    # 1. Check balances
    cex_balance = await exchange.get_balance()
    wallet_balance = await wallet.get_balance(NetworkType.ETHEREUM)
    
    assert cex_balance['USDT'] > 0
    assert wallet_balance > 0
    
    # 2. Execute trade on CEX
    order = await exchange.place_order(
        symbol="ETH/USDT",
        side="buy",
        order_type=OrderType.LIMIT,
        amount=Decimal("0.1"),
        price=Decimal("2000")
    )
    
    # 3. Wait for fill
    await asyncio.sleep(1)
    order_status = await exchange.get_order_status(order.order_id, "ETH/USDT")
    
    # 4. Withdraw to wallet
    # (Mock withdrawal process)
    
    # 5. Interact with DeFi
    await defi_protocol.swap(
        token_in="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
        token_out="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
        amount_in=Decimal("0.05"),
        min_amount_out=Decimal("95")  # 5% slippage
    )
    
    # 6. Check final balances
    final_cex_balance = await exchange.get_balance()
    final_wallet_balance = await wallet.get_all_balances()
    
    assert final_wallet_balance[NetworkType.ETHEREUM]['USDC'] > 0

async def test_multi_exchange_aggregation():
    """Test multi-exchange price aggregation"""
    
    exchanges = {
        'binance': create_test_exchange('binance'),
        'coinbase': create_test_exchange('coinbase'),
        'kraken': create_test_exchange('kraken')
    }
    
    # Get best price across exchanges
    symbol = "BTC/USDT"
    best_bid = Decimal("0")
    best_ask = Decimal("999999")
    best_bid_exchange = None
    best_ask_exchange = None
    
    for name, exchange in exchanges.items():
        ticker = await exchange.get_ticker(symbol)
        
        if ticker['bid'] > best_bid:
            best_bid = ticker['bid']
            best_bid_exchange = name
        
        if ticker['ask'] < best_ask:
            best_ask = ticker['ask']
            best_ask_exchange = name
    
    spread = (best_ask - best_bid) / best_bid
    
    assert best_bid_exchange is not None
    assert best_ask_exchange is not None
    assert spread < Decimal("0.01")  # Less than 1% spread
```

### 8. System Tests

```python
# tests/system/test_step12_production_crypto.py
async def test_high_frequency_crypto_trading():
    """Test high-frequency crypto trading capabilities"""
    
    # Setup
    strategy = create_production_crypto_strategy()
    
    # Performance tracking
    start_time = time.time()
    trades_executed = 0
    errors = 0
    
    # Run for 1 minute
    while time.time() - start_time < 60:
        try:
            # Check for opportunities
            opportunities = await strategy.scan_opportunities()
            
            for opp in opportunities[:5]:  # Limit concurrent trades
                if opp['expected_profit'] > 10:  # $10 minimum
                    await strategy.execute_opportunity(opp)
                    trades_executed += 1
            
            await asyncio.sleep(0.1)  # 100ms loop
            
        except Exception as e:
            errors += 1
            logger.error(f"Trading error: {e}")
    
    # Verify performance
    execution_time = time.time() - start_time
    trades_per_second = trades_executed / execution_time
    
    assert trades_per_second > 0.1  # At least 1 trade per 10 seconds
    assert errors < trades_executed * 0.1  # Less than 10% error rate

async def test_defi_protocol_resilience():
    """Test DeFi protocol interaction resilience"""
    
    protocols = {
        'uniswap': create_uniswap_protocol(),
        'aave': create_aave_protocol(),
        'compound': create_compound_protocol()
    }
    
    # Test each protocol
    results = {}
    
    for name, protocol in protocols.items():
        try:
            # Test basic operations
            start = time.time()
            
            # Get pool/market info
            info = await protocol.get_pool_info("USDC-ETH")
            
            # Get user position
            position = await protocol.get_position(TEST_ADDRESS)
            
            # Calculate response time
            response_time = time.time() - start
            
            results[name] = {
                'status': 'success',
                'response_time': response_time,
                'has_liquidity': info.get('liquidity', 0) > 0
            }
            
        except Exception as e:
            results[name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # At least 2/3 protocols should work
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    assert successful >= 2
    
    # Response times should be reasonable
    for name, result in results.items():
        if result['status'] == 'success':
            assert result['response_time'] < 5  # 5 seconds max
```

## ✅ Validation Checklist

### Exchange Integration
- [ ] CEX connections working (Binance, Coinbase, etc.)
- [ ] Order placement and cancellation functional
- [ ] Balance and position tracking accurate
- [ ] WebSocket feeds operational
- [ ] Rate limiting implemented

### DeFi Protocols
- [ ] Smart contract interactions working
- [ ] Gas estimation accurate
- [ ] Transaction signing secure
- [ ] Multiple protocols supported
- [ ] Error handling comprehensive

### Multi-Chain Support
- [ ] Multiple networks configured
- [ ] Wallet management secure
- [ ] Balance tracking across chains
- [ ] Transaction routing optimized
- [ ] Bridge integrations working

### Trading Features
- [ ] CEX-DEX arbitrage detection
- [ ] Cross-chain arbitrage working
- [ ] Yield optimization functional
- [ ] Risk management active
- [ ] Performance tracking accurate

### Security
- [ ] Private keys encrypted
- [ ] Transaction validation working
- [ ] Slippage protection active
- [ ] Approval management secure
- [ ] Audit trail complete

## 📊 Performance Benchmarks

### Exchange Operations
- Order placement: < 100ms
- Balance update: < 50ms
- Ticker update: < 10ms
- Order book depth: 20+ levels

### DeFi Operations
- Gas estimation: < 500ms
- Transaction broadcast: < 1 second
- Confirmation wait: Configurable
- Multi-call batching: Working

### System Performance
- Concurrent exchanges: 10+
- Updates per second: 100+
- Memory usage: < 500MB
- Network efficiency: Optimized

## 🐛 Common Issues

1. **Network Congestion**
   - Implement dynamic gas pricing
   - Use flashbots for MEV protection
   - Queue transactions appropriately
   - Monitor mempool status

2. **Exchange API Limits**
   - Implement proper rate limiting
   - Use WebSocket where available
   - Cache frequently accessed data
   - Distribute requests across time

3. **Smart Contract Failures**
   - Always simulate transactions first
   - Handle revert reasons properly
   - Implement retry mechanisms
   - Monitor contract upgrades

## 🎯 Success Criteria

Step 12 is complete when:
1. ✅ Multiple crypto exchanges integrated
2. ✅ DeFi protocols accessible
3. ✅ Multi-chain operations working
4. ✅ Arbitrage strategies functional
5. ✅ Security measures implemented

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 13: Cross-Exchange Arbitrage](step-13-cross-exchange-arbitrage.md)

## 📚 Additional Resources

- [Cryptocurrency Exchange APIs](../references/crypto-exchange-apis.md)
- [DeFi Protocol Documentation](../references/defi-protocols.md)
- [Web3 Security Best Practices](../references/web3-security.md)
- [MEV Protection Strategies](../references/mev-protection.md)
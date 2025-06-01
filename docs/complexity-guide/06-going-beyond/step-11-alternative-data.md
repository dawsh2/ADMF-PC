# Step 11: Alternative Data Integration

**Status**: Advanced Step  
**Complexity**: Very High  
**Prerequisites**: [Step 10.7: Advanced Visualization](../05-intermediate-complexity/step-10.7-visualization.md) completed  
**Architecture Ref**: [Alternative Data Architecture](../architecture/alternative-data-architecture.md)

## 🎯 Objective

Integrate non-traditional data sources for trading insights:
- Social media sentiment analysis (Twitter, Reddit, StockTwits)
- News sentiment and event extraction
- Satellite imagery for economic indicators
- Web scraping for supply chain data
- SEC filings and corporate communications
- Weather data for commodity trading

## 📋 Required Reading

Before starting:
1. [Alternative Data in Finance](../references/alternative-data-guide.md)
2. [NLP for Financial Markets](../references/financial-nlp.md)
3. [Data Quality and Validation](../references/data-quality.md)

## 🏗️ Implementation Tasks

### 1. Alternative Data Framework

```python
# src/data/alternative/base.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp

class DataSourceType(Enum):
    """Types of alternative data sources"""
    SOCIAL_MEDIA = "social_media"
    NEWS = "news"
    SATELLITE = "satellite"
    WEB_SCRAPING = "web_scraping"
    REGULATORY = "regulatory"
    WEATHER = "weather"
    ECONOMIC = "economic"

class DataQuality(Enum):
    """Data quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"

@dataclass
class AlternativeDataPoint:
    """Single alternative data observation"""
    source: str
    source_type: DataSourceType
    timestamp: datetime
    
    # Data content
    raw_data: Any
    processed_data: Dict[str, Any]
    
    # Metadata
    symbols: List[str]  # Related trading symbols
    confidence: float  # 0-1 confidence score
    quality: DataQuality
    
    # Processing info
    processing_time: float
    version: str
    
    # Optional fields
    location: Optional[Dict[str, float]] = None  # Lat/lon for geo data
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlternativeDataBatch:
    """Batch of alternative data points"""
    data_points: List[AlternativeDataPoint]
    batch_id: str
    start_time: datetime
    end_time: datetime
    
    # Batch statistics
    total_points: int
    quality_distribution: Dict[DataQuality, int]
    symbol_coverage: Dict[str, int]
    
    # Processing metrics
    fetch_duration: float
    process_duration: float
    error_count: int

class AlternativeDataSource(ABC):
    """Base class for alternative data sources"""
    
    def __init__(self, name: str, source_type: DataSourceType):
        self.name = name
        self.source_type = source_type
        self.is_active = False
        
        # Rate limiting
        self.rate_limiter = RateLimiter()
        
        # Data quality
        self.quality_monitor = DataQualityMonitor()
        
        # Caching
        self.cache = DataCache()
        
        # Metrics
        self.fetch_count = 0
        self.error_count = 0
        self.last_fetch_time = None
        
        self.logger = ComponentLogger(f"AltData_{name}", "alternative_data")
    
    @abstractmethod
    async def fetch_data(self, symbols: List[str], 
                        start_time: datetime,
                        end_time: datetime) -> AlternativeDataBatch:
        """Fetch data from source"""
        pass
    
    @abstractmethod
    async def validate_data(self, data_batch: AlternativeDataBatch) -> bool:
        """Validate fetched data"""
        pass
    
    @abstractmethod
    def process_data(self, raw_data: Any) -> Dict[str, Any]:
        """Process raw data into structured format"""
        pass
    
    async def get_data(self, symbols: List[str],
                      start_time: datetime,
                      end_time: datetime,
                      use_cache: bool = True) -> AlternativeDataBatch:
        """Main interface to get data with caching and validation"""
        
        # Check cache first
        cache_key = self._generate_cache_key(symbols, start_time, end_time)
        
        if use_cache and self.cache.has(cache_key):
            self.logger.debug(f"Cache hit for {cache_key}")
            return self.cache.get(cache_key)
        
        # Rate limiting
        await self.rate_limiter.acquire()
        
        try:
            # Fetch data
            data_batch = await self.fetch_data(symbols, start_time, end_time)
            
            # Validate
            if not await self.validate_data(data_batch):
                raise ValueError("Data validation failed")
            
            # Update quality metrics
            self.quality_monitor.update(data_batch)
            
            # Cache valid data
            if use_cache:
                self.cache.set(cache_key, data_batch)
            
            # Update metrics
            self.fetch_count += 1
            self.last_fetch_time = datetime.now()
            
            return data_batch
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Failed to fetch data: {e}")
            raise
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality metrics"""
        return {
            'fetch_count': self.fetch_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.fetch_count, 1),
            'last_fetch': self.last_fetch_time,
            'quality_stats': self.quality_monitor.get_stats()
        }

class RateLimiter:
    """Rate limiting for API calls"""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.call_times = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        async with self.lock:
            now = datetime.now()
            
            # Remove old calls
            cutoff = now - timedelta(minutes=1)
            self.call_times = [t for t in self.call_times if t > cutoff]
            
            # Check if we need to wait
            if len(self.call_times) >= self.calls_per_minute:
                wait_time = (self.call_times[0] + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            # Record this call
            self.call_times.append(now)

class DataQualityMonitor:
    """Monitor alternative data quality"""
    
    def __init__(self):
        self.quality_scores = []
        self.completeness_scores = []
        self.timeliness_scores = []
        self.accuracy_scores = []
    
    def update(self, data_batch: AlternativeDataBatch):
        """Update quality metrics with new batch"""
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(data_batch)
        self.quality_scores.append(quality_score)
        
        # Calculate completeness
        completeness = len(data_batch.data_points) / max(data_batch.total_points, 1)
        self.completeness_scores.append(completeness)
        
        # Calculate timeliness
        fetch_time = data_batch.fetch_duration + data_batch.process_duration
        timeliness = 1.0 / (1.0 + fetch_time / 60)  # Decay over minutes
        self.timeliness_scores.append(timeliness)
    
    def _calculate_quality_score(self, data_batch: AlternativeDataBatch) -> float:
        """Calculate overall quality score"""
        
        # Weight by quality level
        quality_weights = {
            DataQuality.HIGH: 1.0,
            DataQuality.MEDIUM: 0.7,
            DataQuality.LOW: 0.3,
            DataQuality.UNVERIFIED: 0.1
        }
        
        total_weight = 0
        total_score = 0
        
        for quality, count in data_batch.quality_distribution.items():
            weight = quality_weights.get(quality, 0)
            total_weight += count * weight
            total_score += count
        
        return total_weight / max(total_score, 1)
    
    def get_stats(self) -> Dict[str, float]:
        """Get quality statistics"""
        return {
            'avg_quality': np.mean(self.quality_scores) if self.quality_scores else 0,
            'avg_completeness': np.mean(self.completeness_scores) if self.completeness_scores else 0,
            'avg_timeliness': np.mean(self.timeliness_scores) if self.timeliness_scores else 0,
            'quality_trend': self._calculate_trend(self.quality_scores)
        }
    
    def _calculate_trend(self, scores: List[float]) -> float:
        """Calculate trend in scores"""
        if len(scores) < 2:
            return 0
        
        # Simple linear regression
        x = np.arange(len(scores))
        y = np.array(scores)
        
        if len(x) > 0:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        return 0
```

### 2. Social Media Sentiment

```python
# src/data/alternative/social_sentiment.py
import tweepy
import praw  # Reddit API
from textblob import TextBlob
from transformers import pipeline
import re
from collections import Counter

class SocialSentimentAnalyzer(AlternativeDataSource):
    """Analyze sentiment from social media sources"""
    
    def __init__(self, credentials: Dict[str, str]):
        super().__init__("SocialSentiment", DataSourceType.SOCIAL_MEDIA)
        
        # API clients
        self.twitter_client = self._init_twitter(credentials.get('twitter', {}))
        self.reddit_client = self._init_reddit(credentials.get('reddit', {}))
        
        # NLP models
        self.sentiment_pipeline = pipeline("sentiment-analysis", 
                                         model="finiteautomata/bertweet-base-sentiment-analysis")
        
        # Symbol mapping
        self.symbol_mapper = SymbolMapper()
        
        # Metrics
        self.sentiment_cache = {}
    
    def _init_twitter(self, creds: Dict) -> Optional[tweepy.Client]:
        """Initialize Twitter client"""
        if not creds:
            return None
        
        return tweepy.Client(
            bearer_token=creds.get('bearer_token'),
            consumer_key=creds.get('consumer_key'),
            consumer_secret=creds.get('consumer_secret'),
            access_token=creds.get('access_token'),
            access_token_secret=creds.get('access_token_secret')
        )
    
    def _init_reddit(self, creds: Dict) -> Optional[praw.Reddit]:
        """Initialize Reddit client"""
        if not creds:
            return None
        
        return praw.Reddit(
            client_id=creds.get('client_id'),
            client_secret=creds.get('client_secret'),
            user_agent=creds.get('user_agent', 'TradingBot/1.0')
        )
    
    async def fetch_data(self, symbols: List[str],
                        start_time: datetime,
                        end_time: datetime) -> AlternativeDataBatch:
        """Fetch social media data"""
        
        data_points = []
        fetch_start = datetime.now()
        
        # Fetch from multiple sources concurrently
        tasks = []
        
        if self.twitter_client:
            tasks.append(self._fetch_twitter_data(symbols, start_time, end_time))
        
        if self.reddit_client:
            tasks.append(self._fetch_reddit_data(symbols, start_time, end_time))
        
        # StockTwits
        tasks.append(self._fetch_stocktwits_data(symbols, start_time, end_time))
        
        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                data_points.extend(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Fetch error: {result}")
        
        # Calculate statistics
        quality_dist = Counter(dp.quality for dp in data_points)
        symbol_coverage = Counter()
        for dp in data_points:
            for symbol in dp.symbols:
                symbol_coverage[symbol] += 1
        
        fetch_duration = (datetime.now() - fetch_start).total_seconds()
        
        return AlternativeDataBatch(
            data_points=data_points,
            batch_id=f"social_{datetime.now().timestamp()}",
            start_time=start_time,
            end_time=end_time,
            total_points=len(data_points),
            quality_distribution=dict(quality_dist),
            symbol_coverage=dict(symbol_coverage),
            fetch_duration=fetch_duration,
            process_duration=0,  # Will be updated
            error_count=len([r for r in results if isinstance(r, Exception)])
        )
    
    async def _fetch_twitter_data(self, symbols: List[str],
                                 start_time: datetime,
                                 end_time: datetime) -> List[AlternativeDataPoint]:
        """Fetch Twitter data"""
        data_points = []
        
        # Build query
        cashtags = [f"${symbol}" for symbol in symbols]
        query = " OR ".join(cashtags) + " -is:retweet"
        
        try:
            # Search tweets
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                start_time=start_time,
                end_time=end_time,
                max_results=100,
                tweet_fields=['created_at', 'author_id', 'public_metrics']
            )
            
            if tweets.data:
                for tweet in tweets.data:
                    # Extract symbols mentioned
                    mentioned_symbols = self._extract_symbols(tweet.text)
                    
                    # Analyze sentiment
                    sentiment = self._analyze_sentiment(tweet.text)
                    
                    # Create data point
                    dp = AlternativeDataPoint(
                        source="twitter",
                        source_type=DataSourceType.SOCIAL_MEDIA,
                        timestamp=tweet.created_at,
                        raw_data=tweet.data,
                        processed_data={
                            'text': tweet.text,
                            'sentiment': sentiment['sentiment'],
                            'sentiment_score': sentiment['score'],
                            'engagement': tweet.public_metrics['like_count'] + 
                                        tweet.public_metrics['retweet_count'],
                            'author_id': tweet.author_id
                        },
                        symbols=mentioned_symbols,
                        confidence=sentiment['confidence'],
                        quality=self._assess_tweet_quality(tweet),
                        processing_time=0.01,
                        version="1.0",
                        language="en"
                    )
                    
                    data_points.append(dp)
        
        except Exception as e:
            self.logger.error(f"Twitter fetch error: {e}")
        
        return data_points
    
    async def _fetch_reddit_data(self, symbols: List[str],
                                start_time: datetime,
                                end_time: datetime) -> List[AlternativeDataPoint]:
        """Fetch Reddit data"""
        data_points = []
        
        # Subreddits to monitor
        subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
        
        try:
            for subreddit_name in subreddits:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                
                # Get recent posts
                for post in subreddit.new(limit=100):
                    post_time = datetime.fromtimestamp(post.created_utc)
                    
                    if start_time <= post_time <= end_time:
                        # Check if any symbols mentioned
                        mentioned_symbols = self._extract_symbols(post.title + " " + post.selftext)
                        
                        if any(symbol in symbols for symbol in mentioned_symbols):
                            # Analyze sentiment
                            sentiment = self._analyze_sentiment(post.title + " " + post.selftext[:500])
                            
                            dp = AlternativeDataPoint(
                                source=f"reddit_{subreddit_name}",
                                source_type=DataSourceType.SOCIAL_MEDIA,
                                timestamp=post_time,
                                raw_data={
                                    'title': post.title,
                                    'text': post.selftext[:1000],
                                    'score': post.score,
                                    'num_comments': post.num_comments
                                },
                                processed_data={
                                    'sentiment': sentiment['sentiment'],
                                    'sentiment_score': sentiment['score'],
                                    'engagement_score': post.score + post.num_comments * 2,
                                    'subreddit': subreddit_name
                                },
                                symbols=mentioned_symbols,
                                confidence=sentiment['confidence'],
                                quality=self._assess_reddit_quality(post),
                                processing_time=0.02,
                                version="1.0",
                                language="en"
                            )
                            
                            data_points.append(dp)
        
        except Exception as e:
            self.logger.error(f"Reddit fetch error: {e}")
        
        return data_points
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        try:
            # Clean text
            clean_text = self._clean_text(text)
            
            # Use transformer model
            result = self.sentiment_pipeline(clean_text[:512])[0]  # Max 512 tokens
            
            # Map to standard format
            sentiment_map = {
                'POSITIVE': 'bullish',
                'NEGATIVE': 'bearish',
                'NEUTRAL': 'neutral'
            }
            
            return {
                'sentiment': sentiment_map.get(result['label'], 'neutral'),
                'score': result['score'],
                'confidence': result['score']
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            
            # Fallback to TextBlob
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    sentiment = 'bullish'
                elif polarity < -0.1:
                    sentiment = 'bearish'
                else:
                    sentiment = 'neutral'
                
                return {
                    'sentiment': sentiment,
                    'score': abs(polarity),
                    'confidence': min(abs(polarity) * 2, 1.0)
                }
            except:
                return {
                    'sentiment': 'neutral',
                    'score': 0.5,
                    'confidence': 0.1
                }
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        symbols = []
        
        # Cashtags ($AAPL)
        cashtags = re.findall(r'\$([A-Z]{1,5})\b', text)
        symbols.extend(cashtags)
        
        # Common patterns
        patterns = [
            r'\b([A-Z]{1,5})\s+(?:calls?|puts?|stock|shares?)\b',
            r'\b(?:buy|sell|long|short)\s+([A-Z]{1,5})\b',
            r'\b([A-Z]{1,5})\s+(?:to the moon|rocket|diamond hands)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            symbols.extend([m.upper() for m in matches])
        
        # Validate symbols
        valid_symbols = []
        for symbol in set(symbols):
            if self.symbol_mapper.is_valid_symbol(symbol):
                valid_symbols.append(symbol)
        
        return valid_symbols
    
    def _assess_tweet_quality(self, tweet) -> DataQuality:
        """Assess quality of tweet data"""
        
        # Verified account
        if hasattr(tweet, 'author') and tweet.author.verified:
            return DataQuality.HIGH
        
        # High engagement
        metrics = tweet.public_metrics
        if metrics['like_count'] > 100 or metrics['retweet_count'] > 50:
            return DataQuality.MEDIUM
        
        # Low engagement
        if metrics['like_count'] < 10:
            return DataQuality.LOW
        
        return DataQuality.MEDIUM
    
    def process_data(self, raw_data: Any) -> Dict[str, Any]:
        """Process social sentiment data"""
        # Already processed in fetch methods
        return raw_data

class SymbolMapper:
    """Map text mentions to valid trading symbols"""
    
    def __init__(self):
        # Load symbol database
        self.valid_symbols = self._load_symbol_database()
        self.company_names = self._load_company_names()
    
    def is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid"""
        return symbol.upper() in self.valid_symbols
    
    def _load_symbol_database(self) -> set:
        """Load valid trading symbols"""
        # In production, load from database or API
        # For now, common symbols
        return {
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX',
            'SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV', 'TLT',
            'BTC', 'ETH', 'DOGE', 'GME', 'AMC', 'BB', 'NOK'
        }
```

### 3. News Sentiment Analysis

```python
# src/data/alternative/news_sentiment.py
import feedparser
from newsapi import NewsApiClient
from newspaper import Article
import yfinance as yf

class NewsSentimentAnalyzer(AlternativeDataSource):
    """Analyze sentiment from news sources"""
    
    def __init__(self, api_keys: Dict[str, str]):
        super().__init__("NewsSentiment", DataSourceType.NEWS)
        
        # News APIs
        self.newsapi = NewsApiClient(api_key=api_keys.get('newsapi'))
        self.alpha_vantage_key = api_keys.get('alpha_vantage')
        
        # RSS feeds
        self.rss_feeds = {
            'reuters': 'https://www.reutersagency.com/feed/',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
            'wsj': 'https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml',
            'cnbc': 'https://www.cnbc.com/id/10001147/device/rss/rss.html'
        }
        
        # Event extraction
        self.event_extractor = FinancialEventExtractor()
        
        # Entity recognition
        self.entity_recognizer = FinancialEntityRecognizer()
    
    async def fetch_data(self, symbols: List[str],
                        start_time: datetime,
                        end_time: datetime) -> AlternativeDataBatch:
        """Fetch news data from multiple sources"""
        
        data_points = []
        fetch_start = datetime.now()
        
        # Fetch from different sources
        tasks = [
            self._fetch_newsapi_data(symbols, start_time, end_time),
            self._fetch_rss_data(symbols, start_time, end_time),
            self._fetch_financial_news(symbols, start_time, end_time)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                data_points.extend(result)
        
        # Process and enrich data
        data_points = await self._enrich_news_data(data_points)
        
        # Create batch
        return self._create_data_batch(data_points, start_time, end_time, fetch_start)
    
    async def _fetch_newsapi_data(self, symbols: List[str],
                                 start_time: datetime,
                                 end_time: datetime) -> List[AlternativeDataPoint]:
        """Fetch from NewsAPI"""
        data_points = []
        
        try:
            # Build query
            query = ' OR '.join([f'"{symbol}"' for symbol in symbols])
            
            # Fetch articles
            articles = self.newsapi.get_everything(
                q=query,
                from_param=start_time.isoformat(),
                to=end_time.isoformat(),
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            
            for article in articles.get('articles', []):
                # Extract full text
                full_text = await self._extract_article_text(article['url'])
                
                # Analyze
                sentiment = self._analyze_sentiment(
                    article['title'] + ' ' + (article['description'] or '') + ' ' + full_text
                )
                
                # Extract entities and events
                entities = self.entity_recognizer.extract(full_text)
                events = self.event_extractor.extract(full_text)
                
                # Find mentioned symbols
                mentioned_symbols = self._find_mentioned_symbols(
                    full_text, symbols, entities
                )
                
                dp = AlternativeDataPoint(
                    source=article['source']['name'],
                    source_type=DataSourceType.NEWS,
                    timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                    raw_data=article,
                    processed_data={
                        'title': article['title'],
                        'sentiment': sentiment['sentiment'],
                        'sentiment_score': sentiment['score'],
                        'entities': entities,
                        'events': events,
                        'source_credibility': self._assess_source_credibility(article['source']['name'])
                    },
                    symbols=mentioned_symbols,
                    confidence=sentiment['confidence'] * self._assess_source_credibility(article['source']['name']),
                    quality=self._assess_news_quality(article),
                    processing_time=0.1,
                    version="1.0",
                    language="en"
                )
                
                data_points.append(dp)
        
        except Exception as e:
            self.logger.error(f"NewsAPI fetch error: {e}")
        
        return data_points
    
    async def _extract_article_text(self, url: str) -> str:
        """Extract full article text"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text[:2000]  # Limit length
        except:
            return ""
    
    def _assess_source_credibility(self, source_name: str) -> float:
        """Assess news source credibility"""
        
        credibility_scores = {
            'Reuters': 0.95,
            'Bloomberg': 0.95,
            'Wall Street Journal': 0.93,
            'Financial Times': 0.93,
            'CNBC': 0.85,
            'MarketWatch': 0.80,
            'Yahoo Finance': 0.75,
            'Seeking Alpha': 0.70,
            'Benzinga': 0.65
        }
        
        # Check if source matches known sources
        for known_source, score in credibility_scores.items():
            if known_source.lower() in source_name.lower():
                return score
        
        return 0.5  # Default for unknown sources

class FinancialEventExtractor:
    """Extract financial events from text"""
    
    def __init__(self):
        self.event_patterns = {
            'earnings': [
                r'earnings (?:report|call|announcement)',
                r'Q[1-4] (?:earnings|results)',
                r'(?:beat|miss) (?:earnings|EPS) estimates'
            ],
            'merger': [
                r'(?:merger|acquisition|acquire|takeover)',
                r'(?:buy|purchase) .*? for \$?\d+\.?\d*[BMK]?',
                r'deal (?:worth|valued) at'
            ],
            'guidance': [
                r'(?:raises|lowers|maintains) (?:guidance|outlook)',
                r'(?:revenue|earnings) (?:forecast|projection)',
                r'(?:upgrade|downgrade) (?:rating|price target)'
            ],
            'product': [
                r'(?:launch|announce|unveil|introduce) (?:new|latest)',
                r'product (?:launch|announcement|release)'
            ],
            'regulatory': [
                r'(?:SEC|FDA|FTC|DOJ) (?:investigation|approval|filing)',
                r'regulatory (?:approval|clearance|review)'
            ]
        }
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial events from text"""
        events = []
        
        for event_type, patterns in self.event_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    events.append({
                        'type': event_type,
                        'text': match.group(0),
                        'position': match.span(),
                        'confidence': 0.8
                    })
        
        return events
```

### 4. Satellite Data Integration

```python
# src/data/alternative/satellite_data.py
import rasterio
import geopandas as gpd
from sentinel2 import Sentinel2API
import cv2

class SatelliteDataAnalyzer(AlternativeDataSource):
    """Analyze satellite imagery for economic indicators"""
    
    def __init__(self, credentials: Dict[str, str]):
        super().__init__("SatelliteData", DataSourceType.SATELLITE)
        
        # Satellite data APIs
        self.sentinel_api = Sentinel2API(credentials.get('sentinel'))
        self.planet_api = credentials.get('planet_api_key')
        
        # Analysis models
        self.parking_lot_analyzer = ParkingLotAnalyzer()
        self.ship_tracker = ShipTracker()
        self.crop_analyzer = CropYieldAnalyzer()
        self.construction_monitor = ConstructionMonitor()
    
    async def fetch_data(self, symbols: List[str],
                        start_time: datetime,
                        end_time: datetime) -> AlternativeDataBatch:
        """Fetch satellite data for economic analysis"""
        
        data_points = []
        
        # Map symbols to locations of interest
        locations = self._map_symbols_to_locations(symbols)
        
        for symbol, location_data in locations.items():
            for location in location_data:
                # Fetch imagery
                imagery = await self._fetch_satellite_imagery(
                    location, start_time, end_time
                )
                
                # Analyze based on location type
                if location['type'] == 'retail':
                    analysis = self.parking_lot_analyzer.analyze(imagery)
                elif location['type'] == 'port':
                    analysis = self.ship_tracker.analyze(imagery)
                elif location['type'] == 'agriculture':
                    analysis = self.crop_analyzer.analyze(imagery)
                elif location['type'] == 'construction':
                    analysis = self.construction_monitor.analyze(imagery)
                else:
                    continue
                
                dp = AlternativeDataPoint(
                    source="satellite",
                    source_type=DataSourceType.SATELLITE,
                    timestamp=imagery['capture_time'],
                    raw_data=imagery['metadata'],
                    processed_data={
                        'location_type': location['type'],
                        'analysis': analysis,
                        'location_name': location['name'],
                        'coordinates': location['coordinates']
                    },
                    symbols=[symbol],
                    confidence=analysis.get('confidence', 0.7),
                    quality=self._assess_imagery_quality(imagery),
                    processing_time=analysis.get('processing_time', 1.0),
                    version="1.0",
                    location=location['coordinates']
                )
                
                data_points.append(dp)
        
        return self._create_data_batch(data_points, start_time, end_time)
    
    def _map_symbols_to_locations(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        """Map trading symbols to physical locations"""
        
        # In production, this would use a comprehensive database
        location_mapping = {
            'WMT': [
                {'name': 'Walmart HQ', 'coordinates': {'lat': 36.3729, 'lon': -94.2088}, 'type': 'retail'},
                {'name': 'Walmart DC', 'coordinates': {'lat': 35.3859, 'lon': -94.3985}, 'type': 'retail'}
            ],
            'AMZN': [
                {'name': 'Amazon SEA', 'coordinates': {'lat': 47.6062, 'lon': -122.3321}, 'type': 'retail'},
                {'name': 'Amazon Fulfillment', 'coordinates': {'lat': 39.8283, 'lon': -86.2639}, 'type': 'retail'}
            ],
            'XOM': [
                {'name': 'Exxon Refinery', 'coordinates': {'lat': 29.7604, 'lon': -95.3698}, 'type': 'industrial'}
            ],
            'BA': [
                {'name': 'Boeing Factory', 'coordinates': {'lat': 47.9445, 'lon': -122.3045}, 'type': 'industrial'}
            ]
        }
        
        return {symbol: location_mapping.get(symbol, []) for symbol in symbols}

class ParkingLotAnalyzer:
    """Analyze parking lot occupancy from satellite images"""
    
    def __init__(self):
        # Load pre-trained car detection model
        self.car_detector = self._load_car_detection_model()
    
    def analyze(self, imagery: Dict) -> Dict[str, Any]:
        """Analyze parking lot occupancy"""
        
        start_time = datetime.now()
        
        # Load image
        image = cv2.imread(imagery['file_path'])
        
        # Detect cars
        cars = self.car_detector.detect(image)
        
        # Calculate occupancy
        parking_spaces = self._detect_parking_spaces(image)
        occupancy_rate = len(cars) / max(len(parking_spaces), 1)
        
        # Historical comparison
        historical_avg = self._get_historical_average(
            imagery['location'], imagery['capture_time']
        )
        
        relative_activity = occupancy_rate / max(historical_avg, 0.01)
        
        return {
            'car_count': len(cars),
            'parking_spaces': len(parking_spaces),
            'occupancy_rate': occupancy_rate,
            'historical_average': historical_avg,
            'relative_activity': relative_activity,
            'confidence': self._calculate_confidence(image),
            'processing_time': (datetime.now() - start_time).total_seconds()
        }
```

### 5. Alternative Data Fusion

```python
# src/data/alternative/data_fusion.py
class AlternativeDataFusion:
    """Fuse multiple alternative data sources into trading signals"""
    
    def __init__(self):
        self.fusion_models = {}
        self.signal_generator = AlternativeSignalGenerator()
        self.logger = ComponentLogger("DataFusion", "alternative_data")
    
    def fuse_data(self, data_batches: Dict[str, AlternativeDataBatch],
                  symbols: List[str]) -> Dict[str, Signal]:
        """Fuse multiple data sources into trading signals"""
        
        signals = {}
        
        for symbol in symbols:
            # Collect all data points for symbol
            symbol_data = self._collect_symbol_data(data_batches, symbol)
            
            if not symbol_data:
                continue
            
            # Sentiment fusion
            sentiment_signal = self._fuse_sentiment(symbol_data)
            
            # Event-based signals
            event_signal = self._process_events(symbol_data)
            
            # Satellite-based signals
            satellite_signal = self._process_satellite_data(symbol_data)
            
            # Combine into final signal
            combined_signal = self._combine_signals(
                sentiment_signal, event_signal, satellite_signal
            )
            
            signals[symbol] = combined_signal
        
        return signals
    
    def _fuse_sentiment(self, symbol_data: List[AlternativeDataPoint]) -> Signal:
        """Fuse sentiment from multiple sources"""
        
        sentiments = []
        weights = []
        
        for dp in symbol_data:
            if 'sentiment' in dp.processed_data:
                sentiment_score = dp.processed_data.get('sentiment_score', 0)
                
                # Map sentiment to directional score
                if dp.processed_data['sentiment'] == 'bullish':
                    directional_score = sentiment_score
                elif dp.processed_data['sentiment'] == 'bearish':
                    directional_score = -sentiment_score
                else:
                    directional_score = 0
                
                sentiments.append(directional_score)
                
                # Weight by confidence and data quality
                weight = dp.confidence * self._quality_to_weight(dp.quality)
                weights.append(weight)
        
        if not sentiments:
            return Signal(
                direction=SignalDirection.FLAT,
                strength=0,
                metadata={'source': 'alternative_sentiment'}
            )
        
        # Weighted average sentiment
        weighted_sentiment = np.average(sentiments, weights=weights)
        
        # Convert to signal
        if weighted_sentiment > 0.1:
            direction = SignalDirection.LONG
        elif weighted_sentiment < -0.1:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.FLAT
        
        return Signal(
            direction=direction,
            strength=min(abs(weighted_sentiment), 1.0),
            metadata={
                'source': 'alternative_sentiment',
                'sentiment_score': weighted_sentiment,
                'data_points': len(sentiments)
            }
        )
    
    def _quality_to_weight(self, quality: DataQuality) -> float:
        """Convert data quality to weight"""
        return {
            DataQuality.HIGH: 1.0,
            DataQuality.MEDIUM: 0.7,
            DataQuality.LOW: 0.3,
            DataQuality.UNVERIFIED: 0.1
        }.get(quality, 0.5)

class AlternativeDataManager:
    """Manage all alternative data sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sources = {}
        self.fusion = AlternativeDataFusion()
        self.logger = ComponentLogger("AltDataManager", "alternative_data")
        
        # Initialize sources
        self._initialize_sources()
    
    def _initialize_sources(self):
        """Initialize all configured data sources"""
        
        # Social sentiment
        if self.config.get('social_sentiment', {}).get('enabled'):
            self.sources['social'] = SocialSentimentAnalyzer(
                self.config['social_sentiment']['credentials']
            )
        
        # News sentiment
        if self.config.get('news_sentiment', {}).get('enabled'):
            self.sources['news'] = NewsSentimentAnalyzer(
                self.config['news_sentiment']['api_keys']
            )
        
        # Satellite data
        if self.config.get('satellite_data', {}).get('enabled'):
            self.sources['satellite'] = SatelliteDataAnalyzer(
                self.config['satellite_data']['credentials']
            )
    
    async def get_alternative_signals(self, symbols: List[str],
                                    lookback_hours: int = 24) -> Dict[str, Signal]:
        """Get alternative data signals for symbols"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        # Fetch from all sources
        data_batches = {}
        
        for name, source in self.sources.items():
            try:
                batch = await source.get_data(symbols, start_time, end_time)
                data_batches[name] = batch
            except Exception as e:
                self.logger.error(f"Failed to fetch from {name}: {e}")
        
        # Fuse data into signals
        signals = self.fusion.fuse_data(data_batches, symbols)
        
        return signals
```

## 🧪 Testing Requirements

### Unit Tests

Create `tests/unit/test_step11_alternative_data.py`:

```python
class TestAlternativeDataFramework:
    """Test alternative data base framework"""
    
    async def test_rate_limiter(self):
        """Test API rate limiting"""
        limiter = RateLimiter(calls_per_minute=10)
        
        # Should allow 10 calls quickly
        start_time = datetime.now()
        for i in range(10):
            await limiter.acquire()
        duration = (datetime.now() - start_time).total_seconds()
        
        assert duration < 1  # Should be fast
        
        # 11th call should wait
        start_time = datetime.now()
        await limiter.acquire()
        duration = (datetime.now() - start_time).total_seconds()
        
        assert duration > 50  # Should wait ~60 seconds
    
    def test_data_quality_monitor(self):
        """Test data quality monitoring"""
        monitor = DataQualityMonitor()
        
        # Create test batch
        batch = AlternativeDataBatch(
            data_points=[],
            batch_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            total_points=100,
            quality_distribution={
                DataQuality.HIGH: 50,
                DataQuality.MEDIUM: 30,
                DataQuality.LOW: 20
            },
            symbol_coverage={},
            fetch_duration=1.0,
            process_duration=0.5,
            error_count=0
        )
        
        monitor.update(batch)
        stats = monitor.get_stats()
        
        assert stats['avg_quality'] > 0.5  # Should be decent quality
        assert stats['avg_timeliness'] > 0.9  # Fast processing

class TestSocialSentiment:
    """Test social sentiment analysis"""
    
    def test_sentiment_analysis(self):
        """Test sentiment extraction"""
        analyzer = SocialSentimentAnalyzer({})
        
        # Test various texts
        bullish_text = "AAPL to the moon! 🚀 Strong buy signal!"
        result = analyzer._analyze_sentiment(bullish_text)
        assert result['sentiment'] == 'bullish'
        assert result['score'] > 0.5
        
        bearish_text = "Selling all my TSLA, company is doomed"
        result = analyzer._analyze_sentiment(bearish_text)
        assert result['sentiment'] == 'bearish'
        assert result['score'] > 0.5
    
    def test_symbol_extraction(self):
        """Test symbol extraction from text"""
        analyzer = SocialSentimentAnalyzer({})
        
        text = "Buying $AAPL calls and shorting $TSLA. Also long on SPY."
        symbols = analyzer._extract_symbols(text)
        
        assert 'AAPL' in symbols
        assert 'TSLA' in symbols
        assert 'SPY' in symbols

class TestDataFusion:
    """Test alternative data fusion"""
    
    def test_sentiment_fusion(self):
        """Test fusing sentiment from multiple sources"""
        fusion = AlternativeDataFusion()
        
        # Create test data points
        data_points = [
            AlternativeDataPoint(
                source="twitter",
                source_type=DataSourceType.SOCIAL_MEDIA,
                timestamp=datetime.now(),
                raw_data={},
                processed_data={
                    'sentiment': 'bullish',
                    'sentiment_score': 0.8
                },
                symbols=['AAPL'],
                confidence=0.9,
                quality=DataQuality.HIGH,
                processing_time=0.01,
                version="1.0"
            ),
            AlternativeDataPoint(
                source="reddit",
                source_type=DataSourceType.SOCIAL_MEDIA,
                timestamp=datetime.now(),
                raw_data={},
                processed_data={
                    'sentiment': 'bearish',
                    'sentiment_score': 0.6
                },
                symbols=['AAPL'],
                confidence=0.7,
                quality=DataQuality.MEDIUM,
                processing_time=0.01,
                version="1.0"
            )
        ]
        
        signal = fusion._fuse_sentiment(data_points)
        
        # Should be slightly bullish (weighted by confidence/quality)
        assert signal.direction == SignalDirection.LONG
        assert signal.strength > 0
```

### Integration Tests

Create `tests/integration/test_step11_alternative_integration.py`:

```python
async def test_complete_alternative_data_flow():
    """Test full alternative data pipeline"""
    
    # Setup manager
    config = {
        'social_sentiment': {
            'enabled': True,
            'credentials': {
                'twitter': {'bearer_token': 'test'},
                'reddit': {'client_id': 'test'}
            }
        },
        'news_sentiment': {
            'enabled': True,
            'api_keys': {'newsapi': 'test'}
        }
    }
    
    manager = AlternativeDataManager(config)
    
    # Mock data sources
    manager.sources['social'] = MockSocialSentiment()
    manager.sources['news'] = MockNewsSentiment()
    
    # Get signals
    symbols = ['AAPL', 'TSLA', 'GOOGL']
    signals = await manager.get_alternative_signals(symbols, lookback_hours=1)
    
    # Verify signals generated
    assert len(signals) > 0
    assert all(isinstance(s, Signal) for s in signals.values())
    
    # Check signal metadata
    for symbol, signal in signals.items():
        assert 'source' in signal.metadata
        assert signal.strength >= 0 and signal.strength <= 1

async def test_data_quality_degradation():
    """Test handling of degraded data quality"""
    
    analyzer = SocialSentimentAnalyzer({})
    
    # Simulate degraded API responses
    analyzer.twitter_client = MockDegradedTwitterClient()
    
    batch = await analyzer.fetch_data(['AAPL'], 
                                     datetime.now() - timedelta(hours=1),
                                     datetime.now())
    
    # Should still return data but with lower quality
    assert batch.total_points > 0
    assert batch.quality_distribution[DataQuality.LOW] > 0
    assert batch.error_count > 0
```

### System Tests

Create `tests/system/test_step11_production_alternative.py`:

```python
async def test_high_volume_alternative_data():
    """Test alternative data system under high load"""
    
    # Create manager with multiple sources
    manager = create_production_alt_data_manager()
    
    # Test with many symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 
               'NVDA', 'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'PYPL', 'SQ']
    
    # Performance tracking
    start_time = time.time()
    memory_start = get_memory_usage_mb()
    
    # Fetch data multiple times
    for i in range(10):
        signals = await manager.get_alternative_signals(symbols)
        
        # Verify all symbols processed
        assert len(signals) == len(symbols)
    
    # Performance checks
    total_time = time.time() - start_time
    memory_used = get_memory_usage_mb() - memory_start
    
    assert total_time < 300  # 5 minutes for 10 iterations
    assert memory_used < 1000  # Less than 1GB memory growth
    
    # Quality checks
    quality_stats = {}
    for source in manager.sources.values():
        quality_stats[source.name] = source.get_quality_metrics()
    
    # All sources should maintain quality
    for source_name, stats in quality_stats.items():
        assert stats['error_rate'] < 0.1  # Less than 10% errors
        assert stats['quality_stats']['avg_quality'] > 0.5

async def test_real_time_signal_generation():
    """Test real-time alternative data signal generation"""
    
    manager = AlternativeDataManager(load_production_config())
    
    # Simulate real-time updates
    symbols = ['SPY', 'QQQ', 'IWM']
    signal_history = []
    
    for minute in range(60):  # One hour simulation
        # Get latest signals
        signals = await manager.get_alternative_signals(
            symbols, 
            lookback_hours=0.5  # 30 minutes
        )
        
        signal_history.append({
            'timestamp': datetime.now(),
            'signals': signals
        })
        
        # Wait for next update
        await asyncio.sleep(60)  # 1 minute
    
    # Analyze signal stability
    signal_changes = 0
    for i in range(1, len(signal_history)):
        prev_signals = signal_history[i-1]['signals']
        curr_signals = signal_history[i]['signals']
        
        for symbol in symbols:
            if symbol in prev_signals and symbol in curr_signals:
                if prev_signals[symbol].direction != curr_signals[symbol].direction:
                    signal_changes += 1
    
    # Signals should be relatively stable
    assert signal_changes < 10  # Less than 10 direction changes per hour
```

## ✅ Validation Checklist

### Core Framework
- [ ] Alternative data source abstraction working
- [ ] Rate limiting functional
- [ ] Data quality monitoring active
- [ ] Caching system operational
- [ ] Error handling comprehensive

### Social Sentiment
- [ ] Twitter integration working
- [ ] Reddit data fetching functional
- [ ] Sentiment analysis accurate
- [ ] Symbol extraction reliable
- [ ] Quality assessment working

### News Analysis
- [ ] Multiple news sources integrated
- [ ] Article extraction working
- [ ] Event detection functional
- [ ] Entity recognition accurate
- [ ] Source credibility scoring

### Satellite Data
- [ ] Location mapping working
- [ ] Image analysis functional
- [ ] Economic indicators extracted
- [ ] Quality assessment accurate
- [ ] Processing optimized

### Data Fusion
- [ ] Multi-source fusion working
- [ ] Signal generation logical
- [ ] Weighting system effective
- [ ] Confidence scoring accurate
- [ ] Performance acceptable

## 📊 Performance Benchmarks

### Data Fetching
- Social media: < 5 seconds per batch
- News articles: < 10 seconds per batch
- Satellite data: < 30 seconds per location
- Total pipeline: < 60 seconds for all sources

### Processing Performance
- Sentiment analysis: < 100ms per text
- Event extraction: < 200ms per article
- Image analysis: < 5 seconds per image
- Data fusion: < 1 second per symbol

### Quality Metrics
- Data completeness: > 80%
- Sentiment accuracy: > 70%
- Event detection precision: > 60%
- Signal correlation with price: > 0.3

## 🐛 Common Issues

1. **API Rate Limits**
   - Implement proper rate limiting
   - Use caching effectively
   - Distribute requests over time
   - Have fallback data sources

2. **Data Quality Issues**
   - Validate all incoming data
   - Handle missing/corrupt data gracefully
   - Monitor quality metrics continuously
   - Weight by data quality in fusion

3. **Processing Delays**
   - Use asynchronous processing
   - Implement timeouts
   - Cache processed results
   - Prioritize real-time data

## 🎯 Success Criteria

Step 11 is complete when:
1. ✅ Multiple alternative data sources integrated
2. ✅ Data quality monitoring operational
3. ✅ Sentiment analysis accurate and timely
4. ✅ Data fusion producing meaningful signals
5. ✅ Performance benchmarks met

## 🚀 Next Steps

Once all validations pass, proceed to:
[Step 12: Crypto & DeFi Integration](step-12-crypto-defi.md)

## 📚 Additional Resources

- [Alternative Data Handbook](../references/alt-data-handbook.md)
- [Financial NLP Guide](../references/financial-nlp-guide.md)
- [Satellite Data Analysis](../references/satellite-analysis.md)
- [Social Media Analytics](../references/social-analytics.md)
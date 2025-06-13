Scalable Trading Strategy Data Infrastructure

  A Comprehensive Guide to Building High-Performance Analytics for Quantitative Finance

  ---
  Table of Contents

  1. #executive-summary
  2. #architecture-overview
  3. #data-storage-layer
  4. #data-pipeline-architecture
  5. #query-interface--apis
  6. #automated-analytics-workflows
  7. #performance-optimization
  8. #machine-learning-integration
  9. #monitoring--observability
  10. #security--compliance
  11. #cost-optimization
  12. #implementation-roadmap
  13. #appendices

  ---
  Executive Summary

  As quantitative trading strategies become increasingly sophisticated, the need for robust, scalable data infrastructure becomes critical. This document outlines a comprehensive architecture for handling large-scale strategy backtesting, real-time analytics, and
  automated insights generation capable of supporting 5,000+ concurrent strategies while maintaining sub-second query performance.

  Key Requirements

  - Scale: Support 5,000+ strategies generating millions of events daily
  - Performance: Sub-second queries for real-time analytics
  - Reliability: 99.9% uptime with automatic failover
  - Flexibility: Support ad-hoc analysis and automated workflows
  - Cost-Effectiveness: Optimize storage and compute costs
  - ML-Ready: Enable machine learning workflows and feature engineering

  Architecture Highlights

  - Hybrid Storage: TimescaleDB for hot data, Parquet/S3 for cold storage
  - Event-Driven: Real-time processing with Apache Kafka
  - Microservices: Containerized services with Kubernetes orchestration
  - Analytics-First: Pre-computed aggregations and materialized views
  - ML Integration: Built-in feature store and model serving capabilities

  ---
  Architecture Overview

  High-Level Components

  graph TB
      subgraph "Data Ingestion"
          A[Strategy Engine] --> B[Apache Kafka]
          C[Market Data] --> B
          D[External Signals] --> B
      end

      subgraph "Stream Processing"
          B --> E[Apache Beam/Flink]
          E --> F[Real-time Enrichment]
          E --> G[Feature Engineering]
      end

      subgraph "Storage Layer"
          F --> H[TimescaleDB<br/>Hot Data]
          F --> I[S3/MinIO<br/>Cold Storage]
          G --> J[Feature Store<br/>Feast/Redis]
      end

      subgraph "Analytics Layer"
          H --> K[Query API<br/>FastAPI]
          I --> K
          J --> K
          K --> L[Analytics Dashboard]
          K --> M[Automated Workflows]
      end

      subgraph "ML Platform"
          J --> N[Model Training<br/>MLflow]
          N --> O[Model Serving<br/>Seldon/KServe]
          O --> P[Predictions API]
      end

  Technology Stack

  | Layer             | Technology             | Purpose                          |
  |-------------------|------------------------|----------------------------------|
  | Message Queue     | Apache Kafka           | Event streaming, data ingestion  |
  | Stream Processing | Apache Beam/Flink      | Real-time data processing        |
  | Time-Series DB    | TimescaleDB            | Hot data storage, fast queries   |
  | Data Lake         | S3/MinIO + Parquet     | Cold storage, historical data    |
  | Feature Store     | Feast + Redis          | ML features, caching             |
  | API Layer         | FastAPI + asyncpg      | High-performance query interface |
  | Orchestration     | Apache Airflow/Prefect | Workflow automation              |
  | Monitoring        | Prometheus + Grafana   | System observability             |
  | ML Platform       | MLflow + Seldon Core   | Model lifecycle management       |

  ---
  Data Storage Layer

  1. Time-Series Database (TimescaleDB)

  TimescaleDB serves as the primary storage for hot data (last 90 days) requiring fast access.

  Schema Design

  -- Core signals table
  CREATE TABLE signals (
      time TIMESTAMPTZ NOT NULL,
      strategy_id TEXT NOT NULL,
      symbol TEXT NOT NULL,
      signal_type TEXT NOT NULL, -- 'ENTRY', 'EXIT', 'SCALE_IN', 'SCALE_OUT'
      direction TEXT, -- 'LONG', 'SHORT', 'FLAT'
      strength FLOAT CHECK (strength BETWEEN 0 AND 1),
      price FLOAT,
      volume BIGINT,
      metadata JSONB,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      PRIMARY KEY (time, strategy_id, symbol)
  );

  -- Create hypertable for automatic partitioning
  SELECT create_hypertable('signals', 'time', chunk_time_interval => INTERVAL '1 day');

  -- Add indexes for common query patterns
  CREATE INDEX idx_signals_strategy_time ON signals (strategy_id, time DESC);
  CREATE INDEX idx_signals_symbol_time ON signals (symbol, time DESC);
  CREATE INDEX idx_signals_metadata_gin ON signals USING GIN (metadata);

  -- Regime classifications table
  CREATE TABLE regime_classifications (
      time TIMESTAMPTZ NOT NULL,
      classifier_id TEXT NOT NULL,
      symbol TEXT NOT NULL,
      regime TEXT NOT NULL,
      confidence FLOAT CHECK (confidence BETWEEN 0 AND 1),
      features JSONB,
      previous_regime TEXT,
      regime_duration INTERVAL,
      PRIMARY KEY (time, classifier_id, symbol)
  );

  SELECT create_hypertable('regime_classifications', 'time', chunk_time_interval => INTERVAL '1 day');

  -- Strategy performance metrics
  CREATE TABLE strategy_performance (
      time TIMESTAMPTZ NOT NULL,
      strategy_id TEXT NOT NULL,
      symbol TEXT NOT NULL,
      pnl FLOAT,
      cumulative_pnl FLOAT,
      returns FLOAT,
      sharpe_ratio_20d FLOAT,
      max_drawdown FLOAT,
      win_rate_20d FLOAT,
      trade_count_20d INTEGER,
      volatility_20d FLOAT,
      beta FLOAT,
      alpha FLOAT,
      information_ratio FLOAT,
      calmar_ratio FLOAT,
      sortino_ratio FLOAT,
      metadata JSONB,
      PRIMARY KEY (time, strategy_id, symbol)
  );

  SELECT create_hypertable('strategy_performance', 'time', chunk_time_interval => INTERVAL '1 day');

  -- Market data table
  CREATE TABLE market_data (
      time TIMESTAMPTZ NOT NULL,
      symbol TEXT NOT NULL,
      open FLOAT,
      high FLOAT,
      low FLOAT,
      close FLOAT,
      volume BIGINT,
      vwap FLOAT,
      bid FLOAT,
      ask FLOAT,
      spread FLOAT,
      PRIMARY KEY (time, symbol)
  );

  SELECT create_hypertable('market_data', 'time', chunk_time_interval => INTERVAL '1 hour');

  Continuous Aggregates for Fast Analytics

  -- Daily strategy performance rollup
  CREATE MATERIALIZED VIEW strategy_daily_performance
  WITH (timescaledb.continuous) AS
  SELECT
      time_bucket('1 day', time) AS day,
      strategy_id,
      symbol,
      AVG(sharpe_ratio_20d) as avg_sharpe,
      SUM(pnl) as daily_pnl,
      MAX(cumulative_pnl) - MIN(cumulative_pnl) as daily_range,
      MAX(max_drawdown) as max_daily_drawdown,
      COUNT(*) as signal_count,
      STDDEV(returns) as daily_volatility
  FROM strategy_performance
  GROUP BY day, strategy_id, symbol;

  -- Refresh policy for real-time updates
  SELECT add_continuous_aggregate_policy('strategy_daily_performance',
      start_offset => INTERVAL '3 days',
      end_offset => INTERVAL '1 hour',
      schedule_interval => INTERVAL '1 hour');

  -- Regime transition analysis
  CREATE MATERIALIZED VIEW regime_transitions
  WITH (timescaledb.continuous) AS
  SELECT
      time_bucket('1 hour', time) AS hour,
      classifier_id,
      symbol,
      regime,
      COUNT(*) as transition_count,
      AVG(confidence) as avg_confidence,
      LAG(regime) OVER (PARTITION BY classifier_id, symbol ORDER BY hour) as previous_regime
  FROM regime_classifications
  GROUP BY hour, classifier_id, symbol, regime;

  -- Strategy-regime performance correlation
  CREATE MATERIALIZED VIEW strategy_regime_performance
  WITH (timescaledb.continuous) AS
  SELECT
      time_bucket('4 hours', sp.time) AS period,
      sp.strategy_id,
      sp.symbol,
      rc.regime,
      AVG(sp.sharpe_ratio_20d) as avg_sharpe_in_regime,
      SUM(sp.pnl) as total_pnl_in_regime,
      COUNT(*) as signals_in_regime,
      STDDEV(sp.returns) as volatility_in_regime
  FROM strategy_performance sp
  JOIN regime_classifications rc
      ON sp.time = rc.time AND sp.symbol = rc.symbol
  GROUP BY period, sp.strategy_id, sp.symbol, rc.regime;

  Data Retention Policy

  -- Automatic data retention for older chunks
  SELECT add_retention_policy('signals', INTERVAL '90 days');
  SELECT add_retention_policy('strategy_performance', INTERVAL '90 days');
  SELECT add_retention_policy('regime_classifications', INTERVAL '90 days');
  SELECT add_retention_policy('market_data', INTERVAL '30 days');

  -- Compression policy for older data
  SELECT add_compression_policy('signals', INTERVAL '7 days');
  SELECT add_compression_policy('strategy_performance', INTERVAL '7 days');

  2. Data Lake (S3/MinIO + Parquet)

  For long-term storage and historical analysis, we use a data lake architecture with Parquet files.

  Partitioning Strategy

  s3://trading-data-lake/
  ├── signals/
  │   ├── year=2024/
  │   │   ├── month=01/
  │   │   │   ├── day=01/
  │   │   │   │   ├── strategy_type=momentum/
  │   │   │   │   │   └── part-00000.parquet
  │   │   │   │   └── strategy_type=mean_reversion/
  │   │   │   │       └── part-00000.parquet
  │   │   │   └── day=02/
  │   │   └── month=02/
  │   └── year=2025/
  ├── regime_classifications/
  │   └── [similar partitioning]
  ├── strategy_performance/
  │   └── [similar partitioning]
  └── market_data/
      └── [similar partitioning]

  Data Lake Management

  import pandas as pd
  import pyarrow as pa
  import pyarrow.parquet as pq
  from datetime import datetime, timedelta
  import s3fs

  class DataLakeManager:
      def __init__(self, bucket_name: str, aws_access_key: str, aws_secret_key: str):
          self.bucket = bucket_name
          self.fs = s3fs.S3FileSystem(
              key=aws_access_key,
              secret=aws_secret_key
          )

      def write_parquet_partitioned(self, df: pd.DataFrame, table_name: str, 
                                   partition_cols: list = None):
          """Write DataFrame to partitioned Parquet files."""

          if partition_cols is None:
              partition_cols = ['year', 'month', 'day']

          # Add partition columns if not present
          if 'year' not in df.columns:
              df['year'] = df['time'].dt.year
              df['month'] = df['time'].dt.month
              df['day'] = df['time'].dt.day

          # Convert to PyArrow Table
          table = pa.Table.from_pandas(df)

          # Write partitioned dataset
          pq.write_to_dataset(
              table,
              root_path=f's3://{self.bucket}/{table_name}',
              partition_cols=partition_cols,
              filesystem=self.fs,
              compression='snappy',
              use_dictionary=True,
              write_statistics=True
          )

      def read_parquet_filtered(self, table_name: str, 
                               start_date: datetime, end_date: datetime,
                               strategy_types: list = None) -> pd.DataFrame:
          """Read Parquet data with predicate pushdown for efficiency."""

          filters = [
              ('year', '>=', start_date.year),
              ('year', '<=', end_date.year),
              ('month', '>=', start_date.month if start_date.year == end_date.year else 1),
              ('month', '<=', end_date.month if start_date.year == end_date.year else 12)
          ]

          if strategy_types:
              filters.append(('strategy_type', 'in', strategy_types))

          dataset = pq.ParquetDataset(
              f's3://{self.bucket}/{table_name}',
              filesystem=self.fs,
              filters=filters
          )

          return dataset.read_pandas().to_pandas()

      def optimize_partition(self, table_name: str, partition_path: str):
          """Optimize Parquet files by compacting small files."""

          # Read all files in partition
          files = self.fs.glob(f'{self.bucket}/{table_name}/{partition_path}/*.parquet')

          if len(files) > 10:  # Only optimize if many small files
              dfs = []
              for file in files:
                  df = pd.read_parquet(f's3://{file}')
                  dfs.append(df)

              # Combine and rewrite as single file
              combined_df = pd.concat(dfs, ignore_index=True)
              combined_df = combined_df.sort_values('time')

              # Write optimized file
              optimized_path = f's3://{self.bucket}/{table_name}/{partition_path}/optimized.parquet'
              combined_df.to_parquet(optimized_path, compression='snappy')

              # Remove old files
              for file in files:
                  self.fs.rm(f's3://{file}')

  3. Feature Store (Feast + Redis)

  The feature store provides low-latency access to ML features and caches frequently accessed data.

  Feature Store Configuration

  # feature_store.yaml
  project: trading_strategies
  registry: s3://trading-feature-store/registry.db
  provider: aws
  online_store:
    type: redis
    connection_string: "redis://redis-cluster:6379"
  offline_store:
    type: s3
    s3_endpoint_override: "http://minio:9000"

  entities:
    - name: strategy
      value_type: STRING
      description: "Trading strategy identifier"

    - name: symbol
      value_type: STRING
      description: "Financial instrument symbol"

  feature_views:
    - name: strategy_features
      entities:
        - strategy
        - symbol
      ttl: 3600  # 1 hour
      features:
        - name: sharpe_ratio_1d
          value_type: FLOAT
        - name: sharpe_ratio_7d
          value_type: FLOAT
        - name: sharpe_ratio_30d
          value_type: FLOAT
        - name: max_drawdown_7d
          value_type: FLOAT
        - name: volatility_7d
          value_type: FLOAT
        - name: win_rate_7d
          value_type: FLOAT
        - name: trade_frequency_7d
          value_type: FLOAT
        - name: avg_trade_duration_7d
          value_type: FLOAT

    - name: regime_features
      entities:
        - symbol
      ttl: 300  # 5 minutes
      features:
        - name: current_regime
          value_type: STRING
        - name: regime_confidence
          value_type: FLOAT
        - name: regime_duration_minutes
          value_type: INT64
        - name: regime_stability_1h
          value_type: FLOAT
        - name: transition_probability
          value_type: FLOAT

  Feature Store Implementation

  from feast import FeatureStore
  from feast.feature_view import FeatureView
  from feast.entity import Entity
  from feast.field import Field
  from feast.types import Float64, String, Int64
  import pandas as pd
  from datetime import datetime, timedelta

  class TradingFeatureStore:
      def __init__(self, repo_path: str = "."):
          self.store = FeatureStore(repo_path=repo_path)

      def get_strategy_features(self, strategy_ids: list, symbols: list, 
                              feature_names: list = None) -> pd.DataFrame:
          """Get latest features for strategies."""

          entity_df = pd.DataFrame({
              'strategy': strategy_ids * len(symbols),
              'symbol': symbols * len(strategy_ids),
              'event_timestamp': [datetime.now()] * len(strategy_ids) * len(symbols)
          })

          if feature_names is None:
              feature_names = [
                  'strategy_features:sharpe_ratio_1d',
                  'strategy_features:sharpe_ratio_7d',
                  'strategy_features:max_drawdown_7d',
                  'strategy_features:volatility_7d'
              ]

          return self.store.get_online_features(
              features=feature_names,
              entity_df=entity_df
          ).to_df()

      def get_regime_features(self, symbols: list) -> pd.DataFrame:
          """Get current regime information for symbols."""

          entity_df = pd.DataFrame({
              'symbol': symbols,
              'event_timestamp': [datetime.now()] * len(symbols)
          })

          return self.store.get_online_features(
              features=[
                  'regime_features:current_regime',
                  'regime_features:regime_confidence',
                  'regime_features:regime_duration_minutes'
              ],
              entity_df=entity_df
          ).to_df()

      def batch_score_strategies(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
          """Get historical features for batch ML training."""

          entity_df = pd.DataFrame({
              'strategy': self._get_all_strategies(),
              'symbol': ['SPY'] * len(self._get_all_strategies()),
              'event_timestamp': pd.date_range(start_date, end_date, freq='1H')
          })

          return self.store.get_historical_features(
              entity_df=entity_df,
              features=[
                  'strategy_features:sharpe_ratio_7d',
                  'strategy_features:volatility_7d',
                  'regime_features:current_regime',
                  'regime_features:regime_confidence'
              ]
          ).to_df()

      def materialize_features(self, start_date: datetime, end_date: datetime):
          """Materialize features to online store."""

          self.store.materialize(
              feature_views=['strategy_features', 'regime_features'],
              start_date=start_date,
              end_date=end_date
          )

      def _get_all_strategies(self) -> list:
          """Helper to get all strategy IDs from database."""
          # Implementation to fetch from TimescaleDB
          pass

  ---
  Data Pipeline Architecture

  1. Real-Time Stream Processing

  Apache Kafka Configuration

  # kafka-cluster.yaml
  apiVersion: kafka.strimzi.io/v1beta2
  kind: Kafka
  metadata:
    name: trading-kafka
  spec:
    kafka:
      version: 3.4.0
      replicas: 3
      listeners:
        - name: plain
          port: 9092
          type: internal
          tls: false
        - name: tls
          port: 9093
          type: internal
          tls: true
      config:
        offsets.topic.replication.factor: 3
        transaction.state.log.replication.factor: 3
        transaction.state.log.min.isr: 2
        default.replication.factor: 3
        min.insync.replicas: 2
        inter.broker.protocol.version: "3.4"
        log.retention.hours: 168  # 7 days
        log.segment.bytes: 1073741824  # 1GB
        log.retention.check.interval.ms: 300000
        num.partitions: 12
      storage:
        type: persistent-claim
        size: 100Gi
        class: fast-ssd
    zookeeper:
      replicas: 3
      storage:
        type: persistent-claim
        size: 10Gi
        class: fast-ssd

  # Topics configuration
  topics:
    - name: strategy-signals
      partitions: 12
      replicas: 3
      config:
        compression.type: snappy
        retention.ms: 604800000  # 7 days

    - name: regime-classifications
      partitions: 6
      replicas: 3
      config:
        compression.type: snappy
        retention.ms: 2592000000  # 30 days

    - name: market-data
      partitions: 24
      replicas: 3
      config:
        compression.type: snappy
        retention.ms: 259200000  # 3 days

  Apache Beam Pipeline

  import apache_beam as beam
  from apache_beam.options.pipeline_options import PipelineOptions
  from apache_beam.transforms import window
  import json
  from datetime import datetime, timedelta
  import asyncpg
  import asyncio

  class TradingDataPipeline:
      def __init__(self, pipeline_options: PipelineOptions):
          self.options = pipeline_options

      def create_pipeline(self):
          """Create the main data processing pipeline."""

          with beam.Pipeline(options=self.options) as pipeline:

              # Read strategy signals from Kafka
              signals = (pipeline
                  | 'ReadSignals' >> beam.io.ReadFromKafka(
                      consumer_config={
                          'bootstrap.servers': 'kafka-cluster:9092',
                          'group.id': 'strategy-pipeline',
                          'auto.offset.reset': 'latest'
                      },
                      topics=['strategy-signals'])
                  | 'ParseSignalJSON' >> beam.Map(self._parse_json)
                  | 'FilterValidSignals' >> beam.Filter(self._is_valid_signal)
              )

              # Read regime classifications
              regimes = (pipeline
                  | 'ReadRegimes' >> beam.io.ReadFromKafka(
                      consumer_config={
                          'bootstrap.servers': 'kafka-cluster:9092',
                          'group.id': 'regime-pipeline',
                          'auto.offset.reset': 'latest'
                      },
                      topics=['regime-classifications'])
                  | 'ParseRegimeJSON' >> beam.Map(self._parse_json)
                  | 'FilterValidRegimes' >> beam.Filter(self._is_valid_regime)
              )

              # Read market data
              market_data = (pipeline
                  | 'ReadMarketData' >> beam.io.ReadFromKafka(
                      consumer_config={
                          'bootstrap.servers': 'kafka-cluster:9092',
                          'group.id': 'market-pipeline'
                      },
                      topics=['market-data'])
                  | 'ParseMarketJSON' >> beam.Map(self._parse_json)
              )

              # Enrich signals with market context
              enriched_signals = (
                  {
                      'signals': signals,
                      'market_data': market_data
                  }
                  | 'CoGroupBySymbol' >> beam.CoGroupByKey()
                  | 'EnrichSignals' >> beam.ParDo(EnrichSignalsWithMarketData())
              )

              # Calculate real-time strategy metrics
              strategy_metrics = (enriched_signals
                  | 'WindowIntoFixedWindows' >> beam.WindowInto(
                      window.FixedWindows(300))  # 5-minute windows
                  | 'GroupByStrategy' >> beam.GroupByKey()
                  | 'CalculateMetrics' >> beam.ParDo(CalculateStrategyMetrics())
              )

              # Write to TimescaleDB
              (enriched_signals
                  | 'FormatSignalsForDB' >> beam.Map(self._format_signal_for_db)
                  | 'WriteSignalsToDB' >> beam.ParDo(WriteToTimescaleDB('signals'))
              )

              (regimes
                  | 'FormatRegimesForDB' >> beam.Map(self._format_regime_for_db)
                  | 'WriteRegimesToDB' >> beam.ParDo(WriteToTimescaleDB('regime_classifications'))
              )

              (strategy_metrics
                  | 'WriteMetricsToDB' >> beam.ParDo(WriteToTimescaleDB('strategy_performance'))
              )

              # Update feature store
              (strategy_metrics
                  | 'UpdateFeatureStore' >> beam.ParDo(UpdateFeatureStore())
              )

              # Archive to data lake
              (enriched_signals
                  | 'WindowForArchive' >> beam.WindowInto(
                      window.FixedWindows(3600))  # 1-hour windows for batching
                  | 'GroupForArchive' >> beam.GroupByKey()
                  | 'WriteToDataLake' >> beam.ParDo(WriteToDataLake())
              )

      def _parse_json(self, kafka_message):
          """Parse JSON message from Kafka."""
          try:
              return json.loads(kafka_message[1].decode('utf-8'))
          except:
              return None

      def _is_valid_signal(self, signal):
          """Validate signal structure."""
          required_fields = ['strategy_id', 'symbol', 'time', 'signal_type']
          return signal and all(field in signal for field in required_fields)

      def _is_valid_regime(self, regime):
          """Validate regime classification structure."""
          required_fields = ['classifier_id', 'symbol', 'time', 'regime']
          return regime and all(field in regime for field in required_fields)

  class EnrichSignalsWithMarketData(beam.DoFn):
      """Enrich signals with market context."""

      def process(self, element):
          symbol, grouped_data = element
          signals = grouped_data.get('signals', [])
          market_data = grouped_data.get('market_data', [])

          # Create market data lookup by timestamp
          market_lookup = {md['time']: md for md in market_data}

          for signal in signals:
              # Find closest market data
              signal_time = signal['time']
              closest_market_data = self._find_closest_market_data(signal_time, market_lookup)

              if closest_market_data:
                  signal.update({
                      'market_price': closest_market_data['close'],
                      'market_volume': closest_market_data['volume'],
                      'bid_ask_spread': closest_market_data.get('spread', 0),
                      'market_volatility': self._calculate_volatility(market_lookup, signal_time)
                  })

              yield signal

      def _find_closest_market_data(self, signal_time, market_lookup):
          """Find market data closest to signal time."""
          # Implementation for finding closest timestamp
          pass

      def _calculate_volatility(self, market_lookup, current_time):
          """Calculate rolling volatility."""
          # Implementation for volatility calculation
          pass

  class CalculateStrategyMetrics(beam.DoFn):
      """Calculate real-time strategy performance metrics."""

      def process(self, element):
          strategy_id, signals = element

          if not signals:
              return

          # Calculate metrics for this window
          metrics = self._calculate_window_metrics(signals)

          yield {
              'time': datetime.now(),
              'strategy_id': strategy_id,
              'symbol': signals[0]['symbol'],  # Assuming single symbol per strategy
              **metrics
          }

      def _calculate_window_metrics(self, signals):
          """Calculate performance metrics for signal window."""
          # Implementation for Sharpe ratio, drawdown, etc.
          pass

  class WriteToTimescaleDB(beam.DoFn):
      """Write data to TimescaleDB."""

      def __init__(self, table_name):
          self.table_name = table_name
          self.connection_pool = None

      def setup(self):
          """Initialize database connection pool."""
          self.connection_pool = asyncpg.create_pool(
              host='timescaledb-service',
              database='trading',
              user='postgres',
              password='password',
              min_size=5,
              max_size=20
          )

      def process(self, element):
          """Write element to database."""
          asyncio.run(self._write_to_db(element))

      async def _write_to_db(self, data):
          """Async write to database."""
          async with self.connection_pool.acquire() as conn:
              # Implement table-specific insert logic
              pass

  2. Batch Processing for Historical Analysis

  from airflow import DAG
  from airflow.operators.python import PythonOperator
  from airflow.operators.bash import BashOperator
  from datetime import datetime, timedelta
  import pandas as pd

  # DAG for daily batch processing
  default_args = {
      'owner': 'trading-team',
      'depends_on_past': False,
      'start_date': datetime(2024, 1, 1),
      'email_on_failure': True,
      'email_on_retry': False,
      'retries': 2,
      'retry_delay': timedelta(minutes=5)
  }

  dag = DAG(
      'daily_strategy_analysis',
      default_args=default_args,
      description='Daily strategy performance analysis',
      schedule_interval='0 6 * * *',  # 6 AM daily
      catchup=False,
      max_active_runs=1
  )

  def extract_daily_performance(**context):
      """Extract daily performance data from TimescaleDB."""

      execution_date = context['execution_date']
      start_date = execution_date - timedelta(days=1)
      end_date = execution_date

      query = """
      SELECT 
          strategy_id,
          symbol,
          SUM(pnl) as daily_pnl,
          AVG(sharpe_ratio_20d) as avg_sharpe,
          MAX(max_drawdown) as max_drawdown,
          COUNT(*) as signal_count
      FROM strategy_performance
      WHERE time >= %s AND time < %s
      GROUP BY strategy_id, symbol
      """

      df = pd.read_sql(query, connection, params=[start_date, end_date])

      # Save to S3 for further processing
      df.to_parquet(f's3://trading-data/daily-performance/{execution_date.strftime("%Y-%m-%d")}.parquet')

      return f"Processed {len(df)} strategy-symbol combinations"

  def calculate_regime_attribution(**context):
      """Calculate performance attribution by regime."""

      execution_date = context['execution_date']

      # Load daily performance data
      df = pd.read_parquet(f's3://trading-data/daily-performance/{execution_date.strftime("%Y-%m-%d")}.parquet')

      # Calculate regime-based attribution
      attribution = analyze_regime_attribution(df)

      # Save results
      attribution.to_parquet(f's3://trading-data/regime-attribution/{execution_date.strftime("%Y-%m-%d")}.parquet')

  def update_strategy_rankings(**context):
      """Update strategy rankings and send alerts."""

      execution_date = context['execution_date']

      # Load performance data for last 30 days
      end_date = execution_date
      start_date = end_date - timedelta(days=30)

      rankings = calculate_strategy_rankings(start_date, end_date)

      # Update rankings table
      save_rankings_to_db(rankings)

      # Send alerts for significant changes
      send_ranking_alerts(rankings)

  # Define tasks
  extract_task = PythonOperator(
      task_id='extract_daily_performance',
      python_callable=extract_daily_performance,
      dag=dag
  )

  regime_task = PythonOperator(
      task_id='calculate_regime_attribution',
      python_callable=calculate_regime_attribution,
      dag=dag
  )

  ranking_task = PythonOperator(
      task_id='update_strategy_rankings',
      python_callable=update_strategy_rankings,
      dag=dag
  )

  # Set dependencies
  extract_task >> regime_task >> ranking_task

  ---
  Query Interface & APIs

  1. High-Performance Query API

  from fastapi import FastAPI, Query, HTTPException, Depends
  from fastapi.middleware.cors import CORSMiddleware
  from fastapi.responses import StreamingResponse
  import asyncpg
  import asyncio
  from typing import List, Optional, Dict, Any
  from datetime import datetime, timedelta
  import pandas as pd
  import json
  from pydantic import BaseModel
  import uvicorn

  app = FastAPI(
      title="Trading Strategy Analytics API",
      description="High-performance API for trading strategy analysis",
      version="1.0.0"
  )

  app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )

  # Connection pool
  connection_pool = None

  async def get_db_pool():
      """Get database connection pool."""
      global connection_pool
      if connection_pool is None:
          connection_pool = await asyncpg.create_pool(
              host='timescaledb-service',
              database='trading',
              user='postgres',
              password='password',
              min_size=10,
              max_size=50,
              command_timeout=60
          )
      return connection_pool

  # Request/Response models
  class StrategyPerformanceQuery(BaseModel):
      strategy_ids: Optional[List[str]] = None
      symbols: Optional[List[str]] = None
      start_date: datetime
      end_date: datetime
      regime: Optional[str] = None
      min_sharpe: Optional[float] = None

  class StrategyRanking(BaseModel):
      strategy_id: str
      symbol: str
      sharpe_ratio: float
      total_pnl: float
      max_drawdown: float
      win_rate: float
      trade_count: int
      rank: int

  class RegimeAnalysis(BaseModel):
      regime: str
      duration_hours: float
      avg_confidence: float
      transition_count: int
      top_strategies: List[StrategyRanking]

  @app.get("/api/v1/strategies/rankings", response_model=List[StrategyRanking])
  async def get_strategy_rankings(
      lookback_days: int = Query(30, ge=1, le=365),
      regime: Optional[str] = Query(None),
      symbol: Optional[str] = Query(None),
      min_trades: int = Query(10, ge=1),
      limit: int = Query(50, le=1000),
      pool: asyncpg.Pool = Depends(get_db_pool)
  ):
      """Get top-performing strategies ranked by Sharpe ratio."""

      async with pool.acquire() as conn:
          query = """
          WITH strategy_metrics AS (
              SELECT 
                  sp.strategy_id,
                  sp.symbol,
                  AVG(sp.sharpe_ratio_20d) as avg_sharpe,
                  SUM(sp.pnl) as total_pnl,
                  MAX(sp.max_drawdown) as max_drawdown,
                  AVG(sp.win_rate_20d) as avg_win_rate,
                  SUM(sp.trade_count_20d) as total_trades,
                  COUNT(*) as data_points
              FROM strategy_performance sp
              LEFT JOIN regime_classifications rc 
                  ON sp.time = rc.time AND sp.symbol = rc.symbol
              WHERE sp.time >= NOW() - INTERVAL '%s days'
                  AND ($2::text IS NULL OR rc.regime = $2)
                  AND ($3::text IS NULL OR sp.symbol = $3)
              GROUP BY sp.strategy_id, sp.symbol
              HAVING SUM(sp.trade_count_20d) >= $4
                  AND COUNT(*) > 10  -- Minimum data points for reliability
          )
          SELECT 
              strategy_id,
              symbol,
              avg_sharpe as sharpe_ratio,
              total_pnl,
              max_drawdown,
              avg_win_rate as win_rate,
              total_trades as trade_count,
              ROW_NUMBER() OVER (ORDER BY avg_sharpe DESC) as rank
          FROM strategy_metrics
          ORDER BY avg_sharpe DESC
          LIMIT $5
          """

          rows = await conn.fetch(query, lookback_days, regime, symbol, min_trades, limit)

          return [StrategyRanking(**dict(row)) for row in rows]

  @app.get("/api/v1/strategies/{strategy_id}/performance")
  async def get_strategy_performance(
      strategy_id: str,
      start_date: datetime = Query(...),
      end_date: datetime = Query(...),
      symbol: Optional[str] = Query(None),
      pool: asyncpg.Pool = Depends(get_db_pool)
  ):
      """Get detailed performance metrics for a specific strategy."""

      async with pool.acquire() as conn:
          query = """
          SELECT 
              time,
              symbol,
              pnl,
              cumulative_pnl,
              sharpe_ratio_20d,
              max_drawdown,
              win_rate_20d,
              trade_count_20d,
              volatility_20d
          FROM strategy_performance
          WHERE strategy_id = $1
              AND time >= $2
              AND time <= $3
              AND ($4::text IS NULL OR symbol = $4)
          ORDER BY time
          """

          rows = await conn.fetch(query, strategy_id, start_date, end_date, symbol)

          if not rows:
              raise HTTPException(status_code=404, detail="Strategy not found")

          return [dict(row) for row in rows]

  @app.get("/api/v1/regimes/analysis", response_model=List[RegimeAnalysis])
  async def get_regime_analysis(
      lookback_days: int = Query(30, ge=1, le=365),
      symbol: str = Query("SPY"),
      classifier_id: Optional[str] = Query(None),
      pool: asyncpg.Pool = Depends(get_db_pool)
  ):
      """Analyze strategy performance across different market regimes."""

      async with pool.acquire() as conn:
          # Get regime periods and statistics
          regime_query = """
          WITH regime_periods AS (
              SELECT 
                  regime,
                  time,
                  LEAD(time) OVER (PARTITION BY classifier_id ORDER BY time) as next_time,
                  confidence,
                  classifier_id
              FROM regime_classifications
              WHERE symbol = $1
                  AND time >= NOW() - INTERVAL '%s days'
                  AND ($2::text IS NULL OR classifier_id = $2)
          ),
          regime_stats AS (
              SELECT 
                  regime,
                  COUNT(*) as transition_count,
                  AVG(confidence) as avg_confidence,
                  AVG(EXTRACT(EPOCH FROM (next_time - time))/3600) as avg_duration_hours
              FROM regime_periods
              WHERE next_time IS NOT NULL
              GROUP BY regime
          )
          SELECT * FROM regime_stats
          ORDER BY avg_duration_hours DESC
          """

          regime_rows = await conn.fetch(regime_query, symbol, classifier_id, lookback_days)

          results = []
          for regime_row in regime_rows:
              regime = regime_row['regime']

              # Get top strategies for this regime
              strategy_query = """
              WITH regime_performance AS (
                  SELECT 
                      sp.strategy_id,
                      sp.symbol,
                      AVG(sp.sharpe_ratio_20d) as avg_sharpe,
                      SUM(sp.pnl) as total_pnl,
                      MAX(sp.max_drawdown) as max_drawdown,
                      AVG(sp.win_rate_20d) as win_rate,
                      SUM(sp.trade_count_20d) as trade_count
                  FROM strategy_performance sp
                  JOIN regime_classifications rc 
                      ON sp.time = rc.time AND sp.symbol = rc.symbol
                  WHERE rc.regime = $1
                      AND sp.symbol = $2
                      AND sp.time >= NOW() - INTERVAL '%s days'
                  GROUP BY sp.strategy_id, sp.symbol
                  HAVING SUM(sp.trade_count_20d) > 5
              )
              SELECT 
                  strategy_id,
                  symbol,
                  avg_sharpe as sharpe_ratio,
                  total_pnl,
                  max_drawdown,
                  win_rate,
                  trade_count,
                  ROW_NUMBER() OVER (ORDER BY avg_sharpe DESC) as rank
              FROM regime_performance
              ORDER BY avg_sharpe DESC
              LIMIT 10
              """

              strategy_rows = await conn.fetch(strategy_query, regime, symbol, lookback_days)
              top_strategies = [StrategyRanking(**dict(row)) for row in strategy_rows]

              results.append(RegimeAnalysis(
                  regime=regime,
                  duration_hours=regime_row['avg_duration_hours'] or 0,
                  avg_confidence=regime_row['avg_confidence'] or 0,
                  transition_count=regime_row['transition_count'] or 0,
                  top_strategies=top_strategies
              ))

          return results

  @app.get("/api/v1/strategies/parameter-sensitivity/{strategy_type}")
  async def analyze_parameter_sensitivity(
      strategy_type: str,
      lookback_days: int = Query(30, ge=1, le=90),
      pool: asyncpg.Pool = Depends(get_db_pool)
  ):
      """Analyze parameter sensitivity for a strategy type."""

      async with pool.acquire() as conn:
          query = """
          WITH strategy_variants AS (
              SELECT 
                  strategy_id,
                  metadata->>'parameters' as parameters,
                  AVG(sharpe_ratio_20d) as avg_sharpe,
                  STDDEV(sharpe_ratio_20d) as sharpe_std,
                  SUM(pnl) as total_pnl,
                  COUNT(*) as data_points
              FROM strategy_performance
              WHERE strategy_id LIKE $1 || '%'
                  AND time >= NOW() - INTERVAL '%s days'
              GROUP BY strategy_id, metadata->>'parameters'
              HAVING COUNT(*) > 10
          )
          SELECT 
              strategy_id,
              parameters,
              avg_sharpe,
              sharpe_std,
              total_pnl,
              data_points,
              avg_sharpe / NULLIF(sharpe_std, 0) as information_ratio
          FROM strategy_variants
          ORDER BY avg_sharpe DESC
          """

          rows = await conn.fetch(query, strategy_type, lookback_days)

          # Analyze parameter impact
          sensitivity_analysis = analyze_parameter_impact([dict(row) for row in rows])

          return sensitivity_analysis

  @app.get("/api/v1/strategies/realtime-metrics")
  async def get_realtime_metrics(
      strategy_ids: List[str] = Query(...),
      pool: asyncpg.Pool = Depends(get_db_pool)
  ):
      """Get real-time performance metrics for specified strategies."""

      async with pool.acquire() as conn:
          # Get latest metrics from continuous aggregates
          query = """
          SELECT 
              strategy_id,
              symbol,
              avg_sharpe,
              daily_pnl,
              signal_count,
              daily_volatility,
              day
          FROM strategy_daily_performance
          WHERE strategy_id = ANY($1)
              AND day >= CURRENT_DATE - INTERVAL '7 days'
          ORDER BY day DESC, avg_sharpe DESC
          """

          rows = await conn.fetch(query, strategy_ids)

          return [dict(row) for row in rows]

  @app.get("/api/v1/export/performance-report")
  async def export_performance_report(
      start_date: datetime = Query(...),
      end_date: datetime = Query(...),
      format: str = Query("csv", regex="^(csv|excel|parquet)$"),
      strategy_ids: Optional[List[str]] = Query(None),
      pool: asyncpg.Pool = Depends(get_db_pool)
  ):
      """Export comprehensive performance report."""

      async with pool.acquire() as conn:
          query = """
          SELECT 
              sp.time,
              sp.strategy_id,
              sp.symbol,
              sp.pnl,
              sp.cumulative_pnl,
              sp.sharpe_ratio_20d,
              sp.max_drawdown,
              sp.win_rate_20d,
              sp.trade_count_20d,
              rc.regime,
              rc.confidence as regime_confidence
          FROM strategy_performance sp
          LEFT JOIN regime_classifications rc 
              ON sp.time = rc.time AND sp.symbol = rc.symbol
          WHERE sp.time >= $1 
              AND sp.time <= $2
              AND ($3::text[] IS NULL OR sp.strategy_id = ANY($3))
          ORDER BY sp.time, sp.strategy_id
          """

          rows = await conn.fetch(query, start_date, end_date, strategy_ids)

          # Convert to DataFrame
          df = pd.DataFrame([dict(row) for row in rows])

          # Export in requested format
          if format == "csv":
              return StreamingResponse(
                  io.StringIO(df.to_csv(index=False)),
                  media_type="text/csv",
                  headers={"Content-Disposition": "attachment; filename=performance_report.csv"}
              )
          elif format == "excel":
              buffer = io.BytesIO()
              df.to_excel(buffer, index=False, engine='openpyxl')
              buffer.seek(0)
              return StreamingResponse(
                  buffer,
                  media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                  headers={"Content-Disposition": "attachment; filename=performance_report.xlsx"}
              )
          elif format == "parquet":
              buffer = io.BytesIO()
              df.to_parquet(buffer, index=False)
              buffer.seek(0)
              return StreamingResponse(
                  buffer,
                  media_type="application/octet-stream",
                  headers={"Content-Disposition": "attachment; filename=performance_report.parquet"}
              )

  def analyze_parameter_impact(strategy_variants: List[Dict]) -> Dict:
      """Analyze the impact of different parameters on strategy performance."""

      if not strategy_variants:
          return {"error": "No strategy variants found"}

      # Group by parameter combinations
      param_groups = {}
      for variant in strategy_variants:
          params = variant.get('parameters', '{}')
          if params not in param_groups:
              param_groups[params] = []
          param_groups[params].append(variant)

      # Calculate sensitivity metrics
      sensitivity = {
          "total_variants": len(strategy_variants),
          "parameter_combinations": len(param_groups),
          "best_performance": max(strategy_variants, key=lambda x: x['avg_sharpe']),
          "worst_performance": min(strategy_variants, key=lambda x: x['avg_sharpe']),
          "performance_range": {
              "sharpe_min": min(v['avg_sharpe'] for v in strategy_variants),
              "sharpe_max": max(v['avg_sharpe'] for v in strategy_variants),
              "sharpe_std": pd.Series([v['avg_sharpe'] for v in strategy_variants]).std()
          },
          "parameter_impact": []
      }

      # Analyze individual parameter impact (simplified)
      for params, variants in param_groups.items():
          avg_sharpe = sum(v['avg_sharpe'] for v in variants) / len(variants)
          sensitivity["parameter_impact"].append({
              "parameters": params,
              "variant_count": len(variants),
              "avg_sharpe": avg_sharpe,
              "total_pnl": sum(v['total_pnl'] for v in variants)
          })

      sensitivity["parameter_impact"].sort(key=lambda x: x['avg_sharpe'], reverse=True)

      return sensitivity

  if __name__ == "__main__":
      uvicorn.run(app, host="0.0.0.0", port=8000)

  2. GraphQL API for Complex Queries

  import strawberry
  from strawberry.fastapi import GraphQLRouter
  from typing import List, Optional
  from datetime import datetime
  import asyncpg

  @strawberry.type
  class Strategy:
      id: str
      name: str
      type: str
      parameters: str
      created_at: datetime

  @strawberry.type
  class PerformanceMetric:
      time: datetime
      strategy_id: str
      symbol: str
      pnl: float
      sharpe_ratio: float
      max_drawdown: float
      win_rate: float

  @strawberry.type
  class RegimeClassification:
      time: datetime
      classifier_id: str
      symbol: str
      regime: str
      confidence: float

  @strawberry.type
  class Query:
      @strawberry.field
      async def strategies(self, 
                          strategy_type: Optional[str] = None,
                          min_sharpe: Optional[float] = None) -> List[Strategy]:
          """Get strategies with optional filtering."""
          # Implementation
          pass

      @strawberry.field
      async def performance_metrics(self,
                                   strategy_ids: List[str],
                                   start_date: datetime,
                                   end_date: datetime,
                                   symbol: Optional[str] = None) -> List[PerformanceMetric]:
          """Get performance metrics for strategies."""
          # Implementation
          pass

      @strawberry.field
      async def regime_classifications(self,
                                     classifier_id: str,
                                     symbol: str,
                                     start_date: datetime,
                                     end_date: datetime) -> List[RegimeClassification]:
          """Get regime classifications."""
          # Implementation
          pass

      @strawberry.field
      async def strategy_performance_by_regime(self,
                                             strategy_id: str,
                                             regime: str,
                                             lookback_days: int = 30) -> List[PerformanceMetric]:
          """Get strategy performance filtered by regime."""
          # Implementation with complex JOIN
          pass

  schema = strawberry.Schema(query=Query)
  graphql_app = GraphQLRouter(schema)

  ---
  Automated Analytics Workflows

  1. Strategy Performance Monitoring

  from prefect import flow, task, get_run_logger
  from prefect.task_runners import ConcurrentTaskRunner
  from prefect.deployments import Deployment
  from prefect.server.schemas.schedules import CronSchedule
  import pandas as pd
  import numpy as np
  from typing import Dict, List, Tuple
  import asyncpg
  import asyncio
  from datetime import datetime, timedelta
  import plotly.graph_objects as go
  import plotly.express as px
  from plotly.subplots import make_subplots
  import smtplib
  from email.mime.multipart import MIMEMultipart
  from email.mime.text import MIMEText
  from email.mime.base import MIMEBase
  import boto3

  @task
  async def extract_daily_performance_data(date: datetime) -> pd.DataFrame:
      """Extract daily performance data for all strategies."""
      logger = get_run_logger()

      connection = await asyncpg.connect(
          host='timescaledb-service',
          database='trading',
          user='postgres',
          password='password'
      )

      query = """
      SELECT 
          strategy_id,
          symbol,
          SUM(pnl) as daily_pnl,
          AVG(sharpe_ratio_20d) as avg_sharpe,
          MAX(max_drawdown) as max_drawdown,
          AVG(win_rate_20d) as win_rate,
          SUM(trade_count_20d) as trade_count,
          STDDEV(returns) as daily_volatility
      FROM strategy_performance
      WHERE DATE(time) = $1
      GROUP BY strategy_id, symbol
      HAVING SUM(trade_count_20d) > 0
      """

      rows = await connection.fetch(query, date.date())
      await connection.close()

      df = pd.DataFrame([dict(row) for row in rows])
      logger.info(f"Extracted {len(df)} strategy performance records for {date.date()}")

      return df

  @task
  async def identify_performance_anomalies(df: pd.DataFrame, 
                                         historical_window: int = 30) -> Dict[str, List]:
      """Identify strategies with unusual performance."""
      logger = get_run_logger()

      anomalies = {
          'high_performers': [],
          'underperformers': [],
          'high_volatility': [],
          'drawdown_alerts': []
      }

      # Get historical performance for comparison
      connection = await asyncpg.connect(
          host='timescaledb-service',
          database='trading',
          user='postgres',
          password='password'
      )

      for _, row in df.iterrows():
          strategy_id = row['strategy_id']
          symbol = row['symbol']

          # Get historical performance
          hist_query = """
          SELECT 
              AVG(sharpe_ratio_20d) as avg_historical_sharpe,
              STDDEV(sharpe_ratio_20d) as std_historical_sharpe,
              AVG(max_drawdown) as avg_historical_drawdown,
              STDDEV(daily_pnl) as std_historical_pnl
          FROM (
              SELECT 
                  DATE(time) as date,
                  AVG(sharpe_ratio_20d) as sharpe_ratio_20d,
                  MAX(max_drawdown) as max_drawdown,
                  SUM(pnl) as daily_pnl
              FROM strategy_performance
              WHERE strategy_id = $1 
                  AND symbol = $2
                  AND time >= NOW() - INTERVAL '%s days'
              GROUP BY DATE(time)
          ) daily_stats
          """

          hist_result = await connection.fetchrow(hist_query, strategy_id, symbol, historical_window)

          if hist_result:
              # Check for anomalies
              current_sharpe = row['avg_sharpe']
              hist_sharpe_mean = hist_result['avg_historical_sharpe'] or 0
              hist_sharpe_std = hist_result['std_historical_sharpe'] or 1

              # High performer (> 2 std above mean)
              if current_sharpe > hist_sharpe_mean + 2 * hist_sharpe_std:
                  anomalies['high_performers'].append({
                      'strategy_id': strategy_id,
                      'symbol': symbol,
                      'current_sharpe': current_sharpe,
                      'historical_mean': hist_sharpe_mean,
                      'z_score': (current_sharpe - hist_sharpe_mean) / hist_sharpe_std
                  })

              # Underperformer (> 2 std below mean)
              elif current_sharpe < hist_sharpe_mean - 2 * hist_sharpe_std:
                  anomalies['underperformers'].append({
                      'strategy_id': strategy_id,
                      'symbol': symbol,
                      'current_sharpe': current_sharpe,
                      'historical_mean': hist_sharpe_mean,
                      'z_score': (current_sharpe - hist_sharpe_mean) / hist_sharpe_std
                  })

              # High drawdown alert
              current_drawdown = row['max_drawdown']
              hist_drawdown_mean = hist_result['avg_historical_drawdown'] or 0
              if current_drawdown > hist_drawdown_mean * 1.5:  # 50% worse than average
                  anomalies['drawdown_alerts'].append({
                      'strategy_id': strategy_id,
                      'symbol': symbol,
                      'current_drawdown': current_drawdown,
                      'historical_mean': hist_drawdown_mean
                  })

      await connection.close()

      logger.info(f"Identified {len(anomalies['high_performers'])} high performers, "
                 f"{len(anomalies['underperformers'])} underperformers")

      return anomalies

  @task
  async def analyze_regime_impact(date: datetime) -> Dict[str, any]:
      """Analyze how regime changes impact strategy performance."""
      logger = get_run_logger()

      connection = await asyncpg.connect(
          host='timescaledb-service',
          database='trading',
          user='postgres',
          password='password'
      )

      # Get regime transitions for the day
      regime_query = """
      WITH regime_changes AS (
          SELECT 
              time,
              classifier_id,
              symbol,
              regime,
              LAG(regime) OVER (PARTITION BY classifier_id, symbol ORDER BY time) as prev_regime,
              confidence
          FROM regime_classifications
          WHERE DATE(time) = $1
      )
      SELECT * FROM regime_changes 
      WHERE prev_regime IS NOT NULL AND prev_regime != regime
      ORDER BY time
      """

      regime_transitions = await connection.fetch(regime_query, date.date())

      # Analyze strategy performance around regime changes
      regime_impact = {
          'transitions': len(regime_transitions),
          'regime_performance': {},
          'transition_events': []
      }

      for transition in regime_transitions:
          transition_time = transition['time']
          classifier_id = transition['classifier_id']
          symbol = transition['symbol']
          new_regime = transition['regime']
          old_regime = transition['prev_regime']

          # Get strategy performance before/after transition
          perf_query = """
          SELECT 
              strategy_id,
              AVG(CASE WHEN time <= $1 THEN sharpe_ratio_20d END) as sharpe_before,
              AVG(CASE WHEN time > $1 THEN sharpe_ratio_20d END) as sharpe_after,
              COUNT(*) as data_points
          FROM strategy_performance
          WHERE symbol = $2
              AND time >= $1 - INTERVAL '2 hours'
              AND time <= $1 + INTERVAL '2 hours'
          GROUP BY strategy_id
          HAVING COUNT(*) >= 4
          """

          perf_results = await connection.fetch(perf_query, transition_time, symbol)

          transition_impact = {
              'time': transition_time,
              'classifier_id': classifier_id,
              'symbol': symbol,
              'old_regime': old_regime,
              'new_regime': new_regime,
              'confidence': transition['confidence'],
              'strategy_impacts': []
          }

          for perf in perf_results:
              if perf['sharpe_before'] and perf['sharpe_after']:
                  impact = perf['sharpe_after'] - perf['sharpe_before']
                  transition_impact['strategy_impacts'].append({
                      'strategy_id': perf['strategy_id'],
                      'sharpe_before': perf['sharpe_before'],
                      'sharpe_after': perf['sharpe_after'],
                      'impact': impact
                  })

          regime_impact['transition_events'].append(transition_impact)

      await connection.close()

      logger.info(f"Analyzed {len(regime_transitions)} regime transitions")
      return regime_impact

  @task
  def generate_performance_visualizations(df: pd.DataFrame, 
                                         anomalies: Dict, 
                                         regime_impact: Dict) -> str:
      """Generate comprehensive performance visualizations."""
      logger = get_run_logger()

      # Create subplots
      fig = make_subplots(
          rows=3, cols=2,
          subplot_titles=(
              'Strategy Performance Distribution',
              'Sharpe Ratio vs Drawdown',
              'Top/Bottom Performers',
              'Regime Transition Impact',
              'Daily PnL Distribution',
              'Trade Frequency vs Performance'
          ),
          specs=[
              [{"secondary_y": True}, {"type": "scatter"}],
              [{"type": "bar"}, {"type": "scatter"}],
              [{"type": "histogram"}, {"type": "scatter"}]
          ]
      )

      # 1. Performance distribution
      fig.add_trace(
          go.Histogram(x=df['avg_sharpe'], name='Sharpe Distribution', nbinsx=30),
          row=1, col=1
      )

      # 2. Risk-Return scatter
      fig.add_trace(
          go.Scatter(
              x=df['max_drawdown'],
              y=df['avg_sharpe'],
              mode='markers',
              text=df['strategy_id'],
              name='Strategies',
              marker=dict(
                  size=df['trade_count']/10,
                  color=df['daily_pnl'],
                  colorscale='RdYlGn',
                  colorbar=dict(title="Daily PnL")
              )
          ),
          row=1, col=2
      )

      # 3. Top/Bottom performers
      top_performers = df.nlargest(10, 'avg_sharpe')
      bottom_performers = df.nsmallest(10, 'avg_sharpe')

      fig.add_trace(
          go.Bar(
              x=top_performers['strategy_id'],
              y=top_performers['avg_sharpe'],
              name='Top Performers',
              marker_color='green'
          ),
          row=2, col=1
      )

      # 4. Regime impact visualization
      if regime_impact['transition_events']:
          impact_data = []
          for event in regime_impact['transition_events']:
              for impact in event['strategy_impacts']:
                  impact_data.append({
                      'strategy': impact['strategy_id'],
                      'impact': impact['impact'],
                      'regime_change': f"{event['old_regime']} → {event['new_regime']}"
                  })

          if impact_data:
              impact_df = pd.DataFrame(impact_data)
              fig.add_trace(
                  go.Scatter(
                      x=impact_df['strategy'],
                      y=impact_df['impact'],
                      mode='markers',
                      text=impact_df['regime_change'],
                      name='Regime Impact',
                      marker=dict(color='blue')
                  ),
                  row=2, col=2
              )

      # 5. Daily PnL distribution
      fig.add_trace(
          go.Histogram(x=df['daily_pnl'], name='PnL Distribution', nbinsx=30),
          row=3, col=1
      )

      # 6. Trade frequency vs performance
      fig.add_trace(
          go.Scatter(
              x=df['trade_count'],
              y=df['avg_sharpe'],
              mode='markers',
              text=df['strategy_id'],
              name='Frequency vs Performance',
              marker=dict(color='purple')
          ),
          row=3, col=2
      )

      # Update layout
      fig.update_layout(
          height=1200,
          title_text="Daily Strategy Performance Analysis",
          showlegend=False
      )

      # Save to file
      html_content = fig.to_html()

      logger.info("Generated performance visualizations")
      return html_content

  @task
  def send_daily_report(performance_df: pd.DataFrame,
                       anomalies: Dict,
                       regime_impact: Dict,
                       visualizations_html: str):
      """Send daily performance report via email."""
      logger = get_run_logger()

      # Prepare email content
      email_body = f"""
      <html>
      <head><title>Daily Strategy Performance Report</title></head>
      <body>
      <h1>Daily Strategy Performance Report - {datetime.now().strftime('%Y-%m-%d')}</h1>
      
      <h2>Executive Summary</h2>
      <ul>
          <li>Total Strategies Analyzed: {len(performance_df)}</li>
          <li>Average Sharpe Ratio: {performance_df['avg_sharpe'].mean():.3f}</li>
          <li>Total Daily PnL: ${performance_df['daily_pnl'].sum():,.2f}</li>
          <li>Strategies with Positive PnL: {(performance_df['daily_pnl'] > 0).sum()}</li>
          <li>Regime Transitions: {regime_impact['transitions']}</li>
      </ul>
      
      <h2>Performance Anomalies</h2>
      <h3>High Performers ({len(anomalies['high_performers'])} strategies)</h3>
      <ul>
      """

      for performer in anomalies['high_performers'][:5]:  # Top 5
          email_body += f"<li>{performer['strategy_id']}: Sharpe {performer['current_sharpe']:.3f} (Z-score: {performer['z_score']:.2f})</li>"

      email_body += """
      </ul>
      
      <h3>Underperformers ({len(anomalies['underperformers'])} strategies)</h3>
      <ul>
      """.format(len=len(anomalies['underperformers']))

      for performer in anomalies['underperformers'][:5]:  # Bottom 5
          email_body += f"<li>{performer['strategy_id']}: Sharpe {performer['current_sharpe']:.3f} (Z-score: {performer['z_score']:.2f})</li>"

      email_body += f"""
      </ul>
      
      <h2>Top Performing Strategies</h2>
      <table border="1" style="border-collapse: collapse;">
      <tr><th>Strategy ID</th><th>Sharpe Ratio</th><th>Daily PnL</th><th>Max Drawdown</th><th>Win Rate</th></tr>
      """

      top_strategies = performance_df.nlargest(10, 'avg_sharpe')
      for _, row in top_strategies.iterrows():
          email_body += f"""
          <tr>
              <td>{row['strategy_id']}</td>
              <td>{row['avg_sharpe']:.3f}</td>
              <td>${row['daily_pnl']:,.2f}</td>
              <td>{row['max_drawdown']:.3f}</td>
              <td>{row['win_rate']:.1%}</td>
          </tr>
          """

      email_body += """
      </table>
      
      <h2>Interactive Visualizations</h2>
      <p>Please see the attached HTML file for interactive charts and detailed analysis.</p>
      
      </body>
      </html>
      """

      # Send email
      msg = MIMEMultipart()
      msg['From'] = 'trading-system@company.com'
      msg['To'] = 'trading-team@company.com'
      msg['Subject'] = f'Daily Strategy Performance Report - {datetime.now().strftime("%Y-%m-%d")}'

      msg.attach(MIMEText(email_body, 'html'))

      # Attach visualizations
      attachment = MIMEBase('application', 'octet-stream')
      attachment.set_payload(visualizations_html.encode())
      attachment.add_header(
          'Content-Disposition',
          f'attachment; filename="performance_charts_{datetime.now().strftime("%Y%m%d")}.html"'
      )
      msg.attach(attachment)

      # Send via SMTP
      try:
          server = smtplib.SMTP('smtp.company.com', 587)
          server.starttls()
          server.login('trading-system@company.com', 'password')
          text = msg.as_string()
          server.sendmail('trading-system@company.com', 'trading-team@company.com', text)
          server.quit()
          logger.info("Daily report sent successfully")
      except Exception as e:
          logger.error(f"Failed to send email: {e}")

  @flow(task_runner=ConcurrentTaskRunner())
  async def daily_performance_analysis(date: datetime = None):
      """Main flow for daily performance analysis."""
      if date is None:
          date = datetime.now() - timedelta(days=1)  # Previous day

      logger = get_run_logger()
      logger.info(f"Starting daily performance analysis for {date.date()}")

      # Extract performance data
      performance_df = await extract_daily_performance_data(date)

      # Identify anomalies
      anomalies = await identify_performance_anomalies(performance_df)

      # Analyze regime impact
      regime_impact = await analyze_regime_impact(date)

      # Generate visualizations
      visualizations = generate_performance_visualizations(
          performance_df, anomalies, regime_impact
      )

      # Send report
      send_daily_report(performance_df, anomalies, regime_impact, visualizations)

      logger.info("Daily performance analysis completed")

  # Create deployment
  deployment = Deployment.build_from_flow(
      flow=daily_performance_analysis,
      name="daily-strategy-analysis",
      schedule=CronSchedule(cron="0 7 * * *"),  # 7 AM daily
      tags=["analytics", "daily", "performance"]
  )

  if __name__ == "__main__":
      deployment.apply()

  2. Strategy Discovery and Optimization

  @flow
  async def strategy_optimization_workflow():
      """Automated strategy discovery and optimization workflow."""

      logger = get_run_logger()

      # 1. Identify underperforming strategies
      underperformers = await identify_underperforming_strategies()

      # 2. Generate parameter variations
      variations = await generate_parameter_variations(underperformers)

      # 3. Run backtests for variations
      backtest_results = await run_parameter_backtests(variations)

      # 4. Analyze results and recommend optimizations
      recommendations = await analyze_optimization_results(backtest_results)

      # 5. Generate optimization report
      await generate_optimization_report(recommendations)

      logger.info("Strategy optimization workflow completed")

  @task
  async def identify_underperforming_strategies(lookback_days: int = 30) -> List[str]:
      """Identify strategies that have been underperforming."""

      connection = await asyncpg.connect(
          host='timescaledb-service',
          database='trading',
          user='postgres',
          password='password'
      )

      query = """
      WITH strategy_performance AS (
          SELECT 
              strategy_id,
              AVG(sharpe_ratio_20d) as avg_sharpe,
              STDDEV(sharpe_ratio_20d) as sharpe_stability,
              SUM(pnl) as total_pnl,
              MAX(max_drawdown) as worst_drawdown
          FROM strategy_performance
          WHERE time >= NOW() - INTERVAL '%s days'
          GROUP BY strategy_id
          HAVING COUNT(*) > 100  -- Sufficient data
      ),
      performance_percentiles AS (
          SELECT 
              strategy_id,
              avg_sharpe,
              PERCENT_RANK() OVER (ORDER BY avg_sharpe) as sharpe_percentile,
              PERCENT_RANK() OVER (ORDER BY total_pnl) as pnl_percentile
          FROM strategy_performance
      )
      SELECT strategy_id
      FROM performance_percentiles
      WHERE sharpe_percentile < 0.25  -- Bottom quartile
          OR pnl_percentile < 0.25
      ORDER BY sharpe_percentile
      """

      rows = await connection.fetch(query, lookback_days)
      await connection.close()

      return [row['strategy_id'] for row in rows]

  @task
  async def generate_parameter_variations(strategy_ids: List[str]) -> Dict[str, List]:
      """Generate parameter variations for optimization."""

      variations = {}

      for strategy_id in strategy_ids:
          # Extract strategy type and current parameters
          strategy_type = strategy_id.split('_')[1] if '_' in strategy_id else strategy_id

          # Generate parameter grids based on strategy type
          if strategy_type == 'momentum':
              variations[strategy_id] = generate_momentum_variations()
          elif strategy_type == 'mean_reversion':
              variations[strategy_id] = generate_mean_reversion_variations()
          # Add more strategy types as needed

      return variations

  def generate_momentum_variations() -> List[Dict]:
      """Generate parameter variations for momentum strategies."""
      import itertools

      rsi_periods = [10, 14, 20, 25]
      rsi_overbought = [65, 70, 75, 80]
      rsi_oversold = [20, 25, 30, 35]
      momentum_windows = [5, 10, 15, 20]

      variations = []
      for combo in itertools.product(rsi_periods, rsi_overbought, rsi_oversold, momentum_windows):
          if combo[1] > combo[2] + 20:  # Ensure reasonable spread
              variations.append({
                  'rsi_period': combo[0],
                  'rsi_overbought': combo[1],
                  'rsi_oversold': combo[2],
                  'momentum_window': combo[3]
              })

      return variations[:50]  # Limit to 50 variations

  @task
  async def run_parameter_backtests(variations: Dict[str, List]) -> Dict:
      """Run backtests for parameter variations."""

      # This would integrate with your backtesting engine
      # For now, we'll simulate the results

      backtest_results = {}

      for strategy_id, param_sets in variations.items():
          strategy_results = []

          for params in param_sets:
              # Simulate backtest results
              # In reality, this would call your backtesting system
              result = {
                  'parameters': params,
                  'sharpe_ratio': np.random.normal(0.5, 0.3),
                  'total_return': np.random.normal(0.1, 0.15),
                  'max_drawdown': np.random.uniform(0.05, 0.25),
                  'win_rate': np.random.uniform(0.4, 0.7),
                  'trade_count': np.random.randint(50, 500)
              }
              strategy_results.append(result)

          backtest_results[strategy_id] = strategy_results

      return backtest_results

  @task
  async def analyze_optimization_results(backtest_results: Dict) -> Dict:
      """Analyze backtest results and generate recommendations."""

      recommendations = {}

      for strategy_id, results in backtest_results.items():
          if not results:
              continue

          # Find best performing parameter set
          best_result = max(results, key=lambda x: x['sharpe_ratio'])

          # Calculate improvement potential
          current_performance = await get_current_strategy_performance(strategy_id)
          potential_improvement = best_result['sharpe_ratio'] - current_performance.get('sharpe_ratio', 0)

          recommendations[strategy_id] = {
              'current_performance': current_performance,
              'best_parameters': best_result['parameters'],
              'expected_improvement': {
                  'sharpe_ratio': potential_improvement,
                  'total_return': best_result['total_return'],
                  'max_drawdown': best_result['max_drawdown']
              },
              'confidence': calculate_optimization_confidence(results),
              'recommended_action': 'optimize' if potential_improvement > 0.1 else 'monitor'
          }

      return recommendations

  def calculate_optimization_confidence(results: List[Dict]) -> float:
      """Calculate confidence in optimization recommendation."""

      if len(results) < 10:
          return 0.0

      sharpe_ratios = [r['sharpe_ratio'] for r in results]
      best_sharpe = max(sharpe_ratios)

      # Count how many results are within 10% of the best
      near_optimal_count = sum(1 for s in sharpe_ratios if s >= best_sharpe * 0.9)

      # Confidence based on consistency of good results
      return min(near_optimal_count / len(results), 1.0)

  ---
  Performance Optimization

  1. Database Optimization

  Query Optimization Strategies

  -- 1. Specialized indexes for common query patterns
  CREATE INDEX CONCURRENTLY idx_strategy_perf_time_strategy_symbol
  ON strategy_performance (time DESC, strategy_id, symbol)
  INCLUDE (sharpe_ratio_20d, pnl, max_drawdown);

  -- 2. Partial indexes for active strategies
  CREATE INDEX CONCURRENTLY idx_active_strategies_recent
  ON strategy_performance (strategy_id, time DESC, sharpe_ratio_20d)
  WHERE time >= NOW() - INTERVAL '30 days'
    AND sharpe_ratio_20d IS NOT NULL;

  -- 3. Expression indexes for computed queries
  CREATE INDEX CONCURRENTLY idx_strategy_performance_score
  ON strategy_performance ((sharpe_ratio_20d / NULLIF(max_drawdown, 0)));

  -- 4. Optimized continuous aggregates
  CREATE MATERIALIZED VIEW strategy_hourly_performance
  WITH (timescaledb.continuous) AS
  SELECT
      time_bucket('1 hour', time) AS hour,
      strategy_id,
      symbol,
      FIRST(sharpe_ratio_20d, time) as latest_sharpe,
      AVG(pnl) as avg_hourly_pnl,
      SUM(trade_count_20d) as hourly_trades,
      MAX(max_drawdown) as max_hourly_drawdown
  FROM strategy_performance
  GROUP BY hour, strategy_id, symbol;

  -- Refresh every 5 minutes
  SELECT add_continuous_aggregate_policy('strategy_hourly_performance',
      start_offset => INTERVAL '2 hours',
      end_offset => INTERVAL '5 minutes',
      schedule_interval => INTERVAL '5 minutes');

  Connection Pooling Configuration

  import asyncpg
  import asyncio
  from asyncio import Queue
  from typing import Optional
  import time

  class OptimizedConnectionPool:
      """High-performance connection pool with health monitoring."""

      def __init__(self, 
                   dsn: str,
                   min_size: int = 10,
                   max_size: int = 50,
                   max_queries: int = 50000,
                   max_inactive_connection_lifetime: float = 300.0):
          self.dsn = dsn
          self.min_size = min_size
          self.max_size = max_size
          self.max_queries = max_queries
          self.max_inactive_connection_lifetime = max_inactive_connection_lifetime

          self._pool: Optional[asyncpg.Pool] = None
          self._health_check_task: Optional[asyncio.Task] = None

      async def initialize(self):
          """Initialize connection pool with optimal settings."""

          self._pool = await asyncpg.create_pool(
              self.dsn,
              min_size=self.min_size,
              max_size=self.max_size,
              max_queries=self.max_queries,
              max_inactive_connection_lifetime=self.max_inactive_connection_lifetime,
              command_timeout=60,
              server_settings={
                  'application_name': 'trading_analytics',
                  'tcp_keepalives_idle': '600',
                  'tcp_keepalives_interval': '30',
                  'tcp_keepalives_count': '3',
              }
          )

          # Start health monitoring
          self._health_check_task = asyncio.create_task(self._health_monitor())

      async def _health_monitor(self):
          """Monitor pool health and recreate if necessary."""

          while True:
              try:
                  await asyncio.sleep(30)  # Check every 30 seconds

                  # Test pool health
                  async with self._pool.acquire() as conn:
                      await conn.fetchval('SELECT 1')

                  # Log pool statistics
                  print(f"Pool stats: size={self._pool.get_size()}, "
                        f"idle={self._pool.get_idle_size()}")

              except Exception as e:
                  print(f"Pool health check failed: {e}")
                  # Could implement pool recreation logic here

      async def execute_query(self, query: str, *args, timeout: float = 30.0):
          """Execute query with retry logic."""

          max_retries = 3
          for attempt in range(max_retries):
              try:
                  async with asyncio.timeout(timeout):
                      async with self._pool.acquire() as conn:
                          return await conn.fetch(query, *args)

              except asyncio.TimeoutError:
                  if attempt == max_retries - 1:
                      raise
                  await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff

              except asyncpg.PostgresError as e:
                  if attempt == max_retries - 1:
                      raise
                  if 'connection' in str(e).lower():
                      await asyncio.sleep(0.5)
                  else:
                      raise

      async def close(self):
          """Clean shutdown of pool."""
          if self._health_check_task:
              self._health_check_task.cancel()

          if self._pool:
              await self._pool.close()

  2. Caching Strategy

  import redis.asyncio as redis
  import json
  import pickle
  from typing import Any, Optional, Union
  import hashlib
  from datetime import timedelta
  import asyncio

  class AnalyticsCache:
      """Multi-level caching for analytics queries."""

      def __init__(self, redis_url: str = "redis://localhost:6379"):
          self.redis = redis.from_url(redis_url, decode_responses=False)
          self.local_cache = {}
          self.local_cache_ttl = {}

      async def get(self, key: str, use_local: bool = True) -> Optional[Any]:
          """Get value from cache with fallback hierarchy."""

          # 1. Check local cache first (fastest)
          if use_local and key in self.local_cache:
              if key not in self.local_cache_ttl or \
                 time.time() < self.local_cache_ttl[key]:
                  return self.local_cache[key]
              else:
                  # Expired, remove from local cache
                  del self.local_cache[key]
                  del self.local_cache_ttl[key]

          # 2. Check Redis cache
          try:
              cached_data = await self.redis.get(key)
              if cached_data:
                  data = pickle.loads(cached_data)

                  # Store in local cache for future requests
                  if use_local:
                      self.local_cache[key] = data
                      self.local_cache_ttl[key] = time.time() + 300  # 5 min local TTL

                  return data
          except Exception as e:
              print(f"Redis cache error: {e}")

          return None

      async def set(self, key: str, value: Any, ttl: int = 3600, use_local: bool = True):
          """Set value in cache."""

          try:
              # Store in Redis
              serialized = pickle.dumps(value)
              await self.redis.setex(key, ttl, serialized)

              # Store in local cache
              if use_local:
                  self.local_cache[key] = value
                  self.local_cache_ttl[key] = time.time() + min(ttl, 300)

          except Exception as e:
              print(f"Cache set error: {e}")

      def cache_key(self, prefix: str, **kwargs) -> str:
          """Generate consistent cache key."""

          # Sort kwargs for consistent key generation
          sorted_params = sorted(kwargs.items())
          param_string = json.dumps(sorted_params, sort_keys=True, default=str)
          hash_suffix = hashlib.md5(param_string.encode()).hexdigest()[:8]

          return f"{prefix}:{hash_suffix}"

      async def cached_query(self, key: str, query_func, ttl: int = 3600, **query_kwargs):
          """Execute query with caching."""

          # Try to get from cache
          result = await self.get(key)
          if result is not None:
              return result

          # Execute query
          result = await query_func(**query_kwargs)

          # Cache result
          await self.set(key, result, ttl)

          return result

  # Usage example
  cache = AnalyticsCache()

  async def get_strategy_rankings_cached(lookback_days: int = 30, 
                                       regime: str = None,
                                       **kwargs):
      """Get strategy rankings with caching."""

      cache_key = cache.cache_key(
          "strategy_rankings",
          lookback_days=lookback_days,
          regime=regime,
          **kwargs
      )

      async def query_rankings():
          # Your actual database query here
          return await execute_strategy_ranking_query(lookback_days, regime, **kwargs)

      return await cache.cached_query(
          cache_key,
          query_rankings,
          ttl=1800,  # 30 minutes
          lookback_days=lookback_days,
          regime=regime,
          **kwargs
      )

  3. Parallel Processing

  import asyncio
  import aiofiles
  from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
  from typing import List, Dict, Any, Callable
  import pandas as pd
  import numpy as np
  from multiprocessing import cpu_count

  class ParallelAnalyticsProcessor:
      """High-performance parallel processing for analytics."""

      def __init__(self, max_workers: int = None):
          self.max_workers = max_workers or cpu_count()
          self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
          self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers * 2)

      async def process_strategies_parallel(self, 
                                          strategy_ids: List[str],
                                          analysis_func: Callable,
                                          chunk_size: int = 100) -> List[Dict]:
          """Process strategies in parallel chunks."""

          # Split into chunks
          chunks = [strategy_ids[i:i + chunk_size]
                   for i in range(0, len(strategy_ids), chunk_size)]

          # Process chunks in parallel
          tasks = []
          for chunk in chunks:
              task = asyncio.create_task(
                  self._process_strategy_chunk(chunk, analysis_func)
              )
              tasks.append(task)

          # Wait for all chunks to complete
          results = await asyncio.gather(*tasks)

          # Flatten results
          return [item for sublist in results for item in sublist]

      async def _process_strategy_chunk(self, 
                                      strategy_ids: List[str],
                                      analysis_func: Callable) -> List[Dict]:
          """Process a single chunk of strategies."""

          loop = asyncio.get_event_loop()

          # Use process executor for CPU-intensive work
          return await loop.run_in_executor(
              self.process_executor,
              self._cpu_intensive_analysis,
              strategy_ids,
              analysis_func
          )

      def _cpu_intensive_analysis(self, 
                                 strategy_ids: List[str],
                                 analysis_func: Callable) -> List[Dict]:
          """CPU-intensive analysis in separate process."""

          results = []
          for strategy_id in strategy_ids:
              try:
                  result = analysis_func(strategy_id)
                  results.append(result)
              except Exception as e:
                  print(f"Error processing {strategy_id}: {e}")
                  results.append({'strategy_id': strategy_id, 'error': str(e)})

          return results

      async def parallel_file_processing(self, 
                                       file_paths: List[str],
                                       process_func: Callable) -> List[Any]:
          """Process multiple files in parallel."""

          async def process_file(file_path: str):
              async with aiofiles.open(file_path, 'rb') as f:
                  content = await f.read()
                  loop = asyncio.get_event_loop()
                  return await loop.run_in_executor(
                      self.thread_executor,
                      process_func,
                      content
                  )

          tasks = [process_file(path) for path in file_paths]
          return await asyncio.gather(*tasks)

      async def close(self):
          """Clean shutdown of executors."""
          self.process_executor.shutdown(wait=True)
          self.thread_executor.shutdown(wait=True)

  # Example usage
  processor = ParallelAnalyticsProcessor()

  async def analyze_strategy_performance_parallel(strategy_ids: List[str]):
      """Analyze strategy performance using parallel processing."""

      def single_strategy_analysis(strategy_id: str) -> Dict:
          """Analyze single strategy (CPU-intensive)."""

          # Simulate complex calculations
          # In reality, this would do statistical analysis, risk calculations, etc.
          np.random.seed(hash(strategy_id) % 2**31)

          returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
          cumulative_returns = np.cumprod(1 + returns)

          # Calculate metrics
          total_return = cumulative_returns[-1] - 1
          volatility = np.std(returns) * np.sqrt(252)
          sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility > 0 else 0

          # Calculate maximum drawdown
          peak = np.maximum.accumulate(cumulative_returns)
          drawdown = (cumulative_returns - peak) / peak
          max_drawdown = np.min(drawdown)

          return {
              'strategy_id': strategy_id,
              'total_return': total_return,
              'volatility': volatility,
              'sharpe_ratio': sharpe_ratio,
              'max_drawdown': abs(max_drawdown),
              'calmar_ratio': total_return / abs(max_drawdown) if max_drawdown != 0 else 0
          }

      results = await processor.process_strategies_parallel(
          strategy_ids,
          single_strategy_analysis,
          chunk_size=50
      )

      return results

  ---
  Machine Learning Integration

  1. Feature Engineering Pipeline

  import pandas as pd
  import numpy as np
  from sklearn.preprocessing import StandardScaler, RobustScaler
  from sklearn.feature_selection import SelectKBest, f_regression
  from typing import Dict, List, Tuple, Optional
  import asyncpg
  from datetime import datetime, timedelta
  import asyncio

  class TradingFeatureEngineer:
      """Advanced feature engineering for trading strategies."""

      def __init__(self, connection_pool):
          self.pool = connection_pool
          self.scalers = {}
          self.feature_selectors = {}

      async def extract_base_features(self, 
                                    start_date: datetime,
                                    end_date: datetime,
                                    symbols: List[str] = None) -> pd.DataFrame:
          """Extract base features from strategy performance data."""

          symbols_filter = "AND symbol = ANY($3)" if symbols else ""
          symbol_param = symbols if symbols else None

          query = f"""
          SELECT 
              time,
              strategy_id,
              symbol,
              pnl,
              sharpe_ratio_20d,
              max_drawdown,
              win_rate_20d,
              trade_count_20d,
              volatility_20d,
              returns
          FROM strategy_performance
          WHERE time >= $1 AND time <= $2
          {symbols_filter}
          ORDER BY time, strategy_id, symbol
          """

          async with self.pool.acquire() as conn:
              if symbol_param:
                  rows = await conn.fetch(query, start_date, end_date, symbol_param)
              else:
                  rows = await conn.fetch(query, start_date, end_date)

          return pd.DataFrame([dict(row) for row in rows])

      async def extract_regime_features(self,
                                      start_date: datetime,
                                      end_date: datetime,
                                      symbols: List[str] = None) -> pd.DataFrame:
          """Extract regime classification features."""

          symbols_filter = "AND symbol = ANY($3)" if symbols else ""
          symbol_param = symbols if symbols else None

          query = f"""
          WITH regime_windows AS (
              SELECT 
                  time,
                  symbol,
                  classifier_id,
                  regime,
                  confidence,
                  -- Previous regime
                  LAG(regime) OVER w as prev_regime,
                  -- Regime duration
                  time - LAG(time) OVER (PARTITION BY classifier_id, symbol, regime ORDER BY time) as regime_duration,
                  -- Regime stability (confidence trend)
                  AVG(confidence) OVER (w ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as regime_stability_5,
                  -- Transition probability
                  CASE WHEN LAG(regime) OVER w != regime THEN 1 ELSE 0 END as regime_transition
              FROM regime_classifications
              WHERE time >= $1 AND time <= $2
              {symbols_filter}
              WINDOW w AS (PARTITION BY classifier_id, symbol ORDER BY time)
          )
          SELECT 
              time,
              symbol,
              classifier_id,
              regime,
              confidence,
              prev_regime,
              EXTRACT(EPOCH FROM regime_duration)/3600 as regime_duration_hours,
              regime_stability_5,
              regime_transition
          FROM regime_windows
          ORDER BY time, symbol, classifier_id
          """

          async with self.pool.acquire() as conn:
              if symbol_param:
                  rows = await conn.fetch(query, start_date, end_date, symbol_param)
              else:
                  rows = await conn.fetch(query, start_date, end_date)

          return pd.DataFrame([dict(row) for row in rows])

      def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
          """Create technical analysis features."""

          df = df.copy()

          # Rolling statistics for performance metrics
          windows = [5, 10, 20, 50]

          for window in windows:
              # Rolling Sharpe ratio statistics
              df[f'sharpe_ma_{window}'] = df.groupby('strategy_id')['sharpe_ratio_20d'].transform(
                  lambda x: x.rolling(window, min_periods=1).mean()
              )
              df[f'sharpe_std_{window}'] = df.groupby('strategy_id')['sharpe_ratio_20d'].transform(
                  lambda x: x.rolling(window, min_periods=1).std()
              )

              # Rolling PnL statistics
              df[f'pnl_ma_{window}'] = df.groupby('strategy_id')['pnl'].transform(
                  lambda x: x.rolling(window, min_periods=1).mean()
              )
              df[f'pnl_volatility_{window}'] = df.groupby('strategy_id')['pnl'].transform(
                  lambda x: x.rolling(window, min_periods=1).std()
              )

              # Rolling drawdown statistics
              df[f'drawdown_max_{window}'] = df.groupby('strategy_id')['max_drawdown'].transform(
                  lambda x: x.rolling(window, min_periods=1).max()
              )

          # Momentum features
          df['sharpe_momentum_5'] = df.groupby('strategy_id')['sharpe_ratio_20d'].transform(
              lambda x: x.diff(5)
          )
          df['pnl_momentum_10'] = df.groupby('strategy_id')['pnl'].transform(
              lambda x: x.rolling(10).sum()
          )

          # Volatility features
          df['performance_volatility'] = df.groupby('strategy_id')['returns'].transform(
              lambda x: x.rolling(20, min_periods=1).std()
          )

          # Risk-adjusted features
          df['information_ratio'] = df['sharpe_ratio_20d'] / (df['volatility_20d'] + 1e-6)
          df['calmar_ratio'] = df['pnl'] / (df['max_drawdown'] + 1e-6)

          return df

      def create_regime_features(self, performance_df: pd.DataFrame, 
                                regime_df: pd.DataFrame) -> pd.DataFrame:
          """Create regime-based features."""

          # Merge performance and regime data
          merged = pd.merge_asof(
              performance_df.sort_values('time'),
              regime_df.sort_values('time'),
              on='time',
              by='symbol',
              direction='backward'
          )

          # Create regime transition features
          merged['regime_changed'] = (merged['regime'] != merged['prev_regime']).astype(int)
          merged['time_since_regime_change'] = merged.groupby(['strategy_id', 'regime']).cumcount()

          # Regime stability features
          merged['regime_confidence_trend'] = merged.groupby(['strategy_id', 'regime'])['confidence'].transform(
              lambda x: x.diff()
          )

          # Performance in current regime
          merged['performance_in_regime'] = merged.groupby(['strategy_id', 'regime'])['pnl'].transform('cumsum')
          merged['sharpe_in_regime'] = merged.groupby(['strategy_id', 'regime'])['sharpe_ratio_20d'].transform('mean')

          # Cross-regime performance comparison
          merged['sharpe_vs_other_regimes'] = merged.groupby('strategy_id')['sharpe_ratio_20d'].transform('mean') - \
                                            merged['sharpe_in_regime']

          return merged

      def create_cross_strategy_features(self, df: pd.DataFrame) -> pd.DataFrame:
          """Create features comparing strategies to each other."""

          df = df.copy()

          # Strategy type groupings
          df['strategy_type'] = df['strategy_id'].str.extract(r'([^_]+_[^_]+)')[0]

          # Relative performance within strategy type
          df['sharpe_percentile_in_type'] = df.groupby(['time', 'strategy_type'])['sharpe_ratio_20d'].transform(
              lambda x: x.rank(pct=True)
          )

          # Market-wide performance metrics
          df['market_avg_sharpe'] = df.groupby('time')['sharpe_ratio_20d'].transform('mean')
          df['market_std_sharpe'] = df.groupby('time')['sharpe_ratio_20d'].transform('std')

          # Relative performance vs market
          df['sharpe_vs_market'] = df['sharpe_ratio_20d'] - df['market_avg_sharpe']
          df['sharpe_zscore'] = (df['sharpe_ratio_20d'] - df['market_avg_sharpe']) / (df['market_std_sharpe'] + 1e-6)

          # Correlation-based features (simplified)
          df['performance_divergence'] = abs(df['sharpe_vs_market'])

          return df

      def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
          """Create time-based features."""

          df = df.copy()
          df['time'] = pd.to_datetime(df['time'])

          # Basic time features
          df['hour'] = df['time'].dt.hour
          df['day_of_week'] = df['time'].dt.dayofweek
          df['month'] = df['time'].dt.month
          df['quarter'] = df['time'].dt.quarter

          # Cyclical encoding for time features
          df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
          df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
          df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
          df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
          df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
          df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

          # Market session features
          df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] <= 16) &
                                 (df['day_of_week'] < 5)).astype(int)
          df['time_to_close'] = np.where(df['is_market_open'], 16 - df['hour'], 0)

          return df

      async def create_feature_dataset(self,
                                     start_date: datetime,
                                     end_date: datetime,
                                     symbols: List[str] = None,
                                     target_column: str = 'sharpe_ratio_20d') -> Tuple[pd.DataFrame, pd.DataFrame]:
          """Create complete feature dataset for ML."""

          # Extract base data
          performance_df = await self.extract_base_features(start_date, end_date, symbols)
          regime_df = await self.extract_regime_features(start_date, end_date, symbols)

          if performance_df.empty:
              raise ValueError("No performance data found for the specified period")

          # Create features
          df_with_technical = self.create_technical_features(performance_df)

          if not regime_df.empty:
              df_with_regime = self.create_regime_features(df_with_technical, regime_df)
          else:
              df_with_regime = df_with_technical

          df_with_cross_strategy = self.create_cross_strategy_features(df_with_regime)
          final_df = self.create_time_features(df_with_cross_strategy)

          # Prepare features and target
          feature_columns = [col for col in final_df.columns
                            if col not in ['time', 'strategy_id', 'symbol', target_column]]

          # Remove rows with NaN in target
          final_df = final_df.dropna(subset=[target_column])

          X = final_df[feature_columns]
          y = final_df[[target_column, 'strategy_id', 'time']]  # Include metadata

          return X, y

      def preprocess_features(self, X: pd.DataFrame, 
                            fit_scalers: bool = True) -> pd.DataFrame:
          """Preprocess features for ML."""

          X_processed = X.copy()

          # Handle missing values
          X_processed = X_processed.fillna(X_processed.median())

          # Separate numeric and categorical columns
          numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
          categorical_columns = X_processed.select_dtypes(exclude=[np.number]).columns

          # Scale numeric features
          if fit_scalers:
              self.scalers['robust'] = RobustScaler()
              X_processed[numeric_columns] = self.scalers['robust'].fit_transform(X_processed[numeric_columns])
          else:
              if 'robust' in self.scalers:
                  X_processed[numeric_columns] = self.scalers['robust'].transform(X_processed[numeric_columns])

          # Encode categorical features (if any)
          for col in categorical_columns:
              if col in X_processed.columns:
                  # Simple label encoding for now
                  unique_values = X_processed[col].unique()
                  mapping = {val: idx for idx, val in enumerate(unique_values)}
                  X_processed[col] = X_processed[col].map(mapping)

          return X_processed

      def select_features(self, X: pd.DataFrame, y: pd.Series,
                         k: int = 50, fit_selector: bool = True) -> pd.DataFrame:
          """Select best features using statistical tests."""

          if fit_selector:
              self.feature_selectors['k_best'] = SelectKBest(f_regression, k=k)
              X_selected = self.feature_selectors['k_best'].fit_transform(X, y)
              selected_features = X.columns[self.feature_selectors['k_best'].get_support()]
          else:
              if 'k_best' in self.feature_selectors:
                  X_selected = self.feature_selectors['k_best'].transform(X)
                  selected_features = X.columns[self.feature_selectors['k_best'].get_support()]
              else:
                  return X

          return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

  2. Model Training and Serving

  import mlflow
  import mlflow.sklearn
  from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
  from sklearn.linear_model import Ridge, ElasticNet
  from sklearn.model_selection import TimeSeriesSplit, cross_val_score
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  import joblib
  import numpy as np
  from typing import Dict, Any, List, Tuple
  import asyncio
  import json

  class TradingMLPipeline:
      """Complete ML pipeline for trading strategy prediction."""

      def __init__(self, experiment_name: str = "strategy_performance_prediction"):
          mlflow.set_experiment(experiment_name)
          self.models = {}
          self.feature_engineer = None

      def train_ensemble_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
          """Train ensemble of models for robust predictions."""

          models = {
              'random_forest': RandomForestRegressor(
                  n_estimators=200,
                  max_depth=10,
                  min_samples_split=20,
                  min_samples_leaf=10,
                  random_state=42,
                  n_jobs=-1
              ),
              'gradient_boosting': GradientBoostingRegressor(
                  n_estimators=200,
                  learning_rate=0.1,
                  max_depth=6,
                  min_samples_split=20,
                  min_samples_leaf=10,
                  random_state=42
              ),
              'ridge': Ridge(alpha=1.0),
              'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
          }

          # Time series split for validation
          tscv = TimeSeriesSplit(n_splits=5)

          model_results = {}

          with mlflow.start_run():
              mlflow.log_param("n_features", X.shape[1])
              mlflow.log_param("n_samples", X.shape[0])

              for name, model in models.items():
                  print(f"Training {name}...")

                  # Cross-validation
                  cv_scores = cross_val_score(model, X, y, cv=tscv,
                                            scoring='neg_mean_squared_error', n_jobs=-1)
                  cv_rmse = np.sqrt(-cv_scores)

                  # Full model training
                  model.fit(X, y)

                  # Predictions
                  y_pred = model.predict(X)

                  # Metrics
                  mse = mean_squared_error(y, y_pred)
                  rmse = np.sqrt(mse)
                  mae = mean_absolute_error(y, y_pred)
                  r2 = r2_score(y, y_pred)

                  model_results[name] = {
                      'model': model,
                      'cv_rmse_mean': cv_rmse.mean(),
                      'cv_rmse_std': cv_rmse.std(),
                      'train_rmse': rmse,
                      'train_mae': mae,
                      'train_r2': r2
                  }

                  # Log to MLflow
                  with mlflow.start_run(nested=True, run_name=name):
                      mlflow.log_params(model.get_params())
                      mlflow.log_metric("cv_rmse_mean", cv_rmse.mean())
                      mlflow.log_metric("cv_rmse_std", cv_rmse.std())
                      mlflow.log_metric("train_rmse", rmse)
                      mlflow.log_metric("train_mae", mae)
                      mlflow.log_metric("train_r2", r2)

                      # Feature importance for tree-based models
                      if hasattr(model, 'feature_importances_'):
                          feature_importance = pd.DataFrame({
                              'feature': X.columns,
                              'importance': model.feature_importances_
                          }).sort_values('importance', ascending=False)

                          mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")

                      # Save model
                      mlflow.sklearn.log_model(model, f"model_{name}")

                  print(f"{name} - CV RMSE: {cv_rmse.mean():.4f} (±{cv_rmse.std():.4f})")

          self.models = {name: result['model'] for name, result in model_results.items()}
          return model_results

      def create_ensemble_predictions(self, X: pd.DataFrame, 
                                    weights: Dict[str, float] = None) -> np.ndarray:
          """Create ensemble predictions from multiple models."""

          if weights is None:
              weights = {name: 1.0 for name in self.models.keys()}

          predictions = {}
          for name, model in self.models.items():
              predictions[name] = model.predict(X)

          # Weighted average
          ensemble_pred = np.zeros(len(X))
          total_weight = sum(weights.values())

          for name, pred in predictions.items():
              weight = weights.get(name, 0) / total_weight
              ensemble_pred += weight * pred

          return ensemble_pred

      def analyze_model_performance(self, X: pd.DataFrame, y: pd.Series,
                                  strategy_ids: pd.Series) -> Dict[str, Any]:
          """Analyze model performance across different strategies."""

          analysis = {}

          for name, model in self.models.items():
              predictions = model.predict(X)

              # Overall metrics
              mse = mean_squared_error(y, predictions)
              mae = mean_absolute_error(y, predictions)
              r2 = r2_score(y, predictions)

              # Per-strategy analysis
              strategy_performance = []
              unique_strategies = strategy_ids.unique()

              for strategy_id in unique_strategies:
                  mask = strategy_ids == strategy_id
                  if mask.sum() > 10:  # Minimum samples for reliable metrics
                      y_strategy = y[mask]
                      pred_strategy = predictions[mask]

                      strategy_r2 = r2_score(y_strategy, pred_strategy)
                      strategy_mae = mean_absolute_error(y_strategy, pred_strategy)

                      strategy_performance.append({
                          'strategy_id': strategy_id,
                          'r2': strategy_r2,
                          'mae': strategy_mae,
                          'n_samples': mask.sum()
                      })

              analysis[name] = {
                  'overall_metrics': {
                      'mse': mse,
                      'rmse': np.sqrt(mse),
                      'mae': mae,
                      'r2': r2
                  },
                  'strategy_performance': strategy_performance
              }

          return analysis

  class ModelServingAPI:
      """API for serving ML model predictions."""

      def __init__(self, model_registry_uri: str):
          mlflow.set_tracking_uri(model_registry_uri)
          self.models = {}
          self.feature_engineer = TradingFeatureEngineer(None)  # Initialize with pool later

      async def load_models(self, model_names: List[str], stage: str = "Production"):
          """Load models from MLflow registry."""

          for model_name in model_names:
              try:
                  model_uri = f"models:/{model_name}/{stage}"
                  model = mlflow.sklearn.load_model(model_uri)
                  self.models[model_name] = model
                  print(f"Loaded model: {model_name}")
              except Exception as e:
                  print(f"Failed to load model {model_name}: {e}")

      async def predict_strategy_performance(self, 
                                           strategy_ids: List[str],
                                           prediction_horizon: int = 24) -> Dict[str, Any]:
          """Predict strategy performance for the next N hours."""

          # Get current features
          end_time = datetime.now()
          start_time = end_time - timedelta(days=7)  # Last 7 days for feature calculation

          X, _ = await self.feature_engineer.create_feature_dataset(
              start_time, end_time, symbols=['SPY']  # Assuming SPY for now
          )

          # Filter for requested strategies
          latest_features = X.groupby('strategy_id').tail(1)  # Get latest features per strategy
          strategy_features = latest_features[latest_features.index.isin(strategy_ids)]

          # Preprocess features
          X_processed = self.feature_engineer.preprocess_features(strategy_features, fit_scalers=False)
          X_selected = self.feature_engineer.select_features(X_processed, None, fit_selector=False)

          # Make predictions with all models
          predictions = {}
          for model_name, model in self.models.items():
              pred = model.predict(X_selected)
              predictions[model_name] = pred.tolist()

          # Create ensemble prediction
          ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)

          return {
              'predictions': predictions,
              'ensemble_prediction': ensemble_pred.tolist(),
              'strategy_ids': strategy_ids,
              'prediction_horizon_hours': prediction_horizon,
              'timestamp': datetime.now().isoformat()
          }

      async def get_feature_importance(self, model_name: str) -> Dict[str, float]:
          """Get feature importance from trained model."""

          if model_name not in self.models:
              raise ValueError(f"Model {model_name} not loaded")

          model = self.models[model_name]

          if hasattr(model, 'feature_importances_'):
              # Get feature names (would need to be stored/tracked)
              # For now, return generic names
              feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

              importance_dict = dict(zip(feature_names, model.feature_importances_))

              # Sort by importance
              sorted_importance = dict(sorted(importance_dict.items(),
                                            key=lambda x: x[1], reverse=True))

              return sorted_importance
          else:
              raise ValueError(f"Model {model_name} does not have feature importance")

  ---
  Monitoring & Observability

  1. System Monitoring

  # prometheus-config.yaml
  global:
    scrape_interval: 15s
    evaluation_interval: 15s

  rule_files:
    - "/etc/prometheus/rules/*.yml"

  alerting:
    alertmanagers:
      - static_configs:
          - targets:
            - alertmanager:9093

  scrape_configs:
    - job_name: 'trading-api'
      static_configs:
        - targets: ['trading-api:8000']
      metrics_path: /metrics
      scrape_interval: 30s

    - job_name: 'timescaledb'
      static_configs:
        - targets: ['timescaledb:9187']

    - job_name: 'kafka'
      static_configs:
        - targets: ['kafka-exporter:9308']

    - job_name: 'redis'
      static_configs:
        - targets: ['redis-exporter:9121']

    - job_name: 'trading-strategy-metrics'
      static_configs:
        - targets: ['strategy-monitor:8080']
      scrape_interval: 60s

  # strategy_monitor.py
  from prometheus_client import Counter, Histogram, Gauge, start_http_server
  import asyncpg
  import asyncio
  import time
  from datetime import datetime, timedelta

  # Prometheus metrics
  STRATEGY_SIGNALS_TOTAL = Counter('strategy_signals_total',
                                  'Total strategy signals generated',
                                  ['strategy_id', 'signal_type', 'symbol'])

  STRATEGY_PERFORMANCE = Gauge('strategy_sharpe_ratio',
                             'Current strategy Sharpe ratio',
                             ['strategy_id', 'symbol'])

  QUERY_DURATION = Histogram('database_query_duration_seconds',
                           'Database query duration',
                           ['query_type'])

  ACTIVE_STRATEGIES = Gauge('active_strategies_total',
                          'Number of active strategies')

  REGIME_TRANSITIONS = Counter('regime_transitions_total',
                             'Total regime transitions',
                             ['classifier_id', 'from_regime', 'to_regime'])

  class StrategyMonitor:
      """Monitor strategy performance and system health."""

      def __init__(self, db_url: str):
          self.db_url = db_url
          self.running = False

      async def start_monitoring(self):
          """Start monitoring loops."""
          self.running = True

          # Start Prometheus metrics server
          start_http_server(8080)

          # Start monitoring tasks
          tasks = [
              asyncio.create_task(self.monitor_strategy_performance()),
              asyncio.create_task(self.monitor_signal_generation()),
              asyncio.create_task(self.monitor_regime_changes()),
              asyncio.create_task(self.monitor_system_health())
          ]

          await asyncio.gather(*tasks)

      async def monitor_strategy_performance(self):
          """Monitor real-time strategy performance metrics."""

          while self.running:
              try:
                  start_time = time.time()

                  connection = await asyncpg.connect(self.db_url)

                  # Get latest performance metrics
                  query = """
                  SELECT 
                      strategy_id,
                      symbol,
                      sharpe_ratio_20d,
                      max_drawdown,
                      win_rate_20d
                  FROM strategy_performance
                  WHERE time >= NOW() - INTERVAL '1 hour'
                  ORDER BY time DESC
                  """

                  rows = await connection.fetch(query)
                  await connection.close()

                  # Update Prometheus metrics
                  strategy_count = 0
                  for row in rows:
                      if row['sharpe_ratio_20d'] is not None:
                          STRATEGY_PERFORMANCE.labels(
                              strategy_id=row['strategy_id'],
                              symbol=row['symbol']
                          ).set(row['sharpe_ratio_20d'])
                          strategy_count += 1

                  ACTIVE_STRATEGIES.set(strategy_count)

                  # Record query duration
                  QUERY_DURATION.labels(query_type='strategy_performance').observe(
                      time.time() - start_time
                  )

              except Exception as e:
                  print(f"Error monitoring strategy performance: {e}")

              await asyncio.sleep(60)  # Update every minute

      async def monitor_signal_generation(self):
          """Monitor signal generation rates."""

          last_check = datetime.now() - timedelta(minutes=5)

          while self.running:
              try:
                  current_time = datetime.now()

                  connection = await asyncpg.connect(self.db_url)

                  # Count signals generated since last check
                  query = """
                  SELECT 
                      payload->>'strategy_id' as strategy_id,
                      payload->>'signal_type' as signal_type,
                      payload->>'symbol' as symbol,
                      COUNT(*) as signal_count
                  FROM events
                  WHERE event_type = 'SIGNAL'
                      AND timestamp >= $1
                      AND timestamp < $2
                  GROUP BY 
                      payload->>'strategy_id',
                      payload->>'signal_type',
                      payload->>'symbol'
                  """

                  rows = await connection.fetch(query, last_check, current_time)
                  await connection.close()

                  # Update signal counters
                  for row in rows:
                      STRATEGY_SIGNALS_TOTAL.labels(
                          strategy_id=row['strategy_id'],
                          signal_type=row['signal_type'],
                          symbol=row['symbol']
                      ).inc(row['signal_count'])

                  last_check = current_time

              except Exception as e:
                  print(f"Error monitoring signal generation: {e}")

              await asyncio.sleep(300)  # Check every 5 minutes

      async def monitor_regime_changes(self):
          """Monitor regime classification changes."""

          last_check = datetime.now() - timedelta(minutes=5)

          while self.running:
              try:
                  current_time = datetime.now()

                  connection = await asyncpg.connect(self.db_url)

                  # Find regime transitions
                  query = """
                  WITH regime_changes AS (
                      SELECT 
                          payload->>'classifier_id' as classifier_id,
                          payload->>'regime' as current_regime,
                          payload->>'previous_regime' as previous_regime
                      FROM events
                      WHERE event_type = 'CLASSIFICATION'
                          AND timestamp >= $1
                          AND timestamp < $2
                          AND payload->>'previous_regime' IS NOT NULL
                          AND payload->>'regime' != payload->>'previous_regime'
                  )
                  SELECT 
                      classifier_id,
                      previous_regime,
                      current_regime,
                      COUNT(*) as transition_count
                  FROM regime_changes
                  GROUP BY classifier_id, previous_regime, current_regime
                  """

                  rows = await connection.fetch(query, last_check, current_time)
                  await connection.close()

                  # Update transition counters
                  for row in rows:
                      REGIME_TRANSITIONS.labels(
                          classifier_id=row['classifier_id'],
                          from_regime=row['previous_regime'],
                          to_regime=row['current_regime']
                      ).inc(row['transition_count'])

                  last_check = current_time

              except Exception as e:
                  print(f"Error monitoring regime changes: {e}")

              await asyncio.sleep(300)  # Check every 5 minutes

      async def monitor_system_health(self):
          """Monitor overall system health."""

          while self.running:
              try:
                  # Check database connectivity
                  connection = await asyncpg.connect(self.db_url)
                  await connection.fetchval('SELECT 1')
                  await connection.close()

                  # Add more health checks as needed
                  # - Kafka connectivity
                  # - Redis connectivity
                  # - API responsiveness

              except Exception as e:
                  print(f"System health check failed: {e}")

              await asyncio.sleep(
  ⎿  API Error: Claude's response exceeded the 32000 output token maximum. To configure this behavior, set the CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable.

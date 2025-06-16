# Strategy Analytics SQL Guide

**A comprehensive guide for analyzing trading strategies with DuckDB**

*Built from lessons learned analyzing 1,157 trading strategies*

---

## 🚨 Critical Pitfalls to Avoid

### 1. **NEVER** Calculate Sharpe Ratios Per-Trade
```sql
-- ❌ WRONG: Inflates Sharpe for low-frequency strategies
ROUND(AVG(return_bps) / STDDEV(return_bps) * SQRT(252 * trades_per_day), 2) as sharpe

-- ✅ CORRECT: Use wall-clock time (daily returns)
WITH daily_returns AS (
    SELECT 
        strategy_id,
        DATE(entry_time) as trade_date,
        SUM(net_return_bps) as daily_net_bps
    FROM trades
    GROUP BY strategy_id, DATE(entry_time)
),
all_trading_days AS (
    -- Include zero-return days between first and last trade
    SELECT strategy_id, calendar_date, COALESCE(daily_net_bps, 0) as daily_net_bps
    FROM date_range 
    LEFT JOIN daily_returns USING (strategy_id, calendar_date)
    WHERE EXTRACT(DOW FROM calendar_date) BETWEEN 1 AND 5
)
SELECT 
    strategy_id,
    AVG(daily_net_bps) / STDDEV(daily_net_bps) * SQRT(252) as correct_sharpe
FROM all_trading_days
GROUP BY strategy_id
```

### 2. **NEVER** Ignore Overnight Trades
```sql
-- ❌ Problem: Overnight gaps can destroy strategies
-- Real trading stops at 4pm, resumes at 9:30am

-- ✅ Solution: Filter out overnight trades
CASE 
    WHEN DATE(entry_time) != DATE(exit_time) THEN 'OVERNIGHT'
    WHEN EXTRACT(hour FROM entry_time) >= 16 AND EXTRACT(hour FROM exit_time) < 9 THEN 'OVERNIGHT'
    ELSE 'INTRADAY'
END as trade_type
```

### 3. **ALWAYS** Use Realistic Transaction Costs
```sql
-- Standard assumption: 0.5 basis points round-trip
net_return_bps = gross_return_bps - 0.5
```

---

## 📊 Strategy Analysis Templates

### Basic Strategy Performance
```sql
WITH trades AS (
    -- Your trade construction logic here
    SELECT strategy_id, entry_time, net_return_bps FROM your_trades
),
daily_performance AS (
    SELECT 
        strategy_id,
        DATE(entry_time) as trade_date,
        SUM(net_return_bps) as daily_net_bps,
        COUNT(*) as daily_trades
    FROM trades
    GROUP BY strategy_id, DATE(entry_time)
)
SELECT 
    strategy_id,
    COUNT(DISTINCT trade_date) as trading_days,
    SUM(daily_trades) as total_trades,
    ROUND(SUM(daily_trades)::DOUBLE / COUNT(DISTINCT trade_date), 2) as trades_per_day,
    ROUND(AVG(daily_net_bps), 3) as avg_daily_return_bps,
    ROUND(STDDEV(daily_net_bps), 3) as daily_volatility_bps,
    ROUND(AVG(daily_net_bps) / NULLIF(STDDEV(daily_net_bps), 0) * SQRT(252), 2) as annual_sharpe,
    CASE WHEN AVG(daily_net_bps) > 0 THEN 'PROFITABLE' ELSE 'UNPROFITABLE' END as status
FROM daily_performance
GROUP BY strategy_id
HAVING SUM(daily_trades) >= 100  -- Minimum for statistical significance
ORDER BY annual_sharpe DESC;
```

### Trade Construction from Sparse Signals
```sql
-- Construct proper entry/exit pairs from signal changes
WITH signals AS (
    SELECT 
        strat as strategy_id,
        ts::timestamp as timestamp,
        val as signal_value,
        LAG(val) OVER (PARTITION BY strat ORDER BY ts) as prev_signal
    FROM read_parquet('signals/*.parquet')
),
signal_changes AS (
    SELECT 
        s.strategy_id,
        s.timestamp,
        s.signal_value,
        s.prev_signal,
        m.close as price
    FROM signals s
    JOIN market_data m ON s.timestamp = m.timestamp
    WHERE s.prev_signal IS NOT NULL 
      AND s.signal_value != s.prev_signal
),
trades AS (
    SELECT 
        sc1.strategy_id,
        sc1.timestamp as entry_time,
        sc1.price as entry_price,
        sc2.timestamp as exit_time,
        sc2.price as exit_price,
        sc1.signal_value as position,
        CASE 
            WHEN sc1.signal_value = 1 THEN (sc2.price / sc1.price - 1) * 10000 - 0.5
            WHEN sc1.signal_value = -1 THEN (sc1.price / sc2.price - 1) * 10000 - 0.5
        END as net_return_bps,
        EXTRACT(EPOCH FROM (sc2.timestamp - sc1.timestamp)) / 60 as duration_minutes
    FROM signal_changes sc1
    JOIN signal_changes sc2 ON sc1.strategy_id = sc2.strategy_id
        AND sc2.timestamp > sc1.timestamp
        AND sc1.signal_value != 0      -- Entry signal
        AND sc2.signal_value = 0       -- Exit signal  
        AND sc1.prev_signal = 0        -- Was neutral before entry
        AND sc2.prev_signal = sc1.signal_value  -- Exit from the position we entered
    WHERE NOT EXISTS (
        -- Ensure this is the immediate next exit
        SELECT 1 FROM signal_changes sc3
        WHERE sc3.strategy_id = sc1.strategy_id
          AND sc3.timestamp > sc1.timestamp
          AND sc3.timestamp < sc2.timestamp
          AND sc3.signal_value = 0
          AND sc3.prev_signal = sc1.signal_value
    )
)
SELECT * FROM trades;
```

### Duration Limit Analysis
```sql
-- Test impact of maximum trade duration limits
WITH duration_analysis AS (
    SELECT 
        t.*,
        d.max_duration
    FROM trades t
    CROSS JOIN (VALUES (30), (60), (120), (240), (9999)) AS d(max_duration)
    WHERE t.duration_minutes <= d.max_duration
      AND t.trade_type = 'INTRADAY'
)
SELECT 
    strategy_id,
    max_duration,
    COUNT(*) as trades_included,
    ROUND(AVG(net_return_bps), 2) as avg_net_bps,
    ROUND(STDDEV(net_return_bps), 2) as volatility,
    ROUND(AVG(net_return_bps) / NULLIF(STDDEV(net_return_bps), 0) * SQRT(252 * 390 / max_duration), 2) as annualized_sharpe
FROM duration_analysis
GROUP BY strategy_id, max_duration
ORDER BY strategy_id, max_duration;
```

---

## 🔧 Reusable SQL Functions

### Memory Management
```sql
-- Always start scripts with these settings
PRAGMA memory_limit='3GB';
SET threads=4;

-- For large datasets, process in batches
WITH strategy_batch AS (
    SELECT strategy_id 
    FROM (SELECT DISTINCT strategy_id FROM all_strategies)
    LIMIT 200 OFFSET 0  -- Adjust offset for each batch
)
```

### Market Hours Filter
```sql
-- Standard US market hours: 9:30 AM - 4:00 PM ET
WHERE EXTRACT(hour FROM timestamp) BETWEEN 9 AND 15
   OR (EXTRACT(hour FROM timestamp) = 16 AND EXTRACT(minute FROM timestamp) = 0)
```

### Statistical Significance Filter
```sql
-- Minimum requirements for reliable analysis
HAVING COUNT(*) >= 100          -- Minimum trades
   AND COUNT(DISTINCT DATE(entry_time)) >= 20  -- Minimum trading days
```

---

## 🎯 Quality Filters

### Profitable Strategy Criteria
```sql
WHERE net_return_bps > 0           -- Must be profitable after costs
  AND annual_sharpe >= 0.5         -- Minimum risk-adjusted return
  AND trades_per_day >= 1.0        -- Reasonable frequency
  AND total_trades >= 100          -- Statistical significance
```

### Outlier Detection
```sql
-- Remove extreme price movements (likely errors)
WHERE ABS(return_bps) < 1000       -- Max 10% single trade return
  AND duration_minutes < 1440      -- Max 24 hour trades
```

---

## 📈 Performance Comparison Template

### Train vs Test Analysis
```sql
WITH train_performance AS (
    SELECT strategy_id, annual_sharpe, avg_daily_return_bps
    FROM strategy_analysis 
    WHERE period = 'TRAIN'
),
test_performance AS (
    SELECT strategy_id, annual_sharpe, avg_daily_return_bps
    FROM strategy_analysis 
    WHERE period = 'TEST'
)
SELECT 
    t.strategy_id,
    train.annual_sharpe as train_sharpe,
    t.annual_sharpe as test_sharpe,
    ROUND(t.annual_sharpe - train.annual_sharpe, 2) as sharpe_degradation,
    CASE 
        WHEN t.avg_daily_return_bps > 0 AND train.avg_daily_return_bps > 0 THEN 'CONSISTENT'
        WHEN t.avg_daily_return_bps > 0 AND train.avg_daily_return_bps <= 0 THEN 'IMPROVED'
        WHEN t.avg_daily_return_bps <= 0 AND train.avg_daily_return_bps > 0 THEN 'DEGRADED'
        ELSE 'CONSISTENTLY_POOR'
    END as performance_consistency
FROM test_performance t
JOIN train_performance train ON t.strategy_id = train.strategy_id;
```

---

## ⚠️ Common Debugging Steps

### 1. Signal Validation
```sql
-- Check signal distribution
SELECT 
    strat as strategy_id,
    val as signal_value,
    COUNT(*) as count,
    MIN(ts) as first_signal,
    MAX(ts) as last_signal
FROM read_parquet('signals/*.parquet')
GROUP BY strat, val
ORDER BY strat, val;
```

### 2. Trade Count Verification
```sql
-- Verify entry/exit matching
WITH entries AS (
    SELECT strategy_id, COUNT(*) as entry_count
    FROM signal_changes 
    WHERE prev_signal = 0 AND signal_value != 0
    GROUP BY strategy_id
),
exits AS (
    SELECT strategy_id, COUNT(*) as exit_count
    FROM signal_changes 
    WHERE prev_signal != 0 AND signal_value = 0
    GROUP BY strategy_id
)
SELECT 
    e.strategy_id,
    e.entry_count,
    x.exit_count,
    e.entry_count - x.exit_count as unmatched_entries
FROM entries e
LEFT JOIN exits x ON e.strategy_id = x.strategy_id
WHERE e.entry_count != x.exit_count;
```

### 3. Price Data Validation
```sql
-- Check for missing price data
SELECT 
    DATE(timestamp) as trade_date,
    COUNT(*) as records,
    MIN(timestamp) as first_record,
    MAX(timestamp) as last_record
FROM market_data
GROUP BY DATE(timestamp)
ORDER BY trade_date;
```

---

## 🎁 Ready-to-Use Analysis Scripts

Save these as `.sql` files for repeated use:

1. **`basic_strategy_analysis.sql`** - Standard performance metrics
2. **`duration_limit_optimization.sql`** - Find optimal trade duration limits  
3. **`train_test_comparison.sql`** - Out-of-sample validation
4. **`regime_performance_analysis.sql`** - Performance by market regime (when available)
5. **`sharpe_correction_script.sql`** - Fix incorrectly calculated Sharpe ratios

---

## 🏆 Key Lessons Learned

1. **Transaction costs kill most strategies** - 0.5 bps eliminates ~97% of candidates
2. **Duration limits can dramatically improve performance** - especially for RSI strategies
3. **Wall-clock Sharpe calculation is critical** - don't let low-frequency strategies fool you
4. **Overnight trades are often toxic** - filter them out for realistic backtests
5. **Sample size matters** - require minimum 100 trades and 20 trading days
6. **Most profitable strategies have 1-10 trades/day** - not hundreds

---

*This guide was built from analyzing 1,157 trading strategies and making every mistake so you don't have to.*
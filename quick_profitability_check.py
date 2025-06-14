import duckdb

con = duckdb.connect('workspaces/expansive_grid_search_8d35f12c/analytics.duckdb')

# Check gross returns
query = """
SELECT 
    strat,
    COUNT(*) as trades,
    ROUND(AVG(CASE 
        WHEN val = 1 THEN (m2.close - m1.close) / m1.close * 100
        WHEN val = -1 THEN (m1.close - m2.close) / m1.close * 100
    END), 5) as gross_return_pct
FROM read_parquet('workspaces/expansive_grid_search_8d35f12c/traces/*/signals/*/*.parquet') s
JOIN read_parquet('data/SPY_1m.parquet') m1 ON s.idx = m1.bar_index
JOIN read_parquet('data/SPY_1m.parquet') m2 ON s.idx + 1 = m2.bar_index
WHERE s.val != 0
GROUP BY s.strat
HAVING COUNT(*) >= 250
ORDER BY gross_return_pct DESC
LIMIT 20
"""

df = con.execute(query).df()

print("=== Top 20 Strategies by Gross Return (250+ trades) ===")
print(df.to_string(index=False))

print("\n=== Profitability Analysis ===")
print(f"Total strategies: {len(df)}")
print(f"Positive gross return: {(df['gross_return_pct'] > 0).sum()}")
print(f"Above 0.5bp: {(df['gross_return_pct'] > 0.005).sum()}")
print(f"Above 1bp: {(df['gross_return_pct'] > 0.01).sum()}")
print(f"Above 2bp: {(df['gross_return_pct'] > 0.02).sum()}")

con.close()
import sys
sys.path.append('../../src')
from analytics.simple_analytics import TraceAnalysis

# Load data
ta = TraceAnalysis('results/20250622_180858')

# Just run a simple query
df = ta.sql("SELECT COUNT(DISTINCT strategy_id) as total_strategies FROM traces")
print(df)
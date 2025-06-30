import sys
sys.path.append('../../src')
from analytics.simple_analytics import TraceAnalysis

ta = TraceAnalysis('results/20250622_155944')

# What columns do we actually have?
print(ta.sql("SELECT * FROM traces LIMIT 1"))
#!/usr/bin/env python3
"""Test analytics workspace creation"""

import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test DuckDB import
try:
    import duckdb
    logger.info(f"✓ DuckDB imported successfully: {duckdb.__version__}")
except ImportError as e:
    logger.error(f"✗ Failed to import DuckDB: {e}")
    sys.exit(1)

# Test analytics workspace creation
try:
    from src.analytics.workspace import AnalyticsWorkspace
    from src.analytics.migration import setup_workspace
    
    # Create test workspace
    workspace_path = "workspaces/test_analytics_workspace"
    logger.info(f"Creating test workspace at: {workspace_path}")
    
    workspace = setup_workspace(workspace_path)
    logger.info("✓ Workspace created successfully")
    
    # Test basic query
    tables = workspace.sql("SHOW TABLES")
    logger.info(f"✓ Tables in workspace: {list(tables['name'])}")
    
    # Close workspace
    workspace.close()
    logger.info("✓ Workspace closed successfully")
    
except Exception as e:
    logger.error(f"✗ Failed to create workspace: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

logger.info("\n✓ All tests passed! Analytics workspace is working correctly.")
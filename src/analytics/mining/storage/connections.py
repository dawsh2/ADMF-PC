"""
Database connection management for ADMF-PC analytics.
Handles PostgreSQL connections with optional TimescaleDB extensions.
"""
import logging
from typing import Optional, Dict, Any, List, Iterator
from contextlib import contextmanager
import json
from datetime import datetime
from decimal import Decimal

# Using standard library for compatibility
import sqlite3  # For local development/testing

# Optional imports for production
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    from psycopg2.pool import SimpleConnectionPool
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    psycopg2 = None


class DatabaseConnection:
    """Abstract base for database connections."""
    
    def execute(self, query: str, params: Optional[tuple] = None) -> None:
        """Execute a query without returning results."""
        raise NotImplementedError
        
    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """Fetch a single row as a dictionary."""
        raise NotImplementedError
        
    def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Fetch all rows as a list of dictionaries."""
        raise NotImplementedError
        
    def insert_many(self, table: str, records: List[Dict[str, Any]]) -> None:
        """Bulk insert records into a table."""
        raise NotImplementedError
        
    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        raise NotImplementedError


class SQLiteConnection(DatabaseConnection):
    """SQLite connection for development and testing."""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row
        self.logger = logging.getLogger(f"{__name__}.SQLiteConnection")
        
        # Enable foreign keys
        self.connection.execute("PRAGMA foreign_keys = ON")
        
        # Register adapters for custom types
        self._register_adapters()
        
    def _register_adapters(self):
        """Register SQLite adapters for custom types."""
        # Decimal adapter
        sqlite3.register_adapter(Decimal, lambda d: str(d))
        sqlite3.register_converter("DECIMAL", lambda s: Decimal(s.decode()))
        
        # JSON adapter
        sqlite3.register_adapter(dict, json.dumps)
        sqlite3.register_adapter(list, json.dumps)
        
    def execute(self, query: str, params: Optional[tuple] = None) -> None:
        """Execute a query without returning results."""
        # Convert PostgreSQL-specific syntax
        query = self._convert_query(query)
        
        cursor = self.connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        self.connection.commit()
        
    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """Fetch a single row as a dictionary."""
        query = self._convert_query(query)
        
        cursor = self.connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
            
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
        
    def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Fetch all rows as a list of dictionaries."""
        query = self._convert_query(query)
        
        cursor = self.connection.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
            
        return [dict(row) for row in cursor.fetchall()]
        
    def insert_many(self, table: str, records: List[Dict[str, Any]]) -> None:
        """Bulk insert records into a table."""
        if not records:
            return
            
        # Get column names from first record
        columns = list(records[0].keys())
        placeholders = ", ".join(["?" for _ in columns])
        
        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        
        cursor = self.connection.cursor()
        values = [tuple(record.get(col) for col in columns) for record in records]
        cursor.executemany(query, values)
        self.connection.commit()
        
    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        try:
            yield self
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
            
    def _convert_query(self, query: str) -> str:
        """Convert PostgreSQL-specific syntax to SQLite."""
        # Basic conversions
        conversions = {
            "UUID": "TEXT",
            "JSONB": "TEXT",
            "DECIMAL": "REAL",
            "INTERVAL": "TEXT",
            "daterange": "TEXT",
            "NOW()": "datetime('now')",
            "CURRENT_TIMESTAMP": "datetime('now')",
            "::": " AS ",  # Type casting
            "%s": "?",  # Parameter placeholder
        }
        
        for pg_syntax, sqlite_syntax in conversions.items():
            query = query.replace(pg_syntax, sqlite_syntax)
            
        # Remove unsupported features
        if "CREATE MATERIALIZED VIEW" in query:
            query = query.replace("CREATE MATERIALIZED VIEW", "CREATE VIEW")
        if "USING GIN" in query:
            query = query.split("USING GIN")[0] + ";"
            
        return query
        
    def close(self):
        """Close the connection."""
        self.connection.close()


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL connection for production use."""
    
    def __init__(self, 
                 host: str,
                 port: int,
                 database: str,
                 user: str,
                 password: str,
                 pool_size: int = 5):
        if not HAS_POSTGRES:
            raise ImportError("psycopg2 is required for PostgreSQL support. "
                            "Install with: pip install psycopg2-binary")
                            
        self.config = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        
        # Create connection pool
        self.pool = SimpleConnectionPool(
            1, pool_size,
            **self.config
        )
        
        self.logger = logging.getLogger(f"{__name__}.PostgreSQLConnection")
        
    @contextmanager
    def _get_connection(self):
        """Get a connection from the pool."""
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)
            
    def execute(self, query: str, params: Optional[tuple] = None) -> None:
        """Execute a query without returning results."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                conn.commit()
                
    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """Fetch a single row as a dictionary."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()
                return dict(result) if result else None
                
    def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Fetch all rows as a list of dictionaries."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
                
    def insert_many(self, table: str, records: List[Dict[str, Any]]) -> None:
        """Bulk insert records into a table."""
        if not records:
            return
            
        # Get column names from first record
        columns = list(records[0].keys())
        
        # Build INSERT query with COPY for performance
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                # Use execute_values for efficient bulk insert
                from psycopg2.extras import execute_values
                
                query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s"
                values = [tuple(record.get(col) for col in columns) for record in records]
                
                execute_values(cursor, query, values)
                conn.commit()
                
    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        with self._get_connection() as conn:
            try:
                yield self
                conn.commit()
            except Exception:
                conn.rollback()
                raise
                
    def close(self):
        """Close all connections in the pool."""
        self.pool.closeall()


class DatabaseManager:
    """
    Manages database connections and schema creation.
    Supports both SQLite (development) and PostgreSQL (production).
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection: Optional[DatabaseConnection] = None
        self.logger = logging.getLogger(f"{__name__}.DatabaseManager")
        
    def connect(self) -> DatabaseConnection:
        """Create and return a database connection."""
        db_type = self.config.get('type', 'sqlite')
        
        if db_type == 'sqlite':
            self.connection = SQLiteConnection(
                db_path=self.config.get('path', ':memory:')
            )
        elif db_type == 'postgresql':
            self.connection = PostgreSQLConnection(
                host=self.config['host'],
                port=self.config.get('port', 5432),
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                pool_size=self.config.get('pool_size', 5)
            )
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
            
        self.logger.info(f"Connected to {db_type} database")
        return self.connection
        
    def create_schema(self) -> None:
        """Create the analytics schema."""
        if not self.connection:
            raise RuntimeError("Not connected to database")
            
        from src.core.data_mining.storage.schemas import create_analytics_schema
        
        schema_statements = create_analytics_schema()
        
        for statement in schema_statements:
            try:
                self.connection.execute(statement)
                self.logger.debug(f"Executed schema statement: {statement[:50]}...")
            except Exception as e:
                # Some statements might fail in SQLite (like materialized views)
                # Log but continue
                self.logger.warning(f"Schema statement failed: {e}")
                
        self.logger.info("Analytics schema created successfully")
        
    def check_timescaledb(self) -> bool:
        """Check if TimescaleDB extension is available."""
        if not isinstance(self.connection, PostgreSQLConnection):
            return False
            
        try:
            result = self.connection.fetch_one(
                "SELECT extname FROM pg_extension WHERE extname = 'timescaledb'"
            )
            return result is not None
        except Exception:
            return False
            
    def enable_timescaledb(self, tables: List[str]) -> None:
        """Enable TimescaleDB hypertables for specified tables."""
        if not isinstance(self.connection, PostgreSQLConnection):
            self.logger.warning("TimescaleDB only supported on PostgreSQL")
            return
            
        if not self.check_timescaledb():
            self.logger.warning("TimescaleDB extension not installed")
            return
            
        for table in tables:
            try:
                # Convert to hypertable
                self.connection.execute(
                    f"SELECT create_hypertable('{table}', 'timestamp', "
                    f"migrate_data => true, if_not_exists => true)"
                )
                self.logger.info(f"Converted {table} to TimescaleDB hypertable")
            except Exception as e:
                self.logger.error(f"Failed to create hypertable for {table}: {e}")
                
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.logger.info("Database connection closed")


# Convenience functions for common queries
class AnalyticsQueries:
    """Common analytics queries."""
    
    @staticmethod
    def find_high_sharpe_strategies(
        conn: DatabaseConnection,
        min_sharpe: float = 1.5,
        volatility_regime: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find strategies with high Sharpe ratios."""
        query = """
            SELECT 
                run_id,
                correlation_id,
                strategy_type,
                parameters,
                sharpe_ratio,
                total_return,
                max_drawdown,
                market_regime,
                volatility_regime
            FROM optimization_runs
            WHERE sharpe_ratio > ?
        """
        
        params = [min_sharpe]
        
        if volatility_regime:
            query += " AND volatility_regime = ?"
            params.append(volatility_regime)
            
        query += " ORDER BY sharpe_ratio DESC"
        
        return conn.fetch_all(query, tuple(params))
        
    @staticmethod
    def get_pattern_performance_history(
        conn: DatabaseConnection,
        pattern_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get recent performance history for a pattern."""
        query = """
            SELECT 
                evaluation_date,
                success_rate,
                sample_size,
                avg_return,
                sharpe_contribution,
                market_conditions
            FROM pattern_performance_history
            WHERE pattern_id = ?
            AND evaluation_date >= date('now', ? || ' days')
            ORDER BY evaluation_date DESC
        """
        
        return conn.fetch_all(query, (pattern_id, -days))
        
    @staticmethod
    def find_degrading_patterns(
        conn: DatabaseConnection,
        degradation_threshold: float = 0.2
    ) -> List[Dict[str, Any]]:
        """Find patterns with degrading performance."""
        query = """
            SELECT 
                p.pattern_id,
                p.pattern_name,
                p.pattern_type,
                p.success_rate as original_rate,
                h.success_rate as current_rate,
                (p.success_rate - h.success_rate) as degradation
            FROM discovered_patterns p
            JOIN pattern_performance_history h ON p.pattern_id = h.pattern_id
            WHERE h.evaluation_date = date('now')
            AND h.success_rate < p.success_rate * (1 - ?)
            ORDER BY degradation DESC
        """
        
        return conn.fetch_all(query, (degradation_threshold,))
# Analytics Exceptions
"""
Custom exceptions for analytics module.
"""

class AnalyticsError(Exception):
    """Base exception for analytics operations"""
    pass

class WorkspaceNotFoundError(AnalyticsError):
    """Raised when workspace directory or database is not found"""
    pass

class SchemaError(AnalyticsError):
    """Raised when there are issues with database schema"""
    pass

class QueryError(AnalyticsError):
    """Raised when SQL query execution fails"""
    pass

class MigrationError(AnalyticsError):
    """Raised when workspace migration fails"""
    pass
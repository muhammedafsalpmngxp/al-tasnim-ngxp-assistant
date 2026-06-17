import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src import settings
from src.connection_pool import ConnectionPool
from src.schema_introspector import SchemaIntrospector
from src.value_resolver import ValueResolver
from src.models import Intent, Filter, Operator
from src.sql_builder import SQLBuilder

def test_connection_pool():
    """Test connection pool works"""
    conn_string = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={settings.DB_SERVER};"
        f"DATABASE={settings.DB_NAME};"
        f"UID={settings.DB_READONLY_USER};"
        f"PWD={settings.DB_READONLY_PASSWORD};"
        f"ApplicationIntent=ReadOnly;"
    )
    pool = ConnectionPool(conn_string, max_size=3)
    
    with pool.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1
    
    stats = pool.get_stats()
    assert stats['healthy'] > 0
    pool.close_all()
    print("Connection pool works")

def test_schema_introspection():
    """Test schema loading"""
    conn_string = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={settings.DB_SERVER};"
        f"DATABASE={settings.DB_NAME};"
        f"UID={settings.DB_READONLY_USER};"
        f"PWD={settings.DB_READONLY_PASSWORD};"
        f"ApplicationIntent=ReadOnly;"
    )
    pool = ConnectionPool(conn_string, max_size=3)
    schema = SchemaIntrospector(pool, settings.ALLOWED_TABLES)
    
    well_schema = schema.get_table_schema("2026_Well_Delivery_Scope_Well_Type")
    assert 'Well_ID' in well_schema['columns']
    assert 'RIG' in well_schema['columns']
    pool.close_all()
    print("Schema introspection works")

def test_sql_builder():
    """Test SQL builder"""
    conn_string = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={settings.DB_SERVER};"
        f"DATABASE={settings.DB_NAME};"
        f"UID={settings.DB_READONLY_USER};"
        f"PWD={settings.DB_READONLY_PASSWORD};"
        f"ApplicationIntent=ReadOnly;"
    )
    pool = ConnectionPool(conn_string, max_size=3)
    schema = SchemaIntrospector(pool, settings.ALLOWED_TABLES)
    builder = SQLBuilder(schema, max_rows=1000, pii_patterns=settings.PII_PATTERNS)
    
    intent = Intent(
        table="2026_Well_Delivery_Scope_Well_Type",
        columns=["Well_ID", "Field"],
        filters=[Filter(col="RIG", op=Operator.EQ, value=104)],
        limit=10,
        natural_language="test"
    )
    
    sql, params = builder.build(intent)
    assert "SELECT" in sql.upper()
    assert "TOP 10" in sql
    assert "[RIG] = ?" in sql
    assert params == [104]
    pool.close_all()
    print(f"SQL builder works: {sql}")

if __name__ == "__main__":
    print("Running tests...\n")
    test_connection_pool()
    test_schema_introspection()
    test_sql_builder()
    print("\nAll tests passed!")

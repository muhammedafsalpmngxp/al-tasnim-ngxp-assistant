#!/usr/bin/env python
"""Refresh value resolver lookup caches from the database."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.config import settings
from src.connection_pool import ConnectionPool
from src.value_resolver import ValueResolver
from src.utils import setup_logging

def main():
    setup_logging(settings.LOG_LEVEL, settings.LOG_FILE)
    conn_string = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={settings.DB_SERVER};"
        f"DATABASE={settings.DB_NAME};"
        f"UID={settings.DB_READONLY_USER};"
        f"PWD={settings.DB_READONLY_PASSWORD};"
        f"ApplicationIntent=ReadOnly;"
    )
    pool = ConnectionPool(conn_string, max_size=3)
    resolver = ValueResolver(pool, settings.FUZZY_THRESHOLD, settings.AMBIGUITY_MARGIN)
    resolver.refresh_cache()
    pool.close_all()
    print("Value resolver cache cleared. Fresh lookups will load on next query.")

if __name__ == "__main__":
    main()

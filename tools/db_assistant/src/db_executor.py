import time
import logging
from typing import Dict, Any, List
from .connection_pool import ConnectionPool

logger = logging.getLogger(__name__)

class ReadOnlyExecutor:
    def __init__(self, pool: ConnectionPool, statement_timeout: int = 30):
        self.pool = pool
        self.statement_timeout = statement_timeout
    
    def execute(self, sql: str, params: List) -> Dict[str, Any]:
        start = time.time()
        
        try:
            with self.pool.get_connection() as conn:
                cursor = conn.cursor()
                
                try:
                    cursor.execute(f"SET LOCK_TIMEOUT {self.statement_timeout * 1000}")
                except:
                    pass
                
                cursor.execute(sql, params)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchall()
                
                data = []
                for row in rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        val = row[i]
                        if hasattr(val, "hex"):
                            logger.debug("Skipping binary column '%s'", col)
                            continue
                        row_dict[col] = val
                    data.append(row_dict)
                
                return {
                    'success': True,
                    'data': data,
                    'columns': columns,
                    'row_count': len(data),
                    'execution_time_ms': (time.time() - start) * 1000
                }
                
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time_ms': (time.time() - start) * 1000
            }

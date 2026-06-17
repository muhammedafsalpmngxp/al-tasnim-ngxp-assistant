import threading
import time
import logging
from queue import Queue, Empty
from contextlib import contextmanager
import pyodbc

logger = logging.getLogger(__name__)

class ConnectionPool:
    def __init__(self, connection_string: str, max_size: int = 10):
        self.connection_string = connection_string
        self.max_size = max_size
        self._pool = Queue(maxsize=max_size)
        self._lock = threading.Lock()
        self._created = 0
        self._closed = False
        self._healthy_count = 0
        
        for _ in range(min(3, max_size)):
            self._add_connection()
    
    def _add_connection(self) -> bool:
        with self._lock:
            if self._closed or self._created >= self.max_size:
                return False
            try:
                from .config import settings as _s
                conn = pyodbc.connect(self.connection_string, timeout=_s.QUERY_TIMEOUT_SEC)
                self._pool.put(conn, block=False)
                self._created += 1
                self._healthy_count += 1
                return True
            except Exception as e:
                logger.error(f"Failed to create connection: {e}")
                return False
    
    def _is_healthy(self, conn) -> bool:
        if conn is None:
            return False
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except Exception:
            try:
                conn.close()
            except:
                pass
            with self._lock:
                self._healthy_count -= 1
            return False
    
    @contextmanager
    def get_connection(self):
        if self._closed:
            raise Exception("Pool is closed")
        
        conn = None
        attempts = 0
        max_attempts = 3
        
        try:
            while attempts < max_attempts:
                try:
                    conn = self._pool.get(block=True, timeout=5)
                    if self._is_healthy(conn):
                        yield conn
                        if self._is_healthy(conn):
                            self._pool.put(conn, block=False)
                        return
                    else:
                        conn = None
                        
                except Empty:
                    with self._lock:
                        if self._created < self.max_size:
                            self._add_connection()
                        elif self._healthy_count == 0:
                            self._created = 0
                            self._healthy_count = 0
                            self._add_connection()
                
                attempts += 1
                time.sleep(0.1 * attempts)
            
            raise Exception("Unable to obtain database connection")
            
        except Exception as e:
            if conn is not None:
                try:
                    self._pool.put(conn, block=False)
                except:
                    pass
            raise e
    
    def close_all(self):
        self._closed = True
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                try:
                    conn.close()
                except:
                    pass
            except:
                pass
        with self._lock:
            self._created = 0
            self._healthy_count = 0
    
    def get_stats(self) -> dict:
        return {
            'pool_size': self._pool.qsize(),
            'created': self._created,
            'healthy': self._healthy_count,
            'max_size': self.max_size
        }

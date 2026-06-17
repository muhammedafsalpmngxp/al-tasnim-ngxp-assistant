import re
import time
from typing import Tuple, Optional, List, Any
from rapidfuzz import process, fuzz
from .connection_pool import ConnectionPool

class ValueResolver:
    def __init__(self, pool: ConnectionPool, threshold: int = 80, ambiguity_margin: int = 10):
        self.pool = pool
        self.threshold = threshold
        self.ambiguity_margin = ambiguity_margin
        self._cache = {}
        self._cache_ttl = 3600
        self._cache_timestamps = {}
    
    def _normalize(self, text: str) -> str:
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9]', '', text)
        return text
    
    def _get_distinct_values(self, table: str, column: str, limit: int = 10000) -> List[Any]:
        cache_key = f"{table}:{column}"
        
        if cache_key in self._cache:
            if time.time() - self._cache_timestamps.get(cache_key, 0) < self._cache_ttl:
                return self._cache[cache_key]
        
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            safe_table = table.replace(']', ']]')
            safe_col = column.replace(']', ']]')
            sql = f"""
            SELECT DISTINCT TOP ({limit}) [{safe_col}]
            FROM [{safe_table}]
            WHERE [{safe_col}] IS NOT NULL
            ORDER BY [{safe_col}]
            """
            cursor.execute(sql)
            rows = cursor.fetchall()
            values = [row[0] for row in rows]
            
            self._cache[cache_key] = values
            self._cache_timestamps[cache_key] = time.time()
            return values
    
    def resolve_value(self, table: str, column: str, user_value: Any) -> Tuple[Optional[Any], str, Optional[List[Any]]]:
        user_norm = self._normalize(user_value)
        # Use explicit empty-string check — `not user_norm` would wrongly reject 0 or False
        if user_norm == "":
            return None, 'not_found', None
        
        db_values = self._get_distinct_values(table, column)
        if not db_values:
            return None, 'not_found', None
        
        lookup = {}
        for val in db_values:
            norm = self._normalize(val)
            lookup[norm] = val
        
        if user_norm in lookup:
            return lookup[user_norm], 'exact', None
        
        try:
            num_val = int(user_value)
            for val in db_values:
                if isinstance(val, (int, float)) and val == num_val:
                    return val, 'numeric', None
                if isinstance(val, str):
                    digits = re.sub(r'[^0-9]', '', val)
                    if digits and int(digits) == num_val:
                        return val, 'digit_match', None
        except (ValueError, TypeError):
            pass
        
        db_norms = [self._normalize(v) for v in db_values]
        matches = process.extract(
            user_norm,
            db_norms,
            scorer=fuzz.token_sort_ratio,
            limit=5
        )
        
        good_matches = [(m, s) for m, s, _ in matches if s >= self.threshold]
        
        if good_matches:
            if len(good_matches) >= 2 and good_matches[0][1] - good_matches[1][1] < self.ambiguity_margin:
                suggestions = []
                for match, _ in good_matches[:3]:
                    for val in db_values:
                        if self._normalize(val) == match:
                            suggestions.append(val)
                            break
                return None, 'ambiguous', suggestions
            
            best_match = good_matches[0][0]
            for val in db_values:
                if self._normalize(val) == best_match:
                    return val, 'fuzzy', None
        
        suggestions = []
        for match, score, _ in matches[:5]:
            if score > 50:
                for val in db_values:
                    if self._normalize(val) == match:
                        suggestions.append(val)
                        break
        
        return None, 'not_found', suggestions[:5]
    
    def refresh_cache(self, table: str = None, column: str = None):
        if table and column:
            cache_key = f"{table}:{column}"
            self._cache.pop(cache_key, None)
            self._cache_timestamps.pop(cache_key, None)
        else:
            self._cache.clear()
            self._cache_timestamps.clear()

from typing import Dict, List, Set, Optional
from .connection_pool import ConnectionPool

class SchemaIntrospector:
    def __init__(self, pool: ConnectionPool, allowed_tables: List[str]):
        self.pool = pool
        self.allowed_tables = set(allowed_tables)
        self.snapshot_tables = {"WMR", "Job_Progress_PlanSnapshot"}
        self.snapshot_date_columns = {
            "WMR": "Week_Number",
            "Job_Progress_PlanSnapshot": "SnapshotMonth"
        }
        self._cache: Dict[str, Dict] = {}
    
    def get_table_schema(self, table_name: str) -> Dict:
        if table_name not in self.allowed_tables:
            raise ValueError(f"Table '{table_name}' not allowed")
        
        if table_name in self._cache:
            return self._cache[table_name]
        
        with self.pool.get_connection() as conn:
            cursor = conn.cursor()
            safe_table = table_name.replace(']', ']]')
            sql = f"""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE,
                   COLUMNPROPERTY(OBJECT_ID('{safe_table}'), COLUMN_NAME, 'IsIdentity') AS IsIdentity
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{safe_table}'
            ORDER BY ORDINAL_POSITION
            """
            cursor.execute(sql)
            rows = cursor.fetchall()
            
            schema = {
                'table_name': table_name,
                'columns': {},
                'column_list': []
            }
            
            for row in rows:
                col_name = row[0]
                schema['columns'][col_name] = {
                    'type': row[1],
                    'max_length': row[2],
                    'nullable': row[3] == 'YES',
                    'identity': row[4] == 1
                }
                schema['column_list'].append(col_name)
            
            self._cache[table_name] = schema
            return schema
    
    def is_snapshot_table(self, table_name: str) -> bool:
        return table_name in self.snapshot_tables
    
    def get_snapshot_date_column(self, table_name: str) -> Optional[str]:
        return self.snapshot_date_columns.get(table_name)

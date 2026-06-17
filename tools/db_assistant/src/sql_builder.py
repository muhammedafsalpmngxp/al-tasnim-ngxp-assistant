import re
from typing import Tuple, List
from .models import Intent, Filter, Aggregate, OrderBy

class SQLBuilder:
    def __init__(self, schema_introspector, max_rows: int = 1000, pii_patterns: List[str] = None):
        self.schema = schema_introspector
        self.max_rows = max_rows
        self.pii_patterns = pii_patterns or []
    
    def _escape_identifier(self, name: str) -> str:
        return name.replace(']', ']]')
    
    def _safe_column(self, table: str, column: str) -> str:
        schema = self.schema.get_table_schema(table)
        if column not in schema['columns']:
            raise ValueError(f"Column '{column}' not found in table '{table}'")
        return f"[{self._escape_identifier(column)}]"
    
    def _is_pii(self, table: str, column: str) -> bool:
        col_lower = column.lower()
        for pattern in self.pii_patterns:
            if re.search(pattern, col_lower, re.IGNORECASE):
                return True
        return False
    
    def build(self, intent: Intent) -> Tuple[str, List]:
        if intent.table not in self.schema.allowed_tables:
            raise ValueError(f"Table '{intent.table}' not allowed")
        
        table = intent.table
        schema = self.schema.get_table_schema(table)
        
        is_snapshot = self.schema.is_snapshot_table(table)
        snapshot_col = self.schema.get_snapshot_date_column(table)
        
        valid_columns = []
        for col in intent.columns:
            if col in schema['columns'] and not self._is_pii(table, col):
                valid_columns.append(col)
        
        if not valid_columns and not intent.aggregate:
            for col in schema['column_list']:
                if not self._is_pii(table, col):
                    valid_columns.append(col)
        
        top = f"TOP {min(intent.limit, self.max_rows)}"
        select = self._build_select(table, valid_columns, intent.aggregate)
        
        if is_snapshot and snapshot_col and snapshot_col in schema['columns']:
            return self._build_snapshot_query(table, valid_columns, intent, top, snapshot_col)
        
        from_clause = f"FROM [{self._escape_identifier(table)}]"
        where, params = self._build_where(table, intent.filters)
        group = self._build_group_by(table, intent.aggregate)
        order = self._build_order_by(table, intent.order_by)
        
        sql = f"SELECT {top} {select} {from_clause}"
        if where:
            sql += f" {where}"
        if group:
            sql += f" {group}"
        if order:
            sql += f" {order}"
        
        return sql, params
    
    def _build_snapshot_query(self, table: str, columns: List[str], intent: Intent, top: str, snapshot_col: str) -> Tuple[str, List]:
        partition_col = None
        schema_cols = self.schema.get_table_schema(table)["columns"]
        for col in ["pdo_well_id", "Well_ID", "well_id"]:
            if col in schema_cols:
                partition_col = col
                break

        if not partition_col:
            return self._build_normal_query(table, columns, intent, top)

        if columns:
            select_cols = ", ".join([self._safe_column(table, c) for c in columns])
        else:
            select_cols = "*"

        where, params = self._build_where(table, intent.filters)
        order = self._build_order_by(table, intent.order_by)
        esc_table = self._escape_identifier(table)
        esc_partition = self._escape_identifier(partition_col)
        esc_snapshot = self._escape_identifier(snapshot_col)

        # TOP applied in outer query — snapshot queries must respect max_rows cap
        sql = f"""
        SELECT {top} {select_cols}
        FROM (
            SELECT {select_cols},
                   ROW_NUMBER() OVER (PARTITION BY [{esc_partition}] ORDER BY [{esc_snapshot}] DESC) AS rn
            FROM [{esc_table}]
            {where}
        ) AS latest
        WHERE rn = 1
        """

        if order:
            sql += f" {order}"

        return sql, params
    
    def _build_normal_query(self, table: str, columns: List[str], intent: Intent, top: str) -> Tuple[str, List]:
        select = ", ".join([self._safe_column(table, c) for c in columns]) if columns else "*"
        where, params = self._build_where(table, intent.filters)
        order = self._build_order_by(table, intent.order_by)
        
        sql = f"SELECT {top} {select} FROM [{self._escape_identifier(table)}]"
        if where:
            sql += f" {where}"
        if order:
            sql += f" {order}"
        return sql, params
    
    def _build_select(self, table: str, columns: List[str], aggregate: Aggregate) -> str:
        if aggregate:
            func = aggregate.func.value
            if aggregate.group_by:
                group_cols = []
                for col in aggregate.group_by:
                    if col in self.schema.get_table_schema(table)['columns']:
                        if not self._is_pii(table, col):
                            group_cols.append(self._safe_column(table, col))
                if group_cols:
                    group_cols_str = ", ".join(group_cols)
                    if func == 'count':
                        return f"{group_cols_str}, COUNT(*) AS count"
                    valid_cols = [c for c in columns if c in self.schema.get_table_schema(table)['columns']]
                    if valid_cols and not self._is_pii(table, valid_cols[0]):
                        return f"{group_cols_str}, {func.upper()}({self._safe_column(table, valid_cols[0])}) AS {func}"
                    return f"{group_cols_str}, COUNT(*) AS count"
            
            valid_cols = [c for c in columns if c in self.schema.get_table_schema(table)['columns']]
            valid_cols = [c for c in valid_cols if not self._is_pii(table, c)]
            if func in ['sum', 'avg', 'max', 'min'] and valid_cols:
                return f"{func.upper()}({self._safe_column(table, valid_cols[0])}) AS {func}"
            return "COUNT(*) AS count"
        
        if not columns:
            return "*"
        return ", ".join([self._safe_column(table, col) for col in columns])
    
    def _build_where(self, table: str, filters: List[Filter]) -> Tuple[str, List]:
        if not filters:
            return "", []
        
        conditions = []
        params = []
        schema = self.schema.get_table_schema(table)
        
        for f in filters:
            if f.col not in schema['columns']:
                continue
            col_ref = self._safe_column(table, f.col)
            op = f.op.value
            value = f.resolved_value if f.resolved_value is not None else f.value
            
            if op == "is_null":
                conditions.append(f"{col_ref} IS NULL")
            elif op == "is_not_null":
                conditions.append(f"{col_ref} IS NOT NULL")
            elif op == "in":
                if isinstance(value, list):
                    placeholders = ", ".join(["?" for _ in value])
                    conditions.append(f"{col_ref} IN ({placeholders})")
                    params.extend(value)
                else:
                    conditions.append(f"{col_ref} = ?")
                    params.append(value)
            elif op == "between":
                if isinstance(value, list) and len(value) >= 2:
                    conditions.append(f"{col_ref} BETWEEN ? AND ?")
                    params.extend(value[:2])
                else:
                    conditions.append(f"{col_ref} = ?")
                    params.append(value)
            elif op == "like":
                conditions.append(f"{col_ref} LIKE ?")
                params.append(f"%{value}%")
            else:
                conditions.append(f"{col_ref} {op} ?")
                params.append(value)
        
        if conditions:
            return "WHERE " + " AND ".join(conditions), params
        return "", []
    
    def _build_group_by(self, table: str, aggregate: Aggregate) -> str:
        if not aggregate or not aggregate.group_by:
            return ""
        valid_groups = []
        schema = self.schema.get_table_schema(table)
        for col in aggregate.group_by:
            if col in schema['columns'] and not self._is_pii(table, col):
                valid_groups.append(self._safe_column(table, col))
        if valid_groups:
            return "GROUP BY " + ", ".join(valid_groups)
        return ""
    
    def _build_order_by(self, table: str, order_by: List[OrderBy]) -> str:
        if not order_by:
            return ""
        parts = []
        schema = self.schema.get_table_schema(table)
        for o in order_by:
            if o.col in schema['columns'] and not self._is_pii(table, o.col):
                parts.append(f"{self._safe_column(table, o.col)} {o.direction}")
        if parts:
            return "ORDER BY " + ", ".join(parts)
        return ""

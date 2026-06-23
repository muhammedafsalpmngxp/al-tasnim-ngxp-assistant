import pyodbc
import pandas as pd
from pathlib import Path

# ==========================
# DATABASE CONFIG
# ==========================
SERVER = "20.98.112.250"
DATABASE = "AppMasterDB_Local"
USERNAME = "SA"
PASSWORD = "Complex@123"

TABLES = [
    "ActivityCodesNorms",
    "ActivityMasterCSV",
    "ActivityMasterMapping",
    "ActivityMasterMapping_New",
    "ActivityTaskPlan",
    "Company",
    "CrewEmployee",
    "CrewEquipment",
    "crews",
    "CrewType",
    "CrewTypeEmployee",
    "CrewTypeEquipment",
    "Dashboarding_Job_Progress",
    "Employee",
    "EmployeeType",
    "Job_Progress_PlanSnapshot",
    "OLD_NEW_CREW_MAPPING",
    "PH_Productivity",
    "ProjectIDs",
    "Revenue",
    "SAP_DRILLING_SEQUENCE_History",
    "task_daily",
    "TaskCrew",
    "TaskPlanCSVImport",
    "WBS_Master_Tracker_",
    "WellMonitoringReport_Latest",
    "WellMonitoringReportOptimized",
    "WMR"
]

# ==========================
# CONNECT
# ==========================

conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"UID={USERNAME};"
    f"PWD={PASSWORD};"
    f"TrustServerCertificate=yes;"
)

conn = pyodbc.connect(conn_str)

output_file = Path("database_schema_details.txt")

with open(output_file, "w", encoding="utf-8") as f:

    f.write("=" * 120 + "\n")
    f.write("DATABASE SCHEMA DETAILS\n")
    f.write("=" * 120 + "\n\n")

    for table in TABLES:

        print(f"Processing {table}")

        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write(f"TABLE: {table}\n")
        f.write("=" * 120 + "\n\n")

        # ------------------------
        # Row Count
        # ------------------------
        try:
            row_count = pd.read_sql(
                f"SELECT COUNT(*) cnt FROM [{table}]",
                conn
            ).iloc[0]["cnt"]

            f.write(f"ROW COUNT: {row_count}\n\n")

        except Exception as e:
            f.write(f"ROW COUNT ERROR: {e}\n\n")

        # ------------------------
        # COLUMN DETAILS
        # ------------------------

        column_query = f"""
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            CHARACTER_MAXIMUM_LENGTH,
            NUMERIC_PRECISION,
            NUMERIC_SCALE,
            IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table}'
        ORDER BY ORDINAL_POSITION
        """

        try:

            cols = pd.read_sql(column_query, conn)

            f.write("COLUMN DETAILS\n")
            f.write("-" * 80 + "\n")

            for _, row in cols.iterrows():

                col = row["COLUMN_NAME"]

                f.write(f"\nCOLUMN: {col}\n")
                f.write(f"TYPE: {row['DATA_TYPE']}\n")
                f.write(
                    f"MAX_LENGTH: {row['CHARACTER_MAXIMUM_LENGTH']}\n"
                )
                f.write(
                    f"PRECISION: {row['NUMERIC_PRECISION']}\n"
                )
                f.write(
                    f"SCALE: {row['NUMERIC_SCALE']}\n"
                )
                f.write(
                    f"NULLABLE: {row['IS_NULLABLE']}\n"
                )

                # Sample Values

                try:

                    sample_query = f"""
                    SELECT TOP 5 [{col}]
                    FROM [{table}]
                    WHERE [{col}] IS NOT NULL
                    """

                    samples = pd.read_sql(
                        sample_query,
                        conn
                    )

                    vals = (
                        samples[col]
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    f.write(
                        f"SAMPLE_VALUES: {vals}\n"
                    )

                except:
                    pass

            f.write("\n")

        except Exception as e:
            f.write(f"COLUMN ERROR: {e}\n")

        # ------------------------
        # PRIMARY KEYS
        # ------------------------

        pk_query = f"""
        SELECT KU.COLUMN_NAME
        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS AS TC
        INNER JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS KU
            ON TC.CONSTRAINT_NAME = KU.CONSTRAINT_NAME
        WHERE
            TC.TABLE_NAME = '{table}'
            AND TC.CONSTRAINT_TYPE = 'PRIMARY KEY'
        """

        try:

            pk = pd.read_sql(pk_query, conn)

            f.write("\nPRIMARY KEYS\n")
            f.write("-" * 80 + "\n")

            if len(pk):
                for c in pk["COLUMN_NAME"]:
                    f.write(f"{c}\n")
            else:
                f.write("NONE\n")

        except:
            pass

        # ------------------------
        # FOREIGN KEYS
        # ------------------------

        fk_query = f"""
        SELECT
            parent_col.name AS ColumnName,
            referenced_obj.name AS ReferencedTable,
            referenced_col.name AS ReferencedColumn
        FROM sys.foreign_key_columns fkc
        JOIN sys.objects parent_obj
            ON fkc.parent_object_id = parent_obj.object_id
        JOIN sys.columns parent_col
            ON fkc.parent_object_id = parent_col.object_id
            AND fkc.parent_column_id = parent_col.column_id
        JOIN sys.objects referenced_obj
            ON fkc.referenced_object_id = referenced_obj.object_id
        JOIN sys.columns referenced_col
            ON fkc.referenced_object_id = referenced_col.object_id
            AND fkc.referenced_column_id = referenced_col.column_id
        WHERE parent_obj.name = '{table}'
        """

        try:

            fk = pd.read_sql(fk_query, conn)

            f.write("\nFOREIGN KEYS\n")
            f.write("-" * 80 + "\n")

            if len(fk):

                for _, row in fk.iterrows():

                    f.write(
                        f"{row['ColumnName']} -> "
                        f"{row['ReferencedTable']}."
                        f"{row['ReferencedColumn']}\n"
                    )

            else:
                f.write("NONE\n")

        except:
            pass

        # ------------------------
        # SAMPLE DATA
        # ------------------------

        try:

            sample_rows = pd.read_sql(
                f"SELECT TOP 10 * FROM [{table}]",
                conn
            )

            f.write("\nTOP 10 SAMPLE ROWS\n")
            f.write("-" * 80 + "\n")

            f.write(sample_rows.to_string())
            f.write("\n\n")

        except Exception as e:

            f.write(f"SAMPLE DATA ERROR: {e}\n")

print(f"\nSaved: {output_file}")
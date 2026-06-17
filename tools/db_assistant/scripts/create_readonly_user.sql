-- scripts/create_readonly_user.sql

USE [AppMasterDB]
GO

-- Create login
CREATE LOGIN [NGXP_ReadOnly] 
WITH PASSWORD = 'REPLACE_WITH_SECURE_PASSWORD', 
     CHECK_POLICY = ON,
     DEFAULT_DATABASE = [AppMasterDB]
GO

-- Create user
CREATE USER [NGXP_ReadOnly] 
FOR LOGIN [NGXP_ReadOnly]
WITH DEFAULT_SCHEMA = [dbo]
GO

-- Add to db_denydatawriter (denies INSERT/UPDATE/DELETE)
EXEC sp_addrolemember 'db_denydatawriter', 'NGXP_ReadOnly'
GO

-- Grant SELECT on required tables
DECLARE @sql NVARCHAR(MAX) = ''
SELECT @sql = @sql + 'GRANT SELECT ON [' + TABLE_SCHEMA + '].[' + TABLE_NAME + '] TO [NGXP_ReadOnly];'
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_NAME IN (
    '2026_Well_Delivery_Scope_Well_Type',
    'ActivityTaskPlan',
    'WMR',
    'Employee',
    'crews',
    'CrewEmployee',
    'Equipment',
    'task_daily',
    'Revenue',
    'SAP_DRILLING_SEQUENCE'
)
EXEC sp_executesql @sql
GO

-- Deny any remaining write operations
DENY INSERT, UPDATE, DELETE, EXECUTE, ALTER TO [NGXP_ReadOnly]
GO

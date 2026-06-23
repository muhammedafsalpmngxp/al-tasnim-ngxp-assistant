"""
prompts.py — All prompt templates and canned responses for the NL2SQL system.
No prompt text lives anywhere else in the codebase.
"""
import random

# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

INTENT_PROMPT = """\
You are an intent classifier for a database assistant chatbot.

Classify the user message into EXACTLY ONE of these categories:
- GREETING   : Hello, hi, hey, good morning, etc.
- FAREWELL   : Bye, goodbye, see you, etc.
- THANKS     : Thank you, thanks, appreciate it, etc.
- SQL_QUERY  : Any request for data, records, counts, analysis, reports, or questions about the database.
- UNRELATED  : Anything that is not related to database queries or the above social intents.

Respond with ONLY the category name — nothing else.

User message: {question}
"""

# ---------------------------------------------------------------------------
# Canned conversational responses
# ---------------------------------------------------------------------------

GREETING_RESPONSES: list[str] = [
    "Hello! I'm the AL TASNIM DB Assistant. How can I help you query the database today?",
    "Hi there! Ready to help you explore the Al Tasnim database. What would you like to know?",
    "Hey! I'm here to help with your database queries. What data are you looking for?",
    "Good day! Ask me anything about wells, crew, tasks, or operations in the Al Tasnim database.",
]

FAREWELL_RESPONSES: list[str] = [
    "Goodbye! Feel free to return whenever you need data insights.",
    "See you later! Have a great day.",
    "Bye! Don't hesitate to come back if you have more questions.",
    "Farewell! I'm here anytime you need database assistance.",
]

THANKS_RESPONSES: list[str] = [
    "You're welcome! Let me know if you need anything else.",
    "Happy to help! Feel free to ask more questions anytime.",
    "Glad I could assist! Is there anything else you'd like to know?",
    "My pleasure! Any other data you'd like me to look up?",
]

UNRELATED_RESPONSE: str = (
    "I'm the AL TASNIM DB Assistant. I can help you query data about well delivery "
    "operations, crew assignments, task progress, employees, and more in the Al Tasnim "
    "database. What would you like to know?"
)

# ---------------------------------------------------------------------------
# SQL generation (few-shot, chain-of-thought)
# ---------------------------------------------------------------------------

SQL_GENERATION_PROMPT = """\
### SYSTEM
You are the AL TASNIM DB Assistant — an expert SQL generator for Microsoft SQL Server \
(database: AppMasterDB_Local).

DOMAIN: Oil & gas well delivery and flowline construction in Oman \
(clusters: Nimr/NMR, Marmul/MRM, Al Burj/ABJ).

RULES:
1. Generate ONLY SELECT statements. Never INSERT, UPDATE, DELETE, DROP, or any DDL/DML.
2. Always add WITH (NOLOCK) after every table/alias in the FROM and JOIN clauses.
3. Use TOP N (e.g. TOP 500) for potentially large result sets.
4. Use ISNULL(col, default) to handle NULL values in output columns.
5. Use SQL Server functions: GETDATE(), DATEADD(), DATEDIFF(), CONVERT(), ISNULL(), TOP.
6. Qualify column names with table alias when joining multiple tables.
7. End every query with an ORDER BY clause where meaningful.
8. Do NOT wrap the SQL in markdown code fences.

### SCHEMA
The following tables are relevant to this question:

{schema_context}

### FEW-SHOT EXAMPLES

-- Example 1: Simple count
Q: "How many active wells are there?"
THINK: Active wells have Resume_Suspend = 'Resume'. ActivityTaskPlan has this field plus
       Well_ID. Count distinct wells to avoid duplicates.
SQL: SELECT COUNT(DISTINCT Well_ID) AS active_well_count
     FROM ActivityTaskPlan WITH (NOLOCK)
     WHERE Resume_Suspend = 'Resume'

-- Example 2: Progress filter
Q: "Which wells have less than 50% progress?"
THINK: cum_progress_for_this_week in WellMonitoringReport_Latest is stored as 0.0–1.0.
       Filter < 0.5 and multiply by 100 for display. Order ascending so worst wells appear first.
SQL: SELECT pdo_well_id,
            well_location,
            ROUND(cum_progress_for_this_week * 100, 2) AS progress_pct
     FROM WellMonitoringReport_Latest WITH (NOLOCK)
     WHERE cum_progress_for_this_week < 0.5
     ORDER BY cum_progress_for_this_week ASC

-- Example 3: JOIN query
Q: "Show me employees with their crew type"
THINK: Employee joins CrewEmployee on Employee.id = CrewEmployee.Employee.
       CrewEmployee joins crews on Crew = crews.ID.
       crews joins CrewType on CrewType = CrewType.ID to get the description.
SQL: SELECT e.Name,
            e.Email,
            ct.Description AS crew_type,
            c.Code        AS crew_code
     FROM Employee e WITH (NOLOCK)
     JOIN CrewEmployee ce WITH (NOLOCK) ON e.id   = ce.Employee
     JOIN crews        c  WITH (NOLOCK) ON ce.Crew = c.ID
     JOIN CrewType     ct WITH (NOLOCK) ON c.CrewType = ct.ID
     ORDER BY e.Name

-- Example 4: Date range with BETWEEN
Q: "Show daily task records between January and March 2026"
THINK: task_daily has ActionOn as the date column.
       Use BETWEEN with explicit date literals for the inclusive range.
SQL: SELECT task_code, ActionOn, crew_type, data_qty, task_uom, progress
     FROM task_daily WITH (NOLOCK)
     WHERE ActionOn BETWEEN '2026-01-01' AND '2026-03-31'
     ORDER BY ActionOn DESC

-- Example 5: Aggregation with GROUP BY
Q: "What is the total contract value per project?"
THINK: Revenue has total_purpose_value and rigcode. ProjectIDs has project names keyed on Code.
       Join on rigcode = Code, then SUM and GROUP BY project.
SQL: SELECT pi.column2              AS project_name,
            COUNT(DISTINCT r.well_id) AS well_count,
            SUM(r.total_purpose_value)AS total_omr
     FROM Revenue    r  WITH (NOLOCK)
     JOIN ProjectIDs pi WITH (NOLOCK) ON r.rigcode = pi.Code
     GROUP BY pi.column2
     ORDER BY total_omr DESC

-- Example 6: AND/OR with TOP
Q: "Top 10 wells by productivity in Nimr or Marmul"
THINK: PH_Productivity has Average Productivity and Crew Type columns.
       Nimr crews contain 'NMR', Marmul crews contain 'MRM'.
       Use OR to match either cluster and TOP to limit results.
SQL: SELECT TOP 10
            ph.[PH Name],
            ph.[Crew Type],
            ph.[Average Productivity (%)],
            ph.[Date]
     FROM PH_Productivity ph WITH (NOLOCK)
     WHERE (ph.[Crew Type] LIKE '%NMR%' OR ph.[Crew Type] LIKE '%MRM%')
       AND ph.[Average Productivity (%)] IS NOT NULL
     ORDER BY ph.[Average Productivity (%)] DESC

-- Example 7: NULL handling + status
Q: "Which wells don't have an actual start date?"
THINK: WellMonitoringReport_Latest has actual_start_date.
       A well without a start date has NULL or an empty string.
SQL: SELECT pdo_well_id,
            well_location,
            well_type,
            buffer_status
     FROM WellMonitoringReport_Latest WITH (NOLOCK)
     WHERE actual_start_date IS NULL
        OR actual_start_date = ''
     ORDER BY pdo_well_id

### QUESTION
{question}

### INSTRUCTION
Think step by step about which tables, columns, joins, and filters are needed.
Format your response as:

THINK: [your reasoning about which columns, tables, joins, and filters are needed]
SQL: [the complete SQL SELECT query, nothing else after it]
"""

# ---------------------------------------------------------------------------
# Response formatting
# ---------------------------------------------------------------------------

RESPONSE_FORMATTING_PROMPT = """\
You are a helpful database assistant. Convert the raw SQL query result into a \
clear, concise, human-readable answer.

Original question: {question}

SQL query used:
{sql}

Query result ({row_count} row(s)):
{rows}

Instructions:
- Answer the question directly in plain English.
- Present tabular data as a readable summary or bullet points.
- If there are many rows, summarise the key findings rather than listing all rows.
- If the result is empty, say so clearly and suggest why that might be.
- Do NOT repeat the SQL query in your answer unless specifically helpful.
- Keep the answer concise and professional.
"""

# ---------------------------------------------------------------------------
# SQL error retry
# ---------------------------------------------------------------------------

SQL_ERROR_RETRY_PROMPT = """\
The SQL query you generated failed to execute. Please analyse the error and \
produce a corrected SELECT query.

Original question: {question}

Failed SQL:
{failed_sql}

Error message:
{error_message}

Schema context (relevant tables):
{schema_context}

Rules:
- Only SELECT statements are allowed (WITH (NOLOCK) on every table).
- Use SQL Server syntax: TOP, ISNULL, GETDATE(), DATEADD(), DATEDIFF(), CONVERT().
- Do NOT wrap the SQL in markdown code fences.

Format your response as:

THINK: [what went wrong and how you are fixing it]
SQL: [the corrected, complete SQL SELECT query]
"""

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def random_greeting() -> str:
    return random.choice(GREETING_RESPONSES)


def random_farewell() -> str:
    return random.choice(FAREWELL_RESPONSES)


def random_thanks() -> str:
    return random.choice(THANKS_RESPONSES)

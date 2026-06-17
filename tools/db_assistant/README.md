# Database Assistant - AI-Powered SQL Query Tool

A production-ready database assistant that converts natural language questions to SQL SELECT queries using LLM, with built-in security, clarification handling, and dynamic value resolution.

## Features

- **SELECT-only guaranteed** - LLM never writes SQL, deterministic builder
- **Natural language understanding** - Handles typos, abbreviations, business terms
- **Clarification loop** - Asks for missing information until query is complete
- **Dynamic value resolution** - No hardcoded values, learns from live database
- **Snapshot-aware** - Handles latest-per-well logic for monitoring tables
- **PII protection** - Pattern-based personal data blocking
- **Production ready** - Connection pooling, timeouts, error handling

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/db_assistant.git
cd db_assistant

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.template .env

# Edit .env with your values
nano .env
```

## Quick Start

### 1. Create Read-Only Database User

```bash
sqlcmd -S your_server -U your_admin -P your_password -i scripts/create_readonly_user.sql
```

### 2. Start the Server

```bash
# Using Python directly
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --workers 1

# Or using the run.py script
python run.py
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the status of rig 104?"}'
```

## Usage Examples

```python
questions = [
    "What is the status of rig 104?",
    "Show me all wells in NIMR E field",
    "How many wells are in NIMR E field?",
    "What tasks are planned for Well 33151?",
    "What is the overall progress of Well 33151?",
    "Show me employees in crew 320",
    "Which equipment is at location 55?",
    "whats the status",  # Will ask for clarification
]
```

## Architecture

```
User Question → Intent Parser (LLM) → Structured Intent → SQL Builder → Read-Only DB → Response Formatter (LLM) → Answer
```

## Project Structure

```
db_assistant/
├── src/                    # Core source code
│   ├── config.py           # Configuration
│   ├── models.py           # Pydantic models
│   ├── connection_pool.py  # DB connection pool
│   ├── schema_introspector.py  # Dynamic schema
│   ├── value_resolver.py   # Dynamic value resolution
│   ├── conversation.py     # Stateful conversations
│   ├── intent_parser.py    # LLM → Intent
│   ├── sql_builder.py      # Intent → SQL
│   ├── db_executor.py      # Read-only execution
│   ├── response_formatter.py  # LLM answer formatting
│   └── server.py           # FastAPI service
├── tests/                  # Test suite
│   ├── test_core.py        # Core tests
│   └── eval_harness.py     # Evaluation
├── scripts/                # Utility scripts
│   └── create_readonly_user.sql
├── .env.template           # Environment template
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Running Tests

```bash
python -m pytest tests/
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DB_SERVER` | SQL Server hostname/IP |
| `DB_NAME` | Database name |
| `DB_READONLY_USER` | Read-only database user |
| `DB_READONLY_PASSWORD` | Read-only database password |
| `MODEL_NAME` | Ollama model name |
| `LLM_TIMEOUT_SEC` | LLM request timeout |
| `MAX_ROWS` | Max rows to return |

## Security

- **Read-only database user**: The DB user has only SELECT permissions
- **Parameterized queries**: All values are bound, never interpolated
- **PII protection**: Patterns block personal data from queries
- **SQL injection prevention**: Identifiers are escaped and validated

## License

MIT

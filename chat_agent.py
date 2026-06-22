import os
import re
import sys
import pyodbc
import requests
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in .env")
    sys.exit(1)

genai.configure(api_key=api_key)
# Defaulting to 1.5-pro as fallback for model format just in case
model_name_env = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
# Sanitize: "Gemini 2.5 Pro" -> "gemini-2.5-pro"
model_name = model_name_env.strip().lower().replace(" ", "-")
model = genai.GenerativeModel(model_name)

# Read prompt
try:
    with open("sql_agent_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
except FileNotFoundError:
    print("Error: sql_agent_prompt.txt not found. Please ensure it's in the same directory.")
    sys.exit(1)

# Connect to database
server = os.environ.get("DB_SERVER")
database = os.environ.get("DB_NAME")
username = os.environ.get("DB_USER")
password = os.environ.get("DB_PASSWORD")
driver = os.environ.get("DB_DRIVER", "{ODBC Driver 17 for SQL Server}")

connection_string = f"DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}"
if os.environ.get("DB_ENCRYPT") == "yes":
    connection_string += ";Encrypt=yes"
if os.environ.get("DB_TRUST_CERT") == "yes":
    connection_string += ";TrustServerCertificate=yes"

try:
    print("Connecting to SQL Server...")
    conn = pyodbc.connect(connection_string)
    print("Connected successfully!\n")
except Exception as e:
    print(f"Database connection error: {e}")
    sys.exit(1)

def call_ollama(prompt):
    if os.environ.get("FALLBACK_ENABLED", "1") == "0":
        raise Exception("All fallback LLMs failed and Ollama is disabled.")
        
    url = os.environ.get("FALLBACK_LLM_URL", "http://localhost:11434/api/chat")
    fallback_model = os.environ.get("FALLBACK_LLM_MODEL", "deepseek-r1:8b")
    
    headers = {"Content-Type": "application/json"}
    
    if "v1/chat/completions" in url:
        data = {
            "model": fallback_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            print(f"    [!] Ollama OpenAI API Error: {response.text}")
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    else:
        data = {
            "model": fallback_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 8192
            }
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            print(f"    [!] Ollama API Error: {response.text}")
        response.raise_for_status()
        return response.json()["message"]["content"]

def call_groq(prompt):
    url = os.environ.get("LLM_URL")
    groq_api_key = os.environ.get("LLM_API_KEY")
    groq_model = os.environ.get("LLM_MODEL")
    
    if not groq_api_key:
        return call_ollama(prompt)
        
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": groq_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"    [!] Groq API failed ({e}). Falling back to Local Ollama...")
        return call_ollama(prompt)

def call_llm(prompt):
    # 1. Try Gemini
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower() or "400" in error_str:
            print("    [!] Gemini API quota exceeded or error. Falling back to Groq...")
            # 2. Fallback to Groq -> Ollama
            return call_groq(prompt)
        else:
            raise e

# Interactive loop
print("=====================================================")
print("  [Agent] Welcome to the Agentic SQL Chat Interface!")
print("  Ask questions about the database in plain English.")
print("  Type 'exit' or 'quit' to stop.")
print("=====================================================\n")

while True:
    try:
        user_input = input("\nYou: ")
    except EOFError:
        break
        
    if user_input.strip().lower() in ['exit', 'quit']:
        break
    if not user_input.strip():
        continue
        
    print("\n[Agent] Thinking of the right SQL query...")
    
    # Prompt LLM for SQL
    sql_request = f"{system_prompt}\n\nYou must ONLY respond with the MS SQL query inside a markdown code block (```sql ... ```). Do not include any other text.\nUser: {user_input}\nSQL:\n"
    
    try:
        reply = call_llm(sql_request)
        
        # Extract SQL from markdown block
        match = re.search(r'```(?:sql)?\s*(.*?)\s*```', reply, re.DOTALL | re.IGNORECASE)
        if match:
            sql_query = match.group(1).strip()
        else:
            # Fallback if the LLM didn't use code blocks
            sql_query = reply.strip()
            
        print(f"[Agent] Executing Query:\n{sql_query}\n")
        
        # Execute SQL
        cursor = conn.cursor()
        cursor.execute(sql_query)
        
        try:
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
            
            if not rows:
                results_str = "No rows returned."
            else:
                results_str = "\t".join(columns) + "\n"
                for i, row in enumerate(rows):
                    if i >= 50:
                        results_str += f"... and {len(rows) - 50} more rows truncated.\n"
                        break
                    results_str += "\t".join([str(val) for val in row]) + "\n"
        except pyodbc.ProgrammingError:
            results_str = "Query executed successfully, but returned no result set."

        print(f"[Agent] Database Returned:\n{results_str}\n")
        
        # Send data back to LLM for explanation
        print("[Agent] Interpreting the results for you...")
        explain_request = f"""
The user asked this question: "{user_input}"

I executed the following SQL query:
```sql
{sql_query}
```

And I got these results from the database:
{results_str}

Please provide a friendly, natural language explanation of these results to answer the user's question directly. Keep it concise. Do not show the SQL query in your response.
"""
        explain_reply = call_llm(explain_request)
        print(f"\n[Agent] Answer:\n{explain_reply}")
        print("\n" + "-"*50)
        
    except pyodbc.Error as e:
        print(f"\n[Database Error]: {e}")
        print("The SQL query generated was invalid or there is a database issue. Try rephrasing.")
        print("\n" + "-"*50)
    except Exception as e:
        print(f"\n[Error]: {e}")
        print("\n" + "-"*50)

conn.close()

import os
import json
import re
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Standard Setup ---
load_dotenv()
app = FastAPI(title="Supabase Direct Query Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize Clients ---
try:
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_ANON_KEY")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    supabase: Client = create_client(url, key)
    print("✅ Successfully connected to Supabase and OpenAI.")
except Exception as e:
    print(f"🔥 ERROR: Could not initialize services. Details: {e}")

class ChatRequest(BaseModel):
    message: str

# --- Enhanced Conversational Helper ---
def get_conversational_reply(question: str) -> str | None:
    """Handles simple, non-database questions with greeting and date support."""
    q_lower = question.lower().strip()
    today = datetime.now().date()

    # --- Greeting & Farewell Handling ---
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "namaste"]
    thanks = ["thanks", "thank you", "shukriya"]

    if any(greet in q_lower for greet in greetings):
        return "Hello! How can I help you with your tasks today? 😊"
    
    if any(thank in q_lower for thank in thanks):
        return "You're welcome! Let me know if you need anything else."

    # --- Date Handling ---
    if "parso" in q_lower:
        if "tha" in q_lower or "was" in q_lower:
            day = today - timedelta(days=2)
            return f"The day before yesterday was {day.strftime('%B %d, %Y')}."
        else:
            day = today + timedelta(days=2)
            return f"The day after tomorrow will be {day.strftime('%B %d, %Y')}."

    if "kal" in q_lower:
        if "tha" in q_lower or "was" in q_lower:
            day = today - timedelta(days=1)
            return f"Yesterday was {day.strftime('%B %d, %Y')}."
        else:
            day = today + timedelta(days=1)
            return f"Tomorrow will be {day.strftime('%B %d, %Y')}."

    if "yesterday" in q_lower:
        day = today - timedelta(days=1)
        return f"Yesterday's date was {day.strftime('%B %d, %Y')}."
        
    if "tomorrow" in q_lower or "tommorow" in q_lower:
        day = today + timedelta(days=1)
        return f"Tomorrow's date will be {day.strftime('%B %d, %Y')}."

    today_keywords = ["date", "dinank", "taarikh", "aaj", "today"]
    if any(keyword in q_lower for keyword in today_keywords):
        return f"Today's date is {today.strftime('%B %d, %Y')}."

    return None

@app.post("/chat")
async def ask_question(chat_request: ChatRequest):
    user_question = chat_request.message.strip()
    print(f"\n--- New Request ---")
    print(f"💬 User Question: {user_question}")

    # Step 1: Handle simple conversational questions first
    conversational_reply = get_conversational_reply(user_question)
    if conversational_reply:
        print(f"✅ Responded conversationally: {conversational_reply}")
        return {"reply": conversational_reply}

    # Step 2: If not conversational, use AI to interpret the user's question
    params = None
    for attempt in range(5):
        try:
            print(f"🧠 AI Interpretation Attempt #{attempt + 1}...")
            today = datetime.now().date()
            
            # --- More Accurate & Robust System Prompt ---
            system_prompt = f"""
            You are a precise database query assistant. Your ONLY job is to convert a user's question into a pure JSON object based on the rules below.

            **Context:**
            - Today's date is: {today.isoformat()}.
            - Available tables: 'checklist', 'delegation'.
            - Date-related columns: 'Task Start Date', 'Planned Date', 'Actual'.

            **Column Mapping Guide:**
            - "who", "person", "name", "employee" -> For 'checklist' table use 'Assigned To'. For 'delegation' table use 'Name'.
            - "what", "task", "description" -> 'Task Description'.
            - "status", "done", "pending", "complete" -> 'Status'.
            - "ID", "number" -> 'Task ID'.

            **JSON Output Rules:**
            1.  **`table`**: MUST be 'checklist', 'delegation', or 'both'. If the user does not specify, default to 'both'.
            2.  **`operation`**: MUST be one of 'COUNT', 'SELECT', 'COUNT_TOTAL', 'SELECT_ALL'.
            3.  **`column`**: The specific database column to query.
            4.  **`value`**: The value to search for.
            5.  **Status Normalization**: For the 'Status' column, the `value` in the JSON MUST be exactly `[COMPLETED]` for words like 'complete', 'done', 'yes'. It MUST be `[INCOMPLETE]` for words like 'pending', 'no', 'incomplete', or if the status is blank/null.
            6.  **Date Logic**: For keywords like "today", "yesterday", "tomorrow", the `value` MUST be a JSON object with "gte" and "lte" keys in 'YYYY-MM-DD' format.
            7.  **Number Ranges**: For ranges on "Task ID" (e.g., "tasks 320 to 330"), the `value` MUST be a JSON object with "gte" and "lte" keys.
            8.  **SELECT_ALL**: If the user asks to "list all", "show all", or "describe all" tasks, you MUST use the 'SELECT_ALL' operation. This operation does not require a 'column' or 'value'.

            **Strict Examples:**
            - "how many tasks are done" -> {{"table": "both", "operation": "COUNT", "column": "Status", "value": "[COMPLETED]"}}
            - "how many tasks are pending in the checklist" -> {{"table": "checklist", "operation": "COUNT", "column": "Status", "value": "[INCOMPLETE]"}}
            - "tell me about task id 321" -> {{"table": "checklist", "operation": "SELECT", "column": "Task ID", "value": 321}}
            - "describe all delegation tasks" -> {{"table": "delegation", "operation": "SELECT_ALL"}}
            - "show me tasks 320-330" -> {{"table": "checklist", "operation": "SELECT", "column": "Task ID", "value": {{"gte": 320, "lte": 330}}}}
            - "tasks that started yesterday" -> {{"table": "both", "operation": "SELECT", "column": "Task Start Date", "value": {{"gte": "{(today - timedelta(days=1)).isoformat()}", "lte": "{(today - timedelta(days=1)).isoformat()}"}}}}
            - "list tasks assigned to ankit" -> {{"table": "both", "operation": "SELECT", "column": "Assigned To", "value": "ankit"}}
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ]
            )
            
            raw_response_text = response.choices[0].message.content
            print(f"🤖 Raw AI Response:\n---------------------------------\n{raw_response_text}\n---------------------------------")

            params = json.loads(raw_response_text)
            print(f"✅ Successfully Parsed JSON: {params}")
            break 
        
        except Exception as e:
            print(f"🔥 AI Error on attempt {attempt + 1}: {e}")
            if attempt < 4:
                time.sleep(1) 
            else:
                return {"reply": "Sorry, I'm having trouble understanding the question right now after 5 attempts."}

    # Step 3: Query the database using the interpreted parameters
    try:
        table_param = params.get('table')
        operation = params.get('operation')
        
        if not table_param or not operation:
            return {"reply": "I'm missing a table or operation to perform the query."}
        
        tables_to_query = ['checklist', 'delegation'] if table_param == 'both' else [table_param]
        all_rows = []

        if operation in ['COUNT_TOTAL', 'SELECT_ALL']:
                for table in tables_to_query:
                    result = supabase.table(table).select('*').execute()
                    if result.data:
                        for row in result.data:
                            row['sheetName'] = table
                        all_rows.extend(result.data)
        else: 
            column = params.get('column')
            value = params.get('value')
            if not all([column, value is not None]):
                return {"reply": "I'm missing a column or value to search for."}

            status_synonyms = {
                '[COMPLETED]': ['complete', 'done', 'yes', 'Completed'],
                '[INCOMPLETE]': ['pending', 'no', 'incomplete', 'Pending']
            }
            
            # This logic supports searching for a name in both 'Assigned To' and 'Name' columns if the AI defaults to one.
            columns_to_search = [column]
            if column in ["Assigned To", "Name"]:
                columns_to_search = ["Assigned To", "Name"]

            for table in tables_to_query:
                for col_to_search in columns_to_search:
                    query = supabase.table(table)
                    
                    if isinstance(value, dict) and 'gte' in value and 'lte' in value:
                        result = query.select('*').gte(col_to_search, value['gte']).lte(col_to_search, value['lte']).execute()
                    elif col_to_search == 'Status' and value in status_synonyms:
                        synonyms = status_synonyms[value]
                        filter_string = ",".join([f"{col_to_search}.ilike.%{syn}%" for syn in synonyms])
                        if value == '[INCOMPLETE]':
                            filter_string += f",{col_to_search}.is.null"
                        result = query.select('*').or_(filter_string).execute()
                    elif "Task ID" in col_to_search and str(value).isdigit():
                        result = query.select('*').eq(col_to_search, int(value)).execute()
                    else:
                        result = query.select('*').ilike(col_to_search, f'%{str(value)}%').execute()
                    
                    if result.data:
                        for row in result.data:
                            row['sheetName'] = table
                        all_rows.extend(result.data)

        # --- Process and Format Results ---
        if not all_rows:
            return {"reply": f"I couldn't find any tasks matching your criteria."}

        df = pd.DataFrame(all_rows)
        # Drop duplicates based on all columns except the ones we added
        cols_to_check = [col for col in df.columns if col not in ['id', 'sheetName']]
        df.drop_duplicates(subset=cols_to_check, inplace=True)
        df = df.astype(object).replace({pd.NA: 'N/A', np.nan: 'N/A'})

        if operation in ['COUNT', 'COUNT_TOTAL']:
            total_unique_count = len(df)
            result_text = f"I found a total of {total_unique_count} unique tasks matching your criteria."
            if table_param == 'both' and not df.empty:
                checklist_count = len(df[df['sheetName'] == 'checklist'])
                delegation_count = len(df[df['sheetName'] == 'delegation'])
                if checklist_count > 0 or delegation_count > 0:
                    result_text += f" ({checklist_count} from checklist, {delegation_count} from delegation)."
        
        elif operation in ['SELECT', 'SELECT_ALL']:
            unique_rows = df.to_dict('records')
            result_text = f"I found {len(unique_rows)} unique matching task(s):\n\n"
            for row in unique_rows[:10]: # Limit to first 10 results
                desc = row.get('Task Description', 'N/A')
                given_by = row.get('Given By', 'N/A')
                # Check both possible columns for the assigned person's name
                assigned_to = row.get('Name') if row.get('Name') != 'N/A' else row.get('Assigned To', 'N/A')
                status = row.get('Status', 'N/A')
                sheet_name = row.get('sheetName', 'unknown table') # Get the source table
                
                # --- Clearer Result Formatting ---
                result_text += (
                    f"- **Task**: '{desc}' (from *{sheet_name}*)\n"
                    f"  - **Given by**: {given_by}\n"
                    f"  - **Assigned to**: {assigned_to}\n"
                    f"  - **Status**: {status}\n"
                )
        else:
            result_text = "I understood the query, but the operation is not supported."

        print(f"✅ Final Answer: {result_text}")
        return {"reply": result_text}

    except Exception as e:
        print(f"🔥 An error occurred while querying the database: {e}")
        return {"reply": "Sorry, an error occurred on the server. Please check the terminal for details."}

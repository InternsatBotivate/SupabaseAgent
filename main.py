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

# --- Helper Function for Conversational AI ---
def get_conversational_reply(question: str) -> str | None:
    """Handles simple, non-database questions with improved Hinglish support."""
    q_lower = question.lower()
    today = datetime.now().date()
    
    # --- THIS IS THE ROBUST DATE FIX ---
    # Handle specific Hinglish/English keywords first.
    
    if "parso" in q_lower:
        if "tha" in q_lower or "was" in q_lower:
            day = today - timedelta(days=2)
            return f"The day before yesterday was {day.strftime('%B %d, %Y')}."
        else: # Assumes future if no past tense is given
            day = today + timedelta(days=2)
            return f"The day after tomorrow will be {day.strftime('%B %d, %Y')}."

    if "kal" in q_lower:
        if "tha" in q_lower or "was" in q_lower:
            day = today - timedelta(days=1)
            return f"Yesterday was {day.strftime('%B %d, %Y')}."
        else: # Assumes future if no past tense is given
            day = today + timedelta(days=1)
            return f"Tomorrow will be {day.strftime('%B %d, %Y')}."

    if "yesterday" in q_lower:
        day = today - timedelta(days=1)
        return f"Yesterday's date was {day.strftime('%B %d, %Y')}."
        
    if "tomorrow" in q_lower or "tommorow" in q_lower:
        day = today + timedelta(days=1)
        return f"Tomorrow's date will be {day.strftime('%B %d, %Y')}."

    # General date queries come last
    today_keywords = ["date", "dinank", "taarikh", "aaj", "today"]
    if any(keyword in q_lower for keyword in today_keywords):
        return f"Today's date is {today.strftime('%B %d, %Y')}."
    # ------------------------------------

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
            system_prompt = f"""
            You are a database query assistant. Your only job is to analyze a user's question and convert it into a pure JSON object.
            Today's date is {today.isoformat()}.
            The available tables are 'checklist' and 'delegation'.
            Date-related columns are 'Task Start Date', 'Planned Date', 'Actual'.

            **Operations & Rules:**
            1. Operations: 'COUNT', 'SELECT', 'COUNT_TOTAL', 'SELECT_ALL'.
            2. For number ranges on "Task ID" (e.g., "320-330"), the 'value' MUST be a JSON object with "gte" and "lte".
            3. For date keywords ("today", "yesterday"), 'value' MUST be a JSON object with "gte" and "lte" in 'YYYY-MM-DD' format.
            4. If the user asks to "list all" or "describe all" tasks, you MUST use the 'SELECT_ALL' operation.

            **Status Normalization:**
            - 'complete', 'done', 'yes' -> value becomes '[COMPLETED]'.
            - 'pending', 'no', 'incomplete', blank -> value becomes '[INCOMPLETE]'.

            **Table Selection:**
            - If a table is specified, use it. For general questions (e.g., "how many tasks are complete?"), set the table to 'both'.
            
            Examples:
            - "how many tasks are done" -> {{"table": "both", "operation": "COUNT", "column": "Status", "value": "[COMPLETED]"}}
            - "tell me about task id 321" -> {{"table": "checklist", "operation": "SELECT", "column": "Task ID", "value": 321}}
            - "describe all delegation tasks" -> {{"table": "delegation", "operation": "SELECT_ALL"}}
            - "show me tasks 320-330" -> {{"table": "checklist", "operation": "SELECT", "column": "Task ID", "value": {{"gte": 320, "lte": 330}}}}
            - "tasks that started yesterday" -> {{"table": "both", "operation": "SELECT", "column": "Task Start Date", "value": {{"gte": "{(today - timedelta(days=1)).isoformat()}", "lte": "{(today - timedelta(days=1)).isoformat()}"}}}}
            """
            
            # --- THIS IS THE CORRECT API CALL FOR CHAT MODELS ---
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
            
            for table in tables_to_query:
                query = supabase.table(table)
                
                if isinstance(value, dict) and 'gte' in value and 'lte' in value:
                    result = query.select('*').gte(column, value['gte']).lte(column, value['lte']).execute()
                elif column == 'Status' and value in status_synonyms:
                    synonyms = status_synonyms[value]
                    filter_string = ",".join([f"{column}.ilike.%{syn}%" for syn in synonyms])
                    if value == '[INCOMPLETE]':
                        filter_string += f",{column}.is.null"
                    result = query.select('*').or_(filter_string).execute()
                elif "Task ID" in column and str(value).isdigit():
                    result = query.select('*').eq(column, int(value)).execute()
                else:
                    result = query.select('*').ilike(column, f'%{str(value)}%').execute()
                
                if result.data:
                    for row in result.data:
                        row['sheetName'] = table
                    all_rows.extend(result.data)

        # --- Process and Format Results ---
        if not all_rows:
            return {"reply": f"I couldn't find any tasks matching your criteria."}

        df = pd.DataFrame(all_rows)
        cols_to_check = [col for col in df.columns if col not in ['id', 'sheetName']]
        df.drop_duplicates(subset=cols_to_check, inplace=True)
        df = df.astype(object).replace({pd.NA: 'N/A', np.nan: 'N/A'})


        if operation in ['COUNT', 'COUNT_TOTAL']:
            total_unique_count = len(df)
            result_text = f"I found a total of {total_unique_count} unique tasks matching your criteria."
            if table_param == 'both' and not df.empty:
                checklist_count = len(df.loc[df['sheetName'] == 'checklist'])
                delegation_count = len(df.loc[df['sheetName'] == 'delegation'])
                if checklist_count > 0 or delegation_count > 0:
                    result_text += f" ({checklist_count} from checklist, {delegation_count} from delegation)."
        
        elif operation in ['SELECT', 'SELECT_ALL']:
            unique_rows = df.to_dict('records')
            result_text = f"I found {len(unique_rows)} unique matching task(s):\n\n"
            for row in unique_rows[:10]:
                desc = row.get('Task Description', 'N/A')
                given_by = row.get('Given By', 'N/A')
                assigned_to = row.get('Name') if row.get('Name') != 'N/A' else row.get('Assigned To', 'N/A')
                status = row.get('Status', 'N/A')
                result_text += f"- **Task**: '{desc}'\n  - **Given by**: {given_by}\n  - **Assigned to**: {assigned_to}\n  - **Status**: {status}\n"
        else:
            result_text = "I understood the query, but the operation is not supported."

        print(f"✅ Final Answer: {result_text}")
        return {"reply": result_text}

    except Exception as e:
        print(f"🔥 An error occurred while querying the database: {e}")
        return {"reply": "Sorry, an error occurred on the server. Please check the terminal for details."}


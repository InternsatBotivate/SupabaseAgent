# settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Environment Variables ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")

# --- LlamaIndex Settings ---
PERSIST_DIR = "./storage"
MODEL = "gpt-3.5-turbo"
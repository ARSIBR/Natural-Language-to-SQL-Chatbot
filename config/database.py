# config/database.py
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq

def get_db_connection(db_path: str = "southco.db"):
    """Get a SQLDatabase connection."""
    return SQLDatabase.from_uri(f"sqlite:///{db_path}")

def get_llm(api_key: str = None):
    """Get an instance of the Groq LLM."""
    if not api_key:
        api_key = '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'
    
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=api_key
    )



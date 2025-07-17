import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "SQL Assistant API"
    DEBUG: bool = True
    
    DB_PATH: str = os.getenv("DB_PATH", "southco.db")
    
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "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")
    
    MAX_SQL_RETRIES: int = 3
    MAX_ROWS_RETURN: int = 100
    
    # Chat history settings
    CHAT_HISTORY_PATH: str = os.getenv("CHAT_HISTORY_PATH", "chat_history123.json")
    
    # Model settings
    VANNA_MODEL_PATH: str = os.getenv("VANNA_MODEL_PATH", "vanna_model")
    
    class Config:
        env_file = ".env"

# Instantiate settings
settings = Settings()
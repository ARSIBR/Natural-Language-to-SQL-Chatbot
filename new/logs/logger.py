import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

os.makedirs("logs", exist_ok=True)

def setup_logger(name, log_file=None, level=logging.INFO):
    """Setup logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if log_file:
        file_handler = RotatingFileHandler(
            filename=os.path.join("logs", log_file),
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Create different loggers
app_logger = setup_logger("app", "app.log")
workflow_logger = setup_logger("workflow", "workflow.log")
sql_logger = setup_logger("sql", "sql.log")
chat_logger = setup_logger("chat", "chat.log")
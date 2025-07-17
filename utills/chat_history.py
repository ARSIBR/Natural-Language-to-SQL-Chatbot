import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

class JSONChatHistory:
    def __init__(self, file_path="chat_history123.json"):
        self.file_path = file_path
        self.ensure_file_exists()

    def ensure_file_exists(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump([], f)

    def add_interaction(self, question, response, category=None, sql=None, sql_error=None, df=None, rephrased_question=None):
        try:
            with open(self.file_path, 'r') as f:
                history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            history = []

        interaction = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "rephrased_question": rephrased_question,
            "response": response,
            "metadata": {
                "category": category,
                "sql": sql,
                "sql_error": sql_error,
                "data": df
            }
        }

        history.append(interaction)

        with open(self.file_path, 'w') as f:
            json.dump(history, f, indent=2)

    def get_history(self):
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def get_recent_history(self, n=4):
        history = self.get_history()
        return history[-n:] if len(history) > 0 else []

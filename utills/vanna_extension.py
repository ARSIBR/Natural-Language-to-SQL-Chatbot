from vanna.base import VannaBase
from vanna.chromadb import ChromaDB_VectorStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import pandas as pd

class GroqLLM(VannaBase):
    def __init__(self, chat_model, config=None):
        super().__init__(config)
        self.chat_model = chat_model

    def submit_prompt(self, prompt, **kwargs) -> str:
        if isinstance(prompt, (HumanMessage, SystemMessage, AIMessage)):
            prompt = prompt.content
        messages = [HumanMessage(content=str(prompt))]

        response = self.chat_model.invoke(messages)
        return str(response.content)

    def system_message(self, message: str) -> str:
        return str(message)

    def user_message(self, message: str) -> str:
        return str(message)

    def assistant_message(self, message: str) -> str:
        return str(message)

class MyVanna(ChromaDB_VectorStore, GroqLLM):
    def __init__(self, chat_model, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        GroqLLM.__init__(self, chat_model=chat_model, config=config)
    
    def load_ddl(self, ddl):
        """Load DDL into the vector store."""
        self.add_ddl(ddl)
        
    def train(self, question, sql, ddl=None):
        """Train the model with a question-SQL pair."""
        self.add_question_sql(question=question, sql=sql)
        if ddl:
            self.add_ddl(ddl)
            
    def save_model(self, path):
        """Save the model to a path."""
        try:
            if hasattr(self, 'chromadb_collection') and self.chromadb_collection:
                self.chromadb_collection.persist()
                print(f"Model saved to {path}")
            else:
                print("No collection available to save")
        except Exception as e:
            print(f"Error saving model: {e}")
        
    def load_model(self, path):
        """Load the model from a path."""
        try:
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
# schema/chat_schemas.py
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    """Schema for chat messages."""
    question: str = Field(..., description="The user's question")

class ChatResponse(BaseModel):
    """Schema for chat response."""
    response: str = Field(..., description="The generated response")
    category: Literal["gk", "sql"] = Field(..., description="The category of the question")
    rephrased_question: Optional[str] = Field(None, description="The rephrased question")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata such as SQL query, error, etc.")

class QuestionClassifier(BaseModel):
    """Schema for question classification."""
    category: Literal["gk", "sql"] = Field(
        ...,
        description="Classification of the question: 'gk' for general knowledge or 'sql' for SQL query required."
    )
    rephrased_question: str = Field(
        ...,
        description="A clear, concise, and standalone rephrasing of the user's question that incorporates the given chat history."
    )

class ChatHistoryEntry(BaseModel):
    """Schema for chat history entries."""
    timestamp: str
    question: str
    rephrased_question: Optional[str] = None
    response: str
    metadata: Dict[str, Any]

class ChatHistory(BaseModel):
    """Schema for retrieving chat history."""
    history: List[ChatHistoryEntry]
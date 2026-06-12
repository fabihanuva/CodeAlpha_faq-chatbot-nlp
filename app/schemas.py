from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    score: float
    matched_question: Optional[str] = None
    category: Optional[str] = None
    matched: bool

class FAQItem(BaseModel):
    id: int
    question: str
    category: str

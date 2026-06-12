import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.engine import SemanticEngine
from app.schemas import ChatRequest, ChatResponse, FAQItem
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ShopBot API",
    description="Semantic Search API for FAQ Chatbot",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine = SemanticEngine()

@app.get("/", tags=["Health"])
async def health_check():
    return {"status": "online", "model": "all-MiniLM-L6-v2"}

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    try:
        logger.info(f"Processing query: {request.query}")
        result = engine.get_best_match(request.query)
        return result
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/faqs", response_model=List[FAQItem], tags=["Data"])
async def get_faqs():
    return engine.get_all_faqs()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from services.chat_service import ChatService

app = FastAPI(
    title="Medical Chatbot API",
    description="AI-powered medical chatbot for abdominal pain diagnosis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chat service
chat_service = ChatService()

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    response: str
    suggestions: List[str]
    timestamp: str

# API Routes
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        result = chat_service.process_message(
            session_id=request.session_id or chat_service.create_session(),
            message=request.message
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session")
async def create_session():
    """Create new chat session"""
    session_id = chat_service.create_session()
    return {"session_id": session_id}

@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    """Get chat history"""
    history = chat_service.get_session_history(session_id)
    return {"history": history}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "medical-chatbot"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
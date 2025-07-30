from typing import List, Dict, Optional
import uuid
from datetime import datetime

from models.chatbot_model import MedicalChatbotModel

class ChatService:
    """Main chat service handling conversations"""
    
    def __init__(self):
        self.chatbot_model = MedicalChatbotModel()
        self.active_sessions = {}
        self.load_models()
    
    def load_models(self):
        """Initialize chatbot models"""
        self.chatbot_model.load_models()
    
    def create_session(self) -> str:
        """Create new chat session"""
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            "messages": [],
            "created_at": datetime.now(),
            "user_context": {}
        }
        return session_id
    
    def process_message(self, session_id: str, message: str) -> Dict:
        """Process user message and generate response"""
        if session_id not in self.active_sessions:
            session_id = self.create_session()
        
        session = self.active_sessions[session_id]
        
        # Add user message to history
        session["messages"].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now()
        })
        
        # Generate response
        conversation_history = [msg["content"] for msg in session["messages"][-5:]]  # Last 5 messages
        response = self.chatbot_model.generate_response(message, conversation_history)
        
        # Add bot response to history
        session["messages"].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now()
        })
        
        # Generate suggestions
        suggestions = self._generate_suggestions(message, response)
        
        return {
            "session_id": session_id,
            "response": response,
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_suggestions(self, user_message: str, bot_response: str) -> List[str]:
        """Generate follow-up suggestions"""
        suggestions = []
        
        # Medical condition specific suggestions
        medical_info = self.chatbot_model.find_relevant_medical_info(user_message)
        if medical_info:
            condition = medical_info[0]["condition"].replace("_", " ")
            suggestions.extend([
                f"What causes {condition}?",
                f"How is {condition} treated?",
                f"When should I see a doctor for {condition}?",
                "What are the warning signs I should watch for?"
            ])
        
        # General medical suggestions
        if any(word in user_message.lower() for word in ["pain", "hurt", "ache"]):
            suggestions.extend([
                "How severe is the pain on a scale of 1-10?",
                "Where exactly is the pain located?",
                "When did the pain start?",
                "What makes the pain better or worse?"
            ])
        
        # Symptom-specific suggestions
        if "nausea" in user_message.lower():
            suggestions.append("Are you also experiencing vomiting?")
        
        if "fever" in user_message.lower():
            suggestions.append("What is your current temperature?")
        
        return suggestions[:4]  # Return max 4 suggestions
    
    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get chat history for session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]["messages"]
        return []
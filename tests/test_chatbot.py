import pytest
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.chat_service import ChatService
from app.models.chatbot_model import MedicalChatbotModel

class TestMedicalChatbot:
    """Test suite for medical chatbot"""
    
    def setup_method(self):
        """Setup test environment"""
        self.chat_service = ChatService()
        self.chatbot_model = MedicalChatbotModel()
    
    def test_greeting_responses(self):
        """Test greeting handling"""
        greetings = ["hello", "hi", "hey", "good morning"]
        
        for greeting in greetings:
            response = self.chatbot_model._handle_general_questions(greeting)
            assert response is not None
            assert "hello" in response.lower() or "hi" in response.lower()
    
    def test_medical_knowledge_retrieval(self):
        """Test medical information retrieval"""
        # Load models for testing
        self.chatbot_model.load_models()
        
        test_queries = [
            "sharp pain in upper right abdomen",
            "burning stomach pain",
            "nausea and vomiting",
            "appendicitis symptoms"
        ]
        
        for query in test_queries:
            medical_info = self.chatbot_model.find_relevant_medical_info(query)
            assert len(medical_info) > 0
            assert medical_info[0]["similarity"] > 0.3
    
    def test_session_creation(self):
        """Test chat session management"""
        session_id = self.chat_service.create_session()
        assert session_id is not None
        assert session_id in self.chat_service.active_sessions
    
    def test_message_processing(self):
        """Test message processing"""
        session_id = self.chat_service.create_session()
        
        test_message = "I have stomach pain"
        result = self.chat_service.process_message(session_id, test_message)
        
        assert "response" in result
        assert "suggestions" in result
        assert result["session_id"] == session_id
    
    def test_out_of_scope_questions(self):
        """Test handling of non-medical questions"""
        self.chatbot_model.load_models()
        
        out_of_scope_queries = [
            "What's the weather like?",
            "How do I cook pasta?",
            "What's the capital of France?"
        ]
        
        for query in out_of_scope_queries:
            response = self.chatbot_model.generate_response(query)
            assert "not trained" in response.lower() or "abdominal pain" in response.lower()
    
    def test_emergency_detection(self):
        """Test emergency symptom detection"""
        emergency_queries = [
            "severe persistent pain",
            "blood in vomit",
            "high fever and pain",
            "can't breathe and stomach hurts"
        ]
        
        self.chatbot_model.load_models()
        
        for query in emergency_queries:
            response = self.chatbot_model.generate_response(query)
            assert "emergency" in response.lower() or "immediate" in response.lower() or "urgent" in response.lower()

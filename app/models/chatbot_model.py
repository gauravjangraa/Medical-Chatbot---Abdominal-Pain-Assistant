import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    Trainer, TrainingArguments, 
    GPT2LMHeadModel, GPT2Tokenizer
)
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Dict
import pickle
import json


from data.medical_data import ABDOMINAL_PAIN_KNOWLEDGE

class MedicalChatbotModel:
    """Transformer-based medical chatbot model"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.embedding_model = None
        self.medical_embeddings = None
        self.medical_knowledge = ABDOMINAL_PAIN_KNOWLEDGE
        
    def load_models(self):
        """Load pretrained models"""
        try:
            # Load conversational model
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            
            # Add padding token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Generate medical knowledge embeddings
            self._generate_medical_embeddings()
            
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            
    def _generate_medical_embeddings(self):
        """Generate embeddings for medical knowledge"""
        medical_texts = []
        medical_labels = []
        
        for condition, info in self.medical_knowledge["conditions"].items():
            # Create comprehensive text for each condition
            text = f"{condition}: {info['description']} "
            text += f"Symptoms: {', '.join(info['symptoms'])}. "
            text += f"Causes: {', '.join(info['causes'])}. "
            text += f"Location: {info['location']}. "
            text += f"Severity: {info['severity']}."
            
            medical_texts.append(text)
            medical_labels.append(condition)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(medical_texts)
        self.medical_embeddings = {
            "texts": medical_texts,
            "embeddings": embeddings,
            "labels": medical_labels
        }
        
    def find_relevant_medical_info(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find most relevant medical information for query"""
        if not self.medical_embeddings:
            return []
            
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = np.dot(query_embedding, self.medical_embeddings["embeddings"].T)[0]
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Minimum similarity threshold
                condition = self.medical_embeddings["labels"][idx]
                results.append({
                    "condition": condition,
                    "similarity": float(similarities[idx]),
                    "info": self.medical_knowledge["conditions"][condition]
                })
                
        return results
    
    def generate_response(self, user_input: str, conversation_history: List[str] = None) -> str:
        """Generate chatbot response"""
        try:
            # Handle greetings and general questions
            response = self._handle_general_questions(user_input.lower())
            if response:
                return response
            
            # Find relevant medical information
            medical_info = self.find_relevant_medical_info(user_input)
            
            if medical_info:
                return self._generate_medical_response(user_input, medical_info)
            else:
                return self._generate_conversational_response(user_input, conversation_history)
                
        except Exception as e:
            return "I apologize, but I'm having trouble processing your request. Could you please rephrase your question?"
    
    def _handle_general_questions(self, user_input: str) -> str:
        """Handle greetings and general questions"""
        greetings = self.medical_knowledge["common_questions"]["greetings"]
        how_are_you = self.medical_knowledge["common_questions"]["how_are_you"]
        what_can_you_do = self.medical_knowledge["common_questions"]["what_can_you_do"]
        
        if any(greeting in user_input for greeting in greetings):
            return "Hello! I'm here to help you understand abdominal pain and related symptoms. How can I assist you today?"
        
        if any(phrase in user_input for phrase in how_are_you):
            return "Thank you for asking! I'm doing well and ready to help you with any questions about abdominal pain. How are you feeling?"
        
        if any(phrase in user_input for phrase in what_can_you_do):
            return """I'm a medical assistant specialized in abdominal pain. I can help you:
            
• Understand possible causes of abdominal pain
• Identify warning signs that need immediate attention  
• Learn about different conditions and their symptoms
• Get guidance on when to see a doctor

Please describe your symptoms, and I'll provide relevant information based on my medical training."""
        
        return None
    
    def _generate_medical_response(self, user_input: str, medical_info: List[Dict]) -> str:
        """Generate medical response based on found information"""
        if not medical_info:
            return "I don't have specific information about that in my abdominal pain knowledge base. Could you provide more details about your symptoms?"
        
        # Get the most relevant condition
        top_condition = medical_info[0]
        condition_name = top_condition["condition"].replace("_", " ").title()
        info = top_condition["info"]
        
        response = f"Based on your symptoms, you might be experiencing {condition_name}.\n\n"
        response += f"**Description:** {info['description']}\n\n"
        response += f"**Common symptoms include:**\n"
        for symptom in info['symptoms'][:4]:  # Show top 4 symptoms
            response += f"• {symptom.title()}\n"
        
        response += f"\n**Typical characteristics:**\n"
        response += f"• Location: {info['location']}\n"
        response += f"• Severity: {info['severity']}\n"
        response += f"• Duration: {info['duration']}\n"
        
        # Add warning if severe
        if info['severity'] in ['severe', 'moderate to severe']:
            response += f"\n⚠️ **Important:** This condition may require medical attention. "
            response += f"Please consult a healthcare provider for proper diagnosis and treatment.\n"
        
        # Check for warning signs
        warning_signs = self.medical_knowledge["warning_signs"]
        if any(sign in user_input.lower() for sign in warning_signs):
            response += f"\n🚨 **Urgent:** Your symptoms may indicate a serious condition. "
            response += f"Please seek immediate medical attention or call emergency services.\n"
        
        response += f"\n*This information is for educational purposes only and should not replace professional medical advice.*"
        
        return response
    
    def _generate_conversational_response(self, user_input: str, conversation_history: List[str] = None) -> str:
        """Generate conversational response using transformer model"""
        if not self.model or not self.tokenizer:
            return "I'm not trained to answer questions outside of abdominal pain and related medical topics. Could you ask about symptoms, pain location, or related concerns?"
        
        try:
            # Prepare input with medical context
            medical_context = "You are a medical assistant specialized in abdominal pain. "
            full_input = medical_context + user_input
            
            # Tokenize input
            inputs = self.tokenizer.encode(full_input, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(full_input):].strip()
            
            # Filter non-medical responses
            if not self._is_medical_related(response):
                return "I'm specifically trained to help with abdominal pain and digestive issues. Could you tell me more about any symptoms you're experiencing?"
            
            return response if response else "I'm not sure how to respond to that. Could you ask about abdominal pain symptoms?"
            
        except Exception as e:
            return "I'm not trained to answer questions outside of abdominal pain. Please ask about symptoms, causes, or when to see a doctor."
    
    def _is_medical_related(self, response: str) -> bool:
        """Check if response is medically related"""
        medical_keywords = [
            "pain", "symptom", "condition", "doctor", "medical", "treatment", 
            "abdomen", "stomach", "digestive", "nausea", "fever", "diagnosis"
        ]
        return any(keyword in response.lower() for keyword in medical_keywords)

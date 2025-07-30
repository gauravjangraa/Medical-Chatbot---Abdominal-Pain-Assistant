import streamlit as st
import requests
import time

# =============================================================================
# Page and API Configuration
# =============================================================================
st.set_page_config(
    page_title="Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set the base URL for the FastAPI backend
API_BASE_URL = "http://localhost:8000"

# =============================================================================
# Custom CSS Styling
# =============================================================================
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.st-emotion-cache-1c7y2kd {
    flex-direction: row-reverse;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Main UI Class
# =============================================================================

class MedicalChatbotUI:
    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize Streamlit's session state to store chat history and session ID."""
        if "session_id" not in st.session_state:
            st.session_state.session_id = None
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "suggestions" not in st.session_state:
            st.session_state.suggestions = []

    def create_session(self):
        """Send a request to the API to create a new chat session."""
        try:
            response = requests.post(f"{API_BASE_URL}/session")
            if response.status_code == 200:
                st.session_state.session_id = response.json()["session_id"]
                # Clear previous chat on new session
                st.session_state.messages = []
                st.session_state.suggestions = []
            else:
                st.error("Failed to create a new session. Please ensure the backend is running.")
        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Could not connect to the API server.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    def send_message(self, message: str):
        """Send a user's message to the chatbot API and handle the response."""
        if not st.session_state.session_id:
            self.create_session()
            # Wait a moment for session to be established before sending the message
            time.sleep(0.5)
            if not st.session_state.session_id: # If session creation failed, stop
                return

        # Add user message to the UI immediately
        st.session_state.messages.append({"role": "user", "content": message})

        try:
            payload = {"message": message, "session_id": st.session_state.session_id}
            response = requests.post(f"{API_BASE_URL}/chat", json=payload)

            if response.status_code == 200:
                result = response.json()
                st.session_state.messages.append({"role": "assistant", "content": result["response"]})
                st.session_state.suggestions = result["suggestions"]
            else:
                st.error(f"Error: Received status code {response.status_code} from the server.")
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again."})

        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Could not send message. Please check the backend server.")
            st.session_state.messages.append({"role": "assistant", "content": "I can't connect to my brain right now. Please check the server."})
        except Exception as e:
            st.error(f"An unexpected error occurred while sending the message: {e}")

    def render_sidebar(self):
        """Render the sidebar with controls and information."""
        with st.sidebar:
            st.header("ℹ️ About This Chatbot")
            st.markdown("""
            This AI assistant specializes in **abdominal pain** and can help you:
            - Understand possible causes of abdominal pain
            - Identify symptoms and their severity
            - Learn when to seek medical attention
            """)

            st.header("🚨 Emergency Warning Signs")
            st.warning("""
            **Seek immediate medical attention if you experience:**
            - Severe, persistent abdominal pain
            - High fever or difficulty breathing
            - Blood in vomit or stool
            """)
            
            if st.button("🔄 Start New Conversation", use_container_width=True):
                self.create_session()
                st.rerun()

    def render_chat_interface(self):
        """Render the main chat display area."""
        chat_container = st.container(height=400)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
                    st.markdown(message["content"])

    def render_input_area(self):
        """Render the user input area, resolving the st.form() issue."""
        st.markdown("---")

        # Quick symptom buttons are outside the form
        if st.session_state.suggestions:
            st.markdown("##### 💡 Suggested Questions")
            cols = st.columns(len(st.session_state.suggestions))
            for i, suggestion in enumerate(st.session_state.suggestions):
                if cols[i].button(suggestion, key=f"suggestion_{i}"):
                    self.send_message(suggestion)
                    st.rerun()
        else:
            st.markdown("##### 🔍 Quick Symptom Check")
            quick_symptoms = ["Sharp stomach pain", "Nausea and vomiting", "Bloating and gas", "Burning sensation"]
            symptom_cols = st.columns(4)
            for i, symptom in enumerate(quick_symptoms):
                if symptom_cols[i].button(symptom, key=f"quick_{i}"):
                    self.send_message(f"I'm experiencing {symptom.lower()}")
                    st.rerun()

        # The form now ONLY contains the text input and the submit button
        with st.form(key="chat_input_form"):
            col1, col2 = st.columns([4, 1])
            with col1:
                user_input = st.text_input(
                    "Describe your symptoms or ask a question:",
                    placeholder="e.g., I have sharp pain in my upper right abdomen...",
                    key="user_input",
                    label_visibility="collapsed"
                )
            with col2:
                # The ONLY button inside the form is the submit button
                submitted = st.form_submit_button("📤 Send", use_container_width=True)

        if submitted and user_input:
            with st.spinner("🤔 Analyzing your symptoms..."):
                self.send_message(user_input)
                st.rerun()

        # Other buttons, like the disabled voice button, are also outside the form
        st.button("🎤 Voice Input", disabled=True, help="Voice input feature is coming soon!", use_container_width=True)

# =============================================================================
# Main Application Execution
# =============================================================================
def main():
    """Main function to run the Streamlit application."""
    st.markdown('<h1 class="main-header">🏥 Medical Chatbot - Abdominal Pain Assistant</h1>', unsafe_allow_html=True)
    
    chatbot_ui = MedicalChatbotUI()
    chatbot_ui.render_sidebar()
    chatbot_ui.render_chat_interface()
    chatbot_ui.render_input_area()

if __name__ == "__main__":
    main()
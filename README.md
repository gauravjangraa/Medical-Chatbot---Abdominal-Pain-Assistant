## **🏥 Medical Chatbot - Abdominal Pain Assistant**



**Technology Used 🛠️**
---

Think of these as the tools and materials used to build your chatbot:



**Backend (The Engine Room):**



**Python:** The main programming language used to build everything.



**FastAPI:** A super-efficient "waiter" that takes requests from the website and brings them to the chatbot's brain to get a response.



**Machine Learning (The Brains):**



**PyTorch:** The foundational "engine" that powers the AI models.



Transformers \& Sentence-Transformers: These are like specialized toolkits. One helps the bot chat naturally, and the other helps it understand the meaning and context of sentences, not just the words.



**Frontend (The Face of the App):**



**Streamlit:** A tool that quickly builds the simple, clean webpage where users can actually chat with the bot.


**How the Project Works ⚙️**
---



Imagine you're asking the bot a question. Here’s the step-by-step journey your message takes:



##### **Part 1: The Core Components**



-The Face (streamlit\_app.py): This is the chat window where you type your message. When you hit "Send," it packages up your message and sends it over to the FastAPI "traffic cop."



-The Manager (chat\_service.py): This is the project's manager. It gets the message from FastAPI and keeps track of your entire conversation. Its most important job is to remember the context—if you were just talking about "gallstones," it holds onto that thought.



-The Brain (chatbot\_model.py): This is the real AI. The manager gives it your message and the conversation context. The brain has two key skills:



-The Detective: It uses SentenceTransformer to read your symptoms and find the closest match in its medical library.



-The Conversationalist: It uses GPT-2 to handle hellos, how-are-yous, and to answer specific questions about a topic.



-The Library (medical\_data.py): This is the bot's entire medical knowledge. It’s a single file where all the information about symptoms, causes, and conditions is written down. The "Detective" can only find information that exists in this library.



##### **Part 2: The Workflow in Action**

Let's say you have two questions: one about symptoms and a follow-up.



Question 1: "I have sharp pain in my upper right abdomen"



You type this into the Streamlit web page.



Streamlit sends it to the FastAPI backend.



The Chat Service (the manager) receives it. Since there's no previous context, it asks the Chatbot Model (the brain) for a fresh analysis.



The brain's "Detective" skill reads the sentence, converts it to numbers (embeddings), and finds that "gallstones" is the closest match in its Medical Data library.



The brain generates a summary about gallstones and sends it back to the manager.



The manager updates your conversation history and importantly, sets the context to "gallstones."



The response is sent back to the Streamlit page for you to see.



Question 2: "What causes it?"



You type your follow-up into Streamlit.



-It goes through FastAPI to the Chat Service.



-This time, the manager tells the brain: "The user is asking 'What causes it?' and the context of our conversation   is 'gallstones'."



-The brain sees the context. Instead of using its "Detective" skill, it uses its "Conversationalist" skill. It looks up "gallstones" in its library, finds the causes section, and creates a direct answer.



-This specific, context-aware answer is sent back through the manager to your screen.



-By remembering the context, the bot avoids giving the same generic summary and instead answers your direct follow-up questions, making the conversation feel much more intelligent and natural.


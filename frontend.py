import streamlit as st
#from streamlit_backend import workflow, SmartChatbotState, ChatMessage, log_feedback, create_initial_state, retrieve_all_threads
from langchain_core.messages import HumanMessage
from datetime import datetime
import uuid
import sys
import os
from streamlit_backend import workflow, SmartChatbotState, ChatMessage, log_feedback, create_initial_state, retrieve_all_threads
# **************************************** utility functions *************************
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state["session_id"] = str(thread_id)
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []
    # Reset chatbot state for new thread
    st.session_state.chatbot_state = create_initial_state(st.session_state["session_id"])

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def save_conversation_state(thread_id, chatbot_state):
    """Save the current conversation state for a thread"""
    if 'saved_conversations' not in st.session_state:
        st.session_state['saved_conversations'] = {}
    st.session_state['saved_conversations'][str(thread_id)] = chatbot_state

def load_conversation(thread_id):
    # First try to load from saved conversations in session state
    if 'saved_conversations' in st.session_state:
        saved_state = st.session_state['saved_conversations'].get(str(thread_id))
        if saved_state and 'conversation_history' in saved_state:
            return saved_state['conversation_history']
    
    # Create config for the specific thread
    config = {"configurable": {"thread_id": str(thread_id)}}
    try:
        # Get state from workflow for this thread
        state = workflow.get_state(config)
        # Try to get conversation_history first, then fallback to messages
        if 'conversation_history' in state.values:
            return state.values.get('conversation_history', [])
        else:
            return state.values.get('messages', [])
    except Exception as e:
        print(f"Error loading conversation for thread {thread_id}: {e}")
        # Fallback if workflow doesn't have get_state or thread doesn't exist
        return []

# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'session_id' not in st.session_state:
    st.session_state["session_id"] = str(st.session_state['thread_id'])

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])

#CONFIG = {"configurable": {"thread_id": st.session_state["session_id"]}}

CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_turn",
    }


# Initialize chatbot state once
if "chatbot_state" not in st.session_state:
    st.session_state.chatbot_state = create_initial_state(st.session_state["session_id"])

# **************************************** Sidebar UI *********************************
st.sidebar.title('🤖 Phantom AI')

if st.sidebar.button('New Chat'):
    reset_chat()
    st.rerun()

st.sidebar.header('My Conversations')
for thread_id in st.session_state['chat_threads'][::-1]:
    # Create a more readable button label (first 8 chars of UUID)
    button_label = f"{str(thread_id)[:4]}"
    
    if st.sidebar.button(button_label, key=f"thread_{thread_id}"):
        # Save current conversation state before switching
        save_conversation_state(st.session_state['thread_id'], st.session_state.chatbot_state)
        
        # Switch to selected thread
        st.session_state['thread_id'] = thread_id
        st.session_state["session_id"] = str(thread_id)
        
        # Load conversation for this thread
        conversation_history = load_conversation(thread_id)
        
        # Create or update chatbot state for this thread
        st.session_state.chatbot_state = create_initial_state(st.session_state["session_id"])
        
        if conversation_history:
            # Set the conversation history
            st.session_state.chatbot_state["conversation_history"] = conversation_history
            
            # Convert to message_history format for backward compatibility
            temp_messages = []
            for msg in conversation_history:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    temp_messages.append({'role': msg.role, 'content': msg.content})
                elif isinstance(msg, HumanMessage):
                    temp_messages.append({'role': 'user', 'content': msg.content})
                elif hasattr(msg, 'content'):
                    temp_messages.append({'role': 'assistant', 'content': msg.content})
            st.session_state['message_history'] = temp_messages
        else:
            # No conversation history found
            st.session_state['message_history'] = []
            st.session_state.chatbot_state["conversation_history"] = []
        
        st.rerun()
    


# **************************************** Preferences Controls *************************
st.sidebar.header("🎛️ Preferences")

# Tone Preference
tone_pref = st.sidebar.selectbox(
    "Tone",
    [
        None,
        "Formal", "Casual", "Funny", "Concise", "Storytelling",
        "Inspirational", "Empathetic", "Analytical", "Debater", "Creative"
    ],
    index=0
)

# Depth Preference
depth_pref = st.sidebar.selectbox(
    "Depth",
    [
        None,
        "One-Liner", "Summary", "Moderate Explanation", "Detailed",
        "Step-by-Step", "Research-Grade", "Visual Explanation"
    ],
    index=0
)

# Domain Focus
domain_pref = st.sidebar.selectbox(
    "Domain Focus",
    [
        None,
        "Layman", "General Knowledge", "Tech-Heavy", "Academic / Theoretical",
        "Practical", "Business-Oriented", "Creative", "Scientific Rigor"
    ],
    index=0
)

# Update state immediately if user changes preferences
if tone_pref:
    st.session_state.chatbot_state["tone_preferences"] = tone_pref
if depth_pref:
    st.session_state.chatbot_state["depth_preferences"] = depth_pref
if domain_pref:
    st.session_state.chatbot_state["domain_focus"] = domain_pref


# **************************************** Main Chat Interface *************************


# Display conversation history
conversation_history = st.session_state.chatbot_state.get("conversation_history", [])

for message in conversation_history:
    with st.chat_message(message.role):
        st.markdown(message.content)

# Chat input
user_input = st.chat_input('Type your message here...')

if user_input:
    # Show user message immediately
    current_state = dict(st.session_state.chatbot_state)
    current_state["messages"] = [HumanMessage(content=user_input)]
    
    try:
        # Run workflow
        with st.spinner("🤔 Thinking..."):
            response = workflow.invoke(current_state, config=CONFIG)
        
        # Update session state with the response
        st.session_state.chatbot_state = response
        
        # Debug: Check if conversation history was updated
        history_length = len(st.session_state.chatbot_state.get("conversation_history", []))
        print(f"Frontend: Conversation history length after workflow: {history_length}")
        
        # Update message history for thread switching
        conversation_history = st.session_state.chatbot_state.get("conversation_history", [])
        temp_messages = []
        for msg in conversation_history:
            temp_messages.append({'role': msg.role, 'content': msg.content})
        st.session_state['message_history'] = temp_messages
        
        # Force a rerun to refresh the conversation history display
        st.rerun()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        # Fallback: manually add to conversation history if workflow fails
        if "conversation_history" not in st.session_state.chatbot_state:
            st.session_state.chatbot_state["conversation_history"] = []
        
        st.session_state.chatbot_state["conversation_history"].extend([
            ChatMessage(role="user", content=user_input),
            ChatMessage(role="assistant", content="Sorry, there was an error processing your request.")
        ])

# Display feedback section at the bottom
if st.session_state.chatbot_state.get("conversation_history"):
    st.divider()
    
    with st.expander("💬 Provide Feedback"):
        feedback_text = st.text_area("How was my response?", height=100)
        rating = st.slider("Rate this conversation (1-5)", 1, 5, 3)
        
        if st.button("Submit Feedback"):
            if feedback_text:
                # Add feedback to state
                st.session_state.chatbot_state = log_feedback(
                    st.session_state.chatbot_state, 
                    feedback_text, 
                    rating
                )
                st.success("Thank you for your feedback!")
            else:
                st.warning("Please enter some feedback text.")

# **************************************** Debug Info (Optional) *************************
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.write("**Current Thread ID:**", str(st.session_state['thread_id']))
    st.sidebar.write("**Total Threads:**", len(st.session_state['chat_threads']))
    st.sidebar.write("**Message History Length:**", len(st.session_state.get('message_history', [])))
    st.sidebar.write("**Conversation History Length:**", len(st.session_state.chatbot_state.get("conversation_history", []))) 
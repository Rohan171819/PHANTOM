# %%
#!pip install SpeechRecognition pyttsx3 pyaudio

# %%
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import List, TypedDict, Literal, Dict, Optional, Annotated, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import requests
from pydantic import BaseModel, Field
import os
import json
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from datetime import datetime
import uuid
from datetime import datetime
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph.message import add_messages
import re
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool



# %%
load_dotenv(override=True)
groq_api_key = os.getenv("GROQ_API_KEY")


# %%
llm1 = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"   # Fast intent + draft generator
)

llm2 = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-70b-specdec"    # Deep reasoning + structured response
)

llm3 = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="mixtral-8x7b-32768"  # Creative writing, storytelling
)



# %%
# Pydantic object for intent categorization.
class Intent_Categorization(BaseModel):
    intent: Literal[
        "General Inquiry",
        "Technical Support",
        "Creative Writing",
        "Educational Content",
        "Business Communication",
        "Casual Conversation",
        "Emotional Support",
        "Research Assistance",
        "Entertainment",
        "Other"]
    
    confidence: float = Field(..., ge=0.0, le=1.0)

# %%
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


# %%
class FeedbackEntry(BaseModel):
    feedback: str
    rating: Optional[int]  # 1-5
    timestamp: str


# %%
# Creating the state.
class SmartChatbotState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    conversation_history: List[ChatMessage]    
    draft_responses: List[str]
    final_response: Optional[str]


    # User Intent & Context
    intent_info : Optional[Intent_Categorization]    
    
     # Customization
    tone_preferences: Optional[
        Literal[
        "Formal",
        "Casual",
        "Funny",
        "Concise",
        "Storytelling",
        "Inspirational",
        "Empathetic",
        "Analytical",
        "Debater",
        "Creative"]]
    
    
    depth_preferences: Optional[
        Literal[
        "One-Liner",
        "Summary",
        "Moderate Explanation",
        "Detailed",
        "Step-by-Step",
        "Research-Grade",
        "Visual Explanation"]]
    
    
    domain_focus: Optional[
        Literal[
        "Layman",
        "General Knowledge",
        "Tech-Heavy",
        "Academic / Theoretical",
        "Practical",
        "Business-Oriented",
        "Creative",
        "Scientific Rigor"]]
    

    
    # Meta
    feedback_log:  List[FeedbackEntry]
    timestamp: str
    session_id: str
    unresolved: bool


# %%
def update_conversation_history(state, user_input, assistant_output):
    """Update conversation history with new messages"""
    if "conversation_history" not in state:
        state["conversation_history"] = []
    
    state["conversation_history"].append(ChatMessage(role="user", content=user_input))
    state["conversation_history"].append(ChatMessage(role="assistant", content=assistant_output))
    return state


# %%
def log_feedback(state: SmartChatbotState, feedback: str, rating: int):
    entry = FeedbackEntry(
        feedback=feedback,
        rating=rating,
        timestamp=str(datetime.now())
    )
    state["feedback_log"].append(entry)
    return state


# %%
def preference_detection_node(state: SmartChatbotState) -> SmartChatbotState:
    conversation_history = state.get("conversation_history", [])
    history_text = "\n".join(
        [f"{turn.role.capitalize()}: {turn.content}" for turn in conversation_history]
    )

    prompt = f"""
    Analyze the following conversation history and detect if the user has 
    specified any preferences for:

    1. Tone (Formal, Casual, Funny, Concise, Storytelling, Inspirational,
       Empathetic, Analytical, Debater, Creative)
    2. Depth (One-Liner, Summary, Moderate Explanation, Detailed,
       Step-by-Step, Research-Grade, Visual Explanation)
    3. Domain focus (Layman, General Knowledge, Tech-Heavy, Academic / Theoretical,
       Practical, Business-Oriented, Creative, Scientific Rigor)

    Only return JSON with the structure:
    {{
        "tone_preferences": "...",
        "depth_preferences": "...",
        "domain_focus": "..."
    }}

    Use null for any field not specified.

    Conversation history:
    {history_text}
    """
    result = llm1.invoke(prompt)

    raw_output = result.content.strip()
    # Extract JSON block if wrapped in ```json ... ```
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        print("Preference detection parsing error: no JSON found")
        return state

    try:
        parsed = json.loads(match.group(0))
        if parsed.get("tone_preferences") not in [None, "", "null"]:
            state["tone_preferences"] = parsed["tone_preferences"]
        if parsed.get("depth_preferences") not in [None, "", "null"]:
            state["depth_preferences"] = parsed["depth_preferences"]
        if parsed.get("domain_focus") not in [None, "", "null"]:
            state["domain_focus"] = parsed["domain_focus"]
    except Exception as e:
        print("Preference detection parsing error:", e)

    return state

# %%
def intent_node(state: SmartChatbotState) -> SmartChatbotState:
    user_message = state["messages"][-1].content

    prompt = f"""
    Classify the intent of the following message into one of the categories:
    [General Inquiry, Technical Support, Creative Writing, Educational Content,
    Business Communication, Casual Conversation, Emotional Support,
    Research Assistance, Entertainment, Other].

    Message: {user_message}
    Respond in JSON with fields: intent, confidence (0-1).
    """

    try:
        result = llm1.invoke(prompt)
        raw_output = result.content.strip()
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)

        if match:
            parsed = json.loads(match.group(0))
            state["intent_info"] = Intent_Categorization(**parsed)
        else:
            raise ValueError("No JSON found in response")
    except Exception as e:
        print(f"Intent parsing error: {e}")
        state["intent_info"] = Intent_Categorization(intent="Other", confidence=0.5)

    return state


# %%
def router_node(state: SmartChatbotState) -> SmartChatbotState:
    intent = state["intent_info"].intent if state.get("intent_info") else "Other"
    user_message = state["messages"][-1].content

    if intent in ["Technical Support", "Educational Content", "Research Assistance"]:
        chosen_llm = llm2
    elif intent in ["Creative Writing", "Entertainment", "Emotional Support"]:
        chosen_llm = llm3
    else:
        chosen_llm = llm1

    history_text = "\n".join([f"{msg.role}: {msg.content}" 
                              for msg in state.get("conversation_history", [])])

    tone = state.get("tone_preferences") or "Formal"
    depth = state.get("depth_preferences") or "Summary"
    domain = state.get("domain_focus") or "General Knowledge"



    prompt = f"""
    You are an AI assistant that can call tools when necessary.


    Conversation history:
    {history_text}

    User message:
    {user_message}

    Response Requirements:
    - Tone: {tone}
    - Depth: {depth}
    - Domain focus: {domain}

    If a tool is needed, return a tool call using the correct JSON schema.
    If no tool is needed, answer normally.
    """

    result = chosen_llm.invoke(prompt)
    state["draft_responses"] = [result.content]
    return state


# %%
def finalizer_node(state: SmartChatbotState) -> SmartChatbotState:
    # Find the original user query
    original_query = "Hello!"
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            original_query = msg.content
            break
    
    # Check if we have tool results or draft responses
    if state.get("draft_responses"):
        # Use the latest draft response
        state["final_response"] = state["draft_responses"][-1]
    else:
        # Generate final response incorporating tool results
        tone = state.get("tone_preferences") or "Formal"
        depth = state.get("depth_preferences") or "Summary"
        domain = state.get("domain_focus") or "General Knowledge"
        
        prompt = f"""
        Based on the context below, provide a comprehensive response to the user's query.

        Original query: {original_query}

        Respond with these constraints:
        - Tone: {tone}
        - Depth: {depth}
        - Domain focus: {domain}
        """
        
        if state["intent_info"] and state["intent_info"].intent in ["Technical Support", "Educational Content", "Research Assistance"]:
            chosen_llm = llm2
        elif state["intent_info"] and state["intent_info"].intent in ["Creative Writing", "Entertainment", "Emotional Support"]:
            chosen_llm = llm3
        else:
            chosen_llm = llm1
        
        final_response = chosen_llm.invoke(prompt)
        state["final_response"] = final_response.content

    # Update conversation history
    user_input = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break
    
    if user_input and state.get("final_response"):
        assistant_output = state["final_response"]
        print(f"Finalizer: Adding to history - User: {user_input[:50]}...")
        print(f"Finalizer: Adding to history - Assistant: {assistant_output[:50]}...")
        
        state = update_conversation_history(state, user_input, assistant_output)
        print(f"Finalizer: Total conversation history length: {len(state.get('conversation_history', []))}")

    # Update timestamp
    state["timestamp"] = str(datetime.now())
    
    return state

# %%
# Defining the Graph.
graph = StateGraph(SmartChatbotState)
    
# Adding the Nodes.
graph.add_node("intent_node",intent_node)
graph.add_node("preference_detection_node",preference_detection_node)
graph.add_node("router_node",router_node)
graph.add_node("finalizer_node",finalizer_node)

# Adding Edges.
graph.add_edge(START, "intent_node")
graph.add_edge("intent_node", "preference_detection_node")
graph.add_edge("preference_detection_node", "router_node")
graph.add_edge("router_node", "finalizer_node")
graph.add_edge("finalizer_node", END)


# Creating connection.
conn = sqlite3.connect(database="Chatbot.db", check_same_thread= False)

# %%
# Creating checkpointer object
checkpointer = SqliteSaver(conn= conn)


# Compiling the graph.
workflow = graph.compile(checkpointer= checkpointer)

# Defining the configuration for a new session
config = {"configurable" :{"thread_id": 'thread_123'}}


# %%
workflow


# %%
def retrieve_all_threads():
    all_threads =set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

# %%
def create_initial_state(session_id : str = str(uuid.uuid4()))-> SmartChatbotState:
    return {
        "messages": [HumanMessage(content="Hello!")],
        "conversation_history": [],
        "draft_responses": [],
        "final_response": None,
        "intent_info": None,
        "tone_preferences": None,
        "depth_preferences": None,
        "domain_focus": None,
        "voice_input_mode": False,
        "last_audio_input": None,
        "feedback_log": [],
        "timestamp": str(datetime.now()),
        "session_id": session_id,
        "unresolved": False,}


# %%
def continue_conversation(prev_state, user_message: str, config):
  state = dict(prev_state)

  # update messages for next turn
  state["messages"] = state.get("messages", []) + [HumanMessage(content=user_message)]
  return workflow.invoke(state, config=config)

# %%
state1 = workflow.invoke(create_initial_state(), config= config)
print(state1)
#print("Final Response:", state1["final_response"])
#print("Conversation History:", state1["conversation_history"])

# %%
state2 = continue_conversation(state1, "Tell me today's market.", config)
#print(state2)
print(state2["final_response"])


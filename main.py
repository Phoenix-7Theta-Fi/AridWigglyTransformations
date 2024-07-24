import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from bs4 import BeautifulSoup
import requests
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

# Streamlit configs
st.set_page_config(page_title="AI Text Assistant", page_icon="🤖")

# Get Groq API key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
@st.cache_resource
def initialize_groq_llm():
    return ChatGroq(
        temperature=0,
        model="llama3-70b-8192",
        api_key=groq_api_key
    )

# Initialize DuckDuckGo search
search_tool = DuckDuckGoSearchResults()

# Function to extract content using Beautiful Soup
def extract_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    content = soup.get_text()
    return content[:500]  # Limiting to 500 characters for brevity

# Create LangChain pipeline
def search_and_summarize(query, groq_llm):
    search_results = search_tool.run(query)
    extracted_content = extract_content(search_results[0]['link'])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Summarize the following content:"),
        ("human", "{content}")
    ])
    chain = prompt | groq_llm
    return chain.invoke({"content": extracted_content})

# Streamlit UI
st.title("AI-powered Text Assistant")

# Initialize StreamlitChatMessageHistory
history = StreamlitChatMessageHistory(key="chat_messages")

if groq_api_key:
    groq_llm = initialize_groq_llm()

    # Text input
    user_input = st.text_input("Ask me anything:")
    if user_input:
        # Process user input
        response = search_and_summarize(user_input, groq_llm)
        # Add messages to history
        history.add_user_message(user_input)
        history.add_ai_message(response.content)
        
        # Display the latest response
        st.write("Assistant:", response.content)

    # Display chat history
    st.write("Chat History:")
    for message in history.messages:
        st.write(f"{message.type.capitalize()}: {message.content}")

else:
    st.error("Groq API key not found in environment variables. Please set the GROQ_API_KEY secret in your Repl.it environment.")

import os
import streamlit as st
import speech_recognition as sr
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from bs4 import BeautifulSoup
import requests
import pyttsx3
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

# Streamlit configs
st.set_page_config(page_title="AI Voice Assistant", page_icon="🎙️")

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
    # Extract relevant content (customize this based on your needs)
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

# Initialize text-to-speech engine
@st.cache_resource
def initialize_tts_engine():
    return pyttsx3.init()

# Streamlit UI
st.title("AI-powered Voice Assistant")

# Initialize StreamlitChatMessageHistory
history = StreamlitChatMessageHistory(key="chat_messages")

if groq_api_key:
    groq_llm = initialize_groq_llm()
    tts_engine = initialize_tts_engine()

    # Voice input button
    if st.button("Start Voice Input"):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening...")
            audio = r.listen(source)
            try:
                user_input = r.recognize_google(audio)
                st.write(f"You said: {user_input}")

                # Process user input
                response = search_and_summarize(user_input, groq_llm)

                # Add messages to history
                history.add_user_message(user_input)
                history.add_ai_message(response.content)

                # Convert response to speech
                tts_engine.say(response.content)
                tts_engine.runAndWait()

            except sr.UnknownValueError:
                st.write("Sorry, I couldn't understand that.")
            except sr.RequestError:
                st.write("Sorry, there was an error processing your request.")

    # Display chat history
    for message in history.messages:
        st.chat_message(message.type).write(message.content)
else:
    st.error("Groq API key not found in environment variables. Please set the GROQ_API_KEY secret in your Repl.it environment.")
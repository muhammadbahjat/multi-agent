import streamlit as st
from typing import List, Dict
import os
from multi_agent import MainAgent


st.set_page_config(
    page_title="Bahjat's AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm Bahjat's AI Assistant. I can help you learn about Muhammad Bahjat's work, experience, and projects. I can also assist with general queries. How can I help you today?"
    })

if "assistant" not in st.session_state:
    with st.spinner("Initializing AI Assistant..."):
        st.session_state.assistant = MainAgent()

st.title("🤖 Bahjat's AI Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.assistant.process_query(prompt)
                
                st.session_state.messages.append({"role": "assistant", "content": response})

                message_placeholder.markdown(response)
            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                message_placeholder.markdown(error_message)

# Sidebar
with st.sidebar:
    st.markdown("### About Me")
    st.markdown("""
    I'm an AI assistant created by Muhammad Bahjat to help you:
    - Learn about his work and experience
    - Access his portfolio and documents
    - Find specific files
    - Discuss technology and answer questions
    """)
    
    if st.button("Clear Chat"):
        st.session_state.messages
        # Restore welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm Bahjat's AI Assistant. I can help you learn about Muhammad Bahjat's work, experience, and projects. I can also assist with general queries. How can I help you today?"
        })
        st.experimental_rerun()
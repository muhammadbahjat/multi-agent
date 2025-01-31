# Multi-Agent AI System

This repository contains a **Multi-Agent AI System** designed to integrate various functionalities into a cohesive framework. The system employs AI agents to perform tasks such as file system navigation, document search, web search, and handling general queries. The frontend interface is built using **Streamlit** for an intuitive user experience.

---

## Features
- **File System Navigation**: Securely search and retrieve files while respecting system security restrictions.
- **Document Search**: Process and analyze local documents (PDFs, DOCX) with a powerful document vectorization system.
- **Web Search**: Retrieve relevant and validated information from the internet using DuckDuckGo.
- **General Query Handling**: Maintain conversational identity as "Bahjat's AI Assistant" for engaging and professional interactions.

---

## Technologies Used
- **Python**: Core programming language.
- **LangChain**: For building and managing AI agents.
- **Chroma**: Vector database for efficient document retrieval.
- **Google Generative AI (Gemini-2.0)**: For LLM-powered responses.
- **Streamlit**: For a user-friendly frontend interface.

---

## Installation

Follow these steps to set up the project on your local machine:

### Prerequisites
- Python 3.11 or higher
- Install `pip install -r requirements.txt` for managing dependencies
- An API key for **Google Generative AI**

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/muhammadbahjat/multi-agent.git
   cd multi-agent


#### Set up a virtual environment (recommended):
`python -m venv venv`
`source venv/bin/activate`   # For Linux/macOS
`venv\Scripts\activate` # For Windows

### Set up your environment variables:
 
1. Create a .env file in the project directory.
2. Add your Google API key:
    `GOOGLE_API_KEY=your_google_api_key`

## Place your documents in the docs/ folder for analysis.

## Run the Streamlit app:

`streamlit run ui_streamlit.py`
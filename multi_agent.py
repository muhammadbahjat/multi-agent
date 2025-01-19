import streamlit as st
from dotenv import load_dotenv
import os
from typing import List, Dict
import platform
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun, Tool
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory

class FileSystemTool:
    """File system navigation tool with security restrictions."""
    def __init__(self):
        self.system = platform.system()
        self.restricted_paths = {
            'Linux': ['/root', '/etc/shadow', '/etc/passwd'],
            'Windows': ['C:\\Windows\\System32'],
            'Darwin': ['/etc/shadow', '/etc/passwd']
        }.get(self.system, [])

    def is_path_allowed(self, path: str) -> bool:
        path = os.path.abspath(path)
        return not any(restricted in path for restricted in self.restricted_paths)

    def find_file(self, filename: str) -> Dict[str, List[str]]:
        result = {'found_locations': [], 'errors': []}
        if not filename:
            result['errors'].append("Please provide a filename to search for.")
            return result

        start_paths = ['C:\\'] if self.system == 'Windows' else ['/']
        for start_path in start_paths:
            if not self.is_path_allowed(start_path):
                continue
            try:
                for root, _, files in os.walk(start_path):
                    if any(restricted in root for restricted in self.restricted_paths):
                        continue
                    if filename in files:
                        full_path = os.path.join(root, filename)
                        if self.is_path_allowed(full_path):
                            result['found_locations'].append(full_path)
            except PermissionError:
                continue
            except Exception as e:
                result['errors'].append(f"Error searching in {start_path}: {str(e)}")
        return result

    def locate_path(self, path_query: str) -> str:
        try:
            if not path_query:
                return "Please provide a path or filename to search for."

            if path_query.startswith('/'):
                if self.is_path_allowed(path_query):
                    if os.path.exists(path_query):
                        return f"Found at: {path_query}"
                    return f"Path not found: {path_query}"
                return "Access to this path is restricted for security reasons."

            result = self.find_file(path_query)
            if result['found_locations']:
                locations = "\n".join(result['found_locations'])
                return f"Found in the following location(s):\n{locations}"
            if result['errors']:
                errors = "\n".join(result['errors'])
                return f"Encountered errors while searching:\n{errors}"
            return f"Could not find '{path_query}' in accessible locations."
        except Exception as e:
            return f"Error processing query: {str(e)}"

class BaseAgent:
    """Base class for all specialized agents."""
    def __init__(self, google_api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=google_api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

class DocumentAgent(BaseAgent):
    """Specialized agent for document analysis."""
    def __init__(self, google_api_key: str):
        super().__init__(google_api_key)
        self.setup_document_system()
        
    def setup_document_system(self):
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        folder_path = os.path.join(os.getcwd(), "docs")
        documents = self.load_documents(folder_path)
        self.process_documents(documents)
        self.setup_agent()

    def load_documents(self, folder_path: str) -> List[Document]:
        documents = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif filename.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            else:
                continue
            documents.extend(loader.load())
        return documents

    def process_documents(self, documents: List[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        self.vectorstore = Chroma.from_documents(
            collection_name="my_collection",
            documents=splits,
            embedding=self.embedding_function,
            persist_directory="./chroma_db"
        )

    def search_documents(self, query: str) -> str:
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
            retriever_result = retriever.invoke(query)
            if not retriever_result:
                return "No relevant information found in the documents."
            return "\n\n".join(doc.page_content for doc in retriever_result)
        except Exception as e:
            return f"Error searching documents: {str(e)}"

    def setup_agent(self):
        tools = [
            Tool(
                name="DocumentSearch",
                func=self.search_documents,
                description="Analyzes local documents about Muhammad Bahjat to find relevant information about his experience, skills, and background."
            )
        ]
        
        system_message = """You are the Document Analysis Agent. Your role is to:
        1. Search through local documents about Muhammad Bahjat
        2. Extract relevant information about his experience, skills, and qualifications
        3. Present the information in a clear, organized manner
        4. Focus on professional and public information only"""
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def process_query(self, query: str) -> str:
        response = self.agent.invoke({"input": query})
        return response["output"]
class GeneralqueryAgent(BaseAgent):
    """Specialized agent for general queries analysis."""
    def __init__(self, google_api_key: str):
        super().__init__(google_api_key)
        self.setup_agent()
        
    def handle_general_query(self, query: str) -> str:
        """Handle general queries and maintain identity."""
        if any(greeting in query.lower() for greeting in ["hi", "hello", "hey"]):
            return "Hello! I'm Bahjat's AI Assistant. How can I help you learn about Muhammad Bahjat today?"
        
        if "how are you" in query.lower():
            return "I'm doing well, thank you for asking! As Bahjat's AI Assistant, I'm here to help you learn about Muhammad Bahjat's work and experience. What would you like to know?"
            
        return "As Bahjat's AI Assistant, I'm here to help you learn about Muhammad Bahjat. Would you like to know about his experience, projects, or skills?"
        
    def setup_agent(self):
        """Initialize the general query agent."""
        tools = [
            Tool(
                name="GeneralResponse",
                func=self.handle_general_query,
                description="Handles general queries and greetings while maintaining identity as Bahjat's Assistant"
            )
        ]
        
        system_message = """You are Bahjat's AI Assistant. For any general queries or greetings:

        1. ALWAYS identify yourself as Bahjat's AI Assistant
        2. NEVER mention being an AI model or any technical details about yourself
        3. Keep responses friendly and professional
        4. Include an offer to help learn about Muhammad Bahjat

        Key facts about Bahjat to incorporate:
        - He is a software engineer and AI Engineer
        - Specializes in AI Agents and integrations
        - Experienced in AI Calling implementations
        - Proficient in Python, Flask, FastAPI, and Django
        - Strong background in PostgreSQL
        - Currently focused on AI and Agentic frameworks
        - Helps companies build AI agents and automate workflows

        When handling general queries, always use the GeneralResponse tool to maintain consistent identity."""
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def process_query(self, query: str) -> str:
        """Process general queries and maintain consistent identity."""
        try:
            response = self.agent.invoke({"input": query})
            return response["output"]
        except Exception as e:
            return f"As Bahjat's AI Assistant, I apologize, but I encountered an error: {str(e)}"

class WebSearchAgent(BaseAgent):
    """Specialized agent for web searches."""
    def __init__(self, google_api_key: str):
        super().__init__(google_api_key)
        self.setup_agent()

    def setup_agent(self):
        tools = [
            Tool(
                name="WebSearch",
                func=DuckDuckGoSearchRun(),
                description="Searches the internet for information about Muhammad Bahjat or general topics."
            )
        ]
        
        system_message = """You are the Web Search Agent. Your role is to:
        1. Search the internet for information about Muhammad Bahjat
        2. Find relevant professional information and achievements
        3. Validate and cross-reference information
        4. Focus on recent and reliable sources"""
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def process_query(self, query: str) -> str:
        response = self.agent.invoke({"input": query})
        return response["output"]

class FileSystemAgent(BaseAgent):
    """Specialized agent for file system operations."""
    def __init__(self, google_api_key: str):
        super().__init__(google_api_key)
        self.setup_agent()

    def setup_agent(self):
        self.file_system_tool = FileSystemTool()
        tools = [
            Tool(
                name="FileSystem",
                func=self.file_system_tool.locate_path,
                description="Searches the file system for specific files or directories."
            )
        ]
        
        system_message = """You are the File System Agent. Your role is to:
        1. Safely navigate the file system to find requested files
        2. Focus on finding documents related to Muhammad Bahjat
        3. Respect system security restrictions
        4. Provide clear file locations and access information"""
        
        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def process_query(self, query: str) -> str:
        response = self.agent.invoke({"input": query})
        return response["output"]

class MainAgent:
    """Main coordinator agent that manages all specialized agents."""
    def __init__(self):
        load_dotenv()
        self.setup_api_keys()
        self.setup_agents()
        self.setup_coordinator()

    def setup_api_keys(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("Google API key is missing!")

    def setup_agents(self):
        """Initialize all specialized agents."""
        self.document_agent = DocumentAgent(self.google_api_key)
        self.web_agent = WebSearchAgent(self.google_api_key)
        self.file_agent = FileSystemAgent(self.google_api_key)
        self.generalqueryAgent =  GeneralqueryAgent(self.google_api_key)

    def setup_coordinator(self):
        """Setup the main coordinator agent."""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=self.google_api_key,
            temperature=0.5
        )
        
        tools = [
            Tool(
                name="GeneralqueryAgent",
                func=self.generalqueryAgent.process_query,
                description="Use this agent for general queries like greetings and casual conversation"
            ),
            Tool(
                name="DocumentAnalysis",
                func=self.document_agent.process_query,
                description="Use this agent to analyze local documents about Muhammad Bahjat"
            ),
            Tool(
                name="WebSearch",
                func=self.web_agent.process_query,
                description="Use this agent to find online information about Muhammad Bahjat"
            ),
            Tool(
                name="FileSystem",
                func=self.file_agent.process_query,
                description="Use this agent to locate specific files in the system"
            )
        ]
        
        system_message = """You are Bahjat's AI Assistant, and you must ALWAYS identify yourself as such. Never identify yourself as a Google model or any other AI model.

        IMPORTANT: When asked who you are, ALWAYS respond with:
        "Hello! I am Bahjat's AI Assistant, created by Muhammad Bahjat to help users learn about him and his work. I can help you find information about his experience, projects, and achievements. How can I assist you today?"

        Core Responsibilities:
        1. Identity:
        - ALWAYS identify as "Bahjat's AI Assistant"
        - NEVER mention being trained by Google or being any other type of AI model
        - Focus on your role as Muhammad Bahjat's dedicated assistant

        2. For questions about Muhammad Bahjat:
        - Use DocumentAnalysis for local information
        - Use WebSearch for additional context
        - Combine information into comprehensive responses

        3. For file-related queries:
        - Use FileSystem to locate files
        - Use DocumentAnalysis to analyze contents
        - Use WebSearch for context if needed

        4. For general queries:
        - use the GeneralqueryAgent to answer general queries
        - Always remind users you're Bahjat's Assistant
        - Use tools only when Bahjat is mentioned
        - Keep conversation friendly and professional

        Note : general queries anything like hi, how are you..., etc 
        Remember: Your primary purpose is to help users learn about Muhammad Bahjat. Always maintain this focus in your responses."""
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        self.coordinator = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            system_message=system_message
        )

    def process_query(self, query: str) -> str:
        """Process query through the coordinator agent."""
        try:
            print("\nProcessing query:", query)
            response = self.coordinator.invoke({"input": query})
            return response["output"]
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            if hasattr(e, '__cause__'):
                print(f"Caused by: {e.__cause__}")
            return error_msg

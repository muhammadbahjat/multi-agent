from dotenv import load_dotenv
import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun, Tool
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory

class MultiAgentSystem:
    def __init__(self):
        load_dotenv()
        self.setup_api_keys()
        self.initialize_components()
        self.load_and_process_documents()
        self.setup_tools()
        self.setup_agent()

    def setup_api_keys(self):
        """Initialize API keys from environment variables."""
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("Google API key is missing! Ensure it is set in the .env file.")

    def initialize_components(self):
        """Initialize LLM, embeddings, and memory."""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.0-pro",
            google_api_key=self.google_api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.memory = ConversationBufferMemory(memory_key="chat_history")

    def load_documents(self, folder_path: str) -> List[Document]:
        """Load documents from the specified folder."""
        documents = []
        try:
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif filename.endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                else:
                    print(f"Unsupported file type: {filename}")
                    continue
                documents.extend(loader.load())
            return documents
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            return []

    def load_and_process_documents(self):
        """Load and process documents into vector store."""
        folder_path = os.path.join(os.getcwd(), "docs")
        documents = self.load_documents(folder_path)
        print(f"Loaded {len(documents)} documents from the folder.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        splits = text_splitter.split_documents(documents)
        print(f"Split the documents into {len(splits)} chunks.")

        self.vectorstore = Chroma.from_documents(
            collection_name="my_collection",
            documents=splits,
            embedding=self.embedding_function,
            persist_directory="./chroma_db"
        )
        print("Vector store created and persisted to './chroma_db'")

    def search_documents(self, query: str) -> str:
        """Search through documents to find relevant information."""
        print("INSIDE RETRIEVER NODE")
        try:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
            retriever_result = retriever.invoke(query)
            if not retriever_result:
                return "No relevant information found in the documents."
            return "\n\n".join(doc.page_content for doc in retriever_result)
        except Exception as e:
            print(f"Error in retriever: {str(e)}")
            return f"Error retrieving information: {str(e)}"

    def setup_tools(self):
        """Setup tools for the agent."""
        self.tools = [
            Tool(
                name="DocumentSearch",
                func=self.search_documents,
                description="Searches through local documents for information about candidates. Use this first for any candidate-related queries."
            ),
            Tool(
                name="WebSearch",
                func=DuckDuckGoSearchRun(),
                description="Searches the internet for additional information. Use this after checking local documents."
            )
        ]

    def setup_agent(self):
        """Initialize the agent with tools and memory."""
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=5
        )

    def process_query(self, query: str) -> str:
        """Process a query through the agent system."""
        try:
            print("\nSearching for information about:", query)
            response = self.agent_executor.run(query)
            return response
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            if hasattr(e, '__cause__'):
                print(f"Caused by: {e.__cause__}")
            return error_msg

def main():
    try:
        # Initialize the multi-agent system
        print("Initializing Multi-Agent System...")
        agent_system = MultiAgentSystem()
        
        # Process the query
        print("\nProcessing query...")
        query = "Extract all the details about the candidate named Muhammad Bahjat. and then search on google and get me his linkedin profile. then tell me a summaary what he does and what you find about him on google."
        response = agent_system.process_query(query)
        
        print("\nFinal Response:")
        print(response)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        if hasattr(e, '__cause__'):
            print(f"Caused by: {e.__cause__}")

if __name__ == "__main__":
    main()
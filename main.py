from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from langchain.tools import tool, BaseTool
from pydantic import BaseModel
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain_chroma import Chroma
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.google import Gemini

load_dotenv()

# Get API keys from environment
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not TAVILY_API_KEY or not GOOGLE_API_KEY:
    raise ValueError("API keys are missing! Ensure they are set in the .env file.")

# Initialize AI tools
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY
)
web_search_tool = TavilySearchResults(max_results=2)

# Function to load documents
def load_documents(folder_path: str) -> List[Document]:
    documents = []
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

# Define folder path
folder_path = os.path.join(os.getcwd(), "docs")

# Load documents
documents = load_documents(folder_path)
print(f"Loaded {len(documents)} documents from the folder.")

### Spitting the documents into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

splits = text_splitter.split_documents(documents)
print(f"Split the documents into {len(splits)} chunks.")


collection_name = "my_collection"
vectorstore = Chroma.from_documents(
    collection_name=collection_name,
    documents=splits,
    embedding=embedding_function,
    persist_directory="./chroma_db"
)
print("Vector store created and persisted to './chroma_db'")


class RagToolSchema(BaseModel):
    question: str

class RetrieverTool(BaseTool):
    name: str = "retriever_tool"
    description: str = "Tool to Retrieve Semantically Similar documents to answer User Questions related to FutureSmart AI"
    args_schema: type[BaseModel] = RagToolSchema
    
    def _run(self, question: str) -> str:
        """Execute the tool"""
        print("INSIDE RETRIEVER NODE")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        retriever_result = retriever.invoke(question)
        return "\n\n".join(doc.page_content for doc in retriever_result)

# Create the tool instance
retriever_tool = RetrieverTool()
retriever_tool.__name__ = "retriever_tool"

question = "What are the experinces of the candidates and what is the name? "

result = retriever_tool.invoke({"question": question})
print(result)

google_search = Agent(
    model=Gemini(id="gemini-2.5-flash"),
    tools=[DuckDuckGo()],
    description="You are a web search agent designed to search the web for the user.",
    instructions=[
        "Use the DuckDuckGo tool to search the web and return the results based on the user's query.",
    ],
    show_tool_calls=True,
    debug_mode=True,
)



rag_agent = Agent(
    model=Gemini(id="gemini-2.5-flash"),
    tools=[retriever_tool],
    description="You are a RAG agent to answer user questions",
    instructions=[
        "Given the instruction's from the user you need to search the vector store and return the revelent results.",
    ],
    show_tool_calls=True,
    debug_mode=True,
)

super_agent = Agent(
    model=Gemini(id="gemini-2.5-flash"),
    name="Super Agent",
    team=[rag_agent, google_search],
    description="You are a super agent and you need to call and perform task based on query of user",
    instructions=[
        "Given the instruction's from the user call the relevant agent to perform the task and return the results.",
    ],
    show_tool_calls=True,
    debug_mode=True,
)

def transfer_funtion_RAG():
    print("handling to the RAG Agent")
    return rag_agent

def transfer_funtion_google():
    print("handling to the Google Agent")
    return google_search

# super_agent.functions = [transfer_funtion_RAG, transfer_funtion_google]
# messages = [
#     {"role": "system", "content": "You are a super agent designed to assist the user."},
#     {"role": "user", "content": "What are the experiences of the candidates and what is the name? What did you find on Google about him?"},
# ]

response = super_agent.run("Tell me about  Kathryn I need top skills Certificate Honors location education and experience. By using the retriever tool only no need to search on google or duck duck go.I want you to list only whats inside the doc")

print(response)
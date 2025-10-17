from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any
from langchain.tools import tool
from pydantic import BaseModel
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from phi.agent import Agent
from phi.model.google import Gemini
import json
import re

load_dotenv()

# Get API keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing! Ensure it is set in the .env file.")

# Initialize AI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY
)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

class JobMatchingSystem:
    def __init__(self, docs_folder: str = "docs"):
        self.docs_folder = docs_folder
        self.vectorstore = None
        self.candidates_data = {}
        self._initialize_vectorstore()
        self._extract_candidate_data()
    
    def _initialize_vectorstore(self):
        """Initialize the vector store with candidate documents"""
        # Load documents
        documents = self._load_documents(self.docs_folder)
        print(f"Loaded {len(documents)} documents from the folder.")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        splits = text_splitter.split_documents(documents)
        print(f"Split the documents into {len(splits)} chunks.")
        
        # Create vector store
        collection_name = "candidate_collection"
        self.vectorstore = Chroma.from_documents(
            collection_name=collection_name,
            documents=splits,
            embedding=embedding_function,
            persist_directory="./chroma_db"
        )
        print("Vector store created and persisted to './chroma_db'")
    
    def _load_documents(self, folder_path: str) -> List[Document]:
        """Load documents from the specified folder"""
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
    
    def _extract_candidate_data(self):
        """Extract and structure candidate data from documents"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        
        # Get all documents to extract candidate information
        all_docs = retriever.invoke("candidate resume profile experience education skills")
        
        for doc in all_docs:
            content = doc.page_content
            # Extract candidate name (look for common patterns)
            name_match = re.search(r'([A-Z][a-z]+ [A-Z][a-z]+)', content)
            if name_match:
                candidate_name = name_match.group(1)
                if candidate_name not in self.candidates_data:
                    self.candidates_data[candidate_name] = {
                        'name': candidate_name,
                        'content': content,
                        'skills': [],
                        'experience': [],
                        'education': [],
                        'location': '',
                        'certifications': []
                    }
                else:
                    # Merge content if candidate already exists
                    self.candidates_data[candidate_name]['content'] += "\n\n" + content
        
        print(f"Extracted data for {len(self.candidates_data)} candidates: {list(self.candidates_data.keys())}")
    
    def _create_candidate_search_tool(self):
        """Create a tool for searching candidate information"""
        
        class CandidateSearchSchema(BaseModel):
            query: str
            candidate_name: str = None
        
        @tool(args_schema=CandidateSearchSchema)
        def candidate_search_tool(query: str, candidate_name: str = None):
            """Tool to search for specific candidate information"""
            print(f"Searching for: {query}")
            if candidate_name:
                print(f"Looking specifically for candidate: {candidate_name}")
            
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # If specific candidate name is provided, include it in the search
            search_query = query
            if candidate_name:
                search_query = f"{candidate_name} {query}"
            
            results = retriever.invoke(search_query)
            
            # Filter results by candidate name if specified
            if candidate_name:
                filtered_results = []
                for doc in results:
                    if candidate_name.lower() in doc.page_content.lower():
                        filtered_results.append(doc)
                results = filtered_results
            
            return "\n\n".join(doc.page_content for doc in results)
        
        return candidate_search_tool
    
    def _create_job_analysis_tool(self):
        """Create a tool for analyzing job requirements"""
        
        class JobAnalysisSchema(BaseModel):
            job_description: str
        
        @tool(args_schema=JobAnalysisSchema)
        def job_analysis_tool(job_description: str):
            """Tool to analyze job requirements and extract key skills, experience, and qualifications"""
            analysis_prompt = f"""
            Analyze the following job description and extract:
            1. Required skills (technical and soft skills)
            2. Required experience level and years
            3. Required education/qualifications
            4. Preferred qualifications
            5. Job location
            6. Industry/domain
            
            Job Description:
            {job_description}
            
            Return the analysis in JSON format.
            """
            
            response = llm.invoke(analysis_prompt)
            return response.content
        
        return job_analysis_tool
    
    def _create_candidate_matching_tool(self):
        """Create a tool for matching candidates to job requirements"""
        
        class CandidateMatchingSchema(BaseModel):
            job_requirements: str
            candidate_name: str
        
        @tool(args_schema=CandidateMatchingSchema)
        def candidate_matching_tool(job_requirements: str, candidate_name: str):
            """Tool to match a specific candidate against job requirements"""
            if candidate_name not in self.candidates_data:
                return f"Candidate {candidate_name} not found in the database."
            
            candidate_data = self.candidates_data[candidate_name]
            
            matching_prompt = f"""
            Analyze how well the following candidate matches the job requirements:
            
            Job Requirements:
            {job_requirements}
            
            Candidate Information:
            Name: {candidate_data['name']}
            Content: {candidate_data['content']}
            
            Provide a detailed analysis including:
            1. Match percentage (0-100%)
            2. Strengths that align with the job
            3. Areas where the candidate might be lacking
            4. Overall recommendation
            
            Be specific and detailed in your analysis.
            """
            
            response = llm.invoke(matching_prompt)
            return response.content
        
        return candidate_matching_tool
    
    def find_best_candidate(self, job_description: str) -> Dict[str, Any]:
        """Find the best matching candidate for a given job"""
        
        # Create tools
        candidate_search_tool = self._create_candidate_search_tool()
        job_analysis_tool = self._create_job_analysis_tool()
        candidate_matching_tool = self._create_candidate_matching_tool()
        
        # Create the job matching agent
        job_matching_agent = Agent(
            model=Gemini(id="gemini-2.5-flash"),
            tools=[candidate_search_tool, job_analysis_tool, candidate_matching_tool],
            description="You are a job matching specialist that finds the best candidates for job positions",
            instructions=[
                "1. First, analyze the job description to understand requirements",
                "2. Search for relevant candidates in the database",
                "3. For each candidate, perform a detailed matching analysis",
                "4. Rank candidates based on their match percentage",
                "5. Provide the top 3 or 1 if no other candidate is a best fit candidates with short 1 or 2 line desc and there name. Nothing else is required.",
                "6. Be specific about why each candidate is a good or poor match"
            ],
            show_tool_calls=True,
            debug_mode=True,
        )
        
        # Create the matching prompt
        matching_prompt = f"""
        I need you to find the best candidate for this job position:
        
        {job_description}
        
        Please:
        1. Analyze the job requirements
        2. Search through all available candidates
        3. Match each candidate against the requirements
        4. Provide the top 3 candidates ranked by match quality
        5. For each candidate, provide:
           - Match percentage
           - Key strengths
           - Areas of concern
           - Overall recommendation
        
        Available candidates: {list(self.candidates_data.keys())}
        """
        
        # Get the response from the agent
        response = job_matching_agent.run(matching_prompt)
        
        return {
            "job_description": job_description,
            "available_candidates": list(self.candidates_data.keys()),
            "recommendation": response,
            "total_candidates_analyzed": len(self.candidates_data)
        }
    
    def get_candidate_details(self, candidate_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific candidate"""
        if candidate_name not in self.candidates_data:
            return {"error": f"Candidate {candidate_name} not found"}
        
        candidate_data = self.candidates_data[candidate_name]
        
        # Use AI to extract structured information
        extraction_prompt = f"""
        Extract the following information from this candidate's resume:
        
        {candidate_data['content']}
        
        Please provide:
        1. Full name
        2. Contact information (email, phone, location)
        3. Professional summary
        4. Technical skills
        5. Work experience (with dates and descriptions)
        6. Education
        7. Certifications
        8. Languages
        9. Any other relevant information
        
        Format the response in a clear, structured way.
        """
        
        response = llm.invoke(extraction_prompt)
        
        return {
            "candidate_name": candidate_name,
            "raw_content": candidate_data['content'],
            "structured_info": response.content
        }

def main():
    """Main function to demonstrate the job matching system"""
    print("Initializing Job Matching System...")
    
    # Initialize the system
    job_matcher = JobMatchingSystem()
    
    # Example job description
    job_description = """
    We are looking for a 
    Machine Learning Engineer
    Reducto
    •
    full-time
    ShareShare
    Download
    Download as PDF
    ID:
    SRN2025-11856
    Location:
    San Francisco, CA
    Work Type:
    onsite
    Total Fee:
    $21000
    About the Company
    The vast majority of enterprise data is in files like PDFs and spreadsheets. That includes everything from financial statements to medical records. Reducto helps AI teams turn those really complex documents into LLM-ready inputs with exceptional accuracy. Hundreds of companies have signed up to use Reducto since our launch, and we’re now processing tens of millions of pages every month for teams ranging from startups to Fortune 10 enterprises. We’re hiring founding software engineers to help us continue to serve our customers as we build the ingestion layer that connects human data with LLMs.

    Roles and Responsibilities:
    Training and deploying new state-of-the-art models for parsing and interpreting unstructured data

    Experimenting with novel techniques to improve LLM accuracy

    Building data pipelines, evaluating model performance, and integrating models into the product

    Working directly with the founders and customers to shape the product direction and engineering strategy

    Job Requirements:
    Experience at companies with strong ML talent

    Strong Python skills

    Experience deploying models to production

    Experience implementing ML systems for high-stakes domains (healthcare, finance, government)

    Graduated from a top CS 15 school

    Experience training models and working with ML pipelines

    Familiarity with modern large language models

    Comfortable with both software engineering and ML research

    Passion for staying up-to-date with the latest research and state-of-the-art models

    Great communication and collaboration skills

    Comfortable and will mesh well with existing intense startup culture

    Able to work in-office 5 days a week in San Francisco

    Interview Process:
    Recruiter Initial Screen

    HM / Coding / ML Knowledge

    [Onsite] Working Session

    [Onsite] Final Culture Interview
    """
    
    print("\n" + "="*50)
    print("JOB MATCHING ANALYSIS")
    print("="*50)
    
    # Find the best candidate
    result = job_matcher.find_best_candidate(job_description)
    
    print(f"\nJob Description: {result['job_description']}")
    print(f"\nAvailable Candidates: {result['available_candidates']}")
    print(f"\nTotal Candidates Analyzed: {result['total_candidates_analyzed']}")
    print(f"\nRecommendation:\n{result['recommendation']}")
    
    # Example: Get details for a specific candidate
    if result['available_candidates']:
        print("\n" + "="*50)
        print("CANDIDATE DETAILS")
        print("="*50)
        
        first_candidate = result['available_candidates'][0]
        candidate_details = job_matcher.get_candidate_details(first_candidate)
        
        if "error" not in candidate_details:
            print(f"\nDetailed Information for {first_candidate}:")
            print(candidate_details['structured_info'])

if __name__ == "__main__":
    main()

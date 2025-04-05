import os
import uuid
import sqlite3
import time
from datetime import datetime
import streamlit as st
import arxiv
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
from io import BytesIO

# Langchain imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Research Paper RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# App title and description
st.title("Research Paper Assistant")
st.markdown("""
This app helps you analyze research papers. You can:
- Upload a PDF research paper
- Search for papers on arXiv
- Ask questions about the papers
""")

# Initialize session state variables
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_paper' not in st.session_state:
    st.session_state.current_paper = None
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'last_arxiv_papers' not in st.session_state:
    st.session_state.last_arxiv_papers = []

# Database functions
DB_NAME = "rag_app.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     gpt_response TEXT,
                     model TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_application_logs(session_id, user_query, gpt_response, model):
    conn = get_db_connection()
    conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                 (session_id, user_query, gpt_response, model))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['gpt_response']}
        ])
    conn.close()
    return messages

# Initialize database
create_application_logs()

# Initialize model and embeddings
@st.cache_resource
def initialize_model():
    # Check if API key is available
    if not os.environ.get("TOGETHERAI_API_KEY"):
        st.error("TOGETHERAI_API_KEY is not set. Please set this environment variable.")
    
    model = ChatTogether(model="meta-llama/Llama-3-70b-chat-hf")
    embedding_function = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return model, embedding_function, text_splitter

model, embedding_function, text_splitter = initialize_model()

# Function to process uploaded PDF
def process_uploaded_pdf(uploaded_file):
    if st.session_state.current_paper:
        return "Error: Only one PDF can be processed at a time. Clear the current PDF first."

    if uploaded_file is None:
        return "Error: No file uploaded."

    try:
        # Save the uploaded file to a temporary path
        pdf_bytes = uploaded_file.getvalue()
        temp_path = f"temp_{uuid.uuid4()}.pdf"
        with open(temp_path, "wb") as f:
            f.write(pdf_bytes)
            
        # Use PyMuPDFLoader instead of PyPDFLoader for better compatibility
        loader = PyMuPDFLoader(temp_path)
        documents = loader.load()
        
        if not documents:
            os.remove(temp_path)
            return "Error: Failed to load the PDF. Please upload a valid PDF."

        splits = text_splitter.split_documents(documents)
        
        st.session_state.vectorstore = Chroma.from_documents(
            collection_name="research_paper",
            documents=splits,
            embedding=embedding_function,
            persist_directory="./chroma_db"
        )
        
        st.session_state.current_paper = {"source": uploaded_file.name}
        setup_rag_chain()
        
        # Clean up the temp file
        os.remove(temp_path)
        
        return f"PDF uploaded successfully! Loaded {len(documents)} page(s) and split into {len(splits)} chunks. What would you like to know about it?"
    except Exception as e:
        return f"Error: Failed to process PDF - {str(e)}. Please upload a valid PDF."

# Function to fetch arXiv papers
def fetch_arxiv_papers(query, max_results=3):
    if st.session_state.current_paper and "processed" in st.session_state.current_paper:
        return "Error: Clear the current paper or PDF before fetching new papers from arXiv."

    try:
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
        papers = list(search.results())
        st.session_state.last_arxiv_papers = papers

        if not papers:
            return "No arXiv results found for your query."

        response = "Here are some recent papers from arXiv:\n"
        for i, paper in enumerate(papers, 1):
            authors = ", ".join([author.name for author in paper.authors])
            arxiv_link = paper.entry_id
            response += f"""
            {i}. **{paper.title}**
               - **Abstract**: {paper.summary[:200]}...
               - **DOI**: {paper.doi or 'Not found'}
               - **Published**: {paper.published.year}
               - **Authors**: {authors}
               - **arXiv Link**: {arxiv_link}
               - Select this paper by clicking the button below
            """
        return response
    except Exception as e:
        return f"Error fetching arXiv papers: {str(e)}"

# Function to select an arXiv paper
def select_arxiv_paper(paper_number):
    try:
        if not st.session_state.last_arxiv_papers or paper_number < 1 or paper_number > len(st.session_state.last_arxiv_papers):
            return "Error: Invalid paper selection."

        paper = st.session_state.last_arxiv_papers[paper_number - 1]
        st.session_state.current_paper = {"source": paper.entry_id, "title": paper.title, "metadata": paper}

        authors_str = ", ".join([author.name for author in paper.authors])
        arxiv_link = paper.entry_id

        response = f"""
        **Selected Paper #{paper_number}: {paper.title}**
        - **Abstract**: {paper.summary[:200]}...
        - **DOI**: {paper.doi or 'Not found'}
        - **Publish Year**: {paper.published.year}
        - **Authors**: {authors_str}
        - **arXiv Link**: {arxiv_link}

        This paper has been selected! What would you like to know about it? (Note: Content will be processed on your first question.)
        """
        return response
    except Exception as e:
        return f"Error selecting paper: {str(e)}"

# Function to process selected paper
def process_selected_paper(paper):
    try:
        # Create a temporary file for the downloaded PDF
        temp_file = f"temp_{uuid.uuid4()}.pdf"
        paper.download_pdf(filename=temp_file)
        
        # Use PyMuPDFLoader instead of PyPDFLoader
        loader = PyMuPDFLoader(temp_file)
        documents = loader.load()
        
        if not documents:
            os.remove(temp_file)
            return False

        splits = text_splitter.split_documents(documents)
        
        st.session_state.vectorstore = Chroma.from_documents(
            collection_name="research_paper",
            documents=splits,
            embedding=embedding_function,
            persist_directory="./chroma_db"
        )
        
        st.session_state.current_paper["processed"] = True
        
        # Clean up the temporary file
        os.remove(temp_file)
        
        return True
    except Exception as e:
        st.error(f"Error processing paper: {str(e)}")
        return False

# Function to set up RAG chain
def setup_rag_chain():
    if st.session_state.vectorstore and not st.session_state.rag_chain:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question, formulate a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
            ("system", "Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
        st.session_state.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Function to handle user input
def handle_input(user_input):
    if not user_input or user_input.strip() == "":
        return "Please enter text or process the uploaded PDF."

    # Command to clear the current PDF/Paper
    if user_input.lower() == "clear pdf":
        st.session_state.current_paper = None
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.last_arxiv_papers = []
        response = "Current paper or PDF cleared. You can upload a new one or enter a topic."
        insert_application_logs(st.session_state.session_id, user_input, response, "meta-llama/Llama-3-70b-chat-hf")
        st.session_state.chat_history = get_chat_history(st.session_state.session_id)
        return response

    # Process selected paper if not already processed
    if st.session_state.current_paper and "metadata" in st.session_state.current_paper and "processed" not in st.session_state.current_paper:
        if not process_selected_paper(st.session_state.current_paper["metadata"]):
            return "Error: Failed to process the selected paper. Please try again or select another paper."
        setup_rag_chain()
        response = st.session_state.rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})["answer"]
        insert_application_logs(st.session_state.session_id, user_input, response, "meta-llama/Llama-3-70b-chat-hf")
        st.session_state.chat_history = get_chat_history(st.session_state.session_id)
        return response

    # Answer questions about the current paper using the RAG chain
    elif st.session_state.rag_chain and st.session_state.current_paper:
        response = st.session_state.rag_chain.invoke({"input": user_input, "chat_history": st.session_state.chat_history})["answer"]
        insert_application_logs(st.session_state.session_id, user_input, response, "meta-llama/Llama-3-70b-chat-hf")
        st.session_state.chat_history = get_chat_history(st.session_state.session_id)
        return response

    # Handle arXiv search if no paper is selected
    else:
        response = fetch_arxiv_papers(user_input)
        insert_application_logs(st.session_state.session_id, user_input, response, "meta-llama/Llama-3-70b-chat-hf")
        st.session_state.chat_history = get_chat_history(st.session_state.session_id)
        return response

# Function to generate a visualization
def generate_visualization(page_count):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Pages"], [page_count], color='blue')
    ax.set_title("PDF Page Count Summary")
    ax.set_ylabel("Number of Pages")
    
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return buf

# Sidebar
with st.sidebar:
    st.header("Upload & Options")
    
    # PDF Upload
    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                result = process_uploaded_pdf(uploaded_file)
                st.success(result)
                
                # If PDF was successfully processed, offer visualization
                if "Error" not in result and st.session_state.vectorstore:
                    if st.button("Generate Summary Visualization"):
                        # Get the number of pages from the document
                        page_count = len(PyMuPDFLoader(uploaded_file.name).load())
                        image_buf = generate_visualization(page_count)
                        st.image(image_buf, caption="PDF Page Count Summary")
    
    # Clear button
    if st.session_state.current_paper:
        if st.button("Clear Current Paper"):
            st.session_state.current_paper = None
            st.session_state.vectorstore = None
            st.session_state.rag_chain = None
            st.session_state.last_arxiv_papers = []
            st.success("Cleared current paper.")
    
    # Display current paper info
    if st.session_state.current_paper:
        st.subheader("Current Paper")
        if "title" in st.session_state.current_paper:
            st.write(f"**Title:** {st.session_state.current_paper['title']}")
        else:
            st.write(f"**Source:** {st.session_state.current_paper['source']}")

# Main chat interface
st.header("Chat")

# Display arXiv search results and selection buttons
if st.session_state.last_arxiv_papers and not st.session_state.current_paper:
    st.subheader("Search Results")
    for i, paper in enumerate(st.session_state.last_arxiv_papers, 1):
        st.write(f"**{i}. {paper.title}**")
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(f"Authors: {', '.join([author.name for author in paper.authors])}")
            st.write(f"Published: {paper.published.year}")
        with col2:
            if st.button(f"Select #{i}", key=f"select_{i}"):
                response = select_arxiv_paper(i)
                st.success(response)
                st.experimental_rerun()

# Chat history display
st.subheader("Conversation")
for message in get_chat_history(st.session_state.session_id):
    if message["role"] == "human":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Assistant:** {message['content']}")

# User input
user_query = st.text_input("Ask a question or enter an arXiv search term:", key="user_input")
if st.button("Submit"):
    if user_query:
        with st.spinner("Processing..."):
            response = handle_input(user_query)
            st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("Research Paper Assistant powered by LangChain and Together AI")

if __name__ == "__main__":
    # This will use the PORT environment variable if it exists (needed for Render)
    port = int(os.environ.get("PORT", 8501))
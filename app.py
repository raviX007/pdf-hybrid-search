import streamlit as st
import os
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import tempfile

# Set page configuration
st.set_page_config(page_title="PDF Hybrid Search", layout="wide")

# Title and description
st.title("PDF Hybrid Search")
st.markdown("""
Upload PDF documents and search through their contents using hybrid search technology 
that combines keyword-based (BM25) and semantic (FAISS) search methods.
""")

# Initialize session state for storing documents
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = []

# Sidebar for OpenAI API key
with st.sidebar:
    st.header("Configuration")
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to proceed")
        st.stop()
    
    os.environ["OPENAI_API_KEY"] = openai_api_key

def process_pdf_text(text):
    """Process extracted text into appropriate chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def extract_text_from_pdf(pdf_file):
    """Extract and process text from uploaded PDF file"""
    try:
        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            
            # Open the PDF file
            pdf_reader = PyPDF2.PdfReader(tmp_file.name)
            
            # Extract text from all pages
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text() + "\n\n"
            
            # Process the extracted text into chunks
            if full_text.strip():
                chunks = process_pdf_text(full_text)
                return chunks
            
        os.unlink(tmp_file.name)  # Delete the temporary file
        return []
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return []

# PDF Upload Section
st.header("Upload PDFs")
uploaded_files = st.file_uploader(
    "Upload PDF files", 
    type=['pdf'],
    accept_multiple_files=True,
    help="Upload one or more PDF files to search through their contents"
)

if uploaded_files:
    pdf_docs = []
    progress_bar = st.progress(0)
    for i, uploaded_file in enumerate(uploaded_files):
        with st.spinner(f'Processing {uploaded_file.name}...'):
            extracted_texts = extract_text_from_pdf(uploaded_file)
            if extracted_texts:
                pdf_docs.extend(extracted_texts)
                st.success(f"Successfully extracted {len(extracted_texts)} segments from {uploaded_file.name}")
            else:
                st.warning(f"No text could be extracted from {uploaded_file.name}")
            progress_bar.progress((i + 1) / len(uploaded_files))
    
    if pdf_docs:
        st.session_state.processed_docs = pdf_docs
        st.write("Preview of extracted text:")
        with st.expander("Show extracted text"):
            for i, text in enumerate(pdf_docs[:5], 1):
                st.text(f"{i}. {text[:200]}...")
            if len(pdf_docs) > 5:
                st.text(f"... and {len(pdf_docs) - 5} more segments")
    else:
        st.error("No text could be extracted from any of the uploaded PDFs")
        st.stop()

# Verify we have documents to work with
if not st.session_state.processed_docs:
    st.info("Please upload PDF documents to begin searching")
    st.stop()

# Initialize retrievers
@st.cache_resource(show_spinner=False)
def initialize_retrievers(_docs, api_key):
    with st.spinner("Initializing search engines..."):
        # BM25 Retriever
        bm25_retriever = BM25Retriever.from_texts(_docs)
        bm25_retriever.k = 3
        
        # FAISS Retriever
        embedding = OpenAIEmbeddings(openai_api_key=api_key)
        faiss_vectorstore = FAISS.from_texts(_docs, embedding)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Ensemble Retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.3, 0.7]  # Emphasize semantic search
        )
        
        return bm25_retriever, faiss_retriever, ensemble_retriever

# Initialize retrievers
try:
    bm25_retriever, faiss_retriever, ensemble_retriever = initialize_retrievers(
        st.session_state.processed_docs, 
        openai_api_key
    )
except Exception as e:
    st.error(f"Error initializing search engines: {str(e)}")
    st.stop()

# Search interface
st.header("Search PDF Contents")
search_query = st.text_input("Enter your search query")

if search_query:
    st.subheader("Search Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Keyword Search (BM25)")
        bm25_results = bm25_retriever.get_relevant_documents(search_query)
        for i, doc in enumerate(bm25_results, 1):
            with st.expander(f"Result {i}"):
                st.markdown(doc.page_content)
    
    with col2:
        st.markdown("### Semantic Search (FAISS)")
        faiss_results = faiss_retriever.get_relevant_documents(search_query)
        for i, doc in enumerate(faiss_results, 1):
            with st.expander(f"Result {i}"):
                st.markdown(doc.page_content)
    
    with col3:
        st.markdown("### Hybrid Search")
        ensemble_results = ensemble_retriever.get_relevant_documents(search_query)
        for i, doc in enumerate(ensemble_results, 1):
            with st.expander(f"Result {i}"):
                st.markdown(doc.page_content)

# Information about search methods
with st.expander("About the Search Methods"):
    st.markdown("""
    This application uses three different search methods to find relevant information in your PDFs:
    
    ### Keyword Search (BM25)
    - Traditional keyword-based search
    - Good for finding exact matches and word variations
    - Best for specific terms and phrases
    
    ### Semantic Search (FAISS)
    - Uses AI embeddings to understand meaning
    - Can find related content even with different words
    - Better for conceptual and thematic searches
    
    ### Hybrid Search
    - Combines both methods above
    - Uses 30% keyword search and 70% semantic search
    - Provides balanced results for most queries
    """)
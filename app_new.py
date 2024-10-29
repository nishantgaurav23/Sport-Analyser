import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Callable
import glob
from tqdm import tqdm
import pickle
import torch.nn.functional as F
from llama_cpp import Llama
import streamlit as st
import functools
from datetime import datetime
import re
import time

# Force CPU device
torch.device('cpu')

# Logging configuration
LOGGING_CONFIG = {
    'enabled': True,
    'functions': {
        'encode': True,
        'store_embeddings': True,
        'search': True,
        'load_and_process_csvs': True,
        'process_query': True
    }
}

def log_function(func: Callable) -> Callable:
    """Decorator to log function inputs and outputs"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not LOGGING_CONFIG['enabled'] or not LOGGING_CONFIG['functions'].get(func.__name__, False):
            return func(*args, **kwargs)
        
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
        else:
            class_name = func.__module__

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        log_args = args[1:] if class_name != func.__module__ else args
        
        def format_arg(arg):
            if isinstance(arg, torch.Tensor):
                return f"Tensor(shape={list(arg.shape)}, device={arg.device})"
            elif isinstance(arg, list):
                return f"List(len={len(arg)})"
            elif isinstance(arg, str) and len(arg) > 100:
                return f"String(len={len(arg)}): {arg[:100]}..."
            return arg

        formatted_args = [format_arg(arg) for arg in log_args]
        formatted_kwargs = {k: format_arg(v) for k, v in kwargs.items()}

        print(f"\n{'='*80}")
        print(f"[{timestamp}] FUNCTION CALL: {class_name}.{func.__name__}")
        print(f"INPUTS:")
        print(f"  args: {formatted_args}")
        print(f"  kwargs: {formatted_kwargs}")

        result = func(*args, **kwargs)

        formatted_result = format_arg(result)
        print(f"OUTPUT:")
        print(f"  {formatted_result}")
        print(f"{'='*80}\n")

        return result
    return wrapper

def check_environment():
    """Check if the environment is properly set up"""
    try:
        import numpy as np
        import torch
        import sentence_transformers
        import llama_cpp
        return True
    except ImportError as e:
        st.error(f"Missing required package: {str(e)}")
        st.stop()
        return False

@st.cache_resource
def initialize_model():
    """Initialize the Llama model once"""
    model_path = "mistral-7b-v0.1.Q4_K_M.gguf"
    if not os.path.exists(model_path):
        st.error(f"Model file {model_path} not found!")
        st.stop()
    
    llm_config = {
        "n_ctx": 2048,
        "n_threads": 4,
        "n_batch": 512,
        "n_gpu_layers": 0,
        "verbose": False
    }
    
    return Llama(model_path=model_path, **llm_config)


class SentenceTransformerRetriever:
    @st.cache_resource
    def __init__(_self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = "embeddings_cache"):
        # Force CPU device and suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _self.device = torch.device("cpu")
            _self.model = SentenceTransformer(model_name, device="cpu")
            _self.doc_embeddings = None
            _self.cache_dir = cache_dir
            _self.cache_file = "embeddings.pkl"
            os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, data_folder: str = None) -> str:
        return os.path.join(self.cache_dir, self.cache_file)
    
    @log_function
    def save_cache(self, data_folder: str, cache_data: dict):
        cache_path = self.get_cache_path()
        if os.path.exists(cache_path):
            os.remove(cache_path)
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
            print(f"Cache saved at: {cache_path}")
    
    @log_function
    @st.cache_data
    def load_cache(_self, data_folder: str = None) -> dict:
        cache_path = _self.get_cache_path()
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                print(f"Loading cache from: {cache_path}")
                return pickle.load(f)
        return None
    
    @log_function
    def encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
        return F.normalize(embeddings, p=2, dim=1)

    @log_function
    def store_embeddings(self, embeddings: torch.Tensor):
        self.doc_embeddings = embeddings

    @log_function
    def search(self, query_embedding: torch.Tensor, k: int, documents: List[str]):
        if self.doc_embeddings is None:
            raise ValueError("No document embeddings stored!")
        
        # Compute similarities
        similarities = F.cosine_similarity(query_embedding, self.doc_embeddings)
        
        # Get top k scores and indices
        k = min(k, len(documents))
        scores, indices = torch.topk(similarities, k=k)
        
        # Log similarity statistics
        print(f"\nSimilarity Stats:")
        print(f"Max similarity: {similarities.max().item():.4f}")
        print(f"Mean similarity: {similarities.mean().item():.4f}")
        print(f"Selected similarities: {scores.tolist()}")
        
        return indices.cpu(), scores.cpu()
    



class RAGPipeline:
    def __init__(self, data_folder: str, k: int = 5):
        self.data_folder = data_folder
        self.k = k
        self.retriever = SentenceTransformerRetriever()
        self.documents = []
        self.device = torch.device("cpu")
        self.llm = initialize_model()
        
    @log_function
    @st.cache_data
    def load_and_process_csvs(_self):
        cache_data = _self.retriever.load_cache(_self.data_folder)
        if cache_data is not None:
            _self.documents = cache_data['documents']
            _self.retriever.store_embeddings(cache_data['embeddings'])
            return

        csv_files = glob.glob(os.path.join(_self.data_folder, "*.csv"))
        all_documents = []
        
        for csv_file in tqdm(csv_files, desc="Reading CSV files"):
            try:
                df = pd.read_csv(csv_file)
                texts = df.apply(lambda x: " ".join(x.astype(str)), axis=1).tolist()
                all_documents.extend(texts)
            except Exception as e:
                print(f"Error processing file {csv_file}: {e}")
                continue
        
        _self.documents = all_documents
        embeddings = _self.retriever.encode(all_documents)
        _self.retriever.store_embeddings(embeddings)
        
        cache_data = {
            'embeddings': embeddings,
            'documents': _self.documents
        }
        _self.retriever.save_cache(_self.data_folder, cache_data)

    def preprocess_query(self, query: str) -> str:
        """Clean and prepare the query"""
        query = query.lower().strip()
        query = re.sub(r'\s+', ' ', query)
        return query

    def postprocess_response(self, response: str) -> str:
        """Clean up the generated response"""
        response = response.strip()
        response = re.sub(r'\s+', ' ', response)
        response = re.sub(r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}(?:\+\d{2}:?\d{2})?', '', response)
        return response

    @log_function
    def process_query(self, query: str, placeholder) -> str:
        try:
            # Preprocess query
            query = self.preprocess_query(query)
            
            # Show retrieval status
            status = placeholder.empty()
            status.write("🔍 Finding relevant information...")
            
            # Retrieve relevant documents
            query_embedding = self.retriever.encode([query])
            indices, scores = self.retriever.search(query_embedding, self.k, self.documents)
            
            # Print search results for debugging
            print("\nSearch Results:")
            for idx, score in zip(indices.tolist(), scores.tolist()):
                print(f"Score: {score:.4f} | Document: {self.documents[idx][:100]}...")
            
            relevant_docs = [self.documents[idx] for idx in indices.tolist()]
            
            # Update status
            status.write("💭 Generating response...")
            
            # Prepare context and prompt
            context = "\n".join(relevant_docs)
            prompt = f"""Context information is below:
            {context}
            
            Given the context above, please answer the following question:
            {query}

            Guidelines:
            - If you cannot answer based on the context, say so politely
            - Keep the response concise and focused
            - Only include sports-related information
            - No dates or timestamps in the response
            - Use clear, natural language
            
            Answer:"""
            
            # Generate response
            response_placeholder = placeholder.empty()
            generated_text = ""
            
            try:
                response = self.llm(
                    prompt,
                    max_tokens=512,
                    temperature=0.4,
                    top_p=0.95,
                    echo=False,
                    stop=["Question:", "\n\n"]
                )
                
                if response and 'choices' in response and len(response['choices']) > 0:
                    generated_text = response['choices'][0].get('text', '').strip()
                    
                    if generated_text:
                        final_response = self.postprocess_response(generated_text)
                        response_placeholder.markdown(final_response)
                        return final_response
                    else:
                        message = "No relevant answer found. Please try rephrasing your question."
                        response_placeholder.warning(message)
                        return message
                else:
                    message = "Unable to generate response. Please try again."
                    response_placeholder.warning(message)
                    return message
                    
            except Exception as e:
                print(f"Generation error: {str(e)}")
                message = "Had some trouble generating the response. Please try again."
                response_placeholder.warning(message)
                return message
                
        except Exception as e:
            print(f"Process error: {str(e)}")
            message = "Something went wrong. Please try again with a different question."
            placeholder.warning(message)
            return message
        


@st.cache_resource
def initialize_rag_pipeline():
    """Initialize the RAG pipeline once"""
    data_folder = "ESPN_data"  # Update this path as needed
    rag = RAGPipeline(data_folder)
    rag.load_and_process_csvs()
    return rag

def main():
    # Environment check
    if not check_environment():
        return

    # Page config
    st.set_page_config(
        page_title="The Sport Chatbot",
        page_icon="🏆",
        layout="wide"  # Changed back to wide for more space
    )

    # Improved CSS styling
    st.markdown("""
        <style>
        /* Container styling */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Text input styling */
        .stTextInput > div > div > input {
            width: 100%;
        }
        
        /* Button styling */
        .stButton > button {
            width: 200px;
            margin: 0 auto;
            display: block;
            background-color: #FF4B4B;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        
        /* Title styling */
        .main-title {
            text-align: center;
            padding: 1rem 0;
            font-size: 3rem;
            color: #1F1F1F;
        }
        
        .sub-title {
            text-align: center;
            padding: 0.5rem 0;
            font-size: 1.5rem;
            color: #4F4F4F;
        }
        
        /* Description styling */
        .description {
            text-align: center;
            color: #666666;
            padding: 0.5rem 0;
            font-size: 1.1rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }

        /* Answer container styling */
        .stMarkdown {
            max-width: 100%;
        }

        /* Streamlit default overrides */
        .st-emotion-cache-16idsys p {
            font-size: 1.1rem;
            line-height: 1.6;
        }
        
        /* Container for main content */
        .main-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header section with improved styling
    st.markdown("<h1 class='main-title'>🏆 The Sport Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-title'>Using ESPN API</h3>", unsafe_allow_html=True)
    st.markdown("""
        <p class='description'>
            Hey there! 👋 I can help you with information on Ice Hockey, Baseball, American Football, Soccer, and Basketball. 
            With access to the ESPN API, I'm up to date with the latest details for these sports up until October 2024.
        </p>
        <p class='description'>
            Got any general questions? Feel free to ask—I'll do my best to provide answers based on the information I've been trained on!
        </p>
    """, unsafe_allow_html=True)

    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)


    # Initialize the pipeline
    try:
        with st.spinner("Loading resources..."):
            rag = initialize_rag_pipeline()
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        st.error("Unable to initialize the system. Please check if all required files are present.")
        st.stop()

    # Create columns for layout with golden ratio
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col2:
        # Query input with label styling
        query = st.text_input("What would you like to know about sports?")
        
        # Centered button
        if st.button("Get Answer"):
            if query:
                response_placeholder = st.empty()
                try:
                    response = rag.process_query(query, response_placeholder)
                    print(f"Generated response: {response}")
                except Exception as e:
                    print(f"Query processing error: {str(e)}")
                    response_placeholder.warning("Unable to process your question. Please try again.")
            else:
                st.warning("Please enter a question!")

    # Footer with improved styling
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <p style='text-align: center; color: #666666; padding: 1rem 0;'>
            Powered by ESPN Data & Mistral AI 🚀
        </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
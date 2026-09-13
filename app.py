import os
import re
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
# Load environment variables from .env
load_dotenv()
# Set Streamlit Page Configuration
st.set_page_config(page_title="Streamlit RAG with Groq",layout="wide")
st.title("Streamlit RAG Example")
# --- INITIALIZE GROQ CLIENT ---
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error(" `GROQ_API_KEY` not found! Please make sure it's set in your `.env` file.")
    st.stop()
# Initialize official Groq client
client = Groq(api_key=api_key)
# --- SESSION STATE INITIALIZATION ---
if "documents" not in st.session_state:
    st.session_state.documents = []  # List to store document text chunks
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []  # List to store vector representations


# --- HELPER FUNCTIONS ---

def get_text_embedding(text: str) -> np.ndarray:
    """
    Generates a simple, lightweight semantic term frequency vector.
    
    """
    words = re.findall(r'\w+', text.lower())
    # Create a deterministic fixed-dimensional vector hash space
    vec = np.zeros(128, dtype=float)
    for word in words:
        index = hash(word) % 128
        vec[index] += 1.0
    
    # Normalize the vector for cosine similarity calculation
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculates cosine similarity between two numpy vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return float(dot_product / norm_product) if norm_product > 0 else 0.0

def chunk_text(text: str, chunk_size: int = 250) -> list[str]:
    """Splits raw input text into readable chunks."""
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def query_groq_llm(user_query: str, retrieved_context: str) -> str:
    """Queries Groq Llama 3.3 using retrieved context for RAG response."""
    system_prompt = (
        "You are an expert AI assistant. Answer the user's question accurately using ONLY "
        "the provided context. If the answer cannot be found in the context, clearly state "
        "that you don't know based on the provided text."
    )
    
    user_prompt = f"Context:\n{retrieved_context}\n\nUser Question: {user_query}"

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# --- STREAMLIT USER INTERFACE ---

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Ingest Knowledge Base")
    raw_text = st.text_area(
        "Paste your context/knowledge here:",
        height=220,
        placeholder="Paste text, meeting notes, articles, or documentation..."
    )

    if st.button(" Process & Embed Text", use_container_width=True):
        if not raw_text.strip():
            st.warning("Please enter some text to process.")
        else:
            with st.spinner("Chunking text and creating vector embeddings..."):
                chunks = chunk_text(raw_text)
                vectors = [get_text_embedding(c) for c in chunks]

                st.session_state.documents = chunks
                st.session_state.embeddings = vectors

                st.success(f"Indexed {len(chunks)} chunks into vector memory successfully!")

    # Display Indexed Chunks
    if st.session_state.documents:
        with st.expander(" Inspect Stored Context Chunks"):
            for i, chunk in enumerate(st.session_state.documents):
                st.write(f"**Chunk {i+1}:** {chunk}")

with col2:
    st.subheader("2. Ask Questions (RAG System)")
    user_query = st.text_input(
        "Ask something about your context:", 
        placeholder="What is discussed in the text?"
    )

    if st.button(" Retrieve & Answer", use_container_width=True):
        if not st.session_state.documents:
            st.warning("Please process and embed text on the left first.")
        elif not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching vectors & generating Groq response..."):
                # 1. Embed the query vector
                query_vec = get_text_embedding(user_query)

                # 2. Vector Similarity Search (Retrieval)
                similarities = [
                    cosine_similarity(query_vec, doc_vec) 
                    for doc_vec in st.session_state.embeddings
                ]

                # Extract Top-2 most relevant chunks
                top_indices = np.argsort(similarities)[::-1][:2]
                retrieved_chunks = [st.session_state.documents[idx] for idx in top_indices]
                retrieved_context_str = "\n---\n".join(retrieved_chunks)

                # 3. Generate Answer via Groq
                answer = query_groq_llm(user_query, retrieved_context_str)

                # Output Answer
                st.markdown("###  Answer")
                st.info(answer)

                # Display Retrieved Context Source Chunks
                with st.expander(" Retrieved Source Context"):
                    for idx in top_indices:
                        st.write(f"- **[Similarity Score: {similarities[idx]:.3f}]** {st.session_state.documents[idx]}")

#pip install streamlit groq python-dotenv numpy
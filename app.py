import os
import re

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# Reads GROQ_API_KEY from a local .env file (never commit it) or from the host's environment/secrets.
load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found. Add it to a local .env file, or to the app's Secrets when deployed.")
    st.stop()

client = Groq(api_key=api_key)

LLM_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------- The RAG pipeline: chunk -> embed -> index -> retrieve -> answer ----------

@st.cache_resource  # load the embedding model once, not on every click
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Turn texts into normalised vectors, so inner product equals cosine similarity."""
    vectors = load_embedder().encode(texts, normalize_embeddings=True)
    return np.asarray(vectors, dtype="float32")


def chunk_text(text: str, sentences_per_chunk: int = 3) -> list[str]:
    """Split into sentences (or lines), then group a few sentences per chunk."""
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+|\n+", text) if s.strip()]
    return [
        " ".join(sentences[i : i + sentences_per_chunk])
        for i in range(0, len(sentences), sentences_per_chunk)
    ]


def build_index(chunks: list[str]) -> faiss.IndexFlatIP:
    """Store all chunk vectors in a FAISS index for fast similarity search."""
    vectors = embed(chunks)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(question: str, index: faiss.IndexFlatIP, chunks: list[str], k: int = 3):
    """Return the k most similar chunks to the question, with their scores."""
    scores, ids = index.search(embed([question]), min(k, len(chunks)))
    return [(chunks[i], float(score)) for i, score in zip(ids[0], scores[0])]


def ask_llm(question: str, context_chunks: list[str]) -> str:
    """Ask the LLM to answer using only the retrieved chunks."""
    context = "\n---\n".join(context_chunks)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using ONLY the provided context. "
                    "If the answer is not in the context, say you don't know based on the provided text."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ---------- User interface ----------

left, right = st.columns(2, gap="large")

with left:
    st.subheader("1. Add your text")
    raw_text = st.text_area("Paste notes, an article, or documentation:", height=220)

    if st.button("Process text", use_container_width=True):
        chunks = chunk_text(raw_text)
        if not chunks:
            st.warning("Please paste some text first.")
        else:
            with st.spinner("Embedding and indexing..."):
                st.session_state.chunks = chunks
                st.session_state.index = build_index(chunks)
            st.success(f"Indexed {len(chunks)} chunks.")

    if "chunks" in st.session_state:
        with st.expander("View chunks"):
            for number, chunk in enumerate(st.session_state.chunks, start=1):
                st.write(f"**{number}.** {chunk}")

with right:
    st.subheader("2. Ask a question")
    question = st.text_input("Your question:")

    if st.button("Get answer", use_container_width=True):
        if "index" not in st.session_state:
            st.warning("Process some text first.")
        elif not question.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("Searching and answering..."):
                results = retrieve(question, st.session_state.index, st.session_state.chunks)
                try:
                    answer = ask_llm(question, [chunk for chunk, _ in results])
                except Exception as error:
                    st.error(f"The language model request failed: {error}")
                    st.stop()

            st.markdown("### Answer")
            st.info(answer)
            with st.expander("Sources used"):
                for chunk, score in results:
                    st.write(f"**Similarity {score:.2f}:** {chunk}")
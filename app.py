import streamlit as st
import os
import fitz  # PyMuPDF
import google.generativeai as genai
import time
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

# --- MODERN LANGCHAIN IMPORTS ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
PAGE_TITLE = "Clinical Warfarin Bot"
PDF_FILE = "Warfarin MTAC 2020.pdf" 
ICON = "🩺"

st.set_page_config(page_title=PAGE_TITLE, page_icon=ICON, layout="wide")

# --- 1. SECURE API SETUP ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    # Removed genai.configure to avoid conflicts with LangChain
except FileNotFoundError:
    st.error("Secrets file not found.")
    st.stop()
except Exception as e:
    st.error(f"API Error: {e}")
    st.stop()

# --- 2. LOGIC: PDF PROCESSING ---
@st.cache_resource
def load_and_index_pdf(pdf_path):
    if not os.path.exists(pdf_path): return None, None
    doc = fitz.open(pdf_path)
    raw_documents = []
    
    # Inject Page Numbers
    for page_num, page in enumerate(doc):
        text = page.get_text()
        text_with_meta = f"[PAGE INDEX {page_num}]\n{text}"
        raw_documents.append(Document(page_content=text_with_meta, metadata={"page": page_num}))
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(raw_documents)

    # Use standard embedding model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=api_key
    )
    
    vector_store = None
    batch_size = 5 
    total_docs = len(final_documents)
    my_bar = st.progress(0, text="Initializing AI...")

    for i in range(0, total_docs, batch_size):
        batch = final_documents[i : i + batch_size]
        for attempt in range(3):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
                break 
            except Exception:
                time.sleep(1)
        my_bar.progress(min((i + batch_size) / total_docs, 1.0))
        time.sleep(0.1)

    my_bar.empty()
    return doc, vector_store

# --- 3. MODEL SETUP (STABLE CONFIGURATION) ---
@st.cache_resource
def get_chat_model():
    # We use the specific stable version requested
    # We REMOVE transport='rest' to fix the 404 errors
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-002", 
        temperature=0.0, 
        google_api_key=api_key
    )

# --- 4. CITATION EXTRACTOR ---
def extract_page_from_answer(answer_text):
    match = re.search(r"Page (?:Index )?(\d+)", answer_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

# --- 5. UI & CHAT ---
st

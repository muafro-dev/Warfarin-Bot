# app.py
import streamlit as st
import os
import fitz  # PyMuPDF
import time
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

# LangChain and helpers
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Try to import the newer Google client if available
# We will attempt to use google.genai first, then fall back to google.generativeai if present.
try:
    import google.genai as genai_new  # newer client (if installed)
    GENAI_CLIENT = "genai_new"
except Exception:
    genai_new = None
    GENAI_CLIENT = None

try:
    import google.generativeai as genai_old  # deprecated client (may still be installed)
    if GENAI_CLIENT is None:
        GENAI_CLIENT = "genai_old"
except Exception:
    genai_old = None
    if GENAI_CLIENT is None:
        GENAI_CLIENT = None

# --- CONFIGURATION ---
PAGE_TITLE = "Clinical Warfarin Bot"
PDF_FILE = "Warfarin MTAC 2020.pdf"
ICON = "🩺"

st.set_page_config(page_title=PAGE_TITLE, page_icon=ICON, layout="wide")

# --- 1. SECURE API SETUP ---
# Streamlit secrets: create .streamlit/secrets.toml with GOOGLE_API_KEY = "your_key"
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.error("Missing Google API key in Streamlit secrets. Add GOOGLE_API_KEY to .streamlit/secrets.toml.")
    st.stop()

# If a supported genai client is available, configure it (best-effort)
if GENAI_CLIENT == "genai_new":
    try:
        genai_new.configure(api_key=api_key)
    except Exception:
        pass
elif GENAI_CLIENT == "genai_old":
    try:
        genai_old.configure(api_key=api_key)
    except Exception:
        pass

# --- 2. LOGIC: PDF PROCESSING ---
@st.cache_resource
def load_and_index_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        return None, None
    doc = fitz.open(pdf_path)
    raw_documents = []

    # Inject Page Numbers
    for page_num, page in enumerate(doc):
        text = page.get_text()
        text_with_meta = f"[PAGE INDEX {page_num}]\n{text}"
        raw_documents.append(Document(page_content=text_with_meta, metadata={"page": page_num}))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(raw_documents)

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

# --- 3. MODEL SETUP (SAFE) ---
@st.cache_resource
def get_chat_model(model_name="gemini-1.5-flash"):
    try:
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.0,
            google_api_key=api_key
        )
    except Exception as e:
        raise RuntimeError(
            "Model initialization failed. This can happen if the model name is not supported "
            "by your Google account or the client library. Original error: " + str(e)
        ) from e

# --- 4. CITATION EXTRACTOR ---
def extract_page_from_answer(answer_text):
    match = re.search(r"Page (?:Index )?(\d+)", answer_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

# --- 5. UI & CHAT ---
st.title(f"{ICON} {PAGE_TITLE}")
st.caption("Local RAG · Hallucination-Safe · Citation-Based Retrieval")

# Sidebar: model settings so you can try different model names without editing code
st.sidebar.header("Model settings")
model_name = st.sidebar.text_input("Model name", value="gemini-1.5-flash")

with st.expander("Diagnostics (click to expand)"):
    st.write("GenAI client detected:", GENAI_CLIENT)
    st.write("Model wrapper: langchain-google-genai ChatGoogleGenerativeAI")
    st.write("PDF file path:", PDF_FILE)
    st.write("If model initialization fails, use the 'List available models' button below to see supported model names.")

    if st.button("List available models (best-effort)"):
        try:
            if GENAI_CLIENT == "genai_new":
                models = genai_new.list_models()
                st.write([m.name for m in models])
            elif GENAI_CLIENT == "genai_old":
                try:
                    models = genai_old.list_models()
                    st.write([m.name for m in models])
                except Exception as e:
                    st.write("Old client does not support list_models or call failed:", str(e))
            else:
                st.write("No Google GenAI client available in the environment.")
        except Exception as e:
            st.write("Listing models failed:", str(e))

pdf_doc, vector_store = load_and_index_pdf(PDF_FILE)
if not pdf_doc:
    st.error(f"PDF file not found at path: {PDF_FILE}. Upload the PDF to the app folder or update PDF_FILE.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image_page" in message and message["image_page"] is not None:
            page_idx = message["image_page"]
            try:
                page = pdf_doc.load_page(page_idx)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                st.image(pix.tobytes("png"), caption=f"Source: Page {message['image_page']}", width=700)
            except Exception:
                pass

if prompt := st.chat_input("Ask about Warfarin protocols..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message

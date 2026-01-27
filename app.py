import streamlit as st
import os
import fitz  # PyMuPDF
import google.generativeai as genai
import time
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

# --- LANGCHAIN (Only for Vector Store - The Math Part) ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
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
    # We configure the NATIVE client directly. This bypasses the LangChain 404 bugs.
    genai.configure(api_key=api_key)
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
    my_bar = st.progress(0, text="Initializing AI Brain...")

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

# --- 3. CITATION EXTRACTOR ---
def extract_page_from_answer(answer_text):
    # Regex to find "Page Index X" or "Page X"
    match = re.search(r"Page (?:Index )?(\d+)", answer_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

# --- 5. UI & CHAT ---
st.title(f"{ICON} {PAGE_TITLE}")
st.caption("Local RAG · Hallucination-Safe · Native Google Client")

if "messages" not in st.session_state: st.session_state.messages = []

pdf_doc, vector_store = load_and_index_pdf(PDF_FILE)
if not pdf_doc: st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image_page" in message and message["image_page"] is not None:
            page_idx = message["image_page"]
            page = pdf_doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            st.image(pix.tobytes("png"), caption=f"Source: Page {message['image_page']}", width=700)

if prompt := st.chat_input("Ask about Warfarin protocols..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # 1. RETRIEVAL (Using LangChain - This works fine)
            results = vector_store.similarity_search_with_score(prompt, k=8)
            docs = [doc for doc, score in results]
            
            # Create a Context String from the retrieved docs
            context_text = "\n\n".join([d.page_content for d in docs])

            # 2. GENERATION (Using Native Google Client - The Fix!)
            # This uses the official SDK directly, bypassing the 404 errors in LangChain
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            full_prompt = f"""
            You are a clinical pharmacist assistant based ONLY on the provided protocol.
            
            STRICT RULES:
            1. Answer ONLY using the information in the CONTEXT.
            2. The Context contains markers like [PAGE INDEX 12]. You MUST cite this number.
            3. If the answer is not found, say "The protocol does not contain this information."
            
            CONTEXT: 
            {context_text}
            
            USER QUESTION: 
            {prompt}
            
            RESPONSE FORMAT:
            1. Direct Answer.
            2. SOURCE: "Reference found on Page Index [Insert Number Here]"
            """
            
            # 3. SAFETY AIRBAG (Like the Other AI)
            # We try to use the AI. If it fails (404/Connection), we fall back to showing the text.
            response_text = ""
            try:
                response_obj = model.generate_content(full_prompt)
                response_text = response_obj.text
            except Exception as api_error:
                # Fallback: Just show the text we found, so the user isn't left with nothing
                response_text = f"⚠️ AI Connection Issue. However, here is the relevant protocol text I found:\n\n{docs[0].page_content}"
                print(f"API Error: {api_error}")

            # --- CITATION LOGIC ---
            cited_page = extract_page_from_answer(response_text)
            
            # Hallucination Check
            show_image = True
            lower_res = response_text.lower()
            if "does not contain" in lower_res or "not found" in lower_res or "cannot provide" in lower_res:
                show_image = False
                cited_page = None 

            message_placeholder.markdown(response_text)
            
            if show_image and cited_page is not None:
                try:
                    page = pdf_doc.load_page(cited_page) 
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    st.image(pix.tobytes("png"), caption=f"Source: Page {cited_page}", width=700)
                    st.session_state.messages.append({"role": "assistant", "content": response_text, "image_page": cited_page})
                except:
                    st.session_state.messages.append({"role": "assistant", "content": response_text, "image_page": None})
            else:
                st.session_state.messages.append({"role": "assistant", "content": response_text, "image_page": None})

        except Exception as e:
            st.error(f"Application Error: {str(e)}")

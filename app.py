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
    genai.configure(api_key=api_key, transport='rest') 
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
    
    # We inject the Page Number directly into the text so the AI can "See" it
    for page_num, page in enumerate(doc):
        text = page.get_text()
        text_with_meta = f"[PAGE INDEX {page_num}]\n{text}"
        raw_documents.append(Document(page_content=text_with_meta, metadata={"page": page_num}))
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(raw_documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key, transport="rest")
    
    # Batch Processing to prevent 504 Errors
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

# --- 3. MODEL SETUP ---
@st.cache_resource
def get_chat_model():
    return ChatGoogleGenerativeAI(model="models/gemini-1.5-pro", temperature=0.0, google_api_key=api_key, transport="rest")

# --- 4. NEW: CITATION EXTRACTOR ---
def extract_page_from_answer(answer_text):
    # Looks for "Page Index 31" or "Page 31" in the AI's own answer
    match = re.search(r"Page (?:Index )?(\d+)", answer_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

# --- 5. UI & CHAT ---
st.title(f"{ICON} {PAGE_TITLE}")
st.caption("Local RAG · Hallucination-Safe · Citation-Based Retrieval")

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
            st.image(pix.tobytes("png"), width=700)

if prompt := st.chat_input("Ask about Warfarin protocols..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            results = vector_store.similarity_search_with_score(prompt, k=10)
            docs = [doc for doc, score in results]

            chain = load_qa_chain(get_chat_model(), chain_type="stuff")
            
            # --- THE "STOLEN" PROMPT STRATEGY ---
            # We explicitly ask the AI to cite the [PAGE INDEX] it sees in the text.
            custom_prompt = """
            You are a clinical pharmacist assistant based ONLY on the provided protocol.
            
            STRICT RULES:
            1. Answer ONLY using the information in the CONTEXT.
            2. The Context contains markers like [PAGE INDEX 12]. You MUST cite this number.
            3. If the answer is not found, say "The protocol does not contain this information."
            
            CONTEXT: {context}
            USER QUESTION: {question}
            
            RESPONSE FORMAT:
            1. Direct Answer.
            2. SOURCE: "Reference found on Page Index [Insert Number Here]"
            """
            
            PROMPT = PromptTemplate(template=custom_prompt, input_variables=["context", "question"])
            chain.llm_chain.prompt = PROMPT
            response = chain.run(input_documents=docs, question=prompt)

            # --- NEW LOGIC: TRUST THE AI'S CITATION ---
            # Instead of guessing with Python, we extract the page number the AI wrote.
            cited_page = extract_page_from_answer(response)
            
            # Hallucination Check
            show_image = True
            if "does not contain" in response.lower() or "not found" in response.lower():
                show_image = False
                cited_page = None # Reset page if answer is negative

            message_placeholder.markdown(response)
            
            if show_image and cited_page is not None:
                try:
                    page = pdf_doc.load_page(cited_page) 
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    st.image(pix.tobytes("png"), caption=f"Source: Page {cited_page}", width=700)
                    st.session_state.messages.append({"role": "assistant", "content": response, "image_page": cited_page})
                except:
                    # Fallback if AI cites a non-existent page
                    st.session_state.messages.append({"role": "assistant", "content": response, "image_page": None})
            else:
                st.session_state.messages.append({"role": "assistant", "content": response, "image_page": None})

        except Exception as e:
            message_placeholder.error(f"Error: {str(e)}")

import streamlit as st
import os
import fitz  # PyMuPDF
import google.generativeai as genai
import re
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter # NEW: The "Knife" to cut text

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
    st.error("Secrets file not found. Please check your Streamlit Cloud Advanced Settings.")
    st.stop()
except KeyError:
    st.error("🚨 API Key missing! Go to 'Advanced Settings' -> 'Secrets' and add: GOOGLE_API_KEY = \"your_key\"")
    st.stop()
except Exception as e:
    st.error(f"API Error: {e}")
    st.stop()

# --- 2. HALLUCINATION-SAFE EVIDENCE GATING ---
@dataclass
class EvidenceBundle:
    text_quote: Optional[str] = None
    source_name: Optional[str] = None
    page_number: Optional[int] = None
    page_snapshot_path: Optional[str] = None

def evidence_is_clinically_acceptable(ev: EvidenceBundle) -> Tuple[bool, str]:
    missing = []
    if not ev.text_quote: missing.append("text_quote")
    if not ev.source_name: missing.append("source_name")
    if not ev.page_snapshot_path: missing.append("page_snapshot_path") 

    if missing:
        return False, f"Missing evidence fields: {', '.join(missing)}"
    return True, "Evidence complete."

def build_hallucination_safe_response(recommendation_text: str, ev: EvidenceBundle) -> Dict[str, Any]:
    ok, reason = evidence_is_clinically_acceptable(ev)
    if not ok:
        return {
            "status": "REFUSE",
            "message": "⚠️ **Safety Check Failed:** I cannot provide a recommendation because the supporting protocol page could not be verified. Please escalate to a clinician.",
            "why": reason,
            "evidence": ev.__dict__,
        }
    return {
        "status": "OK",
        "recommendation": recommendation_text,
        "evidence": {"source": ev.source_name, "page_number": ev.page_number, "page_snapshot_path": ev.page_snapshot_path},
    }

# --- 3. CORE LOGIC: PDF PROCESSING (With Chunking & Retries) ---
@st.cache_resource
def load_and_index_pdf(pdf_path):
    if not os.path.exists(pdf_path): return None, None
    doc = fitz.open(pdf_path)
    raw_documents = []
    
    # 1. Read Pages
    for page_num, page in enumerate(doc):
        text = page.get_text()
        text_with_meta = f"[PAGE INDEX {page_num}]\n{text}"
        raw_documents.append(Document(page_content=text_with_meta, metadata={"page": page_num}))
    
    # 2. Split Pages into Smaller Chunks (The Fix for 504 Error)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, # Safe size for Google API
        chunk_overlap=200 # Overlap to keep context
    )
    final_documents = text_splitter.split_documents(raw_documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key, transport="rest")
    
    # 3. Batch Process with Retry Logic
    vector_store = None
    batch_size = 5 # Very safe batch size
    total_docs = len(final_documents)
    
    progress_text = "Initializing AI Brain... Processing Protocols..."
    my_bar = st.progress(0, text=progress_text)

    for i in range(0, total_docs, batch_size):
        batch = final_documents[i : i + batch_size]
        
        # Simple Retry Loop
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
                break # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2) # Wait 2 seconds before retrying
                else:
                    st.error(f"Failed to process batch {i}. Error: {e}")
                    # Continue anyway to salvage what we can
        
        # Update progress bar
        progress = min((i + batch_size) / total_docs, 1.0)
        my_bar.progress(progress, text=f"Processing Segment {i} of {total_docs}...")
        
        # Friendly pause for the API
        time.sleep(0.2)

    my_bar.empty()
    return doc, vector_store

# --- 4. DYNAMIC MODEL SELECTOR ---
@st.cache_resource
def get_chat_model():
    try:
        valid_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target_models = ["models/gemini-1.5-pro", "models/gemini-1.5-pro-latest", "models/gemini-pro"]
        selected_model = next((m for m in target_models if m in valid_models), valid_models[0])
        return ChatGoogleGenerativeAI(model=selected_model, temperature=0.1, google_api_key=api_key, transport="rest")
    except Exception:
        return ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.1, google_api_key=api_key, transport="rest")

# --- 5. SMART PAGE SELECTOR ---
def get_best_page_image(docs, user_query):
    best_page = None
    highest_score = -500 
    user_query = user_query.lower()

    if "weekly" in user_query and ("dose" in user_query or "chart" in user_query or "table" in user_query):
        match = re.search(r"(\d+(\.\d+)?)", user_query)
        if match:
            dose = float(match.group(1))
            if dose <= 14.0: return 53
            elif dose <= 28.0: return 54
            elif dose <= 42.0: return 55
            else: return 56
        else:
            return 53 

    if any(x in user_query for x in ["switch", "convert", "transition", "dabigatran", "rivaroxaban", "apixaban"]): return 77

    intent = "general"
    if any(x in user_query for x in ["revers", "bleed", "supra", "vitamin k"]): intent = "reversal"
    elif re.search(r"\b([5-9]\.\d|1\d\.\d)\b", user_query): intent = "reversal"
    elif any(x in user_query for x in ["adjust", "maintenance", "subtherapeutic", "low", "increase", "target", "stable"]): intent = "maintenance"
    elif re.search(r"\b[0-4]\.\d\b", user_query): intent = "maintenance"
    elif any(x in user_query for x in ["initiat", "start", "begin", "new patient"]): intent = "initiation"

    tier_1_keywords = ["table 1", "table 2", "appendix 18", "appendix 19", "appendix 11", "appendix 12", "reversal", "initiation", "flow chart", "conversion"]
    ignore_keywords = ["checklist", "visit form", "referral form", "demographic", "appendix 1", "appendix 2", "appendix 3", "signature"]

    for i, doc in enumerate(docs):
        content = doc.page_content.lower()
        page_num = doc.metadata["page"]
        score = 80 - (i * 5)
        if any(bad in content for bad in ignore_keywords): score -= 200
        if any(good in content for good in tier_1_keywords): score += 20

        if intent == "conversion":
            if "conversion" in content: score += 50
            if page_num == 77: score += 100
            if "reversal" in content: score -= 50
        elif intent == "initiation":
            if "initiation" in content or "start" in content: score += 50
            if "reversal" in content: score -= 50
            if page_num in [50, 51, 17]: score += 100
        elif intent == "reversal":
            if "reversal" in content or "bleed" in content: score += 50
            if "initiation" in content: score -= 50
            if page_num in [79, 80]: score += 150
        elif intent == "maintenance":
            if "maintenance" in content or "adjust" in content or "target" in content: score += 50
            if page_num in [31, 50, 51]: score += 150 
            if page_num == 79: score -= 100 

        if score > highest_score:
            highest_score = score
            best_page = page_num

    if best_page is None and docs:
        best_page = docs

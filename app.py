import streamlit as st
import os
import fitz  # PyMuPDF
import google.generativeai as genai
import re
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

# --- MODERN IMPORTS (Fixes ModuleNotFoundError) ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# Updated paths for Python 3.13+ compatibility
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
    # Looks for the key in Streamlit Cloud Secrets
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
        return False, f

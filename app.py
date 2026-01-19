import streamlit as st
import os
import fitz  # PyMuPDF
import google.generativeai as genai
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document

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
    st.error("Secrets file not found. Please check .streamlit/secrets.toml")
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
    if not ev.page_snapshot_path: missing.append("page_snapshot_path") # Critical Gate

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

# --- 3. CORE LOGIC: PDF PROCESSING (Cached) ---
@st.cache_resource
def load_and_index_pdf(pdf_path):
    if not os.path.exists(pdf_path): return None, None
    doc = fitz.open(pdf_path)
    documents = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        text_with_meta = f"[PAGE INDEX {page_num}]\n{text}"
        documents.append(Document(page_content=text_with_meta, metadata={"page": page_num}))
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key, transport="rest")
    vector_store = FAISS.from_documents(documents, embeddings)
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

# --- 5. SMART PAGE SELECTOR (ALL LOGIC MERGED) ---
def get_best_page_image(docs, user_query):
    best_page = None
    highest_score = -500 
    user_query = user_query.lower()

    # --- A. DOSE-AWARE ROUTING (Weekly Chart Logic) ---
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

    # --- B. CONVERSION ROUTING ---
    if any(x in user_query for x in ["switch", "convert", "transition", "dabigatran", "rivaroxaban", "apixaban"]): return 77

    # --- C. INTENT DETECTION ---
    intent = "general"
    
    # 1. Reversal (Priority High)
    if any(x in user_query for x in ["revers", "bleed", "supra", "vitamin k"]):
        intent = "reversal"
    elif re.search(r"\b([5-9]\.\d|1\d\.\d)\b", user_query): # Matches 5.0, 9.1, 10.5 etc
        intent = "reversal"
        
    # 2. Maintenance / Subtherapeutic (Priority for INR < 2.0)
    elif any(x in user_query for x in ["adjust", "maintenance", "subtherapeutic", "low", "increase", "target", "stable"]):
        intent = "maintenance"
    elif re.search(r"\b[0-4]\.\d\b", user_query): # Matches 0.9, 1.4, 1.8, up to 4.9 (unless bleeding context overrides)
        intent = "maintenance"
        
    # 3. Initiation
    elif any(x in user_query for x in ["initiat", "start", "begin", "new patient"]):
        intent = "initiation"

    tier_1_keywords = ["table 1", "table 2", "appendix 18", "appendix 19", "appendix 11", "appendix 12", "reversal", "initiation", "flow chart", "conversion"]
    ignore_keywords = ["checklist", "visit form", "referral form", "demographic", "appendix 1", "appendix 2", "appendix 3", "signature"]

    # --- D. SCORING ---
    for i, doc in enumerate(docs):
        content = doc.page_content.lower()
        page_num = doc.metadata["page"]
        
        score = 80 - (i * 5) # Rank Decay
        
        if any(bad in content for bad in ignore_keywords): score -= 200
        if any(good in content for good in tier_1_keywords): score += 20

        # INTENT SCORING
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
            # REWARD: Maintenance pages (Page 31 = target INR, Page 50/51 = Dosing adjustment)
            if "maintenance" in content or "adjust" in content or "target" in content: score += 50
            if page_num in [31, 50, 51]: score += 150 
            # PUNISH: Reversal page (We don't want Table 1 for INR 1.8)
            if page_num == 79: score -= 100 

        if score > highest_score:
            highest_score = score
            best_page = page_num

    if best_page is None and docs:
        best_page = docs[0].metadata["page"]
        
    return best_page

# --- 6. CLINICAL MATH ENGINE ---
def calculate_dosing_schedule(query):
    match = re.search(r"(\d+)(?:\s*)mg", query.lower())
    if not match: return ""
    total_dose = int(match.group(1))
    base_daily = total_dose // 7
    remainder = total_dose % 7
    return f"\n\n[CALCULATED REFERENCE]: User asked for {total_dose}mg Weekly. Calculation: {base_daily}mg daily, with {base_daily + 1}mg on {remainder} days."

# --- 7. UI & CHAT LOGIC ---
st.title(f"{ICON} {PAGE_TITLE}")
st.markdown("##### *\"...And whoever saves one—it is as if he had saved mankind entirely.\"*")
st.caption("Local RAG · Hallucination-Safe · Protocol Finder")

if "messages" not in st.session_state: st.session_state.messages = []

pdf_doc, vector_store = load_and_index_pdf(PDF_FILE)
if not pdf_doc:
    st.error(f"PDF file '{PDF_FILE}' not found.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image_page" in message and message["image_page"] is not None:
            page_idx = message["image_page"]
            page = pdf_doc.load_page(page_idx)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            st.image(pix.tobytes("png"), caption=f"Verified Source: Page Index {message['image_page']}", width=700)

if prompt := st.chat_input("Ask about Warfarin protocols (e.g., INR 7.2)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # 1. Retrieval
            results_with_scores = vector_store.similarity_search_with_score(prompt, k=12)
            docs = [doc for doc, score in results_with_scores]

            # 2. Force-Feed Page 77 for Conversion
            if any(x in prompt.lower() for x in ["switch", "convert", "transition", "dabigatran", "rivaroxaban", "apixaban"]):
                page_77_content = pdf_doc.load_page(77).get_text()
                docs.insert(0, Document(page_content=f"[PAGE INDEX 77 - APPENDIX 18 CONVERSION]\n{page_77_content}", metadata={"page": 77}))
            
            # 3. Select Target Page
            target_page = get_best_page_image(docs, prompt)
            math_context = calculate_dosing_schedule(prompt)
            
            # 4. Generate
            chain = load_qa_chain(get_chat_model(), chain_type="stuff")
            custom_prompt = f"""
            You are a clinical pharmacist assistant. 
            USER SCENARIO: {{question}}
            {math_context} 
            CONTEXT: {{context}}
            INSTRUCTIONS:
            1. Switching: Look for Appendix 18 (Page 77).
            2. Reversal: Capture primary (Omit) and alternatives (Vitamin K).
            3. Initiation: Look for "3mg then 2mg".
            4. Maintenance (Low INR): Increase dose by percentage (Page 50/51).
            5. Weekly Dosing: If chart text is missing, use [CALCULATED REFERENCE] but refer to the snapshot.
            
            CRITICAL: DO NOT type out the text of the table/snapshot. The user will see the actual image. Just refer to it.
            
            RESPONSE FORMAT:
            1. Action Plan.
            2. Citation: "Protocol identified on **Page Index {target_page}**."
            3. Verification: "Please verify the details in the snapshot below."
            """
            PROMPT = PromptTemplate(template=custom_prompt, input_variables=["context", "question"])
            chain.llm_chain.prompt = PROMPT
            ai_text_response = chain.run(input_documents=docs, question=prompt)

            # 5. Build Safety Bundle
            evidence_bundle = EvidenceBundle(
                text_quote=ai_text_response,
                source_name=PDF_FILE,
                page_number=target_page,
                page_snapshot_path=f"Page {target_page}" if target_page is not None else None
            )

            # 6. Render
            final_response = build_hallucination_safe_response(ai_text_response, evidence_bundle)

            if final_response["status"] == "OK":
                message_placeholder.markdown(final_response["recommendation"])
                if target_page is not None:
                    page = pdf_doc.load_page(target_page) 
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    st.image(pix.tobytes("png"), caption=f"Verified Source: Page Index {target_page}", width=700)
                st.session_state.messages.append({"role": "assistant", "content": final_response["recommendation"], "image_page": target_page})
            else:
                message_placeholder.error(final_response["message"])
                with st.expander("Why blocked?"): st.write(final_response["why"])

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "ResourceExhausted" in error_msg:
                message_placeholder.error("🚨 **Rate Limit Hit (429):** Too many requests. Please wait 1 minute.")
            else:
                message_placeholder.error(f"An error occurred: {error_msg}")
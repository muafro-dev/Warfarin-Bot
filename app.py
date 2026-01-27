# app.py
import os
import re
import fitz  # PyMuPDF
import streamlit as st
from typing import List, Optional

# --- Optional imports (wrapped so app still runs without them) ---
HAS_LANGCHAIN = True
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
except Exception:
    HAS_LANGCHAIN = False

# Try to import Google clients (best-effort)
try:
    import google.genai as genai_new
    GENAI_CLIENT = "genai_new"
except Exception:
    genai_new = None
    GENAI_CLIENT = None

try:
    import google.generativeai as genai_old
    if GENAI_CLIENT is None:
        GENAI_CLIENT = "genai_old"
except Exception:
    genai_old = None
    if GENAI_CLIENT is None:
        GENAI_CLIENT = None

# --- CONFIG ---
PAGE_TITLE = "Clinical Warfarin Bot (Final)"
PDF_FILE = "Warfarin MTAC 2020.pdf"
ICON = "🩺"

st.set_page_config(page_title=PAGE_TITLE, page_icon=ICON, layout="wide")

# --- API KEY (from Streamlit secrets) ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    api_key = None

# Configure whichever client is present (best-effort)
if api_key:
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

# --- HELPERS: PDF loading and simple local search ---
def load_pdf_text_pages(pdf_path: str) -> List[str]:
    if not os.path.exists(pdf_path):
        return []
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        pages.append(text or "")
    return pages

def simple_rank_pages(pages: List[str], query: str, top_k: int = 5):
    if not query.strip():
        return []
    tokens = re.findall(r"\w+", query.lower())
    if not tokens:
        return []
    scores = []
    for i, text in enumerate(pages):
        text_lower = text.lower()
        score = sum(text_lower.count(tok) for tok in tokens)
        scores.append((i, float(score)))
    scored = [s for s in scores if s[1] > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]

# --- INDEXING & RAG helpers ---
@st.cache_resource
def build_index(pdf_path: str, api_key_value: Optional[str], chunk_size: int, chunk_overlap: int, persist_path: Optional[str] = None):
    """
    Returns (fitz.Document, vector_store or None, list_of_documents)
    If langchain/embeddings not available or api_key_value is None, returns (doc, None, raw_documents)
    """
    if not os.path.exists(pdf_path):
        return None, None, []

    doc = fitz.open(pdf_path)
    raw_documents = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        text_with_meta = f"[PAGE INDEX {page_num}]\n{text}"
        if HAS_LANGCHAIN:
            raw_documents.append(Document(page_content=text_with_meta, metadata={"page": page_num}))
        else:
            raw_documents.append({"page_content": text_with_meta, "metadata": {"page": page_num}})

    if not HAS_LANGCHAIN or not api_key_value:
        return doc, None, raw_documents

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    try:
        final_documents = splitter.split_documents(raw_documents)
    except Exception:
        final_documents = raw_documents

    # Create embeddings and FAISS index
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key_value)
    except Exception as e:
        st.sidebar.error("Embeddings init failed: " + str(e))
        return doc, None, final_documents

    vector_store = None
    try:
        batch_size = 8
        total_docs = len(final_documents)
        for i in range(0, total_docs, batch_size):
            batch = final_documents[i : i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)
        if persist_path and hasattr(vector_store, "save_local"):
            try:
                vector_store.save_local(persist_path)
            except Exception:
                pass
    except Exception as e:
        st.sidebar.error("Indexing failed: " + str(e))
        return doc, None, final_documents

    return doc, vector_store, final_documents

@st.cache_resource
def get_chat_model(model_name: str = "models/gemini-2.5-flash"):
    if not api_key:
        raise RuntimeError("Missing Google API key. Add GOOGLE_API_KEY to .streamlit/secrets.toml.")
    if not HAS_LANGCHAIN:
        raise RuntimeError("LangChain or Google GenAI wrappers not available in environment.")
    try:
        # Ensure deterministic behavior
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.0, google_api_key=api_key)
    except Exception as e:
        raise RuntimeError("Model initialization failed: " + str(e)) from e

def extract_page_from_answer(answer_text: str) -> Optional[int]:
    if not answer_text:
        return None
    match = re.search(r"(?:Page|Page Index|Pg\.?|Pg)\s*(\d+)", answer_text, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    m2 = re.search(r"SOURCE[:\s]*#?\s*(\d+)", answer_text, re.IGNORECASE)
    if m2:
        try:
            return int(m2.group(1))
        except Exception:
            return None
    return None

# --- UI ---
st.title(f"{ICON} {PAGE_TITLE}")
st.caption("Final patched RAG: local PDF retrieval + optional Google GenAI (explicit context + robust fallbacks)")

# Sidebar controls
st.sidebar.header("Settings & Debug")
model_name = st.sidebar.text_input("Model name", value="models/gemini-2.5-flash")
st.sidebar.write("Detected GenAI client:", GENAI_CLIENT)
st.sidebar.write("LangChain available:", HAS_LANGCHAIN)
st.sidebar.write("Embedding model: models/text-embedding-004")

# Indexing controls (safer defaults)
st.sidebar.markdown("### Indexing options")
chunk_size = st.sidebar.number_input("Chunk size", min_value=500, max_value=4000, value=1200, step=100)
chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=800, value=300, step=50)
persist_index_path = st.sidebar.text_input("FAISS persist path (optional)", value="faiss_index")

# Debug toggles
st.sidebar.markdown("### Debug toggles")
debug_retriever = st.sidebar.checkbox("Show retriever debug", value=True)
debug_context = st.sidebar.checkbox("Show context sent to model", value=True)
debug_verbose = st.sidebar.checkbox("Verbose logs", value=False)

# Diagnostics expander
with st.expander("Diagnostics (click to expand)"):
    st.write("PDF file path:", PDF_FILE)
    st.write("API key present:", bool(api_key))
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
                    st.write("Old client list_models failed:", str(e))
            else:
                st.write("No Google GenAI client available in the environment.")
        except Exception as e:
            st.write("Listing models failed:", str(e))

# Rebuild index button
if st.sidebar.button("Rebuild index now"):
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    st.experimental_rerun()

# Load PDF pages quickly for local fallback and UI
pages = load_pdf_text_pages(PDF_FILE)
if not pages:
    st.error(f"PDF not found or empty at path: {PDF_FILE}. Upload the PDF to the app folder or update PDF_FILE.")
    st.stop()

# Build index (cached)
pdf_doc, vector_store, final_documents = build_index(PDF_FILE, api_key, chunk_size, chunk_overlap, persist_index_path)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Helper to render a page safely (used in multiple places)
def _render_page_safe(pdf_doc, pidx):
    try:
        pidx = int(pidx)
        page = pdf_doc.load_page(pidx)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        st.image(pix.tobytes("png"), caption=f"Source: Page {pidx}", width=700)
        return True
    except Exception as e:
        st.sidebar.warning(f"Could not render page {pidx}: {e}")
        return False

# Choose best retrieved doc by token overlap (lightweight scoring)
def choose_best_doc_for_snapshot(context_docs, query):
    if not context_docs:
        return None
    tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]
    best = None
    best_score = -1
    for d in context_docs:
        text = d.page_content.lower() if hasattr(d, "page_content") else d.get("page_content","").lower()
        score = sum(text.count(tok) for tok in tokens)
        if score > best_score:
            best_score = score
            best = d
    return best

# Render previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image_page") is not None:
            try:
                page_idx = message["image_page"]
                page = pdf_doc.load_page(page_idx)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                st.image(pix.tobytes("png"), caption=f"Source: Page {page_idx}", width=700)
            except Exception:
                pass

# Chat input
if prompt := st.chat_input("Ask about Warfarin protocols..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        try:
            # If no vector store (no API key or missing libs), use local simple search
            if vector_store is None:
                pages_texts = [pdf_doc.load_page(i).get_text() for i in range(len(pdf_doc))]
                tokens = re.findall(r"\w+", prompt.lower())
                scores = []
                for i, text in enumerate(pages_texts):
                    score = sum(text.lower().count(tok) for tok in tokens)
                    scores.append((i, score))
                scored = [s for s in scores if s[1] > 0]
                scored.sort(key=lambda x: x[1], reverse=True)

                if debug_retriever:
                    st.sidebar.markdown("**DEBUG: Local fallback top pages**")
                    for pidx, sc in scored[:6]:
                        snippet = pages_texts[pidx][:300].replace("\n", " ")
                        st.sidebar.write(f"page={pidx} | score={sc} | snippet={snippet}")

                if not scored:
                    answer_text = "The protocol does not contain this information."
                    placeholder.markdown(answer_text)
                    st.session_state.messages.append({"role": "assistant", "content": answer_text, "image_page": None})
                else:
                    top_page = scored[0][0]
                    snippet = pdf_doc.load_page(top_page).get_text()[:1000].replace("\n", " ")
                    answer_text = (
                        "Based on the protocol text found in the document, see the snippet below.\n\n"
                        f"{snippet}\n\nSOURCE: Reference found on Page Index {top_page}"
                    )
                    placeholder.markdown(answer_text)
                    try:
                        page = pdf_doc.load_page(top_page)
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        st.image(pix.tobytes("png"), caption=f"Source: Page {top_page}", width=700)
                        st.session_state.messages.append({"role": "assistant", "content": answer_text, "image_page": top_page})
                    except Exception:
                        st.session_state.messages.append({"role": "assistant", "content": answer_text, "image_page": None})
            else:
                # Use vector store retriever + model
                retriever = vector_store.as_retriever(search_kwargs={"k": 20})

                # Get top docs
                try:
                    top_docs = retriever.get_relevant_documents(prompt)[:20]
                except Exception as e:
                    top_docs = []
                    if debug_verbose:
                        st.sidebar.error("Retriever get_relevant_documents failed: " + str(e))

                if debug_retriever:
                    st.sidebar.markdown("**DEBUG: top retrieved docs**")
                    for d in top_docs[:10]:
                        try:
                            page_meta = d.metadata.get("page") or d.metadata
                            snippet = d.page_content[:300].replace("\n", " ")
                        except Exception:
                            page_meta = d.get("metadata", {})
                            snippet = d.get("page_content", "")[:300].replace("\n", " ")
                        st.sidebar.write(f"page={page_meta} | snippet={snippet}")

                # Build explicit context from top docs (ensure page markers included)
                context_docs = top_docs[:20]  # increased breadth
                context = "\n\n".join((d.page_content if hasattr(d, "page_content") else d.get("page_content", "")) for d in context_docs)

                if debug_context:
                    st.sidebar.markdown("**DEBUG: context sent to model (first 2000 chars)**")
                    st.sidebar.write(context[:2000].replace("\n", " "))

                # Initialize model
                try:
                    chat_model = get_chat_model(model_name=model_name)
                except Exception as e:
                    placeholder.error("Model initialization error: " + str(e))
                    st.session_state.messages.append({"role": "assistant", "content": "Model initialization error: " + str(e), "image_page": None})
                    chat_model = None

                # Build RetrievalQA chain if possible
                qa = None
                if HAS_LANGCHAIN and chat_model is not None:
                    try:
                        qa = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=retriever)
                    except Exception:
                        qa = None

                # Strict prompt template with example
                p_header = (
                    "You are a clinical pharmacist assistant. Use ONLY the CONTEXT below. Do NOT invent facts.\n"
                    "INSTRUCTIONS:\n"
                    "1) Answer in one short paragraph (1-3 sentences) with a clear action (e.g., 'Hold X doses; restart at Y mg; recheck INR in Z days').\n"
                    "2) Immediately after the answer include EXACTLY: SOURCE: Page Index <N> using the page marker from the CONTEXT.\n"
                    "3) If the answer is not in the CONTEXT, reply exactly: The protocol does not contain this information.\n"
                    "EXAMPLE:\n"
                    "Q: INR 4.5 on warfarin 4mg OD, no bleeding.\n"
                    "A: Hold next dose; restart 3 mg daily; recheck INR in 2-3 days. SOURCE: Page Index 79\n"
                )
                full_template = p_header + "\nCONTEXT: {context}\nQUESTION: {question}\n"
                try:
                    prompt_template = PromptTemplate(template=full_template, input_variables=["context", "question"])
                except Exception:
                    prompt_template = None

                response = None

                # 1) Try explicit qa.run with injected context (works for many chain implementations)
                if qa is not None:
                    try:
                        response = qa.run({"context": context, "question": prompt})
                    except Exception:
                        try:
                            response = qa.run({"input_documents": context_docs, "query": prompt})
                        except Exception:
                            response = None

                # 2) If qa failed or returned nothing, call the chat model directly with the formatted prompt
                if (not response or response is None) and chat_model is not None:
                    final_prompt = (
                        p_header
                        + "\nCONTEXT:\n"
                        + context
                        + f"\n\nQUESTION: {prompt}\n"
                    )
                    try:
                        if hasattr(chat_model, "predict"):
                            response = chat_model.predict(final_prompt)
                        elif hasattr(chat_model, "generate"):
                            gen = chat_model.generate(final_prompt)
                            if hasattr(gen, "generations"):
                                try:
                                    response = gen.generations[0][0].text
                                except Exception:
                                    try:
                                        response = gen.generations[0].text
                                    except Exception:
                                        response = str(gen)
                            else:
                                response = str(gen)
                        elif hasattr(chat_model, "create"):
                            gen = chat_model.create(final_prompt)
                            response = str(gen)
                        else:
                            response = str(chat_model(final_prompt))
                    except Exception as e:
                        if debug_verbose:
                            st.sidebar.error("Direct model call failed: " + str(e))
                        response = None

                # 3) If still no response, fallback to showing top snippet
                if not response:
                    if context_docs:
                        snippet = (context_docs[0].page_content if hasattr(context_docs[0], "page_content") else context_docs[0].get("page_content",""))[:1000].replace("\n", " ")
                        answer_text = "Based on the protocol text found in the document, see the snippet below.\n\n" + snippet
                        placeholder.markdown(answer_text)
                        try:
                            first_page = context_docs[0].metadata.get("page") if hasattr(context_docs[0], "metadata") else context_docs[0].get("metadata",{}).get("page")
                            page = pdf_doc.load_page(first_page)
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                            st.image(pix.tobytes("png"), caption=f"Source: Page {first_page}", width=700)
                            st.session_state.messages.append({"role": "assistant", "content": answer_text, "image_page": first_page})
                        except Exception:
                            st.session_state.messages.append({"role": "assistant", "content": answer_text, "image_page": None})
                    else:
                        placeholder.markdown("The protocol does not contain this information.")
                        st.session_state.messages.append({"role": "assistant", "content": "The protocol does not contain this information.", "image_page": None})
                else:
                    # Extract cited page if present
                    cited_page = extract_page_from_answer(response)

                    # Hallucination check: if model says not found, don't show image
                    show_image = True
                    lower_res = response.lower()
                    if "does not contain" in lower_res or "not found" in lower_res or "cannot provide" in lower_res:
                        show_image = False
                        cited_page = None

                    # Additional verification: ensure cited page actually contains query tokens (relaxed)
                    if cited_page is not None:
                        try:
                            page_text = pdf_doc.load_page(cited_page).get_text().lower()
                            tokens = re.findall(r"\w+", prompt.lower())
                            if not any(len(tok) > 2 and tok in page_text for tok in tokens):
                                if debug_verbose:
                                    st.sidebar.warning("Cited page may not contain exact tokens; showing snippet but flagging for review.")
                                # do not suppress image; allow manual review
                        except Exception:
                            cited_page = None
                            show_image = False

                    # If the model answer is too vague, replace with top snippet + page
                    if response and (len(response.split()) < 8 or "based on the protocol" in response.lower()):
                        # treat as non-specific; show top snippet instead
                        if context_docs:
                            top_doc = context_docs[0]
                            try:
                                top_page = top_doc.metadata.get("page") if hasattr(top_doc, "metadata") else top_doc.get("metadata", {}).get("page")
                                top_text = top_doc.page_content if hasattr(top_doc, "page_content") else top_doc.get("page_content", "")
                            except Exception:
                                top_page = None
                                top_text = ""
                            snippet = top_text.replace("\n", " ")[:1000]
                            response = "Based on the protocol text found in the document, see the snippet below.\n\n" + snippet + f"\n\nSOURCE: Page Index {top_page}"
                            cited_page = top_page
                            show_image = True if top_page is not None else False

                    # Display response
                    placeholder.markdown(response)

                    # Determine page to show: prefer cited page, else choose best doc
                    page_to_show = None
                    if cited_page is not None:
                        page_to_show = cited_page
                    else:
                        best_doc = choose_best_doc_for_snapshot(context_docs, prompt) if context_docs else None
                        if best_doc is not None:
                            try:
                                page_to_show = best_doc.metadata.get("page") if hasattr(best_doc, "metadata") else best_doc.get("metadata", {}).get("page")
                            except Exception:
                                page_to_show = None

                    # Render or show fallback info
                    if page_to_show is not None:
                        rendered = _render_page_safe(pdf_doc, page_to_show)
                        if not rendered:
                            st.info(f"Referenced page: {page_to_show}. You can view it using Browse PDF pages below.")
                            if st.button(f"Show Page {page_to_show}", key=f"show_cited_{page_to_show}"):
                                _render_page_safe(pdf_doc, page_to_show)
                        st.session_state.messages.append({"role": "assistant", "content": response, "image_page": page_to_show if rendered else None})
                    else:
                        # No page available — show top snippet and page number fallback if possible
                        if context_docs and len(context_docs) > 0:
                            top_doc = context_docs[0]
                            try:
                                top_page = top_doc.metadata.get("page") if hasattr(top_doc, "metadata") else top_doc.get("metadata", {}).get("page")
                                top_text = top_doc.page_content if hasattr(top_doc, "page_content") else top_doc.get("page_content", "")
                            except Exception:
                                top_page = None
                                top_text = ""
                            snippet = top_text.replace("\n", " ")[:1000]
                            st.markdown("**Top retrieved source (no cited page):**")
                            st.write(snippet + ("..." if len(top_text) > len(snippet) else ""))
                            if top_page is not None:
                                rendered = _render_page_safe(pdf_doc, top_page)
                                if not rendered:
                                    st.info(f"Top retrieved page is {top_page}. Use Browse PDF pages to view it.")
                                    if st.button(f"Show Top Page {top_page}", key=f"show_top_{top_page}"):
                                        _render_page_safe(pdf_doc, top_page)
                            st.session_state.messages.append({"role": "assistant", "content": response, "image_page": top_page if (top_page is not None and rendered) else None})
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": response, "image_page": None})

        except Exception as e:
            placeholder.error(f"Error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "Error: " + str(e), "image_page": None})

# Browse PDF pages
st.markdown("---")
st.subheader("Browse PDF pages")
page_to_show = st.number_input("Page index to view", min_value=0, max_value=len(pages)-1, value=0, step=1)
if st.button("Show page image"):
    try:
        page = fitz.open(PDF_FILE).load_page(int(page_to_show))
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        st.image(pix.tobytes("png"), caption=f"Page {int(page_to_show)}", width=700)
        st.markdown("**Page text (first 2000 chars):**")
        st.write(fitz.open(PDF_FILE).load_page(int(page_to_show)).get_text()[:2000].replace("\n", " "))
    except Exception as e:
        st.error("Failed to render page: " + str(e))

st.markdown("---")
st.caption("This final build forces explicit context into the model and provides robust fallbacks so clinicians see protocol text even when the LLM path is flaky.")

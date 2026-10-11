# =================================================
# 📚 AI Library & Course Assistant – MODERN UI REDESIGN
# Uses: tableofcontent.xlsx (Author, Title, TOC, Category)
# Models: Ollama (gemma2)
# =================================================

import os
import re
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import os
import re
import json
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

# LangChain Imports
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder # Keeping this for reranking


# -------------------------------------------------
# 🧩 Load models (Embeddings, Reranker)
# -------------------------------------------------
# -------------------------------------------------
# 🧩 Load Resources (LangChain + Data)
# -------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_resources():
    # 1. Embeddings & Reranker
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # 2. LLM (ChatOllama via LangChain)
    llm = ChatOllama(model="gemma2", temperature=0.3)
    
    # 3. Vector Store
    if not os.path.exists("faiss_index_langchain"):
        st.error("❌ 'faiss_index_langchain' not found. Run `python langchain_indexing.py` first.")
        st.stop()
    
    try:
        vectorstore = FAISS.load_local(
            "faiss_index_langchain", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        st.stop()
        
    return vectorstore, cross_encoder, llm

@st.cache_data(show_spinner=True)
def load_data():
    if not os.path.exists("tableofcontent_final_merged.xlsx"):
        st.error("❌ 'tableofcontent_final_merged.xlsx' not found.")
        st.stop()

    df = pd.read_excel("tableofcontent_final_merged.xlsx")
    df.columns = [c.strip() for c in df.columns]

    # Map columns
    col_map = {}
    for col in df.columns:
        k = col.strip().lower()
        if "author" in k: col_map[col] = "Author"
        elif "title" in k: col_map[col] = "Title"
        elif k in ("table of contents", "toc", "table_of_contents"): col_map[col] = "Table of Contents"

    df = df.rename(columns=col_map)
    
    # Fill NAs
    df["Author"] = df["Author"].astype(str).fillna("Unknown")
    df["Title"] = df["Title"].astype(str).fillna("").str.strip()
    return df

# Initialize resources
vectorstore, cross_encoder, llm = load_resources()
df = load_data()


# -------------------------------------------------
# 🎯 Load exact QA dataset
# -------------------------------------------------
exact_qa_dict = {}
if os.path.exists("diverse_qa_pairs.json"):
    with open("diverse_qa_pairs.json", "r", encoding="utf-8") as f:
        qa_50_data = json.load(f)
    exact_qa_dict = {
        entry.get("question", "").strip().lower(): entry for entry in qa_50_data
        if isinstance(entry.get("question", ""), str)
    }


# -------------------------------------------------
# 🤖 LLM helpers: Ollama + Gemini 
# -------------------------------------------------
# -------------------------------------------------
# 🤖 RAG Logic (LangChain Retrieval + Cross-Encoder Reranking)
# -------------------------------------------------
def rag_query(query: str) -> dict:
    """
    1. Retrieve via LangChain FAISS
    2. Rerank via CrossEncoder
    3. Generate via LangChain ChatOllama
    """
    # 1. Retrieve (LangChain)
    #    LangChain returns Document objects. We need their page_content.
    docs = vectorstore.similarity_search(query, k=25)
    candidates = [d.page_content for d in docs]

    if not candidates:
        # Fallback
        res = llm.invoke(f"Question: {query}")
        return {
            "query": query, 
            "finetuned_answer": res.content, 
            "context": "", 
            "related_topics": []
        }
    
    # 2. Rerank (Keep existing high-quality reranking)
    pairs = [[query, c] for c in candidates]
    scores = cross_encoder.predict(pairs)
    
    # Sort and take top 5
    scored_docs = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_docs = [doc_text for score, doc_text in scored_docs[:5]]
    
    # 3. Context Construction
    context_text = "\n\n---\n\n".join(top_docs)
    
    system_instruction = (
        "You are a helpful academic assistant. Answer the user's question using the provided context. "
        "CRITICAL: If the user provides a snippet of text (like a Table of Contents) and asks to identify the book/author, "
        "YOU MUST ONLY USE THE PROVIDED CONTEXT. Do not hallucinate or use outside knowledge for identification. "
        "If the context contains the matching TOC entries, state the Book Title and Author from that chunk exactly. "
        "If the context does not contain the answer, say 'I cannot find this in the library.' "
        "For general questions, you may still use general knowledge."
    )
    
    prompt_str = (
        f"{system_instruction}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    
    # 4. Generate Answer (LangChain)
    try:
        response = llm.invoke(prompt_str)
        answer = response.content
    except Exception as e:
        answer = f"Error generating answer: {e}"
        
    return {
        "query": query,
        "finetuned_answer": answer,
        "context": context_text,
        "related_topics": []
    }
# -------------------------------------------------
# 🧱 Sidebar UI (Accordion-style, no default open)
# -------------------------------------------------
CATEGORY_KEYWORDS = {
    "Literature": [
        "epic", "poetry", "novel", "drama",
        "myth", "mahabharata", "ramayana"
    ],
    "Computer Science": [
        "programming", "algorithms",
        "data structures", "operating systems",
        "computer networks"
    ],
    "Cyber Security": [
        "cryptography", "malware",
        "hacking", "network security",
        "forensics"
    ],
    "Artificial Intelligence": [
        "machine learning", "deep learning",
        "neural networks", "ai", "nlp"
    ],
    "Data Science": [
        "data mining", "statistics",
        "analytics", "big data"
    ],
    "Engineering": [
        "mechanical", "electrical",
        "electronics", "civil"
    ],
    "Management": [
        "business", "finance",
        "marketing", "economics"
    ],
    "Law": [
        "constitution", "jurisprudence",
        "legal systems", "rights"
    ],
    "Science": [
        "physics", "chemistry",
        "biology", "mathematics"
    ],
}

with st.sidebar:
    # Initialize theme in session state if missing
    if 'theme' not in st.session_state:
        st.session_state['theme'] = 'Dark'
    st.markdown("### 🧠 Model Selection")
    st.radio("Choose LLM", ["Ollama (gemma2)"], label_visibility="collapsed")

    st.markdown("### 🎓 Disciplines & Categories")
    st.markdown("<p class='small-muted'>Browse academic disciplines</p>", unsafe_allow_html=True)

    for category, keywords in CATEGORY_KEYWORDS.items():
        with st.expander(category, expanded=False):
            for kw in keywords:
                st.markdown(f"<div class='sidebar-keyword'>• {kw}</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin: 24px 0; border: none; border-top: 1px solid rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
    st.markdown("### 🌙 Theme")
    # Bind the radio directly to `st.session_state['theme']` to keep widget state and theme in sync
    st.radio("Choose theme", ["Dark", "Light"], index=0 if st.session_state.theme == 'Dark' else 1, key='theme')
    # Compute theme variables based on current selection, then inject CSS
    theme_choice = st.session_state.get('theme', 'Dark')
    if theme_choice == "Dark":
        bg = "#212529"
        text = "#e9ecef"
        muted = "#adb5bd"
        sidebar_bg = "#343a40"
        button_bg = "#495057"
        header_bg = "#1a1d21"
        header_text = "#e9ecef"
        card_bg = "rgba(255,255,255,0.02)"
        card_border = "rgba(255,255,255,0.04)"
        card_shadow = "0 18px 60px rgba(0,0,0,0.65)"
        input_bg = "#2b3035" # Solid dark color for better visibility
    else:
        bg = "#ffffff"
        text = "#111111"
        muted = "#444444"
        sidebar_bg = "#f6f6f6"
        button_bg = "#e9e9e9"
        header_bg = "#e9ecef"
        header_text = "#111111"
        card_bg = "#f8f9fa"
        card_border = "#e9ecef"
        card_shadow = "0 12px 30px rgba(0,0,0,0.08)"
        input_bg = "#f1f3f5"

    css = f"""
    <style>
    :root {{
      --bg: {bg};
      --text: {text};
      --muted: {muted};
      --sidebar-bg: {sidebar_bg};
      --btn-bg: {button_bg};
      --header-bg: {header_bg};
      --header-text: {header_text};
      --card-bg: {card_bg};
      --card-border: {card_border};
      --card-shadow: {card_shadow};
      --input-bg: {input_bg};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(
    """
<style>
/* ============================================
   GLOBAL STYLES - Theme-aware
   ============================================ */

/* Import modern font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Base application styling uses theme variables */
html, body, [class*="css"], .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Remove Streamlit branding (keep visible) */
#MainMenu {visibility: visible;}
header {visibility: visible;}
footer {visibility: hidden;}

/* ============================================
   SIDEBAR REDESIGN - Theme-aware Navigation Panel
   ============================================ */

/* Sidebar container uses theme variable */
[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid rgba(0,0,0,0.06) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.05) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--text) !important;
}

/* Sidebar headings */
[data-testid="stSidebar"] h3 {
    color: var(--text) !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    margin-top: 24px !important;
    margin-bottom: 12px !important;
    padding-bottom: 8px !important;
    border-bottom: 2px solid rgba(0,0,0,0.06) !important;
}

/* Radio buttons styling */
[data-testid="stSidebar"] .stRadio {
    background: rgba(0, 0, 0, 0.03) !important;
    padding: 12px !important;
    border-radius: 12px !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stSidebar"] .stRadio:hover {
    background: rgba(0, 0, 0, 0.06) !important;
}

[data-testid="stSidebar"] .stRadio label {
    color: var(--text) !important;
    font-weight: 500 !important;
}

/* Expander (accordion) styling */
[data-testid="stSidebar"] .streamlit-expanderHeader {
    background: rgba(0, 0, 0, 0.03) !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    color: var(--text) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    margin-bottom: 8px !important;
}

[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
    transform: translateX(4px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
}

[data-testid="stSidebar"] .streamlit-expanderContent {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(0,0,0,0.04) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 8px 12px !important;
    animation: slideDown 0.3s ease-out !important;
}

@keyframes slideDown {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Sidebar keyword items */
.sidebar-keyword {
    padding: 6px 14px !important;
    font-size: 13px !important;
    color: var(--muted) !important;
    line-height: 1.8 !important;
    transition: all 0.2s ease !important;
    border-radius: 6px !important;
    cursor: default !important;
}

.sidebar-keyword:hover {
    color: var(--text) !important;
    background: rgba(0,0,0,0.03) !important;
    padding-left: 20px !important;
}

/* ============================================
   MAIN HEADER SECTION
   ============================================ */

/* Main title styling */
.stApp h1 {
    color: var(--text) !important;
    font-weight: 700 !important;
    font-size: 42px !important;
    letter-spacing: -0.5px !important;
    margin-bottom: 8px !important;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
}

/* Subtitle/description text */
.stApp p { color: var(--muted) !important; line-height: 1.6 !important; }

/* Top-right controls (Deploy, menu dots, sidebar toggle) */
header, header[data-testid="stHeader"] {
    background-color: var(--bg) !important; /* Match body background or use --header-bg */
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
}

header button, header [data-testid="stToolbar"] button {
    color: var(--text) !important;
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

header svg {
    fill: var(--text) !important;
}

/* ============================================
   INPUT FIELD - Search Box Redesign
   ============================================ */

/* Text input container */
.stTextInput > div > div {
    background: transparent !important;
}

/* Input field itself (theme-aware) */
.stTextInput input, .stTextInput > div > div > input {
    background-color: var(--input-bg) !important;
    color: var(--text) !important;
    caret-color: var(--text) !important; /* Ensure cursor is visible */
    border: 1px solid var(--card-border) !important;
    border-radius: 14px !important;
    padding: 14px 20px !important;
    font-size: 15px !important;
    font-weight: 400 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06) inset !important;
}

.stTextInput input:focus {
    background: var(--input-bg) !important;
    border-color: rgba(0,0,0,0.18) !important;
    box-shadow: 0 10px 28px rgba(0,0,0,0.12) !important;
    outline: none !important;
}

.stTextInput input::placeholder {
    color: var(--muted) !important;
}

/* ============================================
   BUTTONS - Modern Minimal Design
   ============================================ */

/* Primary buttons */
div.stButton > button {
    background: linear-gradient(135deg, #ffffff 0%, #e5e5e5 100%) !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.3px !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #f5f5f5 0%, #d5d5d5 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.6) !important;
}

div.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
}

/* Disabled buttons */
div.stButton > button:disabled {
    background: rgba(255, 255, 255, 0.1) !important;
    color: #505050 !important;
    cursor: not-allowed !important;
    opacity: 0.5 !important;
}

/* Download button styling */
div.stDownloadButton > button {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

div.stDownloadButton > button:hover {
    background: rgba(255, 255, 255, 0.12) !important;
    border-color: #404040 !important;
    transform: translateY(-2px) !important;
}

/* ============================================
   CONTENT CARDS - Glassmorphism Effect
   ============================================ */

/* Answer box */

.answer-box {
    background: var(--card-bg) !important;
    backdrop-filter: blur(14px) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 16px !important;
    padding: 22px 26px !important;
    margin: 18px 0 !important;
    box-shadow: var(--card-shadow) !important;
    color: var(--text) !important;
    line-height: 1.85 !important;
    font-size: 15px !important;
    transition: all 0.28s ease !important;
    animation: fadeIn 0.45s ease-out !important;
}

.answer-box:hover {
    border-color: rgba(255,255,255,0.12) !important;
    box-shadow: 0 20px 50px rgba(0,0,0,0.45) !important;
    transform: translateY(-3px) !important;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Book cards */
.book-card {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    animation: slideIn 0.5s ease-out !important;
}

.book-card:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
    transform: translateY(-4px) !important;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6) !important;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateX(-20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

""",
    unsafe_allow_html=True,
)


# -------------------------------------------------
# 🏠 Main layout
# -------------------------------------------------
st.markdown(
    """
<h1 style="text-align:center; margin-bottom:0;">
    📚 AI Library & Course Assistant
</h1>
<p style="text-align:center; font-size:15px; margin-top:8px; margin-bottom:32px;">
    Ask doubts, get explanations, and discover relevant books across Engineering, Law, Science, Management, and more.
</p>
""",
    unsafe_allow_html=True,
)

query = st.text_input(
    "🔎 Ask your doubt",
    placeholder="e.g., \"Suggest books on Machine Learning for beginners\"",
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_topics" not in st.session_state:
    st.session_state.last_topics = []
if "book_page" not in st.session_state:
    st.session_state.book_page = 0

if query:
    with st.spinner("Thinking with RAG + LLM..."):
        result = rag_query(query)
        st.session_state.chat_history.append(result)
        st.session_state.last_topics = result["related_topics"]
        st.session_state.book_page = 0

    st.markdown("<h3>💡 Answer</h3>", unsafe_allow_html=True)


    if query.strip().lower() in exact_qa_dict:
        st.markdown("**Reference Answer (ground truth):**")
        st.markdown(
            f"<div class='answer-box'>{exact_qa_dict[query.strip().lower()]['answer']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Generated Answer:**")

    st.markdown(
        f"<div class='answer-box'>{result['finetuned_answer']}</div>",
        unsafe_allow_html=True,
    )
    


    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
        if st.button("🔁 Regenerate"):
            with st.spinner("Regenerating..."):
                regen = rag_query(query)
                st.session_state.chat_history[-1] = regen
                st.session_state.last_topics = regen["related_topics"]
                st.rerun()

        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
        # Copy to clipboard button using JavaScript
        # Escape the content for JS and HTML attribute usage
        js_answer = json.dumps(result['finetuned_answer']).replace('"', '&quot;')
        st.markdown(
            f"""
            <button onclick="navigator.clipboard.writeText({js_answer}).then(() => alert('Answer copied to clipboard!')).catch(err => alert('Failed to copy'));" 
            style="background: rgba(0,0,0,0.05); color: var(--text); border: 1px solid rgba(0,0,0,0.1); border-radius: 12px; padding: 10px 24px; font-weight: 600; cursor: pointer; font-size: 14px; transition: all 0.3s ease;">
            📋 Copy answer
            </button>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------
    # 📚 Suggested Books (5 cards)
    # -------------------------------------------------
    st.markdown("<h3 style='color:#3F4749;'>📚 Suggested Books</h3>", unsafe_allow_html=True)

    matched_titles = []
    for line in result["context"].split("\n"):
        if "Book Title:" in line:
            # Extract title after "Book Title:"
            t = line.split("Book Title:", 1)[1].strip()
            matched_titles.append(t)

    matched_books = (
        df[df["Title"].isin(matched_titles)]
        .drop_duplicates(subset="Title")
        .reset_index(drop=True)
    )

    if not matched_books.empty:
        BOOKS_PER_PAGE = 5
        total_books = len(matched_books)
        total_pages = (total_books + BOOKS_PER_PAGE - 1) // BOOKS_PER_PAGE
        page = st.session_state.book_page

        start_idx = page * BOOKS_PER_PAGE
        end_idx = min(start_idx + BOOKS_PER_PAGE, total_books)

        for idx, row in matched_books.iloc[start_idx:end_idx].iterrows():
            toc_text = row.get("TOC", "") or ""
            course_tag = row.get("Category", "General")
            author = row.get("Author", "Unknown")

            # Initialize session state for this book if not present
            book_id = f"book_{page}_{idx}"
            if "summaries" not in st.session_state:
                st.session_state.summaries = {}
            if f"show_toc_{book_id}" not in st.session_state:
                st.session_state[f"show_toc_{book_id}"] = False

            with st.container():
                st.markdown(
                    f"""
                <div class="book-card">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div>
                            <div style="font-weight:600; font-size:15px;">{row['Title']}</div>
                            <div style="font-size:12px; color:#9ca3af;">Author: {author}</div>
                            <div style="font-size:11px; margin-top:4px;">
                                <span class="tag-chip">{course_tag}</span>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                
                has_toc = isinstance(toc_text, str) and len(toc_text.strip()) > 5

                # Render buttons first
                # Only "Show Summary" button remains
                
                # Check if we already have the summary
                summary_key = f"{page}_{idx}"
                has_summary = summary_key in st.session_state.summaries
                
                if st.button(
                    "📘 Show Summary" if not has_summary else "📘 Refresh Summary",
                    key=f"btn_sum_{book_id}",
                    use_container_width=True
                ):
                    with st.spinner("Generating summary..."):
                        if has_toc:
                            summary_prompt = (
                                "You are summarizing a textbook based on its table of contents. "
                                "Write 2–3 sentences describing what this book covers and who it is useful for.\n\n"
                                f"Book: {row['Title']} by {author}\n"
                                f"Table of Contents:\n{toc_text}\n\nSummary:"
                            )
                        else:
                            summary_prompt = (
                                f"Provide a brief 2-3 sentence summary of the book '{row['Title']}' by {author}. "
                                "Describe its main topics and target audience."
                            )
                        
                        try:
                            # Use LangChain LLM to generate summary
                            res = llm.invoke(summary_prompt)
                            summary = res.content
                        except Exception as e:
                            summary = f"Error generating summary: {e}"

                        st.session_state.summaries[summary_key] = summary
                        st.rerun()

                # Render Content Area below buttons
                
                # Show Summary if available
                if summary_key in st.session_state.summaries:
                     st.markdown(
                        f"<div class='answer-box' style='margin-top:10px;'><strong>📝 Summary:</strong> {st.session_state.summaries[summary_key]}</div>",
                        unsafe_allow_html=True,
                    )

        c_prev, c_mid, c_next = st.columns([1, 2, 1])
        with c_prev:
            if page > 0:
                if st.button("⬅️ Previous page"):
                    st.session_state.book_page -= 1
                    st.rerun()

        with c_next:
            if page < total_pages - 1:
                if st.button("Next page ➡️"):
                    st.session_state.book_page += 1
                    st.rerun()

        with c_mid:
            st.markdown(
                f"<p style='text-align:center; color:#9ca3af;'>Page {page+1} of {total_pages}</p>",
                unsafe_allow_html=True,
            )
    else:
        st.warning("No matching books found from the retrieved chunks.")

    if st.session_state.last_topics:
        st.markdown("<h3 style='color:#3F4749;'>🔖 Related Topics</h3>", unsafe_allow_html=True)
        st.markdown(
            " ".join(
                f"<span class='tag-chip'>{tag}</span>"
                for tag in st.session_state.last_topics
            ),
            unsafe_allow_html=True,
        )

# -------------------------------------------------
# 💬 Chat history
# -------------------------------------------------
if st.session_state.chat_history:
    with st.expander("📜 Chat History"):
        st.markdown('<span style="color:#B2F2F2; font-weight:600;">📜 Chat History</span>', unsafe_allow_html=True)
        for item in reversed(st.session_state.chat_history):
            st.markdown(
                f"""
            <div class="chat-history-item">
                <strong>Q:</strong> {item['query']}<br>
                <strong>A:</strong> {item['finetuned_answer'][:160]}...
            </div>
            """,
                unsafe_allow_html=True,
            )
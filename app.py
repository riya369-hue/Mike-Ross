"""
app.py
------
Streamlit UI for the RBI RAG QA system.

Run: streamlit run app.py
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.rag_pipeline import answer_query

st.set_page_config(
    page_title="Q & A",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    
    :root {
        --rbi-green: #006B3F;
        --rbi-gold:  #C9A84C;
        --bg-card:   #F7F9F7;
        --border:    #D8E4D8;
    }
    
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    .main-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: var(--rbi-green);
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    
    .answer-box {
        background: #1a1a2e;
        border-left: 4px solid var(--rbi-green);
        border-radius: 0 8px 8px 0;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #e8f5e8 !important;
    }
    
    .idk-box {
        background: #2a1a00;
        border-left: 4px solid #E07A00;
        border-radius: 0 8px 8px 0;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        color: #ffd580 !important;
        font-size: 0.95rem;
    }
    
    .source-card {
    background: white;
    color: #111111 !important;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.83rem;
}
    
    .confidence-badge {
        display: inline-block;
        background: var(--rbi-green);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
    }
            
    .source-card strong {
        color: #000000 !important;
    }

    .source-card code {
        color: #006B3F !important;
    }
    
    .low-confidence-badge {
        background: #E07A00;
    }
    
    .chunk-text {
        color: #444;
        font-size: 0.8rem;
        line-height: 1.5;
        margin-top: 0.4rem;
        font-style: italic;
        border-top: 1px solid var(--border);
        padding-top: 0.4rem;
    }

    .stTextInput > div > div > input {
        font-family: 'IBM Plex Sans', sans-serif;
        border: 2px solid var(--border);
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--rbi-green);
        box-shadow: 0 0 0 3px rgba(0, 107, 63, 0.1);
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🏦 About this system")
    st.markdown("""
    This RAG (Retrieval-Augmented Generation) system answers questions 
    grounded in **Documents**.
    
    **How it works:**
    1. Your question is embedded into a vector
    2. Most similar document chunks are retrieved
    3. Gemini generates an answer using *only* those chunks
    4. If confidence is low → system says **"I don't know"**
    
    ---
    
    **Design choices:**
    - Embeddings: `all-MiniLM-L6-v2`
    - Vector DB: ChromaDB (cosine similarity)
    - Chunk size: 512 tokens, 50 overlap
    - Confidence threshold: 0.4
    - LLM: Gemini 1.5 Flash
    """)
    
    st.markdown("---")
    st.markdown("**Sample questions to try:**")
    
    sample_qs = [
       " Summarize the Right to Equality.",
        "What is the difference between Article 21 and Article 21A?",
        "Which Fundamental Right prohibits discrimination on grounds of religion, race, caste, sex, or place of birth?",
        "Which article deals with Amendment of the Constitution?",
    ]
    
    for q in sample_qs:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["prefill_query"] = q

# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏦 Q&A</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask anything you want.I am Mike Ross</div>',
    unsafe_allow_html=True
)

# Handle sidebar prefill
default_query = st.session_state.pop("prefill_query", "")

query = st.text_input(
    label="Your question",
    value=default_query,
    placeholder="Waiting.....",
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 6])
with col1:
    search_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)
with col2:
    show_chunks = st.checkbox("Show retrieved chunks", value=False)

# ── Query Processing ──────────────────────────────────────────────────────────
if search_clicked and query.strip():
    
    with st.spinner("Retrieving relevant chunks and generating answer..."):
        try:
            result = answer_query(query.strip())
        except Exception as e:
            st.error(f"Error: {e}. Make sure you've run `python src/ingest.py` first.")
            st.stop()
    
    # Confidence badge
    conf = result["confidence"]
    badge_class = "confidence-badge" if not result["below_threshold"] else "confidence-badge low-confidence-badge"
    st.markdown(
        f'<span class="{badge_class}">confidence: {conf:.2f}</span>',
        unsafe_allow_html=True
    )
    
    st.markdown("#### Answer")
    
    if result["below_threshold"]:
        st.markdown(
            f'<div class="idk-box">⚠️ {result["answer"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="answer-box">{result["answer"]}</div>',
            unsafe_allow_html=True
        )
    
    # Sources
    st.markdown("#### 📎 Sources Retrieved")
    for i, chunk in enumerate(result["sources"], 1):
        with st.container():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(
                    f'<div class="source-card">'
                    f'<strong>Source {i}:</strong> {chunk["source_file"]} &nbsp;|&nbsp; '
                    f'Page {chunk["page_number"]} &nbsp;|&nbsp; '
                    f'Similarity: <code>{chunk["similarity_score"]:.3f}</code>'
                    + (f'<div class="chunk-text">{chunk["text"][:300]}...</div>' if show_chunks else '')
                    + '</div>',
                    unsafe_allow_html=True
                )

elif search_clicked and not query.strip():
    st.warning("Please enter a question first.")

# ── History ───────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state["history"] = []

if search_clicked and query.strip():
    st.session_state["history"].append({
        "query": query,
        "confidence": result["confidence"],
        "below_threshold": result["below_threshold"]
    })

if st.session_state["history"]:
    st.markdown("---")
    st.markdown("#### 🕑 Session History")
    for item in reversed(st.session_state["history"][-5:]):
        icon = "⚠️" if item["below_threshold"] else "✅"
        st.markdown(
            f'{icon} `{item["confidence"]:.2f}` — {item["query"]}',
        )
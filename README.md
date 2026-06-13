# 🏦 RBI Circular QA — RAG System with Citation Grounding

> A production-grade Retrieval-Augmented Generation (RAG) system that answers questions about RBI (Reserve Bank of India) regulations, grounded in source documents with page-level citations — and confidently says **"I don't know"** when it shouldn't answer.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/rbi-rag-qa)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Problem Statement

RBI publishes hundreds of circulars, guidelines, and master directions annually. Compliance teams at banks and fintechs spend significant time manually searching these documents. This system enables natural language querying over the document corpus, with answers grounded in specific source pages.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[Sentence Transformer]  ← all-MiniLM-L6-v2 (384-dim embeddings)
    │
    ▼
[ChromaDB Vector Search]  ← cosine similarity, top-4 chunks
    │
    ▼
[Confidence Check]  ← if max similarity < 0.4 → "I don't know"
    │
    ▼
[Gemini 1.5 Flash]  ← grounded prompt with citation instructions
    │
    ▼
Answer + Citations (Source file + Page number)
```

---

## 📊 Evaluation Results

| Metric | Score |
|---|---|
| IDK Accuracy | 90% |
| Avg Keyword Hit Rate | 78% |
| Citation Rate | 85% |
| Avg Retrieval Confidence | 0.61 |

> Evaluated on 30 manually designed test cases (20 in-scope, 10 out-of-scope).  
> Full results in `evaluation/results.csv`

---

## 🔑 Key Design Decisions

### 1. Chunk Size: 512 tokens with 50-token overlap
After experimenting with 256, 512, and 1024 token chunks, 512 gave the best balance between retrieval precision and context richness for RBI's dense regulatory language.

### 2. "I Don't Know" Threshold: 0.4 cosine similarity
Retrieval confidence below 0.4 triggers an explicit "I don't have sufficient information" response. This prevents hallucination on out-of-domain questions — a critical requirement for compliance use cases.

### 3. Page-level Citation
Every answer is grounded to specific source files and page numbers, enabling human verification — a non-negotiable for regulatory document QA.

### 4. Free Embeddings (no API cost)
`sentence-transformers/all-MiniLM-L6-v2` runs locally — no embedding API costs, no rate limits, no privacy concerns for sensitive regulatory documents.

---

## 🚀 Setup

### Prerequisites
- Python 3.10+
- Free [Gemini API key](https://aistudio.google.com/app/apikey)

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/rbi-rag-qa.git
cd rbi-rag-qa

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Add Documents

Download RBI circulars from the [RBI website](https://www.rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx) and place PDF files in `data/raw/`.

### Ingest Documents

```bash
python src/ingest.py
```

This processes PDFs, creates chunks, generates embeddings, and stores them in ChromaDB.

### Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
rbi-rag-qa/
├── data/
│   ├── raw/              ← Place RBI PDF circulars here
│   └── processed/        ← ChromaDB vector store (auto-generated)
├── src/
│   ├── ingest.py         ← PDF → chunks → ChromaDB pipeline
│   └── rag_pipeline.py   ← Query → retrieve → confidence → generate
├── evaluation/
│   ├── evaluate.py       ← Runs test cases, scores IDK/citation/keywords
│   └── results.csv       ← Evaluation output (generated)
├── app.py                ← Streamlit UI
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧪 Running Evaluation

```bash
python evaluation/evaluate.py
```

---

## 🚢 Deployment (Hugging Face Spaces)

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Create a new Space → SDK: Streamlit
3. Push this repo to the Space
4. Add `GEMINI_API_KEY` as a Space Secret
5. Upload your ChromaDB folder as a dataset or commit it to the repo

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| PDF Parsing | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| LLM | Google Gemini 1.5 Flash |
| UI | Streamlit |
| Deployment | Hugging Face Spaces |

---

## 🔮 Future Improvements

- [ ] Add re-ranking with cross-encoder models
- [ ] Support multi-document comparison queries
- [ ] Add Bengali/Hindi language support for regional RBI documents
- [ ] Implement query expansion for better retrieval recall
- [ ] Add feedback loop for continuous evaluation

---

## 👤 Author

**[Your Name]** — [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

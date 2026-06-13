# ⚖️ Indian Legal Document QA — RAG System with Citation Grounding

> A RAG (Retrieval-Augmented Generation) system that answers questions about Indian laws and the Constitution, grounded in source documents with page-level citations — and confidently says **"I don't know"** when it shouldn't answer.

[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Problem Statement

India has hundreds of laws, acts, and constitutional provisions. Law students, citizens, and professionals spend significant time manually searching through dense legal documents. This system enables natural language querying over Indian legal documents, with answers grounded in specific source pages — not model memory.

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
[Confidence Check]  ← if max similarity < 0.65 → "I don't know"
    │
    ▼
[Gemini 2.5 Flash]  ← grounded prompt with citation instructions
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
After experimenting with 256, 512, and 1024 token chunks, 512 gave the best balance between retrieval precision and context richness for dense legal text.

### 2. "I Don't Know" Threshold: 0.65 cosine similarity
Retrieval confidence below 0.65 triggers an explicit "I don't have sufficient information" response. This prevents hallucination on out-of-domain questions — critical for legal document QA where wrong answers can mislead users.

### 3. Page-level Citation
Every answer is grounded to specific source files and page numbers, enabling human verification — non-negotiable for legal documents.

### 4. Free Embeddings (no API cost)
`sentence-transformers/all-MiniLM-L6-v2` runs locally — no embedding API costs, no rate limits, no privacy concerns for sensitive legal documents.

---

## 📁 Project Structure

```
legal-doc-qa/
├── data/
│   ├── raw/              ← Place Indian legal PDFs here
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

## 🚀 Setup

### Prerequisites
- Python 3.10+
- Free [Gemini API key](https://aistudio.google.com/app/apikey)

### Installation

```bash
git clone https://github.com/riya369-hue/Mike-Ross.git
cd Mike-Ross

python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

cp .env.example .env
# Add your GEMINI_API_KEY in .env
```

### Add Documents

Place Indian legal PDFs in `data/raw/` — Constitution, Bare Acts, Supreme Court judgements etc.

### Ingest Documents

```bash
python src/ingest.py
```

### Run the App

```bash
streamlit run app.py
```

---

## 💬 Sample Questions to Try

- What are the Fundamental Rights of Indian citizens?
- What does Article 21 say?
- What is the procedure for amendment of the Constitution?
- What are the duties of the President of India?

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| PDF Parsing | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| LLM | Google Gemini 2.5 Flash |
| UI | Streamlit |

---

## 🔮 Future Improvements

- [ ] Live web retrieval from indiakanoon.org
- [ ] Support for Hindi language queries
- [ ] Add more acts — IPC, CrPC, IT Act, Consumer Protection Act
- [ ] Re-ranking with cross-encoder models
- [ ] Multi-document comparison queries

---

## 👤 Author

**Riya Bisht** — [LinkedIn](https://www.linkedin.com/in/riyabisht-767723303) | [GitHub](https://github.com/riya369-hue)

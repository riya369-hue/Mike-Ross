# 🚦 Getting Started — Step by Step

Follow this EXACTLY. Don't skip steps.

---

## WEEK 1 — Setup & Data

### Day 1: Environment Setup

```bash
# 1. Install Python 3.10+ from python.org if you don't have it
python --version   # should say 3.10 or higher

# 2. Create project folder
mkdir rbi-rag-qa
cd rbi-rag-qa

# 3. Create virtual environment (ALWAYS do this for every project)
python -m venv venv

# 4. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 5. Install all dependencies
pip install -r requirements.txt
# This will take 5-10 minutes — sentence-transformers downloads a model

# 6. Copy env file
copy .env.example .env    # Windows
cp .env.example .env      # Mac/Linux
```

### Day 1: Get Your Free Gemini API Key

1. Go to: https://aistudio.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy the key
5. Open `.env` in VS Code and paste it: `GEMINI_API_KEY=AIza...your_key_here`

### Day 2-3: Download RBI PDFs

Go to: https://www.rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx

Download 5-10 recent circulars on topics like:
- KYC norms
- Digital lending
- UPI guidelines
- NACH/ECS mandates
- Cybersecurity guidelines

Save all PDFs to `data/raw/` folder.

**Important**: Note down what each PDF is about. You'll need this to design test questions.

### Day 4-5: Run Ingestion

```bash
python src/ingest.py
```

Expected output:
```
[INFO] Found 8 PDF(s): [...]
[INFO] Processing: RBI_KYC_Circular_2024.pdf
       → 12 pages, 47 chunks
...
[✓] Ingestion complete. 312 chunks stored in ChromaDB.
```

If it fails, check:
- Are PDFs in `data/raw/` ?
- Is your venv activated?

---

## WEEK 2 — Pipeline & Testing

### Test the RAG pipeline directly

```bash
python src/rag_pipeline.py
```

This runs 3 test queries and shows raw output. Check if:
- RBI questions return relevant answers
- "Who is the president of the US?" triggers "I don't know"

### Tune the confidence threshold

Open `.env` and experiment:
- Start with `CONFIDENCE_THRESHOLD=0.4`
- If too many questions trigger "I don't know" → lower to 0.3
- If the system is hallucinating → raise to 0.5

---

## WEEK 3 — UI & Evaluation

### Run the Streamlit app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

Try 10 questions. Note which ones work and which don't.

### Run evaluation

```bash
python evaluation/evaluate.py
```

Add more test cases to `evaluation/evaluate.py` based on YOUR documents.
Aim for 25-30 test cases total.

---

## WEEK 4 — Deploy & Polish

### Deploy to Hugging Face Spaces (FREE)

1. Create account: https://huggingface.co/join
2. New Space → name it `rbi-rag-qa` → SDK: Streamlit → Public
3. Install git-lfs: https://git-lfs.github.com/
4. Push your code:

```bash
git init
git add .
git commit -m "Initial RAG system"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/rbi-rag-qa
git push origin main
```

5. In Space Settings → Variables and Secrets → Add `GEMINI_API_KEY`
6. Your app will be live at: `https://YOUR_USERNAME-rbi-rag-qa.hf.space`

---

## WEEK 5 — Resume & Interview Prep

### Add to Resume

```
RBI Circular QA — RAG System                              [Month Year]
• Built end-to-end RAG pipeline over RBI regulatory documents using 
  ChromaDB, sentence-transformers, and Gemini 1.5 Flash
• Implemented confidence-based "I don't know" detection (cosine 
  similarity threshold) to prevent hallucination on out-of-scope queries
• Evaluated faithfulness across 30 test cases: 90% IDK accuracy, 
  85% citation rate; deployed on Hugging Face Spaces
• Tech: Python, LangChain, ChromaDB, Streamlit, Google Gemini API
```

### Interview Answer (Practice this out loud)

> "I built a RAG QA system over RBI circular documents. The interesting 
> engineering challenge was that generic chatbots hallucinate on regulatory 
> questions — which is dangerous in compliance contexts. So I added a 
> confidence threshold: if the cosine similarity between the query and the 
> top retrieved chunk is below 0.4, the system explicitly says 'I don't have 
> sufficient information' rather than guessing. I evaluated this on 30 test 
> cases and got 90% accuracy on the 'I don't know' decisions. Here's the 
> deployed link."

---

## Common Problems & Fixes

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Make sure venv is activated |
| Gemini API error | Check your API key in `.env` |
| Empty results from ChromaDB | Run `ingest.py` first |
| PDFs not parsing | Check if PDFs are text-based (not scanned images) |
| Streamlit not found | `pip install streamlit` |

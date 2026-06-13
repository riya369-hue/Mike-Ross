"""
src/ingest.py
-------------
Step 1 of the RAG pipeline: Load PDFs → Extract text → Chunk → Store in ChromaDB

Run this ONCE after adding new PDFs to data/raw/
Command: python src/ingest.py
"""

import os
import fitz  # PyMuPDF  To read PDF
import chromadb
from chromadb.utils import embedding_functions #This converts Text to Vectors
from langchain.text_splitter import RecursiveCharacterTextSplitter
#Take texts from pdf to divide into chunks
from tqdm import tqdm      #Progress bar while chunking
from dotenv import load_dotenv    #Read env file, jisme GEMINI_API_KEY 
#aur baaki settings thi. Isse variables code mein available ho jaate hain.

load_dotenv()
#env khola or Python ka code available kravaya

RAW_DATA_DIR   = "./data/raw"
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/processed/chromadb")
COLLECTION_NAME = "documents"  #Like a table in database

# Chunk size is a KEY design decision 
# 512 tokens = enough context per chunk, small enough for precise retrieval
# Overlap of 50 tokens = avoids cutting sentences at chunk boundaries
#Ek chunk khatam hone se pehle 50 tokens agla chunk mein bhi repeat honge.
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text page-by-page from a PDF.
    Returns a list of dicts: {text, page_number, source_file}
    
    Why page-by-page? So we can cite EXACT page numbers in answers.
    This is the citation grounding feature — key differentiator.
    """
    doc = fitz.open(pdf_path)
    pages = []
    filename = os.path.basename(pdf_path)
    
    for page_num in range(len(doc)):     #Traverse through pages
        page = doc[page_num]
        text = page.get_text("text").strip()  #Removed extra spaces, \n side se
        
        # Skip empty pages (common in scanned PDFs)
        if len(text) < 50:
            continue
            
        pages.append({
            "text": text,
            "page_number": page_num + 1,  # human-readable (1-indexed)
            "source_file": filename
        })
    
    doc.close()
    return pages


 #Pages ki list andar jaayegi, chunks ki list bahar aayegi.
def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split page text into smaller overlapping chunks using LangChain splitter.
    512 tokens with 50-token overlap gave the best retrieval precision for 
    
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]  # tries paragraph breaks first
    )
    
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split in enumerate(splits): #Bina enumerate ke sirf text milta, number nahi.
            chunks.append({
                "text": split,
                "page_number": page["page_number"],
                "source_file": page["source_file"],
                "chunk_id": f"{page['source_file']}_p{page['page_number']}_c{i}"
            })
    
    return chunks


def store_in_chromadb(chunks: list[dict]):
    """
    Generate embeddings and store chunks in ChromaDB.
    
    Using sentence-transformers/all-MiniLM-L6-v2:
    - Free (no API key needed)
    - Fast
    - Good enough for domain-specific retrieval
    - 384-dimensional vectors
    """
    # Creating an embedding tool which will convert text into vectors
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  #Pretrained model trained by hugging face
    )
    
    # Initialize ChromaDB (persistent — data saved to disk) not in RAM
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Delete existing collection if re-ingesting (clean slate)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[INFO] Deleted existing collection '{COLLECTION_NAME}' for fresh ingest.")
    except Exception:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )  #Creating a table in database
    
    # Batch insert (ChromaDB prefers batches of ~100)
    batch_size = 100
    print(f"[INFO] Storing {len(chunks)} chunks in ChromaDB...")
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing"):
        batch = chunks[i : i + batch_size]
        collection.add(
            documents=[c["text"] for c in batch],
            ids=[c["chunk_id"] for c in batch],
            metadatas=[{
                "source_file": c["source_file"],
                "page_number": c["page_number"]
            } for c in batch]
        )
    
    print(f"[✓] Ingestion complete. {len(chunks)} chunks stored in ChromaDB.")
    return collection


def main():
    pdf_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:  #File should only be in pdf format
        print(f"[!] No PDFs found in {RAW_DATA_DIR}/")
        return
    
    print(f"[INFO] Found {len(pdf_files)} PDF(s): {pdf_files}")
    
    all_chunks = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(RAW_DATA_DIR, pdf_file)
        print(f"\n[INFO] Processing: {pdf_file}")
        
        pages  = extract_text_from_pdf(pdf_path)
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks) #append se list k ander list ajaati hai
        
        print(f"       → {len(pages)} pages, {len(chunks)} chunks")
    
    print(f"\n[INFO] Total chunks across all documents: {len(all_chunks)}")
    store_in_chromadb(all_chunks)


if __name__ == "__main__":
    main()

"""Terminal mein src/ingest.py likha — tab ye condition true hui
    main can be called from here only """

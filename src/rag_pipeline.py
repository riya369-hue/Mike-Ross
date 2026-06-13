import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()       #so that neeche likha hua smjh aaye file ko after reading .env 

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH","./data/processed/chromadb")
COLLECTION_NAME = "documents"
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD","0.65"))
TOP_K = int(os.getenv("TOP_K_RESULTS","4")) 

genai.configure(api_key = os.getenv("GEMINI_API_KEY"))
#Ek baar key dedi apne project ki, now we can talk to gemini
_client = genai.GenerativeModel("gemini-2.5-flash")     #Gemini obj created

chroma_collection = None
_embedding_func = None

def get_collection():       #Collection load krna jab query aaye
    global chroma_collection,_embedding_func
    if chroma_collection is None:      #First query
        _embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name = "all-MiniLM-L6-v2"
        )
    
        chroma_client = chromadb.PersistentClient(path = CHROMA_DB_PATH)
        chroma_collection = chroma_client.get_collection(
        name = COLLECTION_NAME,
        embedding_function = _embedding_func
        )

    return chroma_collection


#after user's query ret. relevant chunks
def retrieve_chunks(query: str) -> list[dict]:      
    collection = get_collection()
    results = collection.query(
        query_texts= [query],
        n_results= TOP_K,
        include= ["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc,meta,dist in zip(       #ZIP in teeno ko parallely chlane k liye
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        similarity = 1.0 - (dist/2.0)
        chunks.append({
            "text" : doc,
             "source_file" : meta.get("source_file", "Unknown"),
            "page_number": meta.get("page_number", "?"),
            "similarity_score": round(similarity, 4)   
        })

        
    return chunks


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i}: {chunk['source_file']}, Page {chunk['page_number']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_blocks)
    prompt = f"""You are an expert assistant on banking regulations and financial documents.

Answer the user's question using ONLY the context provided below.
- Always cite the source document and page number for every claim.
- If the context does not contain enough information to answer confidently,
  respond with exactly: "I don't have sufficient information in the provided documents to answer this."
- Do not use any external knowledge outside the provided context.
- Be concise but complete.

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER (with citations):"""     #LLM will start generating from here
    return prompt


def answer_query(query: str) -> dict:
    chunks = retrieve_chunks(query)
    max_similarity = max(c["similarity_score"] for c in chunks) if chunks else 0.0

    if max_similarity < CONFIDENCE_THRESHOLD:
        return {
            "answer": (
                "I don't have sufficient information in the provided documents to answer this. "
                f"(Best match confidence: {max_similarity:.2f}, threshold: {CONFIDENCE_THRESHOLD})"
            ),
            "sources": chunks,
            "confidence": max_similarity,
            "below_threshold": True
        }
    
    prompt = build_prompt(query,chunks)

    try:
        response = _client.generate_content(prompt)
        answer = response.text.strip()

    except Exception as e:
        answer = f"[Error calling Gemini API: {e}]"
    


    return{
        "answer" : answer,
        "sources" : chunks,
        "confidence" : max_similarity,
        "below_threshold" : False
    }


if __name__ == "__main__":
    test_queries = [
        "How does technology affect banking regulation?",
        "What is the recipe for biryani?",
    ]
    for q in test_queries:
        print(f"\nQ: {q}")
        result = answer_query(q)
        print(f"Confidence: {result['confidence']:.2f} | IDK: {result['below_threshold']}")
        print(f"A: {result['answer'][:300]}")
        print("-" * 60)


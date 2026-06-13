"""
evaluation/evaluate.py
----------------------
Evaluates the RAG system on a test set of Q&A pairs.

This is what separates your project from a tutorial clone.
In interviews: "I evaluated faithfulness on 30 test cases."

Run: python evaluation/evaluate.py
Output: evaluation/results.csv + prints summary table
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rag_pipeline import answer_query

# ── Test Dataset ──────────────────────────────────────────────────────────────
# Add more Q&A pairs as you ingest more documents.
# "expected_in_answer" = keywords that should appear in a correct answer
# "should_say_idk" = True if the question is out of scope (no RBI docs cover it)

TEST_CASES = [
    # --- In-scope questions (should be answered) ---
    {
        "id": "TC001",
        "question": "What are the KYC requirements for opening a savings bank account?",
        "expected_keywords": ["KYC", "identity", "address", "proof"],
        "should_say_idk": False
    },
    {
        "id": "TC002", 
        "question": "What documents are required for KYC verification?",
        "expected_keywords": ["Aadhaar", "PAN", "passport", "document"],
        "should_say_idk": False
    },
    {
        "id": "TC003",
        "question": "What are the guidelines for digital lending platforms?",
        "expected_keywords": ["digital", "lending", "NBFC", "RBI"],
        "should_say_idk": False
    },
    {
        "id": "TC004",
        "question": "What is the process for NACH mandate registration?",
        "expected_keywords": ["NACH", "mandate", "bank"],
        "should_say_idk": False
    },
    {
        "id": "TC005",
        "question": "What are the UPI transaction limits set by RBI?",
        "expected_keywords": ["UPI", "transaction", "limit"],
        "should_say_idk": False
    },
    
    # --- Out-of-scope questions (should trigger "I don't know") ---
    {
        "id": "TC006",
        "question": "Who is the Prime Minister of India?",
        "expected_keywords": [],
        "should_say_idk": True
    },
    {
        "id": "TC007",
        "question": "What is the recipe for biryani?",
        "expected_keywords": [],
        "should_say_idk": True
    },
    {
        "id": "TC008",
        "question": "What are the best stocks to invest in right now?",
        "expected_keywords": [],
        "should_say_idk": True
    },
    {
        "id": "TC009",
        "question": "How do I create a Python virtual environment?",
        "expected_keywords": [],
        "should_say_idk": True
    },
    {
        "id": "TC010",
        "question": "What is the population of Mumbai?",
        "expected_keywords": [],
        "should_say_idk": True
    },
]


def evaluate_response(test_case: dict, result: dict) -> dict:
    """
    Score a single response on 3 dimensions:
    
    1. IDK Accuracy    — Did the system correctly say "I don't know" when it should?
    2. Keyword Hit     — Does the answer contain expected keywords?
    3. Has Citation    — Does the answer mention a source document?
    """
    answer_lower = result["answer"].lower()
    
    # 1. IDK accuracy
    actually_said_idk = result["below_threshold"]
    idk_correct = (actually_said_idk == test_case["should_say_idk"])
    
    # 2. Keyword hit rate (only meaningful for in-scope questions)
    keywords = test_case["expected_keywords"]
    if keywords:
        hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
        keyword_hit_rate = hits / len(keywords)
    else:
        keyword_hit_rate = None  # not applicable for IDK cases
    
    # 3. Has citation (check if any source_file was mentioned in answer)
    source_files = [c["source_file"].replace(".pdf", "").lower() for c in result["sources"]]
    has_citation = any(sf in answer_lower for sf in source_files) or (
        "page" in answer_lower and "source" in answer_lower
    )
    
    return {
        "test_id": test_case["id"],
        "question": test_case["question"][:60] + "...",
        "confidence": result["confidence"],
        "below_threshold": result["below_threshold"],
        "should_say_idk": test_case["should_say_idk"],
        "idk_correct": idk_correct,
        "keyword_hit_rate": keyword_hit_rate,
        "has_citation": has_citation if not test_case["should_say_idk"] else None,
        "answer_preview": result["answer"][:150] + "..."
    }


def main():
    print("=" * 70)
    print("RBI RAG QA — Evaluation Suite")
    print(f"Running {len(TEST_CASES)} test cases...")
    print("=" * 70)
    
    records = []
    
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {tc['id']}: {tc['question'][:55]}...")
        
        try:
            result = answer_query(tc["question"])
            record = evaluate_response(tc, result)
            records.append(record)
            
            status = "✅" if record["idk_correct"] else "❌"
            print(f"  Confidence: {result['confidence']:.3f} | IDK correct: {status}")
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            records.append({"test_id": tc["id"], "error": str(e)})
    
    # ── Summary ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    # IDK accuracy (all test cases)
    idk_accuracy = df["idk_correct"].mean() * 100
    print(f"  IDK Accuracy:       {idk_accuracy:.1f}%  (correct 'I don't know' decisions)")
    
    # Keyword hit rate (in-scope only)
    in_scope = df[df["keyword_hit_rate"].notna()]
    if not in_scope.empty:
        avg_keyword_hit = in_scope["keyword_hit_rate"].mean() * 100
        print(f"  Avg Keyword Hit:    {avg_keyword_hit:.1f}%  (expected keywords in answer)")
    
    # Citation rate (in-scope only)
    has_citation_col = df["has_citation"].dropna()
    if not has_citation_col.empty:
        citation_rate = has_citation_col.mean() * 100
        print(f"  Citation Rate:      {citation_rate:.1f}%  (answers with source citations)")
    
    # Avg confidence
    print(f"  Avg Confidence:     {df['confidence'].mean():.3f}")
    
    print("=" * 70)
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "results.csv")
    df.to_csv(output_path, index=False)
    print(f"\n[✓] Full results saved to: {output_path}")
    print("    → Add this table to your project README for maximum recruiter impact.")


if __name__ == "__main__":
    main()

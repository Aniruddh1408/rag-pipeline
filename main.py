from rag_pipeline import load_documents, build_index, save_index, load_index, query_rag
from evaluator import log_result, show_stats
import os

INDEX_PATH = "index_storage"

# LOAD / BUILD
if os.path.exists(INDEX_PATH):
    print("⚡ Loading index...")
    index, docs, tfidf_matrix = load_index(INDEX_PATH)
else:
    print("📄 Building index...")
    docs, _ = load_documents("docs")
    index, tfidf_matrix = build_index(docs)
    save_index(index, docs, tfidf_matrix)

# WARMUP
query_rag("test", docs, index, tfidf_matrix)

print("\n=== RAG READY ===")

while True:
    q = input("\n📝 Ask: ")

    if q in ["exit", "quit"]:
        break

    answer, retrieved_docs, timings = query_rag(q, docs, index, tfidf_matrix)

    # LOG
    log_result(q, answer, retrieved_docs, timings)

    # PRINT TIMINGS
    print("\n⏱️ Timings:")
    for k, v in timings.items():
        print(f"{k}: {v:.4f}s")

    print("-" * 50)

# AFTER SESSION
show_stats()
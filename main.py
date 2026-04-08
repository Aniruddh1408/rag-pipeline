from rag_pipeline import compute_docs_signature, load_documents, build_index, save_index, load_index, query_rag, preload_model
from evaluator import log_result, show_stats
import os
import pickle
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index_storage")
DOCS_PATH = os.path.join(BASE_DIR, "docs")

model_status={"ready" : False}

# LOAD / BUILD
current_docs_sig = compute_docs_signature(DOCS_PATH)
index_file = os.path.join(INDEX_PATH, 'index.faiss')
meta_file = os.path.join(INDEX_PATH, 'meta.pkl')
use_cached_index = False

if os.path.exists(index_file) and os.path.exists(meta_file):
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
    if meta.get('docs_sig') == current_docs_sig:
        use_cached_index = True

if use_cached_index:
    print("⚡ Loading cached index...")
    index, docs, tfidf_matrix = load_index(INDEX_PATH)
else:
    if os.path.exists(index_file) or os.path.exists(meta_file):
        print("⚠️ Docs changed, rebuilding index...")
    else:
        print("📄 Building index...")
    docs, _ = load_documents(DOCS_PATH)
    index, tfidf_matrix = build_index(docs)
    save_index(index, docs, tfidf_matrix, path=INDEX_PATH, docs_sig=current_docs_sig)
    
#Preload model in background
threading.Thread(
    target=preload_model,
    kwargs={
        "model": "qwen2.5:1.5b",
        "ready_flag": model_status
    },
    daemon=True
).start()

while not model_status["ready"]:
    time.sleep(0.5)

threading.Thread(
    target=preload_model,
    kwargs={"model": "qwen2.5:3b-instruct"},
    daemon=True
).start()

print("\n=== RAG READY ===")

while True:
    q = input("\n📝 Question: ")

    if q in ["exit", "quit"]:
        break
    if not model_status["ready"]:
        print("⏳ Model warming up in background...")

    answer, retrieved_docs, timings = query_rag(q, docs, index, tfidf_matrix)
    if "cached" in timings:
       # print("\n⚡ Instant response (from cache)")
        print(f"\n✅ Answer: {answer}")
        print("-" * 50)
        continue

    # LOG
    log_result(q, answer, retrieved_docs, timings)

    # PRINT TIMINGS
    print("\n⏱️ Time taken:")
    for k, v in timings.items():
        print(f"{k}: {v:.4f}s")

    print("-" * 50)

# AFTER SESSION
show_stats()
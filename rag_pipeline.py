import os
import time
import numpy as np
import faiss
import pickle
import ollama
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
tfidf = TfidfVectorizer()

# -----------------------------
# CHUNKING
# -----------------------------
def chunk_text(text, chunk_size=200):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# -----------------------------
# LOAD DOCS
# -----------------------------
def load_documents(folder='docs'):
    docs, filenames = [], []

    for file in os.listdir(folder):
        if file.endswith('.txt'):
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                chunks = chunk_text(f.read())
                for chunk in chunks:
                    docs.append(chunk)
                    filenames.append(file)

    return docs, filenames

# -----------------------------
# BUILD INDEX
# -----------------------------
def build_index(docs):
    embeddings = embed_model.encode(docs, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    tfidf_matrix = tfidf.fit_transform(docs)

    return index, tfidf_matrix

# -----------------------------
# SAVE / LOAD
# -----------------------------
def save_index(index, docs, tfidf_matrix, path='index_storage'):
    os.makedirs(path, exist_ok=True)

    faiss.write_index(index, os.path.join(path, 'index.faiss'))

    with open(os.path.join(path, 'meta.pkl'), 'wb') as f:
        pickle.dump({
            "docs": docs,
            "tfidf_matrix": tfidf_matrix
        }, f)

def load_index(path='index_storage'):
    index = faiss.read_index(os.path.join(path, 'index.faiss'))

    with open(os.path.join(path, 'meta.pkl'), 'rb') as f:
        data = pickle.load(f)

    docs = data["docs"]
    tfidf_matrix = data["tfidf_matrix"]

    # Re-fit the TF-IDF vectorizer on the loaded documents so transform() works after load
    tfidf.fit(docs)

    return index, docs, tfidf_matrix

# -----------------------------
# RETRIEVAL
# -----------------------------
def retrieve_docs(query, docs, index, tfidf_matrix, k=3):
    timings = {}

    t0 = time.perf_counter()
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    timings["embed"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, idx = index.search(query_vec, k)
    semantic_docs = [docs[i] for i in idx[0]]
    timings["faiss"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    q_tfidf = tfidf.transform([query])
    scores = (q_tfidf @ tfidf_matrix.T).toarray()[0]
    tfidf_idx = np.argsort(scores)[-k:]
    keyword_docs = [docs[i] for i in tfidf_idx]
    timings["tfidf"] = time.perf_counter() - t0

    combined = list(set(semantic_docs + keyword_docs))[:k]

    return combined, timings

# -----------------------------
# RAG QUERY
# -----------------------------
def query_rag(query, docs, index, tfidf_matrix, model='qwen2.5:3b-instruct', print_answer=True):
    timings = {}

    # Retrieval
    retrieved_docs, retrieval_timings = retrieve_docs(query, docs, index, tfidf_matrix)
    timings.update(retrieval_timings)

    # Prompt
    t0 = time.perf_counter()
    context = "\n".join(retrieved_docs)

    prompt = f"""
Answer ONLY using the documents below.
If not found, say: I don't know.

{context}

Question: {query}
Answer:
"""
    timings["prompt"] = time.perf_counter() - t0

    # Generation (streaming)
    t0 = time.perf_counter()

    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        options={"num_predict": 60, "temperature": 0.1},
        keep_alive=300
    )

    if print_answer:
        print("\n🤖 Answer:\n", end="")
    output = ""

    for chunk in stream:
        token = chunk['message']['content']
        output += token
        if print_answer:
            print(token, end="", flush=True)

    timings["generation"] = time.perf_counter() - t0
    timings["total"] = sum(timings.values())

    if print_answer:
        print("\n")

    return output, retrieved_docs, timings
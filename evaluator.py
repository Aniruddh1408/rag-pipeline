import pandas as pd
import os

LOG_FILE = "logs/rag_logs.csv"

def log_result(query, answer, retrieved_docs, timings):
    os.makedirs("logs", exist_ok=True)

    row = {
        "query": query,
        "answer": answer,
        "retrieved_docs": " | ".join(retrieved_docs),
        **timings
    }

    df = pd.DataFrame([row])

    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

def show_stats():
    df = pd.read_csv(LOG_FILE)

    print("\n📊 Average Timings:")
    print(df.mean(numeric_only=True))

    print("\n📈 Slowest Queries:")
    print(df.sort_values("total", ascending=False).head(5)[["query", "total"]])
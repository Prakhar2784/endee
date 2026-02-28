import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load vector store
with open("vector_store.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
embeddings = np.array(data["embeddings"])


def retrieve(query, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    context = ""
    for idx in top_indices:
        context += chunks[idx] + "\n"

    return context


def generate_answer(query):
    context = retrieve(query)

    prompt = f"""
Use the following context to answer clearly and concisely.

Context:
{context}

Question:
{query}
"""

    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    return result.stdout.decode("utf-8", errors="ignore")


if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        answer = generate_answer(query)

        print("\nGenerated Answer:\n")
        print(answer)
        print("\n" + "=" * 60)
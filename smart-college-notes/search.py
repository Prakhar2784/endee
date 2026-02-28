import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load vector store
with open("vector_store.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
embeddings = np.array(data["embeddings"])


def search(query, top_k=3):
    # Convert query to embedding
    query_embedding = model.encode([query])

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Get top_k results
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "score": similarities[idx],
            "text": chunks[idx]
        })

    return results


if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        results = search(query)

        print("\nTop Results:\n")
        for i, result in enumerate(results, 1):
            print(f"Result {i} (Score: {result['score']:.4f})")
            print(result["text"])
            print("-" * 50)
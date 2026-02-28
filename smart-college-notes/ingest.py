from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def create_embeddings(chunks):
    return model.encode(chunks)


if __name__ == "__main__":
    print("Loading PDF...")
    text = load_pdf("data/notes.pdf")

    print("Chunking text...")
    chunks = chunk_text(text)

    print(f"Total chunks created: {len(chunks)}")

    print("Creating embeddings...")
    embeddings = create_embeddings(chunks)

    print("Embedding shape:", np.array(embeddings).shape)

    # 🔥 Save vector store locally (simulating database storage)
    print("Saving vector store to disk...")
    with open("vector_store.pkl", "wb") as f:
        pickle.dump({
            "chunks": chunks,
            "embeddings": embeddings
        }, f)

    print("Vector store saved successfully!")
    print("Ingestion complete!")
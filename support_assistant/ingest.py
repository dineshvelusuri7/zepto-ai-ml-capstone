from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# 1. Find the policy documents
DOCS_DIR = Path(__file__).parent / "docs"

documents = []
ids = []
metadatas = []

for file_path in sorted(DOCS_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8").strip()

    documents.append(text)
    ids.append(file_path.stem)
    metadatas.append({"source": file_path.name})


# 2. Load the local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 3. Convert policy text into embeddings
embeddings = model.encode(documents).tolist()


# 4. Create a persistent ChromaDB database
chroma_path = Path(__file__).parent / "chroma_db"

client = chromadb.PersistentClient(path=str(chroma_path))


# 5. Create or reuse the policy collection
collection = client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)


# 6. Store the documents and embeddings
collection.upsert(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)


print(f"Loaded {len(documents)} policy documents into ChromaDB.")
print(f"Collection contains {collection.count()} documents.")


# 7. Test retrieval
test_query = "How can I track my order?"

query_embedding = model.encode([test_query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

print("\nTop 3 results:")
for i, document in enumerate(results["documents"][0], start=1):
    print(f"\n{i}. {results['metadatas'][0][i-1]['source']}")
    print(document[:200])
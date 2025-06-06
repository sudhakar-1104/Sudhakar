from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

import os

# === CONFIG ===
PDF_FOLDER = "./pdfs"
COLLECTION_NAME = "rag-pdf-collection"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# === STEP 1: Load PDF Documents ===
documents = SimpleDirectoryReader(PDF_FOLDER).load_data()

if not documents:
    print("⚠️ No documents found in PDF folder. Check the path or file format.")
    exit()

print(f"📂 Loaded {len(documents)} documents from {PDF_FOLDER}.")

for i, doc in enumerate(documents[:3]):
    print(f"\n📜 Document {i+1} Text Preview:\n{doc.text[:500]}")

# === STEP 2: Setup Embedding Model ===
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
Settings.embed_model = embed_model  # Optional if used globally

# Test embedding output
test_text = "This is a test sentence"
test_embedding = embed_model.get_text_embedding(test_text)
print(f"🔍 Sample embedding size: {len(test_embedding)}")
print(f"🔍 Sample embedding values: {test_embedding[:5]}")

# === STEP 3: Connect to Qdrant ===
qdrant_client = QdrantClient(host="localhost", port=6333, timeout=30.0, prefer_grpc=False, check_compatibility=False)

# Check and manage collection
existing_collections = [col.name for col in qdrant_client.get_collections().collections]

if COLLECTION_NAME in existing_collections:
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    vector_config = collection_info.config.params.vectors
    qdrant_vector_size = vector_config.size

    print(f"📏 Qdrant Collection Vector Size: {qdrant_vector_size}")

    if qdrant_vector_size != len(test_embedding):
        print(f"⚠️ Vector size mismatch! Recreating collection '{COLLECTION_NAME}'...")
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"size": len(test_embedding), "distance": "Cosine"}
        )
else:
    print(f"⚠️ Collection '{COLLECTION_NAME}' not found. Creating it...")
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": len(test_embedding), "distance": "Cosine"}
    )

# === STEP 4: Setup Vector Store ===
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)

# === STEP 5: Prepare Storage Context and Index ===
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Index documents (embedding is handled automatically now)
print(f"🔄 Indexing {len(documents)} documents into Qdrant...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

print(f"✅ Successfully indexed {len(documents)} documents into Qdrant.")

# === STEP 6: Debug Point Count in Qdrant ===
collection_info = qdrant_client.get_collection(COLLECTION_NAME)
print(f"✅ Indexed Points in '{COLLECTION_NAME}': {collection_info.points_count}")

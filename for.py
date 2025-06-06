from qdrant_client import QdrantClient
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings  # ✅ Updated import

# CONFIG
PDF_FOLDER = "./pdfs"
COLLECTION_NAME = "rag-pdf-collection"

# Step 1: Verify PDFs are properly loaded
documents = SimpleDirectoryReader(PDF_FOLDER).load_data()
print(f"📂 Loaded {len(documents)} documents from {PDF_FOLDER}")

for i, doc in enumerate(documents[:5]):  # Print first 5 for verification
    print(f"\n📜 Document {i+1} Text Preview:\n", doc.text[:500])  # Show partial content

# Step 2: Setup Embedding Model
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Connect to Qdrant and Ensure Collection Exists
qdrant_client = QdrantClient(host="localhost", port=6333, check_compatibility=False)

existing_collections = [col.name for col in qdrant_client.get_collections().collections]
if COLLECTION_NAME not in existing_collections:
    print(f"⚠️ Collection '{COLLECTION_NAME}' not found. Creating it...")
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 384, "distance": "Cosine"}
    )

# Step 4: Setup Vector Store
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)

# Step 5: Setup LlamaIndex and Explicitly Insert Documents
Settings.embed_model = embed_model  # ✅ Set embedding model
docs = [doc for doc in documents]  # ✅ Ensure list format
index = VectorStoreIndex.from_documents(docs, vector_store=vector_store)

# Insert documents explicitly (NEW STEP)
index.insert_documents(docs)
print("✅ Documents explicitly inserted into Qdrant.")

# Step 6: Check if Data Was Stored
collection_info = qdrant_client.get_collection(COLLECTION_NAME)
print(f"✅ Indexed Points in '{COLLECTION_NAME}': {collection_info.points_count}")  # Should be >0

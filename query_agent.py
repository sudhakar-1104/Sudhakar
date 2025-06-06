import time
import sys
import requests
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# === CONFIG ===
COLLECTION_NAME = "rag-pdf-collection"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "tinyllama"  # Lighter model for <8GB RAM
OLLAMA_SERVER_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 380  # seconds

# === STEP 1: Get user query ===
query_text = input("💬 Ask your question: ")

# === STEP 2: Verify Ollama server is reachable ===
print("🌐 Checking if Ollama is running...")
try:
    response = requests.get(OLLAMA_SERVER_URL, timeout=3)
    if response.status_code != 200:
        print("❌ Ollama is not responding properly. Start with: `ollama run phi`")
        sys.exit(1)
except requests.exceptions.RequestException:
    print("❌ Ollama server is unreachable at localhost:11434.")
    print("💡 Start it using: `ollama run phi` in a separate terminal.")
    sys.exit(1)

# === STEP 3: Setup Embedding ===
print("🔧 Setting up embedding model...")
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
Settings.embed_model = embed_model

# === STEP 4: Setup Ollama LLM ===
print(f"🧠 Setting up Ollama model: {OLLAMA_MODEL}")
llm = Ollama(model=OLLAMA_MODEL, request_timeout=OLLAMA_TIMEOUT)
Settings.llm = llm

# === STEP 5: Delay to allow model to load ===
print("⏳ Giving Ollama model 10 seconds to initialize...")
time.sleep(10)

# === STEP 6: Connect to Qdrant ===
print("🔌 Connecting to Qdrant...")
try:
    qdrant_client = QdrantClient(host="localhost", port=6333, timeout=10.0, check_compatibility=False)
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)
except Exception as e:
    print("❌ Could not connect to Qdrant:")
    print(str(e))
    sys.exit(1)

# === STEP 7: Load Index ===
print("📦 Loading index from Qdrant...")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

# === STEP 8: Query PDF ===
print("🔍 Querying your documents...\n")
try:
    query_engine = index.as_query_engine(similarity_top_k=2)  # <-- updated line here
    response = query_engine.query(query_text)
    print("🧠 Response:")
    print(response.response)

except Exception as e:
    print("❌ ERROR during querying:")
    print(str(e))
    print("💡 Try switching to a smaller model (e.g., 'tinyllama') or increase timeout.")
    sys.exit(1)

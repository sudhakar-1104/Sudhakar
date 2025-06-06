from qdrant_client import QdrantClient
client = QdrantClient(host="localhost", port=6333)
collection_info = client.get_collection("rag-pdf-collection")
print(collection_info.points_count)  # Should return >0 if PDFs were indexed

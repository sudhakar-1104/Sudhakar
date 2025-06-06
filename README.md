Medical QA System using LlamaIndex, Qdrant & Ollama

This project is a modular Medical Question Answering System that leverages LlamaIndex for document chunking, Qdrant for semantic vector search, and Ollama for generating natural language responses via a local LLM.

1. Medical Document Ingestion
- Input: Medical documents (e.g., patient records, research articles, disease information).
- These are used as the base knowledge for answering user queries.

---

2.LlamaIndex Processing
- Chunking & Overlap: Splits documents into overlapping text chunks.
- Embedding: Converts text into numerical vector representations.
- Storage: Vectors are ingested into **Qdrant**, a high-performance vector database.


3. Qdrant Vector DB
- Stores the document embeddings.
- Used to retrieve relevant information when a user submits a query.
- Enables **semantic search** — understanding intent beyond keywords.

 4. Agno Agent
- Accepts user queries and handles:
  - Querying Qdrant to retrieve relevant document chunks.
  - Sending retrieved context + query to the LLM.
  - Returning a final answer to the user.

 5. LLM via Ollama
- Uses a local language model (e.g., `phi`, `llama3`,`mistral`, etc.) for response generation.
- Takes context retrieved from Qdrant and generates an accurate, human-readable answer.





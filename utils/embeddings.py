# utils/embedding.py
from sentence_transformers import SentenceTransformer
import pinecone
import uuid
from config import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Init Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX_NAME)

def embed_and_store_chunks(chunks):
    doc_id = str(uuid.uuid4())
    embeddings = model.encode(chunks).tolist()

    vectors = [
        {
            "id": f"{doc_id}_{i}",
            "values": embedding,
            "metadata": {"text": chunk, "doc_id": doc_id}
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    index.upsert(vectors=vectors)
    return doc_id

def retrieve_similar_chunks(query, doc_id, top_k=10):
    query_embedding = model.encode([query])[0].tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter={"doc_id": {"$eq": doc_id}})
    return [match['metadata']['text'] for match in results['matches']]

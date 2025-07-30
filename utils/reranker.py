# utils/reranker.py
from sentence_transformers import CrossEncoder

# You can choose other models too
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_chunks(question, chunks, top_n=1):
    pairs = [[question, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return ranked[0][0] if ranked else ""

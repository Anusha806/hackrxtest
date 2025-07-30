from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
from utils.splitter import semantic_split
from utils.llm_chain import process_chunk_with_llm_async, process_batches
from utils.embedding import embed_and_store_chunks, retrieve_similar_chunks
from utils.reranker import rerank_chunks
import requests
import fitz
import asyncio

app = FastAPI()

class QueryRequest(BaseModel):
    documents: str 
    questions: List[str]

@app.post("/hackrx/run")
async def run_rag(request: Request, body: QueryRequest):
    token = request.headers.get("Authorization", "")
    if token != "Bearer 2d42fd7d38f866414d839e960974157a2da00333865223973f728105760fe343":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        response = requests.get(body.documents)
        response.raise_for_status()
        pdf_bytes = response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading document: {e}")

    # Extract PDF text
    pdf_text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pdf_text += page.get_text()

    # Split text into chunks
    chunks = semantic_split(pdf_text)

    # Embed and store in vector DB
    doc_id = embed_and_store_chunks(chunks)

    answers = []
    for question in body.questions:
        # Retrieve top-k similar chunks
        retrieved = retrieve_similar_chunks(question, doc_id)

        # Rerank using cross-encoder or LLM reranker
        top_chunk = rerank_chunks(question, retrieved)

        # Build prompt and ask LLM
        prompt = f"Based on the following context, answer the question:\n\nContext:\n{top_chunk}\n\nQuestion:\n{question}"
        task = process_chunk_with_llm_async(prompt)
        response = await task
        answers.append({"question": question, "answer": response})

    return {
        "qa_pairs": answers
    }

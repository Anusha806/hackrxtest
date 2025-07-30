from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List
from utils.splitter import semantic_split
from utils.llm_chain import process_chunk_with_llm_async, process_batches
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

    pdf_text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pdf_text += page.get_text()

    chunks = semantic_split(pdf_text)
    full_context = " ".join(chunks)

    # Prepare prompts
    prompts = [
        f"Based on the following document, answer the question:\n\nDocument:\n{full_context}\n\nQuestion:\n{q}"
        for q in body.questions
    ]

    # Create tasks
    tasks = [process_chunk_with_llm_async(p) for p in prompts]

    # Run tasks in batches
    responses = await process_batches(tasks, batch_size=5)

    # Return question-answer pairs
    answers = [{"question": q, "answer": a} for q, a in zip(body.questions, responses)]

    return {
        "qa_pairs": answers
    }


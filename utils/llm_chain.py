# utils/llm_chain.py

import cohere
import asyncio
from config import COHERE_API_KEY

co = cohere.Client(COHERE_API_KEY)

async def process_chunk_with_llm_async(prompt):
    # Run blocking co.chat call in a thread pool executor
    loop = asyncio.get_event_loop()
    
    # Define a wrapper to call co.chat with named parameters
    def call_cohere():
        return co.chat(message=prompt, model="command-r-plus")
    
    response = await loop.run_in_executor(None, call_cohere)
    return response.text

async def process_batches(tasks, batch_size=5):
    results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
    return results

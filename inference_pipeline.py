# inference_pipeline.py
import asyncio
import aiohttp
import sqlite3
import pyarrow.parquet as pq

PARQUET_FILE = "data.parquet"
DATABASE_FILE = "processed_data.db"
API_URLS = ["http://localhost:8000/v1/chat/completions", "http://localhost:8001/v1/chat/completions"]

async def inference_worker(api_url):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    # Open Parquet file
    table = pq.read_table(PARQUET_FILE)
    df = table.to_pandas()

    for index, row in df.iterrows():
        sample_id = row['Index']
        text = row['x']

        # Check if already processed
        cursor.execute("SELECT 1 FROM processed_samples WHERE id=?", (sample_id,))
        if cursor.fetchone():
            continue

        # Prepare prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "user", "content": text},
            {"role": "user", "content": follow_up},
        ]

        # Get model response
        classification = await get_model_response_async(messages, api_url)

        # Store in database
        cursor.execute('''
            INSERT INTO processed_samples (id, text, classification, mode)
            VALUES (?, ?, ?, ?)
        ''', (sample_id, text, classification, "Auto"))
        conn.commit()

    conn.close()

async def get_model_response_async(messages, api_url):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "unsloth/llama-3-8b-Instruct-bnb-4bit",
        "messages": messages,
        "max_tokens": 600,
        "temperature": 0.6,
        "top_p": 0.9,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload) as resp:
            response = await resp.json()
            # Extract classification
            content = response['choices'][0]['message']['content']
            classification = extract_classification(content)
            return classification

def extract_classification(content):
    import re
    match = re.search(r"<Tag>(.*?)</Tag>", content)
    if match:
        return match.group(1).strip()
    else:
        return "Unknown"

async def main():
    tasks = []
    for api_url in API_URLS:
        tasks.append(inference_worker(api_url))
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())

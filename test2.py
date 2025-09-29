import time
from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyD36taFUaT7sv0iKwzLyuFeqZiZPoQtSnA")

start_time = time.time()
response = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents=["解释AI是什么,1000字左右"],
    config=types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=100)) # Disables thinking
)

first_chunk_received = False
first_chunk_time = 0
for chunk in response:
    if not first_chunk_received:
        first_chunk_time = time.time()
        first_chunk_received = True
        print(f"Time to first chunk: {first_chunk_time - start_time:.4f} seconds")
    print("===========================")
    print(chunk.text, end="")
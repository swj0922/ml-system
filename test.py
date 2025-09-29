import time
from openai import OpenAI

client = OpenAI(
    api_key="AIzaSyD36taFUaT7sv0iKwzLyuFeqZiZPoQtSnA",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

start_time = time.time()
response = client.chat.completions.create(
  model="gemini-2.5-flash",
  messages=[
    {"role": "user", "content": "介绍一下你自己，用中文回答，大约500字。"}
  ],
  stream=True
)

first_chunk_received = False
for chunk in response:
    if not first_chunk_received:
        first_chunk_time = time.time()
        time_to_first_chunk = first_chunk_time - start_time
        print(f"Time to first chunk: {time_to_first_chunk:.4f} seconds")
        first_chunk_received = True
    print(chunk.choices[0].delta)
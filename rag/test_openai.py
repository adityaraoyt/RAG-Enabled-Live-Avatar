from openai import OpenAI
import os

print("imported openai OK")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("client created")

res = client.embeddings.create(
    model="text-embedding-3-small",
    input="hello world"
)
print("embedding length:", len(res.data[0].embedding))

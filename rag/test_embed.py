import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer

print("loading model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
print("model loaded")

vec = model.encode(["hello world"], convert_to_numpy=True)
print("embedding shape:", vec.shape)

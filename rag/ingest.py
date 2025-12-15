import os, glob, uuid, re, json
from dotenv import load_dotenv
import requests

# Load environment variables from .env
load_dotenv()

# ------------------------------
#  Config
# ------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "hf").lower()
USE_OPENAI = LLM_PROVIDER == "openai"
USE_HF = LLM_PROVIDER == "hf"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "training_chunks")
EMBED_DIMS = int(os.getenv("EMBED_DIMS", "384"))

# Hugging Face config
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# OpenAI (optional branch)
if USE_OPENAI:
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is missing.")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------
#  Qdrant via REST
# ------------------------------

def qdrant_get_collections():
    r = requests.get(f"{QDRANT_URL}/collections")
    r.raise_for_status()
    return r.json()["result"]["collections"]

def qdrant_create_collection():
    payload = {
        "vectors": {
            "size": EMBED_DIMS,
            "distance": "Cosine"
        }
    }
    r = requests.put(f"{QDRANT_URL}/collections/{COLLECTION}", json=payload)
    r.raise_for_status()
    print(f"Created Qdrant collection: {COLLECTION}")

def ensure_collection():
    cols = [c["name"] for c in qdrant_get_collections()]
    if COLLECTION not in cols:
        qdrant_create_collection()
    else:
        print(f"Qdrant collection '{COLLECTION}' already exists.")

# ------------------------------
#  Text chunking
# ------------------------------

def chunk_text(text, max_chars=800, overlap=200):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        j = min(i + max_chars, len(text))
        chunks.append(text[i:j])
        i = j - overlap
        if i < 0:
            i = 0
    return chunks

# ------------------------------
#  Embeddings: OpenAI or Hugging Face
# ------------------------------

def embed_batch(texts):
    # OpenAI branch (paid)
    if USE_OPENAI:
        res = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [d.embedding for d in res.data]

    # Hugging Face Inference API branch (free tier)
    if USE_HF:
        if not HF_API_TOKEN:
            raise RuntimeError("LLM_PROVIDER=hf but HF_API_TOKEN is missing in .env")

        url = f"https://api-inference.huggingface.co/models/{HF_EMBED_MODEL}"
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        # For sentence-transformers style models, sending {"inputs": [texts]} returns list-of-list embeddings
        response = requests.post(
            url,
            headers=headers,
            json={"inputs": texts}
        )
        response.raise_for_status()
        data = response.json()

        # data is usually a list of list[float], but be defensive:
        # - if single input: data may be list[float]
        # - if multiple inputs: data may be list[list[float]]
        vectors = []
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], list):
                # typical: list of embeddings
                vectors = data
            elif len(data) > 0 and isinstance(data[0], (int, float)):
                # single embedding
                vectors = [data]
        else:
            raise RuntimeError(f"Unexpected HF response format: {data}")

        return vectors

    raise RuntimeError("Unknown LLM_PROVIDER setting; expected 'openai' or 'hf'")

# ------------------------------
#  Qdrant upsert
# ------------------------------

def qdrant_upsert(chunks, vectors, meta):
    points = []
    for chunk, vec in zip(chunks, vectors):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vec,
            "payload": {
                **meta,
                "content": chunk
            }
        })

    payload = {"points": points}
    r = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        json=payload
    )
    r.raise_for_status()

# ------------------------------
#  Ingestion loop
# ------------------------------

def ingest_doc(path):
    print(f"📄 Ingesting {path}")
    # infer metadata from folder structure: data/<course>/<module>/<file>.txt
    parts = path.split(os.sep)
    meta = {
        "doc_id": os.path.splitext(os.path.basename(path))[0],
        "course_id": parts[-3] if len(parts) >= 3 else "general",
        "module_id": parts[-2] if len(parts) >= 2 else "intro",
    }

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    vectors = embed_batch(chunks)
    qdrant_upsert(chunks, vectors, meta)
    print(f"✅ Ingested {len(chunks)} chunks\n")

# ------------------------------
#  MAIN
# ------------------------------

if __name__ == "__main__":
    print("🚀 Starting ingestion")
    if USE_OPENAI:
        print("🔵 Using OpenAI embeddings")
    elif USE_HF:
        print(f"🟣 Using Hugging Face embeddings via API: {HF_EMBED_MODEL}")
    else:
        print("⚠️ Unknown LLM_PROVIDER, defaulting to HF")

    ensure_collection()

    for path in glob.glob("data/**/*.txt", recursive=True):
        ingest_doc(path)

    print("🎉 DONE")

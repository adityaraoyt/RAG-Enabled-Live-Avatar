# RAG Training Assistant – Local MVP Runbook

This project is a **local Retrieval-Augmented Generation (RAG) system** that:
- ingests training documents (PDFs, TXT)
- stores embeddings in Qdrant
- retrieves relevant content per query
- generates grounded answers using a local LLM (Ollama)
- streams responses to a React chatbot UI

This README explains **how to run everything from scratch**.

---

## Architecture Overview

```
Documents (PDF/TXT)
↓
Ingestion (Node.js)
↓
Qdrant (Vector DB)
↓
Backend API (Node + Express)
↓
Ollama (Local LLM)
↓
React UI (Streaming Chat)
```

---

## Prerequisites

### System
- macOS or Linux
- Node.js ≥ 18
- npm ≥ 9
- Docker
- Python 3.11 (optional, only for scripts/logs)

---

## 1. Start Qdrant (Vector Database)

Qdrant runs locally via Docker.

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Verify it is running:

```bash
curl http://127.0.0.1:6333
```

---

## 2. Start Ollama (Local LLM)

### Install Ollama
Download and install from: https://ollama.com

### Start Ollama
```bash
ollama serve
```

Verify:
```bash
curl http://127.0.0.1:11434/api/tags
```

Pull model:
```bash
ollama pull llama3.2:3b
```

---

## 3. Backend Setup (RAG API)

```bash
cd rag
npm install
```

### Environment (`rag/.env`)
```env
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION=training_chunks

HF_API_TOKEN=your_optional_hf_token
HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b
```

---

## 4. Ingest Training Documents

Place documents in:
```
rag/data/
```

Run:
```bash
node ingest.js
```

---

## 5. Start Backend API

```bash
node server.js
```

Health check:
```bash
curl http://localhost:3001/health
```

---

## 6. Test Backend (Optional)

```bash
curl -X POST http://localhost:3001/api/chat   -H "Content-Type: application/json"   -d '{"message":"Summarize the incident investigation process."}'
```

---

## 7. Frontend (React UI)

```bash
npm create vite@latest rag-ui -- --template react
cd rag-ui
npm install
npm run dev
```

Open:
```
http://localhost:5173
```

---

## Running Everything

| Component | Port |
|---------|------|
| Qdrant | 6333 |
| Ollama | 11434 |
| Backend | 3001 |
| React | 5173 |

---

## Notes
- Backend is stateless
- Client manages chat state
- Ready for HeyGen integration

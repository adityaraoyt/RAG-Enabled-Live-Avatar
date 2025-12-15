#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:3001}"
QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"

echo "========================================"
echo "RAG Demo Runbook (Localhost)"
echo "API     : $API_BASE"
echo "Qdrant   : $QDRANT_URL"
echo "Ollama   : $OLLAMA_URL"
echo "========================================"
echo

echo "1) Checking Qdrant..."
curl -sS "$QDRANT_URL" >/dev/null && echo "✅ Qdrant reachable" || (echo "❌ Qdrant not reachable" && exit 1)

echo "2) Checking Ollama..."
curl -sS "$OLLAMA_URL/api/tags" >/dev/null && echo "✅ Ollama reachable" || (echo "❌ Ollama not reachable" && exit 1)

echo "3) Checking API health..."
curl -sS "$API_BASE/health" | sed 's/^/   /'
echo

echo "4) Checking Ollama via API (/debug/ollama)..."
curl -sS "$API_BASE/debug/ollama" | head -c 500 | sed 's/^/   /'
echo
echo

echo "========================================"
echo "DEMO A: Non-streaming RAG answer"
echo "========================================"

curl -X POST "$API_BASE/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"Summarize the incident investigation process."}'

echo



echo "========================================"
echo "DEMO B: Streaming RAG answer (live tokens)"
echo "========================================"
echo "(You should see text appear gradually)"
echo
curl -N -sS -X POST "$API_BASE/api/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"message":"Summarize the incident investigation process in 5 bullets."}'
echo
echo

echo "========================================"
echo "DEMO C: Out-of-scope test (hallucination guardrail)"
echo "========================================"

curl -X POST http://localhost:3001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is our company PTO policy?"}'

echo
echo "✅ Demo complete."


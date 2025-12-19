import "dotenv/config";
import express from "express";
import cors from "cors";
import axios from "axios";
import { InferenceClient } from "@huggingface/inference";

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

// ENV
const QDRANT_URL = process.env.QDRANT_URL || "http://127.0.0.1:6333";
const QDRANT_COLLECTION = process.env.QDRANT_COLLECTION || "training_chunks";

const HF_API_TOKEN = process.env.HF_API_TOKEN;
const HF_EMBED_MODEL =
  process.env.HF_EMBED_MODEL || "sentence-transformers/all-MiniLM-L6-v2";

const hf = new InferenceClient(HF_API_TOKEN);

function normalizeHistory(history) {
  if (!Array.isArray(history)) return [];
  return history
    .filter(
      (m) =>
        m &&
        (m.role === "user" || m.role === "assistant") &&
        typeof m.content === "string"
    )
    .map((m) => ({ role: m.role, content: m.content.slice(0, 4000) }));
}

async function rewriteForRetrieval({ message, history }) {
  if (!history || history.length === 0) return message;

  const convo = history
    .map((m) => `${m.role.toUpperCase()}: ${m.content}`)
    .join("\n");

  const prompt = `
Rewrite the LAST user message into a standalone question that can be understood without the conversation.
Do NOT answer the question. Keep it short and specific.

CONVERSATION:
${convo}

LAST USER MESSAGE:
${message}

Standalone question:
`.trim();

  const resp = await axios.post(`${OLLAMA_URL}/api/generate`, {
    model: OLLAMA_MODEL,
    prompt,
    stream: false,
    options: { temperature: 0 },
  });

  return (resp.data.response || message).trim();
}


// --- Embed user query with same embedder you used for ingestion ---
async function embedQuery(text) {
  const result = await hf.featureExtraction({
    model: HF_EMBED_MODEL,
    inputs: text,
    provider: "hf-inference",
  });

  // normalize to 1D vector
  if (Array.isArray(result) && Array.isArray(result[0])) return result[0];
  if (Array.isArray(result) && typeof result[0] === "number") return result;
  throw new Error("Unexpected embedding format from HF");
}

// --- Search Qdrant ---
async function qdrantSearch(vector, limit = 5) {
  const res = await axios.post(
    `${QDRANT_URL}/collections/${QDRANT_COLLECTION}/points/search`,
    {
      vector,
      limit,
      with_payload: true,
    }
  );
  return res.data.result || [];
}

// --- Simple LLM stub (replace with your provider) ---
const OLLAMA_URL = process.env.OLLAMA_URL || "http://127.0.0.1:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "llama3.2:3b";

async function callLLM({ question, context }) {
  const system = `
You are an enterprise training instructor.

You MUST rely on the provided CONTEXT for training-related questions.
If a training-related answer is not in the CONTEXT, say:
"I don’t have that in the training materials provided."

You MAY respond naturally to simple conversational messages such as greetings, acknowledgements, or closings (for example: "hi", "okay", "thanks") without applying training constraints.

Response guidelines:
- Keep the answer brief.
- Short or conversational inputs should receive brief, natural replies.
- Training or procedural answers must be clear, structured, and actionable.
- When asked about steps or actions, always prioritize IMMEDIATE safety actions first.
- Do not reorder steps unless explicitly instructed.
- Answer ONLY what the question asks.
- Do not add background unless it directly supports the answer.
- Do not use symbols, numbering, headings, or labels.
- Do not include citations, references, or source markers.
- Use plain, natural spoken language suitable for audio delivery.
`.trim();

  const prompt = `
CONTEXT:
${context}

QUESTION:
${question}
`.trim();


  const resp = await axios.post(`${OLLAMA_URL}/api/generate`, {
    model: OLLAMA_MODEL,
    system,
    prompt,
    stream: false,
    options: {
      temperature: 0.2,
      num_ctx: 4096
    },
  });

  return (resp.data.response || "").trim();
}

app.post("/api/chat", async (req, res) => {
  try {
    const { message } = req.body;
    const history = normalizeHistory(req.body.history);

    const retrievalQuery = await rewriteForRetrieval({ message, history });
    console.log("🔁 Retrieval query:", retrievalQuery);

    const queryVec = await embedQuery(retrievalQuery);
    const hits = await qdrantSearch(queryVec, 8);


    const MAX_CHARS_PER_CHUNK = 1200;
    const context = hits
      .map((h, idx) => {
        const p = h.payload || {};
        const src = `${p.source_path || p.doc_id || "unknown"}${p.page_number ? ` (page ${p.page_number})` : ""
          }`;
        const content = (p.content || "").slice(0, MAX_CHARS_PER_CHUNK);
        return `[#${idx + 1}] ${src}\n${content}`;
      })
      .join("\n\n");

    const conversation = history
      .map((m) => `${m.role.toUpperCase()}: ${m.content}`)
      .join("\n");

    const answer = await callLLM({
      question: `${conversation}\nUSER: ${message}`,
      context,
    });

    res.json({ answer, citations: hits });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


app.get("/health", async (_req, res) => res.json({ ok: true }));

app.get("/debug/ollama", async (_req, res) => {
  try {
    const r = await axios.get(`${OLLAMA_URL}/api/tags`);
    res.json({ ok: true, tags: r.data });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.post("/api/chat/stream", async (req, res) => {
  try {
    const { message } = req.body;
    if (!message || typeof message !== "string") {
      return res.status(400).json({ error: "message is required" });
    }

    const history = normalizeHistory(req.body.history);

    // SSE headers
    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders();

    // 0) Rewrite follow-up into standalone query for better retrieval
    const retrievalQuery = await rewriteForRetrieval({ message, history });
    console.log("🔁 Retrieval query (stream):", retrievalQuery);

    // 1) Embed rewritten query
    const queryVec = await embedQuery(retrievalQuery);

    // 2) Retrieve context from Qdrant
    const hits = await qdrantSearch(queryVec, 5);

    const MAX_CHARS_PER_CHUNK = 1200;
    const context = hits
      .map((h, idx) => {
        const p = h.payload || {};
        const src = `${p.source_path || p.doc_id || "unknown"}${p.page_number ? ` (page ${p.page_number})` : ""
          }`;
        const content = (p.content || "").slice(0, MAX_CHARS_PER_CHUNK);
        return `[#${idx + 1}] ${src}\n${content}`;
      })
      .join("\n\n");

    // 3) Build conversation string for memory
    const conversation = history
      .map((m) => `${m.role.toUpperCase()}: ${m.content}`)
      .join("\n");

    // 4) Build prompt for Ollama
    // Use YOUR callLLM rules, but streaming version
    const system = `
You are an enterprise training instructor.

You MUST rely on the provided CONTEXT for training-related questions.
If a training-related answer is not in the CONTEXT, say:
"I don’t have that in the training materials provided."

You MAY respond naturally to simple conversational messages such as greetings, acknowledgements, or closings (for example: "hi", "okay", "thanks") without applying training constraints.

Response guidelines:
- Keep the answer brief.
- Short or conversational inputs should receive brief, natural replies.
- Training or procedural answers must be clear, structured, and actionable.
- When asked about steps or actions, always prioritize IMMEDIATE safety actions first.
- Do not reorder steps unless explicitly instructed.
- Answer ONLY what the question asks.
- Do not add background unless it directly supports the answer.
- Do not use symbols, numbering, headings, or labels.
- Do not include citations, references, or source markers.
- Use plain, natural spoken language suitable for audio delivery.
`.trim();

    const prompt = `
CONTEXT:
${context}

CONVERSATION SO FAR:
${conversation}

USER:
${message}
`.trim();

    // 5) Stream from Ollama -> convert to SSE
    const ollamaResp = await axios.post(
      `${OLLAMA_URL}/api/generate`,
      {
        model: OLLAMA_MODEL,
        system,
        prompt,
        stream: true,
        options: { temperature: 0.2, num_ctx: 4096 },
      },
      { responseType: "stream" }
    );

    // IMPORTANT: Ollama streams NDJSON (one JSON per line)
    // Lines can arrive fragmented -> buffer them
    let buf = "";

    ollamaResp.data.on("data", (chunk) => {
      buf += chunk.toString("utf8");

      const lines = buf.split("\n");
      buf = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        let json;
        try {
          json = JSON.parse(trimmed);
        } catch {
          // ignore broken partial line (should be rare since we buffer)
          continue;
        }

        if (json.response) {
          // Send as SSE data chunks
          res.write(`data: ${json.response}\n\n`);
        }

        if (json.done) {
          // Optional: send citations once at end as a single SSE data message
          // res.write(`event: citations\ndata: ${JSON.stringify(hits)}\n\n`);

          res.write(`data: done\n\n`);
          res.end();
        }
      }
    });

    ollamaResp.data.on("error", (err) => {
      console.error("Ollama stream error", err);
      res.write(`event: error\ndata: Ollama stream error\n\n`);
      res.end();
    });

    ollamaResp.data.on("end", () => {
      // If Ollama ends unexpectedly, still close SSE cleanly
      if (!res.writableEnded) {
        res.write(`data: done\n\n`);
        res.end();
      }
    });
  } catch (err) {
    console.error(err);
    try {
      res.write(`event: error\ndata: ${err.message}\n\n`);
    } finally {
      res.end();
    }
  }
});


console.log("🔥 server.js loaded at", new Date().toISOString());
console.log("Using Ollama model:", OLLAMA_MODEL);

app.listen(3001, () => {
  console.log("✅ API running on http://localhost:3001");
});

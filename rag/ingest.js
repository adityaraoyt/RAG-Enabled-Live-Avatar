import "dotenv/config";
import { InferenceClient } from "@huggingface/inference";
import axios from "axios";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";


//
// ENVIRONMENT
//
const HF_API_TOKEN = process.env.HF_API_TOKEN;
const HF_EMBED_MODEL =
  process.env.HF_EMBED_MODEL || "sentence-transformers/all-MiniLM-L6-v2";
const EMBED_DIMS_INT = parseInt(process.env.EMBED_DIMS || "384", 10);

const QDRANT_URL = process.env.QDRANT_URL || "http://localhost:6333";
const QDRANT_COLLECTION =
  process.env.QDRANT_COLLECTION || "training_chunks";

const hfClient = new InferenceClient(HF_API_TOKEN);

//
// HELPERS
//
function chunkText(text) {
  // normalize whitespace
  const cleaned = text.replace(/\s+/g, " ").trim();

  // SIMPLE MVP: just return the whole doc as a single chunk
  if (!cleaned) return [];
  return [cleaned];
}

async function extractPagesFromPdf(filePath) {
  const data = new Uint8Array(fs.readFileSync(filePath));
  const loadingTask = pdfjsLib.getDocument({ data });
  const pdf = await loadingTask.promise;

  const pages = [];

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);
    const content = await page.getTextContent();

    const pageText = content.items
      .map((item) => (item.str ? item.str : ""))
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();

    if (pageText) pages.push({ pageNum, text: pageText });
  }

  return pages;
}

async function extractTextFromFile(filePath) {
  const ext = path.extname(filePath).toLowerCase();

  if (ext === ".txt") {
    return fs.readFileSync(filePath, "utf8");
  }

  if (ext === ".pdf") {
    return await extractTextFromPdf(filePath);
  }

  console.warn(`Skipping unsupported file type: ${filePath}`);
  return "";
}



//
// HuggingFace embeddings
//
async function embedBatch(texts) {
  const result = await hfClient.featureExtraction({
    model: HF_EMBED_MODEL,
    inputs: texts, // HF client supports batched input
    provider: "hf-inference",
  });

  let vectors = [];

  // Case: [[float,...],[float,...]]
  if (Array.isArray(result) && Array.isArray(result[0])) {
    vectors = result;
  }

  // Case: [float,...] (single embedding)
  else if (Array.isArray(result) && typeof result[0] === "number") {
    vectors = [result];
  } else {
    throw new Error(
      "Unexpected embedding format: " + JSON.stringify(result).slice(0, 200)
    );
  }

  return vectors;
}

//
// QDRANT
//
async function ensureCollection() {
  const res = await axios.get(`${QDRANT_URL}/collections`);
  const collections = res.data.result.collections.map((c) => c.name);

  if (!collections.includes(QDRANT_COLLECTION)) {
    console.log(`Creating collection ${QDRANT_COLLECTION}...`);
    await axios.put(`${QDRANT_URL}/collections/${QDRANT_COLLECTION}`, {
      vectors: { size: EMBED_DIMS_INT, distance: "Cosine" },
    });
    console.log("Created.");
  } else {
    console.log(`Collection '${QDRANT_COLLECTION}' exists.`);
  }
}

async function upsert(chunks, vectors, meta, chunkMeta = []) {
  const points = chunks.map((chunk, i) => ({
    id: crypto.randomUUID(),
    vector: vectors[i],
    payload: { ...meta, ...(chunkMeta[i] || {}), content: chunk },
  }));

  await axios.put(
    `${QDRANT_URL}/collections/${QDRANT_COLLECTION}/points`,
    { points },
    { headers: { "Content-Type": "application/json" } }
  );
}


//
// MAIN INGEST FILE FUNCTION
//
async function ingestFile(filePath) {
  console.log(`📄 Ingesting ${filePath}`);

  const ext = path.extname(filePath).toLowerCase();

  let chunks = [];
  let chunkMeta = []; // per-chunk metadata (e.g., page_number)

  if (ext === ".pdf") {
    const pages = await extractPagesFromPdf(filePath);

    chunks = pages.map((p) => p.text);
    chunkMeta = pages.map((p) => ({ page_number: p.pageNum }));
  } else if (ext === ".txt") {
    const text = fs.readFileSync(filePath, "utf8").replace(/\s+/g, " ").trim();
    if (text) {
      chunks = [text];
      chunkMeta = [{}];
    }
  } else {
    console.warn(`Skipping unsupported file type: ${filePath}`);
    return;
  }

  if (chunks.length === 0) {
    console.warn(`No text extracted from ${filePath}, skipping.`);
    return;
  }

  // Embeddings
  const vectors = await embedBatch(chunks);

  console.log(`Chunks: ${chunks.length}`);
  console.log(
    `First embedding length: ${vectors[0]?.length}, expected: ${EMBED_DIMS_INT}`
  );

  // Metadata inferred from data/<course>/<module>/<file>
  const relParts = path.relative("data", filePath).split(path.sep);
  const meta = {
    doc_id: path.basename(filePath, ext),
    course_id: relParts[0] || "general",
    module_id: relParts[1] || "intro",
    file_type: ext.slice(1), // "pdf" or "txt"
    source_path: relParts.join("/"),
  };

  // Upsert into Qdrant (upsert must merge chunkMeta[i] into payload)
  await upsert(chunks, vectors, meta, chunkMeta);

  console.log(`✅ Ingested ${chunks.length} chunks\n`);
}


//
// RECURSIVE FIND
//
function findDocs(dir) {
  let out = [];
  for (const f of fs.readdirSync(dir)) {
    const full = path.join(dir, f);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) {
      out = out.concat(findDocs(full));
    } else if (stat.isFile()) {
      const ext = path.extname(full).toLowerCase();
      if (ext === ".txt" || ext === ".pdf") {
        out.push(full);
      }
    }
  }
  return out;
}

//
// RUN
//
(async () => {
  console.log("🚀 Starting ingestion");

  await ensureCollection();

  const files = findDocs(path.join(process.cwd(), "data"));


  for (const f of files) {
    await ingestFile(f);
  }

  console.log("🎉 DONE");
})();

import 'dotenv/config';
import { InferenceClient } from '@huggingface/inference';

const HF_API_TOKEN = process.env.HF_API_TOKEN;
const HF_EMBED_MODEL =
  process.env.HF_EMBED_MODEL || 'sentence-transformers/all-MiniLM-L6-v2';

if (!HF_API_TOKEN) {
  console.error('HF_API_TOKEN missing in .env');
  process.exit(1);
}

const client = new InferenceClient(HF_API_TOKEN);

(async () => {
  try {
    console.log('Using model id:', HF_EMBED_MODEL);

    const result = await client.featureExtraction({
      model: HF_EMBED_MODEL,
      inputs: 'hello world',
      provider: 'hf-inference', // this is the new “HF Inference” provider
    });

    // result is usually [ [float, float, ...] ]
    let emb;
    if (Array.isArray(result) && Array.isArray(result[0])) {
      emb = result[0];
    } else if (Array.isArray(result) && typeof result[0] === 'number') {
      emb = result;
    } else {
      throw new Error(
        'Unexpected HF embedding format: ' + JSON.stringify(result).slice(0, 200),
      );
    }

    console.log('Embedding length:', emb.length);
  } catch (err) {
    console.error('HF error:', err);
  }
})();

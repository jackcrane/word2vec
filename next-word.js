import { Dict } from "./dictify.js";
import { extractVectors, cosine } from "./similarity.js";

const MODEL_PATH = process.env.MODEL ?? "model.dat";
const [, , ...rawContext] = process.argv;

const toNonNegativeInt = (value, fallback) => {
  const n = Number(value);
  return Number.isFinite(n) && n >= 0 ? Math.floor(n) : fallback;
};

const MAX_CONTEXT = toNonNegativeInt(process.env.MAX_CONTEXT, 12);
const DELAY_MS = toNonNegativeInt(process.env.DELAY_MS, 300);
const MAX_STEPS_RAW = Number(process.env.MAX_STEPS);
const MAX_STEPS =
  Number.isFinite(MAX_STEPS_RAW) && MAX_STEPS_RAW > 0
    ? Math.floor(MAX_STEPS_RAW)
    : Infinity;
const ALLOW_REPEAT =
  String(process.env.ALLOW_REPEAT ?? "").toLowerCase() === "true" ||
  String(process.env.ALLOW_REPEAT ?? "").toLowerCase() === "1";

process.on("SIGINT", () => {
  process.stdout.write("\n");
  process.exit(0);
});

const sanitizeTokens = (tokens) => {
  const out = [];
  for (const token of tokens) {
    if (!token) continue;
    const lower = token.toLowerCase().replace(/[^a-z-]/g, "");
    if (!lower) continue;
    for (const piece of lower.split("-")) {
      if (piece) out.push(piece);
    }
  }
  return out;
};

const usage = () => {
  console.error(
    [
      "Usage: node next-word.js <word1> <word2> ...",
      "Provide at least one valid word to seed the generator.",
    ].join("\n")
  );
  process.exit(1);
};

const initialTokens = sanitizeTokens(rawContext);
if (initialTokens.length === 0) usage();

const dict = new Dict();
dict.load(MODEL_PATH);

const { keys, vecs } = extractVectors(dict);
if (keys.length === 0) {
  console.error(`Model "${MODEL_PATH}" is empty or missing vectors.`);
  process.exit(1);
}

const vectorByWord = new Map();
for (let i = 0; i < keys.length; i++) {
  vectorByWord.set(keys[i], vecs[i]);
}

let context = [];
for (const token of initialTokens) {
  if (vectorByWord.has(token)) {
    context.push(token);
  } else {
    console.warn(`Skipping "${token}" (no vector in model)`);
  }
}

if (context.length === 0) {
  console.error("None of the provided context words are in the model.");
  process.exit(1);
}

const dimension = vecs[0].length;

const delay = async (ms) =>
  new Promise((resolve) => setTimeout(resolve, Math.max(0, ms)));

const buildMeanUnitVector = (words) => {
  const out = new Float32Array(dimension);
  let count = 0;
  for (const word of words) {
    const vec = vectorByWord.get(word);
    if (!vec || vec.length !== dimension) continue;
    count++;
    for (let i = 0; i < dimension; i++) out[i] += vec[i];
  }
  if (count === 0) return null;
  for (let i = 0; i < dimension; i++) out[i] /= count;
  let norm = 0;
  for (let i = 0; i < dimension; i++) norm += out[i] * out[i];
  norm = Math.sqrt(norm);
  if (!Number.isFinite(norm) || norm === 0) return null;
  for (let i = 0; i < dimension; i++) out[i] /= norm;
  return out;
};

const scoreCandidates = (contextVec, exclude) => {
  const scored = [];
  for (let i = 0; i < keys.length; i++) {
    if (exclude && exclude.has(keys[i])) continue;
    const score = cosine(contextVec, vecs[i]);
    if (!Number.isNaN(score)) {
      scored.push({ word: keys[i], score });
    }
  }
  scored.sort((a, b) => b.score - a.score);
  return scored;
};

const main = async () => {
  console.log("Press Ctrl+C to stop generation.");

  let step = 0;
  const history = context.slice();

  if (history.length > 0) {
    process.stdout.write(history.join(" "));
  }

  while (step < MAX_STEPS) {
    const window =
      MAX_CONTEXT > 0 ? history.slice(-MAX_CONTEXT) : history.slice();
    const ctxVec = buildMeanUnitVector(window);
    if (!ctxVec) {
      process.stdout.write("\n");
      console.error("Unable to build context vector. Stopping.");
      break;
    }

    const exclude = ALLOW_REPEAT ? null : new Set(window);
    const scored = scoreCandidates(ctxVec, exclude);
    if (scored.length === 0) {
      console.error("\nNo candidates could be scored. Stopping.");
      break;
    }

    const nextWord = scored[0].word;
    process.stdout.write(` ${nextWord}`);

    history.push(nextWord);
    step++;

    if (step >= MAX_STEPS) break;

    if (DELAY_MS > 0) {
      await delay(DELAY_MS);
    }
  }
  process.stdout.write("\n");
};

await main().catch((err) => {
  console.error(err);
  process.exit(1);
});

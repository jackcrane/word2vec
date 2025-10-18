import fs from "fs";
import readline from "readline";
import ProgressBar from "progress";
import { Dict } from "./dictify.js";
import { Timer } from "./timer.js";

const SOURCE = process.env.SOURCE ?? "test.txt";
const OUTPUT = process.env.OUTPUT ?? "model.dat";
const EMBED_DIM = Number(process.env.EMBED_DIM ?? 100);
const WINDOW_SIZE = Number(process.env.WINDOW_SIZE ?? 5);
const NEGATIVE_SAMPLES = Number(process.env.NEGATIVE_SAMPLES ?? 5);
const MIN_COUNT = Number(process.env.MIN_COUNT ?? 1);
const EPOCHS = Number(process.env.EPOCHS ?? 10);
const INITIAL_LR = Number(process.env.LEARNING_RATE ?? 0.025);
const MIN_LR =
  Number.isFinite(Number(process.env.MIN_LEARNING_RATE))
    ? Number(process.env.MIN_LEARNING_RATE)
    : INITIAL_LR * 0.001;
const SUBSAMPLE_THRESHOLD = Number(process.env.SUBSAMPLE ?? 0);

if (EMBED_DIM <= 0 || !Number.isFinite(EMBED_DIM)) {
  throw new Error("EMBED_DIM must be a positive finite number");
}

const timer = new Timer();

const sanitizeWords = (line) => {
  if (!line) return [];
  const parts = line.split(/\s+/);
  const clean = [];
  for (const raw of parts) {
    if (!raw) continue;
    const lower = raw.toLowerCase().replace(/[^a-z-]/g, "");
    if (!lower) continue;
    const pieces = lower.split("-");
    for (const piece of pieces) {
      if (piece) clean.push(piece);
    }
  }
  return clean;
};

const iterateCorpus = async (handler) => {
  if (!fs.existsSync(SOURCE)) {
    throw new Error(`Corpus file "${SOURCE}" not found`);
  }

  const rl = readline.createInterface({
    input: fs.createReadStream(SOURCE, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    const tokens = sanitizeWords(line);
    if (tokens.length === 0) continue;
    // Allow handler to be synchronous or async.
    await handler(tokens);
  }
};

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

const binarySearch = (arr, value) => {
  let lo = 0;
  let hi = arr.length - 1;
  while (lo < hi) {
    const mid = lo + Math.floor((hi - lo) / 2);
    if (value <= arr[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
};

const initializeEmbeddings = (vocabSize, dim) => {
  const scale = 1 / Math.sqrt(dim);
  const input = new Float32Array(vocabSize * dim);
  const output = new Float32Array(vocabSize * dim);
  for (let i = 0; i < input.length; i++) {
    input[i] = (Math.random() - 0.5) * 2 * scale;
  }
  // output starts at zero vector; will be learned during training.
  return { input, output };
};

const normalizeVector = (vec) => {
  let sumSq = 0;
  for (const v of vec) sumSq += v * v;
  const norm = Math.sqrt(sumSq);
  if (!Number.isFinite(norm) || norm === 0) return vec;
  return vec.map((v) => v / norm);
};

const main = async () => {
  console.log(`Training word2vec on "${SOURCE}"`);

  // Pass 1: build vocabulary with counts.
  const counts = new Map();
  let totalTokens = 0;
  await iterateCorpus((tokens) => {
    for (const token of tokens) {
      totalTokens++;
      counts.set(token, (counts.get(token) ?? 0) + 1);
    }
  });
  timer.log("Vocabulary collected");

  const vocabEntries = [];
  for (const [word, count] of counts.entries()) {
    if (count >= MIN_COUNT) {
      vocabEntries.push({ word, count });
    }
  }

  if (vocabEntries.length === 0) {
    throw new Error(
      `No vocabulary entries survived MIN_COUNT=${MIN_COUNT}. Lower MIN_COUNT or check corpus.`
    );
  }

  const vocabSize = vocabEntries.length;
  const wordToIndex = new Map();
  vocabEntries.forEach((entry, index) => {
    wordToIndex.set(entry.word, index);
  });

  const totalTrainTokens = vocabEntries.reduce(
    (sum, entry) => sum + entry.count,
    0
  );

  // Subsampling probabilities (optional).
  const discardProb = new Float32Array(vocabSize);
  if (SUBSAMPLE_THRESHOLD > 0) {
    for (let i = 0; i < vocabEntries.length; i++) {
      const freq = vocabEntries[i].count / totalTrainTokens;
      const prob = Math.max(0, 1 - Math.sqrt(SUBSAMPLE_THRESHOLD / freq));
      discardProb[i] = Math.min(1, prob);
    }
  }

  // Negative sampling distribution.
  const cumulative = new Float64Array(vocabSize);
  const power = 0.75;
  let cumulativeTotal = 0;
  for (let i = 0; i < vocabEntries.length; i++) {
    cumulativeTotal += Math.pow(vocabEntries[i].count, power);
    cumulative[i] = cumulativeTotal;
  }

  const sampleNegative = (centerIdx, positiveIdx) => {
    if (vocabSize <= 1) return -1;
    let candidate = -1;
    let attempts = 0;
    do {
      const r = Math.random() * cumulativeTotal;
      candidate = binarySearch(cumulative, r);
      attempts++;
      if (attempts > 6) break;
    } while (candidate === centerIdx || candidate === positiveIdx);
    return candidate;
  };

  const { input, output } = initializeEmbeddings(vocabSize, EMBED_DIM);

  const totalSteps = totalTrainTokens * EPOCHS;
  let processedWords = 0;
  const lrFloor = Math.max(MIN_LR, INITIAL_LR * 0.0001);
  const currentLearningRate = () => {
    if (totalSteps === 0) return INITIAL_LR;
    const progress = processedWords / totalSteps;
    const decayed = INITIAL_LR * (1 - progress);
    return Math.max(lrFloor, decayed);
  };

  console.log(
    `Vocabulary size: ${vocabSize}, tokens: ${totalTrainTokens}, dimensions: ${EMBED_DIM}`
  );

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    const bar = new ProgressBar(
      `Epoch ${epoch + 1}/${EPOCHS} [:bar] :percent :etas`,
      {
        complete: "=",
        incomplete: " ",
        width: 30,
        total: totalTrainTokens,
      }
    );

    await iterateCorpus((tokens) => {
      const indices = [];
      for (const token of tokens) {
        const idx = wordToIndex.get(token);
        if (idx === undefined) continue;
        if (
          SUBSAMPLE_THRESHOLD > 0 &&
          Math.random() < discardProb[idx]
        ) {
          continue;
        }
        indices.push(idx);
      }

      if (indices.length === 0) return;

      for (let pos = 0; pos < indices.length; pos++) {
        const centerIdx = indices[pos];
        const lr = currentLearningRate();
        const maxWindow =
          WINDOW_SIZE > 1
            ? 1 + Math.floor(Math.random() * WINDOW_SIZE)
            : WINDOW_SIZE;
        const left = Math.max(0, pos - maxWindow);
        const right = Math.min(indices.length - 1, pos + maxWindow);

        for (let ctx = left; ctx <= right; ctx++) {
          if (ctx === pos) continue;
          const contextIdx = indices[ctx];
          trainPair(centerIdx, contextIdx, 1, lr, input, output);
          for (let n = 0; n < NEGATIVE_SAMPLES; n++) {
            const negIdx = sampleNegative(centerIdx, contextIdx);
            if (negIdx < 0) continue;
            trainPair(centerIdx, negIdx, 0, lr, input, output);
          }
        }

        processedWords++;
        bar.tick();
      }
    });

    timer.log(`Epoch ${epoch + 1} complete`);
  }

  const dict = new Dict();
  for (let i = 0; i < vocabEntries.length; i++) {
    const merged = new Array(EMBED_DIM);
    const base = i * EMBED_DIM;
    for (let d = 0; d < EMBED_DIM; d++) {
      merged[d] = input[base + d] + output[base + d];
    }
    dict.set(vocabEntries[i].word, normalizeVector(merged));
  }

  await dict.flush(OUTPUT);
  timer.log("Training complete");
  console.log(`Model written to ${OUTPUT}`);
};

const trainPair = (centerIdx, contextIdx, label, lr, input, output) => {
  const baseCenter = centerIdx * EMBED_DIM;
  const baseContext = contextIdx * EMBED_DIM;

  let dot = 0;
  for (let d = 0; d < EMBED_DIM; d++) {
    dot += input[baseCenter + d] * output[baseContext + d];
  }

  const prediction = sigmoid(dot);
  const grad = (label - prediction) * lr;

  for (let d = 0; d < EMBED_DIM; d++) {
    const centerVal = input[baseCenter + d];
    const contextVal = output[baseContext + d];
    input[baseCenter + d] = centerVal + grad * contextVal;
    output[baseContext + d] = contextVal + grad * centerVal;
  }
};

await main().catch((err) => {
  console.error(err);
  process.exit(1);
});

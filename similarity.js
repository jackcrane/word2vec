// similarity.js
// ES modules, named exports, arrow functions. Ready to paste.

/** -------- basic vector math -------- */
export const dot = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

export const l2 = (a) => Math.sqrt(dot(a, a));

export const cosine = (a, b) => {
  if (a.length !== b.length) return NaN;
  const denom = l2(a) * l2(b);
  return denom === 0 ? NaN : dot(a, b) / denom;
};

export const euclidean = (a, b) => {
  if (a.length !== b.length) return NaN;
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
};

export const normalizeVec = (a) => {
  const n = l2(a);
  if (!Number.isFinite(n) || n === 0) return null;
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] / n;
  return out;
};

/** -------- Dict helpers -------- */
export const extractVectors = (dict) => {
  const keys = [];
  const vecs = [];
  for (const [key, value] of dict.entries()) {
    if (
      Array.isArray(value) &&
      value.length > 0 &&
      value.every((v) => Number.isFinite(v))
    ) {
      keys.push(key);
      vecs.push(value);
    }
  }
  return { keys, vecs };
};

export const getVector = (dict, word) => {
  const v = dict.get(word);
  if (!Array.isArray(v) || v.length === 0 || !v.every(Number.isFinite)) {
    throw new Error(`No valid vector for "${word}"`);
  }
  return v;
};

/** v(b) - v(a) */
export const differenceVector = (dict, a, b) => {
  const va = getVector(dict, a);
  const vb = getVector(dict, b);
  if (va.length !== vb.length) throw new Error("Mismatched dimensions");
  const out = new Float32Array(va.length);
  for (let i = 0; i < va.length; i++) out[i] = vb[i] - va[i];
  return out;
};

/** -------- packing / hubness reduction -------- */
export const packVectorsMeanCentered = (dict) => {
  const { keys, vecs } = extractVectors(dict);
  const n = vecs.length;
  const d = n ? vecs[0].length : 0;

  const mean = new Float32Array(d);
  for (let i = 0; i < n; i++) {
    const row = vecs[i];
    for (let k = 0; k < d; k++) mean[k] += row[k];
  }
  for (let k = 0; k < d; k++) mean[k] /= n || 1;

  const data = new Float32Array(n * d);
  const unitData = new Float32Array(n * d);

  for (let i = 0; i < n; i++) {
    const row = vecs[i];
    const base = i * d;
    let s = 0;
    for (let k = 0; k < d; k++) {
      const v = row[k] - mean[k];
      data[base + k] = v;
      s += v * v;
    }
    const norm = Math.sqrt(s) || 1;
    const inv = 1 / norm;
    for (let k = 0; k < d; k++) unitData[base + k] = data[base + k] * inv;
  }
  return { keys, data, unitData, n, d };
};

/** Cosine between (row j - row i) and a unit direction u (length d) */
export const cosineOfDifferenceToDirection = (data, d, i, j, u) => {
  let num = 0;
  let denom2 = 0;
  const baseI = i * d;
  const baseJ = j * d;
  for (let k = 0; k < d; k++) {
    const diff = data[baseJ + k] - data[baseI + k];
    num += diff * u[k];
    denom2 += diff * diff;
  }
  const denom = Math.sqrt(denom2);
  return denom === 0 ? NaN : num / denom;
};

/** -------- percentile utilities -------- */
/** Percentile cut (0..0.5 → bottom/top bands). Guarantees >=1 idx per side. */
export const percentileCut = (arr, p /* 0..1, clamped to 0..0.5 */) => {
  const n = arr.length;
  const pClamped = Math.max(0, Math.min(0.5, p));
  const m = Math.max(1, Math.floor(n * pClamped));
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((i, j) => arr[i] - arr[j]); // ascending
  const lowsIdx = idx.slice(0, m);
  const highsIdx = idx.slice(n - m);
  const lowThresh = arr[idx[m - 1]];
  const highThresh = arr[idx[n - m]];
  return { lowsIdx, highsIdx, lowThresh, highThresh };
};

/** -------- stopwords -------- */
export const buildStopwordSet = () => {
  const words = [
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "for",
    "nor",
    "so",
    "yet",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "into",
    "about",
    "as",
    "per",
    "is",
    "am",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "doing",
    "has",
    "have",
    "had",
    "having",
    "will",
    "would",
    "shall",
    "should",
    "can",
    "could",
    "may",
    "might",
    "must",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "hers",
    "its",
    "our",
    "their",
    "this",
    "that",
    "these",
    "those",
    "here",
    "there",
    "where",
    "when",
    "why",
    "how",
    "not",
    "no",
    "yes",
    "up",
    "down",
    "out",
    "over",
    "under",
    "again",
    "further",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "than",
    "too",
    "very",
    "just",
    "only",
    "own",
    "same",
    "also",
    "ever",
    "never",
  ];
  const s = new Set();
  for (const w of words) s.add(w);
  return s;
};

/** -------- nearest neighbors --------
 * Finds top-k neighbors by cosine similarity for each query word.
 * Options: words?: string[]; k?: number (default 5); minSim?: number (default 0)
 */
export const nearestNeighbors = (dict, opts = {}) => {
  const { words = null, k = 5, minSim = 0 } = opts;
  const { keys, vecs } = extractVectors(dict);
  const index = new Map(keys.map((kk, i) => [kk, i]));
  const queries = words ?? keys;

  const results = [];
  for (const q of queries) {
    const qi = index.get(q);
    if (qi === undefined) continue;
    const qv = vecs[qi];

    const sims = [];
    for (let j = 0; j < vecs.length; j++) {
      if (j === qi) continue;
      const s = cosine(qv, vecs[j]);
      if (!Number.isNaN(s) && s >= minSim) sims.push({ key: keys[j], sim: s });
    }
    sims.sort((a, b) => b.sim - a.sim);
    results.push({ query: q, neighbors: sims.slice(0, k) });
  }
  return results;
};

/** -------- global close pairs --------
 * metric: 'cosine' | 'euclidean' (default 'cosine')
 * threshold: cosine >= t OR euclidean <= t
 * limit: max pairs returned (default Infinity)
 */
export const findClosePairs = (dict, opts = {}) => {
  const {
    metric = "cosine",
    threshold = metric === "cosine" ? 0.8 : 0.5,
    limit = Infinity,
  } = opts;

  const { keys, vecs } = extractVectors(dict);
  const useCosine = metric === "cosine";

  const out = [];
  for (let i = 0; i < vecs.length; i++) {
    for (let j = i + 1; j < vecs.length; j++) {
      const a = vecs[i];
      const b = vecs[j];
      if (a.length !== b.length) continue;

      if (useCosine) {
        const s = cosine(a, b);
        if (!Number.isNaN(s) && s >= threshold) {
          out.push({ a: keys[i], b: keys[j], score: s });
        }
      } else {
        const d = euclidean(a, b);
        if (!Number.isNaN(d) && d <= threshold) {
          out.push({ a: keys[i], b: keys[j], score: d });
        }
      }
    }
  }

  out.sort((x, y) => (useCosine ? y.score - x.score : x.score - y.score));
  return out.slice(0, limit);
};

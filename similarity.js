// similarity.js
// ES modules, named exports, arrow functions. Ready to paste.

/** -------- vector math -------- */
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

/** -------- extract from Dict --------
 * Accepts your Dict instance and returns parallel arrays
 * filtered to entries whose values are numeric arrays (vectors).
 */
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

/** -------- nearest neighbors for selected items --------
 * Finds the top-k closest neighbors for each target word (by cosine similarity).
 * Options:
 *   words?: string[]   — limit search to these query words; omit for "all"
 *   k?: number         — neighbors per word (default 5)
 *   minSim?: number    — minimum cosine similarity to keep (default 0)
 */
export const nearestNeighbors = (dict, opts = {}) => {
  const { words = null, k = 5, minSim = 0 } = opts;
  const { keys, vecs } = extractVectors(dict);

  const index = new Map(keys.map((k, i) => [k, i]));
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
    results.push({
      query: q,
      neighbors: sims.slice(0, k),
    });
  }
  return results;
};

/** -------- global close pairs --------
 * Returns up to `limit` pairs of distinct words whose vectors are "close".
 * You can choose:
 *   metric: 'cosine' | 'euclidean'   (default 'cosine')
 *   threshold: number                (cosine >= threshold, or euclidean <= threshold)
 *   limit?: number                   (default Infinity)
 *
 * Example:
 *   findClosePairs(dict, { metric: 'cosine', threshold: 0.85, limit: 200 })
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

  // Sort best-first: cosine desc, euclidean asc
  out.sort((x, y) => (useCosine ? y.score - x.score : x.score - y.score));

  return out.slice(0, limit);
};

// distance.js
import { Dict } from "./dictify.js";
import {
  differenceVector,
  l2,
  normalizeVec,
  packVectorsMeanCentered,
  cosineOfDifferenceToDirection,
  percentileCut,
  buildStopwordSet,
} from "./similarity.js";

const [, , a, b] = process.argv;
if (!a || !b) {
  console.error("Usage: yarn distance <wordA> <wordB>");
  process.exit(1);
}

const K = Number(process.env.K || 15);
let PERCENTILE = clamp01(Number(process.env.PERCENTILE ?? 0.08));
let MIN_SIM = Number(process.env.MIN_SIM ?? 0.8);
let MIN_GAP = Number(process.env.MIN_GAP ?? 0.25);
const EXCLUDE_STOPWORDS = process.env.EXCLUDE_STOPWORDS !== "0"; // default on
const NO_CENTER = process.env.NO_CENTER === "1"; // set to 1 to disable mean-centering

const dict = new Dict();
dict.load("model.dat");

// Δ = v(b) - v(a)
const delta = differenceVector(dict, a, b);
const deltaL2 = l2(delta);
if (!Number.isFinite(deltaL2) || deltaL2 === 0) throw new Error("Degenerate Δ");
const uDelta = normalizeVec(delta);

const pack = NO_CENTER ? packVectorsNoCenter : packVectorsMeanCentered;
const { keys, data, unitData, n, d } = pack(dict);

const stop = EXCLUDE_STOPWORDS ? buildStopwordSet() : new Set();
stop.add(a.toLowerCase());
stop.add(b.toLowerCase());

// Projections along ûΔ
const proj = new Float32Array(n);
for (let i = 0; i < n; i++) {
  let s = 0;
  const base = i * d;
  for (let k = 0; k < d; k++) s += unitData[base + k] * uDelta[k];
  proj[i] = s;
}

// Attempt with strict gates; if empty, progressively relax.
const attempt = (percentile, minSim, minGap) => {
  const { lowsIdx, highsIdx, lowThresh, highThresh } = percentileCut(
    proj,
    percentile
  );

  const out = [];
  for (let ii = 0; ii < lowsIdx.length; ii++) {
    const i = lowsIdx[ii];
    const ki = keys[i];
    if (stop.has(ki)) continue;

    for (let jj = 0; jj < highsIdx.length; jj++) {
      const j = highsIdx[jj];
      if (i === j) continue;
      const kj = keys[j];
      if (stop.has(kj)) continue;

      const gap = proj[j] - proj[i];
      if (!(gap >= minGap)) continue;

      const sim = cosineOfDifferenceToDirection(data, d, i, j, uDelta);
      if (!Number.isFinite(sim) || sim < minSim) continue;

      out.push({ a: ki, b: kj, sim, gap });
    }
  }

  out.sort((x, y) => y.sim - x.sim || y.gap - x.gap);
  const seen = new Set();
  const dedup = [];
  for (const r of out) {
    const key = `${r.a}\u0000${r.b}`;
    const rev = `${r.b}\u0000${r.a}`;
    if (seen.has(key) || seen.has(rev)) continue;
    seen.add(key);
    dedup.push(r);
    if (dedup.length >= K) break;
  }
  return {
    pairs: dedup,
    gates: { percentile, minSim, minGap, lowThresh, highThresh },
  };
};

// Progressive relaxation loop
let result = attempt(PERCENTILE, MIN_SIM, MIN_GAP);
let rounds = 0;
while (result.pairs.length === 0 && rounds < 8) {
  // widen candidate bands, lower thresholds gently
  PERCENTILE = Math.min(0.5, PERCENTILE * 1.8 + 0.01);
  MIN_SIM = Math.max(0.4, MIN_SIM - 0.05);
  MIN_GAP = Math.max(0.0, MIN_GAP * 0.8 - 0.02);
  result = attempt(PERCENTILE, MIN_SIM, MIN_GAP);
  rounds++;
}

// Fallback: if still empty, return best MxM by projection with no filters
if (result.pairs.length === 0) {
  const { lowsIdx, highsIdx } = percentileCut(
    proj,
    Math.min(0.5, Math.max(0.02, PERCENTILE))
  );
  const brute = [];
  for (const i of lowsIdx) {
    const ki = keys[i];
    if (stop.has(ki)) continue;
    for (const j of highsIdx) {
      if (i === j) continue;
      const kj = keys[j];
      if (stop.has(kj)) continue;
      const sim = cosineOfDifferenceToDirection(data, d, i, j, uDelta);
      if (!Number.isFinite(sim)) continue;
      brute.push({ a: ki, b: kj, sim, gap: proj[j] - proj[i] });
    }
  }
  brute.sort((x, y) => y.sim - x.sim || y.gap - x.gap);
  result.pairs = brute.slice(0, K);
}

const out = {
  query: { a, b, definition: "vector(b) - vector(a)" },
  dim: d,
  l2: deltaL2,
  similarPairs: result.pairs,
  meta: {
    nWords: n,
    rounds,
    gatesUsed: result.gates,
    params: {
      initial: {
        percentile: Number(process.env.PERCENTILE ?? 0.08),
        minSim: Number(process.env.MIN_SIM ?? 0.8),
        minGap: Number(process.env.MIN_GAP ?? 0.25),
        excludeStopwords: EXCLUDE_STOPWORDS,
        noCenter: NO_CENTER,
      },
      final: {
        percentile: PERCENTILE,
        minSim: MIN_SIM,
        minGap: MIN_GAP,
      },
    },
  },
};

console.log(JSON.stringify(out, null, 2));

function clamp01(x) {
  if (!Number.isFinite(x)) return 0.08;
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

// pack without centering (optional)
function packVectorsNoCenter(dict) {
  const { keys, vecs } = (() => {
    const ks = [];
    const vs = [];
    for (const [k, v] of dict.entries()) {
      if (Array.isArray(v) && v.length > 0 && v.every(Number.isFinite)) {
        ks.push(k);
        vs.push(v);
      }
    }
    return { keys: ks, vecs: vs };
  })();

  const n = vecs.length;
  const d = n ? vecs[0].length : 0;
  const data = new Float32Array(n * d);
  const unitData = new Float32Array(n * d);

  for (let i = 0; i < n; i++) {
    const row = vecs[i];
    const base = i * d;
    let s = 0;
    for (let k = 0; k < d; k++) {
      const v = row[k];
      data[base + k] = v;
      s += v * v;
    }
    const norm = Math.sqrt(s) || 1;
    const inv = 1 / norm;
    for (let k = 0; k < d; k++) unitData[base + k] = data[base + k] * inv;
  }
  return { keys, data, unitData, n, d };
}

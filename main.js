import { Dict } from "./dictify.js";
import { Timer } from "./timer.js";
import fs from "fs";
import { Vector } from "./vector.js";
import readline from "readline";
import ProgressBar from "progress";

try {
  fs.rmSync("out", { recursive: true });
} catch (e) {}
try {
  fs.mkdirSync("out");
} catch (e) {}

const SOURCE = "test.txt";
const dict = new Dict();
const t = new Timer();

const sanitizeWords = (dirtyWords) => {
  const clean = [];
  for (let word of dirtyWords) {
    if (!word) continue;
    const w = word.toLowerCase().replace(/[^a-z]/gi, "");
    const splitWords = w.split("-");
    clean.push(...splitWords);
  }
  return clean;
};

const getVecArray = (token) => {
  if (!token) return null;
  const v = dict.get(token);
  return Array.isArray(v) && v.length > 0 ? v : null;
};

t.start();

// Read data into dictionary
await new Promise((resolve, reject) => {
  const stream = fs.createReadStream(SOURCE);

  stream.on("data", (chunk) => {
    const lines = chunk.toString().split("\n");
    for (const line of lines) {
      const words = line.split(" ");
      const sanitized = sanitizeWords(words);
      for (const w of sanitized) {
        dict.add(w);
      }
    }
  });

  stream.on("end", () => {
    t.log("Reading file");
    resolve();
  });

  stream.on("error", reject);
});

await dict.flush("out/-1.dat");

const LEARNING_RATE = 0.001;
const EPOCHS = 100;
const WINDOW_SIZE = 5;

// Start training
const bar = new ProgressBar("Training [:bar] :percent :etas", {
  complete: "=",
  incomplete: " ",
  width: 40,
  total: EPOCHS,
});
t.start();

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  const rl = readline.createInterface({
    input: fs.createReadStream(SOURCE, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    if (!line) continue;

    const sanitized = sanitizeWords(line.split(" "));
    for (let w = 0; w < sanitized.length; w++) {
      // bounds check for context window
      const w0 = sanitized[w];
      const l1 = sanitized[w - 1];
      const l2 = sanitized[w - 2];
      const r1 = sanitized[w + 1];
      const r2 = sanitized[w + 2];

      const v0 = getVecArray(w0);
      const vl1 = getVecArray(l1);
      const vl2 = getVecArray(l2);
      const vr1 = getVecArray(r1);
      const vr2 = getVecArray(r2);

      // skip positions where any vector is missing
      if (!v0 || !vl1 || !vl2 || !vr1 || !vr2) continue;

      const wordVector = new Vector(v0);
      const lneighbor1 = new Vector(vl1);
      const lneighbor2 = new Vector(vl2);
      const rneighbor1 = new Vector(vr1);
      const rneighbor2 = new Vector(vr2);

      const deltaL1 = wordVector.difference(lneighbor1).multiply(LEARNING_RATE);
      const deltaL2 = wordVector.difference(lneighbor2).multiply(LEARNING_RATE);
      const deltaR1 = wordVector.difference(rneighbor1).multiply(LEARNING_RATE);
      const deltaR2 = wordVector.difference(rneighbor2).multiply(LEARNING_RATE);

      // Update neighbors toward center
      dict.set(l1, lneighbor1.add(deltaL1).normalize().values);
      dict.set(l2, lneighbor2.add(deltaL2).normalize().values);
      dict.set(r1, rneighbor1.add(deltaR1).normalize().values);
      dict.set(r2, rneighbor2.add(deltaR2).normalize().values);

      // Update center toward each neighbor — opposite direction
      const newCenter = wordVector
        .subtract(deltaL1)
        .subtract(deltaL2)
        .subtract(deltaR1)
        .subtract(deltaR2)
        .normalize();
      dict.set(w0, newCenter.values);
    }
  }

  bar.tick();
  // dict.flush(`out/${epoch + 1}.dat`);
}
t.log("Training Complete");

await dict.flush("model.dat");

import fs from "fs";
import readline from "readline";
import ProgressBar from "progress";

export const sampleLines = async (inputFile, outputFile, count = 10000) => {
  const lines = [];

  // First, count total lines
  const total = await new Promise((resolve) => {
    let c = 0;
    const rl = readline.createInterface({
      input: fs.createReadStream(inputFile),
      crlfDelay: Infinity,
    });
    rl.on("line", () => c++);
    rl.on("close", () => resolve(c));
  });

  const bar = new ProgressBar("Reading [:bar] :percent :etas", {
    total,
    width: 30,
  });

  const rl = readline.createInterface({
    input: fs.createReadStream(inputFile),
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    lines.push(line);
    bar.tick();
  }

  const selected = [];
  const used = new Set();
  const pickBar = new ProgressBar("Sampling [:bar] :percent :etas", {
    total: count,
    width: 30,
  });

  while (selected.length < count && lines.length > 0) {
    const idx = Math.floor(Math.random() * lines.length);
    if (!used.has(idx)) {
      used.add(idx);
      selected.push(lines[idx]);
      pickBar.tick();
    }
  }

  fs.writeFileSync(outputFile, selected.join("\n"));
  console.log(`\nSaved ${count} random lines to ${outputFile}`);
};

// Example usage:
await sampleLines("wikisent2.txt", "miniwiki.txt", 100000);

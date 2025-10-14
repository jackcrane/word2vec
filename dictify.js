// dictify.js
import fs from "fs";
import chalk from "chalk";

/**
 * case-insensitive (always-lowercase), ordered dictionary with tsv (de)serialization.
 * - all keys are lowercased on add/set/get/has/load.
 * - insertion order is preserved by Map.
 * - file format: one entry per line => <escapedLowerKey>\t<JSON value>
 */
export class Dict {
  constructor() {
    // Map<lowerKey, { key: string, value: any }>
    this._map = new Map();
  }

  // --- utils ---
  _norm = (key) => String(key).toLowerCase();

  _escapeKey = (s) =>
    String(s)
      .toLowerCase()
      .replace(/\\/g, "\\\\")
      .replace(/\t/g, "\\t")
      .replace(/\n/g, "\\n");

  _unescapeKey = (s) =>
    String(s)
      .toLowerCase()
      .replace(/\\n/g, "\n")
      .replace(/\\t/g, "\t")
      .replace(/\\\\/g, "\\");

  _random = (max = 0.01, min = -0.01) => min + (max - min) * Math.random();

  // ensure all existing vectors gain a trailing 0 when a new word is added,
  // and the new word's vector length matches.
  add = (word, v) => {
    const k = this._norm(word);
    if (this._map.has(k)) return;

    // extend all existing vectors by one zero (to keep uniform length)
    for (const entry of this._map.values()) {
      if (Array.isArray(entry.value)) entry.value.push(this._random());
    }

    const newLen = this.length() + 1; // previous size + 1

    let value = v;
    if (value === undefined) {
      value = Array.from({ length: newLen }, () => this._random());
      value[newLen - 1] = 1; // one-hot at the new index
    } else if (Array.isArray(value) && value.length < newLen) {
      // minimally pad user-provided vector if shorter
      value = value.slice();
      while (value.length < newLen) value.push(this._random());
    }

    this._map.set(k, { key: k, value });
  };

  has = (word) => this._map.has(this._norm(word));

  get = (word) => {
    const e = this._map.get(this._norm(word));
    return e ? e.value : undefined;
  };

  set = (word, value) => {
    const k = this._norm(word);
    const existing = this._map.get(k);
    if (existing) {
      existing.value = value;
    } else {
      this._map.set(k, { key: k, value });
    }
  };

  // 0-based insertion index, or -1 if not present
  find = (word) => {
    const k = this._norm(word);
    if (!this._map.has(k)) return -1;
    let i = 0;
    for (const key of this._map.keys()) {
      if (key === k) return i;
      i++;
    }
    return -1;
  };

  length = () => this._map.size;

  // ordered snapshots
  entries = () => Array.from(this._map.values(), (e) => [e.key, e.value]);
  keys = () => Array.from(this._map.values(), (e) => e.key);
  values = () => Array.from(this._map.values(), (e) => e.value);

  // --- persistence (same tsv format & semantics) ---
  flush = (filename = "dict.dat") => {
    return new Promise((resolve, reject) => {
      const stream = fs.createWriteStream(filename, { encoding: "utf8" });
      stream.on("error", reject);
      stream.on("finish", resolve);

      for (const { key, value } of this._map.values()) {
        const keyOut = this._escapeKey(key);
        const json = value === undefined ? "null" : JSON.stringify(value);
        stream.write(`${keyOut}\t${json}\n`);
      }

      stream.end();
    });
  };

  load = (filename = "dict.dat") => {
    if (!fs.existsSync(filename)) return;
    const data = fs.readFileSync(filename, "utf8");
    this._map.clear();

    for (const rawLine of data.split("\n")) {
      if (!rawLine) continue;

      const tabPos = rawLine.indexOf("\t");
      let keyPart, valPart;
      if (tabPos === -1) {
        keyPart = rawLine;
        valPart = "null";
      } else {
        keyPart = rawLine.slice(0, tabPos);
        valPart = rawLine.slice(tabPos + 1);
      }

      const lowerKey = this._unescapeKey(keyPart);
      let value;
      try {
        value = JSON.parse(valPart);
      } catch {
        value = valPart;
      }

      if (!this._map.has(lowerKey)) {
        this._map.set(lowerKey, { key: lowerKey, value });
      }
    }
  };

  // --- logging ---
  log = (key) => {
    const k = this._norm(key);
    const e = this._map.get(k);
    if (!e) {
      console.log(chalk.red(`"${k}" not found`));
      return;
    }

    const isVector = Array.isArray(e.value);
    const rendered =
      e.value === null
        ? "null"
        : typeof e.value === "object"
        ? JSON.stringify(e.value)
        : String(e.value);

    console.log(
      `${chalk.blueBright(k)}: ${
        isVector ? chalk.green("<vector> ") : ""
      }${chalk.yellow(rendered)}`
    );
  };
}

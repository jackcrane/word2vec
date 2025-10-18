// query.js
import { nearestNeighbors } from "./similarity.js";
import { Dict } from "./dictify.js";
import { Vector } from "./vector.js";

const [, , queryWord] = process.argv;

if (!queryWord) {
  console.error("Usage: yarn query <word>");
  process.exit(1);
}

const dict = new Dict();
dict.load("model.dat");

const nn = nearestNeighbors(dict, { words: [queryWord] });
console.log(nn[0]);

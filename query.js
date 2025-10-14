import { nearestNeighbors } from "./similarity.js";
import { Dict } from "./dictify.js";
import { Vector } from "./vector.js";

const dict = new Dict();
dict.load("model.dat");

const nn = nearestNeighbors(dict, { words: ["sherlock"] });
console.log(nn[0]);

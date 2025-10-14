import * as tf from "@tensorflow/tfjs";

// create two random vectors
const v1 = tf.randomUniform([100]);
const v2 = tf.randomUniform([100]);

// cosine similarity
const cosineSim = tf.dot(v1, v2).div(tf.norm(v1).mul(tf.norm(v2)));

cosineSim.print(); // value between 0 and 1

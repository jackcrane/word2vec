export class Vector {
  constructor(values) {
    if (!Array.isArray(values)) {
      throw new Error(
        "Vector expects an array of numbers. Received: " + values
      );
    }
    // Coerce to numbers and validate
    this._values = values.map((v) => Number(v));
    if (
      this._values.length === 0 ||
      this._values.some((v) => Number.isNaN(v))
    ) {
      throw new Error("Vector contains non-numeric or empty values");
    }
  }

  difference(other) {
    if (this._values.length !== other._values.length) {
      throw new Error("Vectors must be of the same length");
    }
    const out = new Array(this._values.length);
    for (let i = 0; i < this._values.length; i++) {
      out[i] = this._values[i] - other._values[i];
    }
    return new Vector(out);
  }

  multiply(scalar) {
    const out = this._values.map((v) => v * scalar);
    return new Vector(out);
  }

  add(other) {
    const out = this._values.map((v, i) => v + other._values[i]);
    return new Vector(out);
  }

  subtract(other) {
    const out = this._values.map((v, i) => v - other._values[i]);
    return new Vector(out);
  }

  log() {
    console.log(this._values);
  }

  values() {
    return this._values;
  }

  get values() {
    return this._values;
  }

  normalize() {
    const norm = Math.sqrt(this._values.reduce((s, v) => s + v * v, 0));
    if (norm === 0) return new Vector(this._values); // Avoid divide-by-zero
    return new Vector(this._values.map((v) => v / norm));
  }
}

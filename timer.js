import chalk from "chalk";

export class Timer {
  constructor() {
    this._start = performance.now();
  }

  start() {
    this._start = performance.now();
  }

  stop() {
    const end = performance.now();
    return end - this._start;
  }

  log(message, clear = true) {
    console.log(
      `${message?.length > 0 ? chalk.red(message) : ""}${
        message?.length > 0 ? ": " : ""
      }Elapsed time: ${this.stop()} ms`
    );
    clear && this.start();
  }
}

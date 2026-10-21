// Generate text through the WASM kernel from the command line.
//   node gen.mjs "It was a dark" 120 0.85 60

import fs from "node:fs";

const wasmBytes = fs.readFileSync("build/model.wasm");
const { instance } = await WebAssembly.instantiate(wasmBytes, {
  env: { abort: () => { throw new Error("abort"); } },
});
const ex = instance.exports;
ex.init();

const buf = fs.readFileSync("model.bin");
const weights = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
new Float32Array(ex.memory.buffer, ex.weightsPtr(), weights.length).set(weights);

const prompt = process.argv[2] ?? "";
const n = Number(process.argv[3] ?? 120);
const T = Number(process.argv[4] ?? 0.85);
const K = Number(process.argv[5] ?? 60);

function sample(ptr) {
  const lg = new Float32Array(ex.memory.buffer, ptr, 256);
  let max = -Infinity;
  for (let i = 0; i < 256; i++) if (lg[i] > max) max = lg[i];
  const p = new Float32Array(256);
  let s = 0;
  for (let i = 0; i < 256; i++) { p[i] = Math.exp((lg[i] - max) / T); s += p[i]; }
  for (let i = 0; i < 256; i++) p[i] /= s;
  const keep = new Set([...p.keys()].sort((a, b) => p[b] - p[a]).slice(0, K));
  let s2 = 0;
  for (let i = 0; i < 256; i++) { if (keep.has(i)) s2 += p[i]; else p[i] = 0; }
  for (let i = 0; i < 256; i++) p[i] /= s2;
  let r = Math.random();
  for (let i = 0; i < 256; i++) { r -= p[i]; if (r <= 0) return i; }
  return 255;
}

ex.reset();
const bytes = prompt ? new TextEncoder().encode(prompt) : new Uint8Array([10]);
let ptr = 0;
for (const b of bytes) ptr = ex.forward(b);

const out = [];
const t0 = performance.now();
for (let i = 0; i < n; i++) { const t = sample(ptr); out.push(t); ptr = ex.forward(t); }
const secs = (performance.now() - t0) / 1000;

process.stdout.write(new TextDecoder().decode(new Uint8Array(out)) + "\n");
console.error(`[${n} tok in ${secs.toFixed(2)}s = ${(n / secs).toFixed(0)} tok/s]`);

// Parity test: run the WASM kernel over the same prompt PyTorch used and
// compare the final logits.
//
//   python export_weights.py --ckpt checkpoints/model.pt
//   python parity_check.py   --ckpt checkpoints/model.pt
//   npm test

import fs from "node:fs";

const wasmBytes = fs.readFileSync("build/model.wasm");
const { instance } = await WebAssembly.instantiate(wasmBytes, {
  env: { abort: () => { throw new Error("wasm abort"); } },
});
const ex = instance.exports;

ex.init();

const buf = fs.readFileSync("model.bin");
const weights = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const view = () => new Float32Array(ex.memory.buffer, ex.weightsPtr(), weights.length);
view().set(weights);

const expected = JSON.parse(fs.readFileSync("expected.json", "utf8"));

ex.reset();
let ptr = 0;
for (const t of expected.tokens) ptr = ex.forward(t);

const got = new Float32Array(ex.memory.buffer, ptr, 256);

let maxAbs = 0;
let maxIdx = 0;
for (let i = 0; i < 256; i++) {
  const d = Math.abs(got[i] - expected.logits[i]);
  if (d > maxAbs) { maxAbs = d; maxIdx = i; }
}

const argmaxPy = expected.logits.indexOf(Math.max(...expected.logits));
const argmaxJs = got.indexOf(Math.max(...got));

console.log(`tokens      ${expected.tokens.length}`);
console.log(`max |diff|  ${maxAbs.toExponential(3)}  (index ${maxIdx})`);
console.log(`argmax      py=${argmaxPy} wasm=${argmaxJs} ${argmaxPy === argmaxJs ? "MATCH" : "MISMATCH"}`);
console.log(`top5 py     ${[...expected.logits.keys()].sort((a,b)=>expected.logits[b]-expected.logits[a]).slice(0,5)}`);
console.log(`top5 wasm   ${[...got.keys()].sort((a,b)=>got[b]-got[a]).slice(0,5)}`);

if (maxAbs > 1e-3) {
  console.error("FAIL: logits diverge");
  process.exit(1);
}
console.log("PASS");

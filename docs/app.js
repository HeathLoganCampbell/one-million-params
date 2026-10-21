// Runs the 1M-parameter transformer entirely in WebAssembly.
// The wasm module holds the forward pass; this file handles sampling and UI.

const $ = (id) => document.getElementById(id);
const transcript = $("transcript");
const promptEl = $("prompt");
const sendBtn = $("send");
const stopBtn = $("stop");
const statusEl = $("status");

let ex = null;          // wasm exports
let meta = null;        // model.json
let maxT = 256;
let running = false;
let abort = false;

const BLURB = "Trained on 508 public-domain novels plus 17 machine-learning papers.";

// Sampling settings are fixed in the UI now.
const TEMPERATURE = 0.85;
const TOP_K = 60;
const MAX_TOKENS = 120;
const LOADER_MS = 7000;

// ---------------------------------------------------------------- loading
async function loadWeights() {
  const res = await fetch("model.bin");
  if (!res.ok) throw new Error(`model.bin missing (${res.status})`);
  const buf = await res.arrayBuffer();
  const weights = new Float32Array(buf);
  if (weights.length !== meta.params) {
    throw new Error(`expected ${meta.params} weights, got ${weights.length}`);
  }
  new Float32Array(ex.memory.buffer, ex.weightsPtr(), weights.length).set(weights);
  ex.reset();
  return buf.byteLength;
}

function addIntro(bytes) {
  addTurn(
    "bot",
    `Model loaded — ${meta.params.toLocaleString()} parameters, ` +
    `${(bytes / 1e6).toFixed(2)} MB, running locally.\n\n` +
    `${BLURB} It continues your text; it doesn't answer questions.`
  );
}

// Drives the progress bar for a fixed duration. Uses a timer rather than
// requestAnimationFrame because rAF is paused in background tabs, which would
// leave the overlay stuck at 0% forever.
function runLoader(ms) {
  const bar = $("loadbar");
  const pct = $("loadpct");
  const started = performance.now();

  return new Promise((resolve) => {
    const timer = setInterval(() => {
      const t = Math.min((performance.now() - started) / ms, 1);
      const p = Math.round(t * 100);
      bar.style.width = p + "%";
      bar.parentElement.setAttribute("aria-valuenow", p);
      pct.textContent = p + "%";
      if (t >= 1) {
        clearInterval(timer);
        resolve();
      }
    }, 50);
  });
}

async function boot() {
  const loader = $("loader");
  try {
    const work = (async () => {
      meta = await (await fetch("model.json")).json();

      const wasmBytes = await (await fetch("build/model.wasm")).arrayBuffer();
      const { instance } = await WebAssembly.instantiate(wasmBytes, {
        env: { abort: () => { throw new Error("wasm aborted"); } },
      });
      ex = instance.exports;
      ex.init();
      maxT = ex.contextLen();

      return await loadWeights();
    })();

    const [bytes] = await Promise.all([work, runLoader(LOADER_MS)]);

    loader.classList.add("gone");
    setTimeout(() => loader.remove(), 400);

    $("specs").textContent =
      `${meta.layers}L · ${meta.heads}H · ${meta.dim}d · ` +
      `${meta.params.toLocaleString()} params`;
    statusEl.textContent = `ready · context ${maxT} bytes`;
    sendBtn.disabled = false;
    addIntro(bytes);
  } catch (e) {
    // Never leave the overlay covering a page that failed to initialise.
    loader.classList.add("gone");
    setTimeout(() => loader.remove(), 400);
    statusEl.textContent = "failed to load: " + e.message;
    console.error(e);
  }
}

// ---------------------------------------------------------------- sampling
function softmaxSample(ptr, temperature, topK) {
  const logits = new Float32Array(ex.memory.buffer, ptr, 256);
  const V = logits.length;

  if (temperature <= 0.01) {
    let best = 0;
    for (let i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }

  let max = -Infinity;
  for (let i = 0; i < V; i++) if (logits[i] > max) max = logits[i];

  const probs = new Float32Array(V);
  let sum = 0;
  for (let i = 0; i < V; i++) {
    const p = Math.exp((logits[i] - max) / temperature);
    probs[i] = p;
    sum += p;
  }
  for (let i = 0; i < V; i++) probs[i] /= sum;

  if (topK > 0 && topK < V) {
    const ranked = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
    const keep = new Set(ranked.slice(0, topK));
    let s = 0;
    for (let i = 0; i < V; i++) {
      if (keep.has(i)) s += probs[i];
      else probs[i] = 0;
    }
    for (let i = 0; i < V; i++) probs[i] /= s;
  }

  let r = Math.random();
  for (let i = 0; i < V; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return V - 1;
}

// ---------------------------------------------------------------- generate
async function generate(text) {
  running = true;
  abort = false;
  sendBtn.disabled = true;
  stopBtn.hidden = false;

  const temperature = TEMPERATURE;
  const topK = TOP_K;
  const budget = Math.min(MAX_TOKENS, maxT);

  const youTurn = addTurn("you", text);
  const botTurn = addTurn("bot", "");
  const body = botTurn.querySelector(".body");
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  body.appendChild(cursor);

  const decoder = new TextDecoder();
  let promptBytes = text.length ? new TextEncoder().encode(text) : new Uint8Array([10]);
  if (promptBytes.length >= maxT) promptBytes = promptBytes.slice(-maxT + 1);

  ex.reset();
  let ptr = 0;
  for (const b of promptBytes) ptr = ex.forward(b);

  let emitted = 0;
  const t0 = performance.now();
  let lastYield = t0;

  while (emitted < budget && ex.getPos() < maxT && !abort) {
    const tok = softmaxSample(ptr, temperature, topK);
    const chunk = decoder.decode(new Uint8Array([tok]), { stream: true });
    if (chunk) body.insertBefore(document.createTextNode(chunk), cursor);
    emitted++;
    ptr = ex.forward(tok);

    const now = performance.now();
    if (now - lastYield > 40) {
      lastYield = now;
      transcript.scrollTop = transcript.scrollHeight;
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  const tail = decoder.decode();
  if (tail) body.insertBefore(document.createTextNode(tail), cursor);
  cursor.remove();

  const secs = (performance.now() - t0) / 1000;
  statusEl.textContent =
    `${emitted} tokens in ${secs.toFixed(1)}s · ${(emitted / secs).toFixed(1)} tok/s · ` +
    `context ${ex.getPos()}/${maxT}`;
  if (ex.getPos() >= maxT) {
    statusEl.textContent += " · window full, press Continue to restart";
  }

  running = false;
  sendBtn.disabled = false;
  stopBtn.hidden = true;
  transcript.scrollTop = transcript.scrollHeight;
  return youTurn;
}

// ---------------------------------------------------------------- UI
function addTurn(kind, text) {
  const el = document.createElement("div");
  el.className = "turn " + kind;
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = kind === "you" ? "you" : "model";
  const bodyEl = document.createElement("div");
  bodyEl.className = "body";
  bodyEl.textContent = text;
  el.append(who, bodyEl);
  transcript.appendChild(el);
  transcript.scrollTop = transcript.scrollHeight;
  return el;
}

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  if (!ex || running) return;
  const text = promptEl.value.trim();
  if (!text) return;
  promptEl.value = "";          // clear the box immediately on submit
  generate(text);
  promptEl.focus();
});

$("stop").addEventListener("click", () => { abort = true; });

promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("composer").requestSubmit();
  }
});

sendBtn.disabled = true;
boot();

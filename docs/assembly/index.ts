// Inference kernel for the 1M-parameter byte-level transformer.
//
// Compiled to WebAssembly with assemblyscript. Dimensions are compile-time
// constants so every buffer is statically sized and no allocator work happens
// during inference.
//
// Weight layout (float32, little-endian) must match export_weights.py exactly:
//
//   tok_emb   [V, D]
//   per layer:
//     norm1   [D]
//     qkv     [3D, D]
//     proj    [D, D]
//     norm2   [D]
//     w1      [MLP, D]
//     w2      [D, MLP]
//     w3      [MLP, D]
//   norm_f    [D]

const D: i32 = 128;
const H: i32 = 4;
const HD: i32 = 32;   // D / H
const HD2: i32 = 16;  // HD / 2
const NL: i32 = 5;
const V: i32 = 256;
const MLP: i32 = 336;
const MAXT: i32 = 256;

const NWEIGHTS: i32 = 1006976;

// ---- weight offsets ---------------------------------------------------------
const LAYER_SIZE: i32 = D + 3 * D * D + D * D + D + MLP * D + D * MLP + MLP * D;

const O_NORM1: i32 = 0;
const O_QKV: i32 = D;
const O_PROJ: i32 = O_QKV + 3 * D * D;
const O_NORM2: i32 = O_PROJ + D * D;
const O_W1: i32 = O_NORM2 + D;
const O_W2: i32 = O_W1 + MLP * D;
const O_W3: i32 = O_W2 + D * MLP;

const O_TOK: i32 = 0;
const O_LAYER0: i32 = V * D;
const O_NORMF: i32 = O_LAYER0 + NL * LAYER_SIZE;

// ---- storage ----------------------------------------------------------------
const W = new Float32Array(NWEIGHTS);

const x = new Float32Array(D);
const xn = new Float32Array(D);
const tmp = new Float32Array(D);
const q = new Float32Array(D);
const k = new Float32Array(D);
const v = new Float32Array(D);
const attn = new Float32Array(D);
const h1 = new Float32Array(MLP);
const h3 = new Float32Array(MLP);
const logits = new Float32Array(V);
const scores = new Float32Array(MAXT);

const kcache = new Float32Array(NL * MAXT * D);
const vcache = new Float32Array(NL * MAXT * D);

const ropeCos = new Float32Array(MAXT * HD2);
const ropeSin = new Float32Array(MAXT * HD2);

let pos: i32 = 0;

// ---- math helpers -----------------------------------------------------------
@inline
function expf(x: f32): f32 {
  return f32(Math.exp(f64(x)));
}

@inline
function sqrtf(x: f32): f32 {
  return f32(Math.sqrt(f64(x)));
}

@inline
function matvec(wOff: i32, src: Float32Array, dst: Float32Array, rows: i32, cols: i32): void {
  for (let r: i32 = 0; r < rows; r++) {
    let acc: f32 = 0.0;
    const base: i32 = wOff + r * cols;
    for (let c: i32 = 0; c < cols; c++) {
      acc += W[base + c] * src[c];
    }
    dst[r] = acc;
  }
}

@inline
function rmsnorm(src: Float32Array, dst: Float32Array, wOff: i32): void {
  let ss: f32 = 0.0;
  for (let i: i32 = 0; i < D; i++) {
    ss += src[i] * src[i];
  }
  const inv: f32 = 1.0 / sqrtf(ss / f32(D) + f32(1e-6));
  for (let i: i32 = 0; i < D; i++) {
    dst[i] = src[i] * inv * W[wOff + i];
  }
}

@inline
function applyRope(vec: Float32Array): void {
  const rowBase: i32 = pos * HD2;
  for (let h: i32 = 0; h < H; h++) {
    const b: i32 = h * HD;
    for (let i: i32 = 0; i < HD2; i++) {
      const c: f32 = ropeCos[rowBase + i];
      const s: f32 = ropeSin[rowBase + i];
      const a: f32 = vec[b + 2 * i];
      const d: f32 = vec[b + 2 * i + 1];
      vec[b + 2 * i] = a * c - d * s;
      vec[b + 2 * i + 1] = a * s + d * c;
    }
  }
}

function attend(layer: i32): void {
  const len: i32 = pos + 1;
  const scale: f32 = 1.0 / sqrtf(f32(HD));
  for (let h: i32 = 0; h < H; h++) {
    const b: i32 = h * HD;

    let maxv: f32 = -1e30;
    for (let t: i32 = 0; t < len; t++) {
      const kb: i32 = (layer * MAXT + t) * D + b;
      let s: f32 = 0.0;
      for (let i: i32 = 0; i < HD; i++) {
        s += q[b + i] * kcache[kb + i];
      }
      s *= scale;
      scores[t] = s;
      if (s > maxv) maxv = s;
    }

    let sum: f32 = 0.0;
    for (let t: i32 = 0; t < len; t++) {
      const e: f32 = expf(scores[t] - maxv);
      scores[t] = e;
      sum += e;
    }
    const invSum: f32 = 1.0 / sum;

    for (let i: i32 = 0; i < HD; i++) {
      let acc: f32 = 0.0;
      for (let t: i32 = 0; t < len; t++) {
        acc += scores[t] * invSum * vcache[(layer * MAXT + t) * D + b + i];
      }
      attn[b + i] = acc;
    }
  }
}

// ---- public API -------------------------------------------------------------
export function init(): void {
  for (let p: i32 = 0; p < MAXT; p++) {
    for (let i: i32 = 0; i < HD2; i++) {
      const invFreq: f64 = 1.0 / Math.pow(10000.0, f64(2 * i) / f64(HD));
      const angle: f64 = f64(p) * invFreq;
      ropeCos[p * HD2 + i] = f32(Math.cos(angle));
      ropeSin[p * HD2 + i] = f32(Math.sin(angle));
    }
  }
}

export function weightsPtr(): usize {
  return W.dataStart;
}

export function logitsPtr(): usize {
  return logits.dataStart;
}

export function reset(): void {
  pos = 0;
}

export function getPos(): i32 {
  return pos;
}

export function contextLen(): i32 {
  return MAXT;
}

export function forward(tok: i32): usize {
  if (pos >= MAXT) pos = 0; // caller is responsible for sliding the window

  const eb: i32 = tok * D;
  for (let i: i32 = 0; i < D; i++) {
    x[i] = W[eb + i];
  }

  for (let l: i32 = 0; l < NL; l++) {
    const lo: i32 = O_LAYER0 + l * LAYER_SIZE;

    rmsnorm(x, xn, lo + O_NORM1);
    matvec(lo + O_QKV, xn, q, D, D);
    matvec(lo + O_QKV + D * D, xn, k, D, D);
    matvec(lo + O_QKV + 2 * D * D, xn, v, D, D);
    applyRope(q);
    applyRope(k);

    const cOff: i32 = (l * MAXT + pos) * D;
    for (let i: i32 = 0; i < D; i++) {
      kcache[cOff + i] = k[i];
      vcache[cOff + i] = v[i];
    }

    attend(l);

    matvec(lo + O_PROJ, attn, tmp, D, D);
    for (let i: i32 = 0; i < D; i++) {
      x[i] += tmp[i];
    }

    rmsnorm(x, xn, lo + O_NORM2);
    matvec(lo + O_W1, xn, h1, MLP, D);
    matvec(lo + O_W3, xn, h3, MLP, D);
    for (let i: i32 = 0; i < MLP; i++) {
      const a: f32 = h1[i];
      h1[i] = (a / (1.0 + expf(-a))) * h3[i];
    }
    matvec(lo + O_W2, h1, tmp, D, MLP);
    for (let i: i32 = 0; i < D; i++) {
      x[i] += tmp[i];
    }
  }

  rmsnorm(x, xn, O_NORMF);
  matvec(O_TOK, xn, logits, V, D);

  pos++;
  return logits.dataStart;
}
